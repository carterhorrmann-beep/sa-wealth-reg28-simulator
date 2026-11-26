"""Post-retirement Living Annuity drawdown & ruin engine.

A South African **Living Annuity** keeps the retiree's capital invested and pays an income of ``x%`` of the
capital value, with ``x`` chosen on every policy anniversary inside the statutory band of **2.5% to 17.5%**.
Because the income is a *percentage of what is left*, the capital can never be exhausted arithmetically; the
real risk is that the income the retiree needs outgrows what the band allows, or that the capital is run so low
that it no longer supports a meaningful income. This module quantifies both with the same Cholesky-correlated
Monte Carlo (6 ZAR asset classes + stochastic SA CPI) that powers the accumulation engine.

Model (annual decisions, monthly cash flows)
--------------------------------------------
* On each anniversary the retiree wants ``D_y = 12 x desired real monthly income x CPI_y`` (the real income
  escalates with the *realised* stochastic CPI path).
* The drawdown rate is ``clip(D_y / K_y, 2.5%, 17.5%)`` where ``K_y`` is the capital on that anniversary, and the
  income ``rate x K_y`` is paid in 12 equal instalments at the start of each month.
* The portfolio is a constant mix (rebalanced monthly, no transaction costs) and may carry an annual fee.
* Living Annuities are **exempt from Regulation 28**, so the offshore allocation is not capped at 45%.

Metrics
-------
``cap breach``  the first year in which the desired income would need more than 17.5% of capital. From then on
                the income is clamped at 17.5% and shrinks in real terms.
``ruin``        the first anniversary on which capital covers less than ``ruin_multiple`` years of the desired
                income (default one year). A literal "capital reaches zero" can never happen in a Living Annuity,
                so this is the practical definition of a depleted fund.
``survival``    the share of paths with no ruin up to an age (retirement age + years).
``bequest``     real (today's rand) capital left at the end of the horizon.

Income is quoted before income tax; estate duty, fees beyond the optional flat fee, and the retiree's other
income are not modelled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import MarketAssumptions, default_assumptions, weights_from_mapping
from .montecarlo import MarketPaths, StressScenario, generate_paths, percentile_frame
from .reg28_tax import STATIC_PRESETS, check_reg28

#: Statutory Living Annuity drawdown band (fraction of capital per year).
LA_MIN_RATE = 0.025
LA_MAX_RATE = 0.175

#: Portfolios offered to a retiree. Only the first two respect Regulation 28; a Living Annuity does not have to.
LA_PORTFOLIOS: dict[str, dict[str, float]] = {
    "Conservative (Reg 28)": STATIC_PRESETS["Conservative (Reg 28)"],
    "Moderate (Reg 28)": STATIC_PRESETS["Moderate (Reg 28)"],
    "Aggressive Offshore (no Reg 28 cap)": STATIC_PRESETS["Offshore Growth (unconstrained)"],
}

#: A -25% equity shock (JSE and global equity, in ZAR; listed property -20%) in the first 12 months of retirement.
RETIREMENT_CRASH = StressScenario(
    key="retirement_crash",
    name="Year-1 Retirement Equity Crash",
    description=(
        "A -25% shock to JSE and global equity (in ZAR) and -20% to listed property hits in the first 12 months of "
        "retirement, with equity volatility 50% higher: the worst moment for a retiree who is drawing an income."
    ),
    anchor="start",
    start_year=0,
    window_years=1,
    shocks={"JSE_EQ": -0.25, "GLB_EQ": -0.25, "SA_PROP": -0.20},
    vol_mult={"JSE_EQ": 1.5, "GLB_EQ": 1.5, "SA_PROP": 1.5},
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def clamp_drawdown_rate(rate, min_rate: float = LA_MIN_RATE, max_rate: float = LA_MAX_RATE):
    """Force a desired drawdown rate into the statutory band (works on scalars and arrays; ``inf`` -> max)."""
    return np.clip(rate, min_rate, max_rate)


@dataclass(frozen=True)
class DrawdownConfig:
    """A retiree's Living Annuity plan plus simulation settings.

    ``monthly_income_real`` is the desired income in today's rand; it escalates with simulated CPI.
    ``horizon_years`` is the retirement length (25 to 35 years is typical; 5 to 40 is accepted).
    """

    initial_capital: float = 8_000_000.0
    monthly_income_real: float = 40_000.0
    horizon_years: int = 30
    retirement_age: int = 65
    n_paths: int = 5_000
    seed: int = 42
    min_rate: float = LA_MIN_RATE
    max_rate: float = LA_MAX_RATE
    annual_fee: float = 0.0
    ruin_multiple: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.monthly_income_real < 0:
            raise ValueError("monthly_income_real must be non-negative")
        if not 5 <= int(self.horizon_years) <= 40:
            raise ValueError("horizon_years must be between 5 and 40")
        if self.n_paths < 100:
            raise ValueError("n_paths must be at least 100")
        if not 0.0 <= self.min_rate < self.max_rate <= 1.0:
            raise ValueError("need 0 <= min_rate < max_rate <= 1")
        if not 0.0 <= self.annual_fee < 0.05:
            raise ValueError("annual_fee must be a fraction between 0 and 0.05")
        if self.ruin_multiple <= 0:
            raise ValueError("ruin_multiple must be positive")

    @property
    def n_months(self) -> int:
        return int(self.horizon_years) * 12

    @property
    def end_age(self) -> int:
        return int(self.retirement_age) + int(self.horizon_years)

    @property
    def initial_withdrawal_rate(self) -> float:
        """Desired first-year income as a fraction of the starting capital."""
        return 12.0 * self.monthly_income_real / self.initial_capital

    @property
    def max_initial_monthly_income(self) -> float:
        """Largest first-year monthly income the statutory ceiling allows on the starting capital."""
        return self.max_rate * self.initial_capital / 12.0


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def resolve_la_portfolio(
    portfolio: str | Mapping[str, float] | Sequence[float],
    assets: Sequence[str],
) -> tuple[str, np.ndarray]:
    """Turn a portfolio name (``"moderate"``, ``"aggressive-offshore"``...), mapping or vector into weights."""
    if isinstance(portfolio, str):
        key = _slug(portfolio)
        names = list(LA_PORTFOLIOS) + [n for n in STATIC_PRESETS if n not in LA_PORTFOLIOS]
        matches = [n for n in names if _slug(n) == key] or [n for n in names if _slug(n).startswith(key)] or [
            n for n in names if key in _slug(n)
        ]
        if not matches:
            raise ValueError(f"unknown portfolio '{portfolio}'. Choose from: " + ", ".join(_slug(n) for n in LA_PORTFOLIOS))
        mapping = LA_PORTFOLIOS.get(matches[0]) or STATIC_PRESETS[matches[0]]
        label, w = matches[0], weights_from_mapping(mapping, assets)
    elif isinstance(portfolio, Mapping):
        label, w = "Custom", weights_from_mapping(portfolio, assets)
    else:
        label, w = "Custom", np.asarray(portfolio, dtype=float)
    if w.shape != (len(assets),) or np.any(w < -1e-9) or w.sum() <= 0:
        raise ValueError("portfolio weights must be non-negative, match the asset universe and sum to a positive number")
    return label, w / w.sum()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True along axis 1 as a float array (NaN where there is none)."""
    first = mask.argmax(axis=1).astype(float)
    first[~mask.any(axis=1)] = np.nan
    return first


@dataclass
class DrawdownResult:
    """Path-wise Living Annuity decumulation. Row = path, column = year since retirement."""

    config: DrawdownConfig
    portfolio: str
    weights: np.ndarray
    reg28_compliant: bool
    years: np.ndarray  # 0..H
    capital: np.ndarray  # (P, H+1) nominal capital on each anniversary (column H = end of the horizon)
    capital_real: np.ndarray  # same, in today's rand
    income: np.ndarray  # (P, H) nominal income paid in each year
    income_real: np.ndarray  # (P, H) in today's rand (deflated at the start of the year)
    desired: np.ndarray  # (P, H+1) nominal income the retiree wants in each year (column H = next year)
    rates: np.ndarray  # (P, H) drawdown rate actually applied, always inside [min_rate, max_rate]
    cpi: np.ndarray  # (P, T+1)
    breach_year: np.ndarray  # (P,) first year the desired income needs more than the ceiling (NaN = never)
    ruin_year: np.ndarray  # (P,) first anniversary with capital below ruin_multiple years of income (NaN = never)
    scenario: StressScenario | None = None

    # -- ages --------------------------------------------------------------------- #
    @property
    def ages(self) -> np.ndarray:
        return self.config.retirement_age + self.years

    # -- survival ----------------------------------------------------------------- #
    def survival_curve(self) -> pd.Series:
        """Share of paths that have not been ruined by each age (anniversaries 0..H)."""
        ruined = np.nan_to_num(self.ruin_year, nan=np.inf)
        surv = np.array([np.mean(ruined > y) for y in self.years])
        return pd.Series(surv, index=pd.Index(self.ages, name="age"), name="survival")

    def survival_prob(self, age: int | None = None) -> float:
        """Portfolio survival probability at ``age`` (default: the end of the horizon)."""
        age = self.config.end_age if age is None else int(age)
        if not self.config.retirement_age <= age <= self.config.end_age:
            raise ValueError(f"age must be between {self.config.retirement_age} and {self.config.end_age}")
        return float(self.survival_curve().loc[age])

    def survival_milestones(self, ages: Sequence[int] = (90, 95)) -> dict[int, float]:
        """Survival at ages 90 and 95 (only those inside the simulated horizon)."""
        return {a: self.survival_prob(a) for a in ages if self.config.retirement_age <= a <= self.config.end_age}

    # -- cap breach --------------------------------------------------------------- #
    @property
    def cap_breach_prob(self) -> float:
        """Probability that the desired income breaches the statutory ceiling within the horizon."""
        return float(np.mean(~np.isnan(self.breach_year)))

    @property
    def median_years_to_cap_breach(self) -> float:
        """Median years until the ceiling is breached, among the paths that breach it (NaN if none do)."""
        hit = self.breach_year[~np.isnan(self.breach_year)]
        return float(np.median(hit)) if hit.size else float("nan")

    def cap_breach_curve(self) -> pd.Series:
        """Cumulative probability that the ceiling has been breached by each age."""
        b = np.nan_to_num(self.breach_year, nan=np.inf)
        return pd.Series(
            [np.mean(b <= y) for y in self.years], index=pd.Index(self.ages, name="age"), name="cap_breach"
        )

    # -- wealth & income ---------------------------------------------------------- #
    @property
    def median_bequest_real(self) -> float:
        """Median real capital left at the end of the horizon (all paths)."""
        return float(np.median(self.capital_real[:, -1]))

    @property
    def p10_bequest_real(self) -> float:
        return float(np.percentile(self.capital_real[:, -1], 10))

    def percentiles(self, real: bool = True) -> pd.DataFrame:
        """P10..P90 of capital by year since retirement."""
        return percentile_frame(self.capital_real if real else self.capital, self.years)

    def income_percentiles(self, real: bool = True) -> pd.DataFrame:
        arr = self.income_real if real else self.income
        return percentile_frame(arr, self.years[:-1])

    @property
    def share_years_income_shortfall(self) -> float:
        """Average share of retirement years in which income is below the desired income."""
        return float(np.mean(self.income < self.desired[:, :-1] * (1 - 1e-9)))

    def summary(self) -> dict:
        cfg = self.config
        out = {
            "portfolio": self.portfolio,
            "reg28_compliant": self.reg28_compliant,
            "initial_capital": cfg.initial_capital,
            "monthly_income_real": cfg.monthly_income_real,
            "initial_withdrawal_rate": cfg.initial_withdrawal_rate,
            "horizon_years": cfg.horizon_years,
            "retirement_age": cfg.retirement_age,
            "end_age": cfg.end_age,
            "n_paths": self.capital.shape[0],
            "survival_prob": self.survival_prob(),
            "median_bequest_real": self.median_bequest_real,
            "p10_bequest_real": self.p10_bequest_real,
            "cap_breach_prob": self.cap_breach_prob,
            "median_years_to_cap_breach": self.median_years_to_cap_breach,
            "share_years_income_shortfall": self.share_years_income_shortfall,
            "median_income_real_final_year": float(np.median(self.income_real[:, -1])),
        }
        for age, p in self.survival_milestones().items():
            out[f"survival_age_{age}"] = p
        return out


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def run_drawdown(
    config: DrawdownConfig,
    assumptions: MarketAssumptions | None = None,
    portfolio: str | Mapping[str, float] | Sequence[float] = "Moderate (Reg 28)",
    paths: MarketPaths | None = None,
    scenario: StressScenario | None = None,
) -> DrawdownResult:
    """Simulate a Living Annuity over ``config.horizon_years`` on Cholesky-correlated Monte Carlo paths.

    Pass pre-generated ``paths`` to reuse the same random numbers (e.g. base vs crash); ``scenario`` applies a
    deterministic stress overlay on top of the random draw.
    """
    assumptions = default_assumptions() if assumptions is None else assumptions
    label, w = resolve_la_portfolio(portfolio, assumptions.assets)
    if paths is None:
        paths = generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed, scenario)
    elif paths.horizon_years != config.horizon_years:
        raise ValueError("paths were generated for a different horizon")
    elif scenario is None:
        scenario = paths.scenario

    h = config.horizon_years
    p = paths.n_paths
    cpi = paths.cpi
    port = np.asarray(paths.returns @ w.astype(paths.returns.dtype), dtype=float)  # (P, T) constant-mix return
    growth = (1.0 + port) * (1.0 - config.annual_fee) ** (1.0 / 12.0)

    cap = np.empty((p, h + 1))
    income = np.zeros((p, h))
    rates = np.zeros((p, h))
    desired = 12.0 * config.monthly_income_real * cpi[:, 12 * np.arange(h + 1)]  # (P, H+1)
    k = np.full(p, float(config.initial_capital))
    cap[:, 0] = k
    for y in range(h):
        want = np.divide(desired[:, y], k, out=np.full(p, np.inf), where=k > 0)
        rate = clamp_drawdown_rate(want, config.min_rate, config.max_rate)
        pay = rate * k / 12.0
        rates[:, y] = rate
        income[:, y] = rate * k
        for m in range(12):
            k = np.maximum(k - pay, 0.0) * growth[:, 12 * y + m]
        cap[:, y + 1] = k

    cpi_year = cpi[:, 12 * np.arange(h + 1)]
    want_rate = np.divide(desired[:, :h], cap[:, :h], out=np.full((p, h), np.inf), where=cap[:, :h] > 0)
    breach = _first_true(want_rate > config.max_rate + 1e-12)
    ruin = _first_true(cap < config.ruin_multiple * desired)
    return DrawdownResult(
        config=config,
        portfolio=label,
        weights=w,
        reg28_compliant=check_reg28(w, assets=assumptions.assets).compliant,
        years=np.arange(h + 1),
        capital=cap,
        capital_real=cap / cpi_year,
        income=income,
        income_real=income / cpi_year[:, :h],
        desired=desired,
        rates=rates,
        cpi=cpi,
        breach_year=breach,
        ruin_year=ruin,
        scenario=scenario,
    )


@dataclass
class CrashImpact:
    """Base run vs the same random numbers with a -25% equity shock in year 1 of retirement."""

    base: DrawdownResult
    crash: DrawdownResult
    scenario: StressScenario

    @property
    def table(self) -> pd.DataFrame:
        b, c = self.base, self.crash
        rows = {
            "Survival probability (end of horizon)": (b.survival_prob(), c.survival_prob()),
            "Median real bequest (today's rand)": (b.median_bequest_real, c.median_bequest_real),
            "P10 real bequest (today's rand)": (b.p10_bequest_real, c.p10_bequest_real),
            "P(income breaches 17.5% ceiling)": (b.cap_breach_prob, c.cap_breach_prob),
            "Median years to ceiling breach": (b.median_years_to_cap_breach, c.median_years_to_cap_breach),
            "Median real income, final year": (
                float(np.median(b.income_real[:, -1])),
                float(np.median(c.income_real[:, -1])),
            ),
        }
        df = pd.DataFrame(rows, index=["Base", self.scenario.name]).T
        df["Change"] = df[self.scenario.name] - df["Base"]
        return df

    def formatted_table(self, money: Callable[[float], str]) -> pd.DataFrame:
        """:attr:`table` as display strings: percentages, years or ``money(x)`` depending on the row."""

        def fmt(row: str, v: float, delta: bool = False) -> str:
            if v != v:
                return "n/a" if delta else "never"
            if row.startswith(("Survival", "P(income")):
                return f"{v * 100:+.1f} pp" if delta else f"{v:.1%}"
            if row.startswith("Median years"):
                return f"{v:+.0f}" if delta else f"{v:.0f}"
            return money(v)

        t = self.table
        out = pd.DataFrame(index=t.index, columns=t.columns, dtype=object)
        for row in t.index:
            for col in t.columns:
                out.loc[row, col] = fmt(row, t.loc[row, col], delta=(col == "Change"))
        return out

    @property
    def survival_drop(self) -> float:
        """Fall in survival probability caused by the crash (percentage points as a fraction)."""
        return self.base.survival_prob() - self.crash.survival_prob()

    @property
    def bequest_drop_pct(self) -> float:
        b = self.base.median_bequest_real
        return (self.crash.median_bequest_real - b) / b if b > 0 else float("nan")


def run_crash_impact(
    config: DrawdownConfig,
    assumptions: MarketAssumptions | None = None,
    portfolio: str | Mapping[str, float] | Sequence[float] = "Moderate (Reg 28)",
    scenario: StressScenario = RETIREMENT_CRASH,
) -> CrashImpact:
    """Sequence-of-returns test: re-run the plan on the same random numbers with the year-1 equity crash."""
    assumptions = default_assumptions() if assumptions is None else assumptions
    base = run_drawdown(config, assumptions, portfolio)
    paths = generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed, scenario)
    crash = run_drawdown(config, assumptions, portfolio, paths=paths, scenario=scenario)
    return CrashImpact(base=base, crash=crash, scenario=scenario)


def withdrawal_rate_grid(
    config: DrawdownConfig,
    rates: Sequence[float] = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08),
    assumptions: MarketAssumptions | None = None,
    portfolio: str | Mapping[str, float] | Sequence[float] = "Moderate (Reg 28)",
) -> pd.DataFrame:
    """Survival, bequest and ceiling-breach risk for a range of initial withdrawal rates (same random numbers)."""
    assumptions = default_assumptions() if assumptions is None else assumptions
    paths = generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed)
    rows = []
    for r in rates:
        cfg = replace(config, monthly_income_real=r * config.initial_capital / 12.0)
        s = run_drawdown(cfg, assumptions, portfolio, paths=paths)
        rows.append(
            {
                "initial_rate": r,
                "monthly_income": cfg.monthly_income_real,
                "survival_prob": s.survival_prob(),
                "cap_breach_prob": s.cap_breach_prob,
                "median_bequest_real": s.median_bequest_real,
            }
        )
    return pd.DataFrame(rows).set_index("initial_rate")
