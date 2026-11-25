"""Vectorised Monte Carlo engine and SA macro stress scenarios.

* :func:`generate_paths` draws Cholesky-correlated monthly ZAR returns for the six asset classes
  *and* SA CPI shocks (seeded, in reproducible blocks of paths).
* :func:`simulate` runs a client's plan through a tax wrapper and reports percentile wealth
  trajectories, probability of reaching the goal, shortfall risk (CVaR 5%) and cumulative tax drag.
* :func:`run_tax_comparison` runs RA vs TFSA vs Discretionary on identical paths.
* :func:`run_stress` re-runs the plan, on the same random numbers, under three SA macro scenarios.

Money is in ZAR. "Real" means deflated to today's purchasing power with the simulated CPI path.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .assets import (
    ASSETS,
    INCOME_CLASS,
    MarketAssumptions,
    default_assumptions,
    nearest_psd,
    simulate_inflation,
    weights_from_mapping,
)
from .reg28_tax import (
    DEFAULT_TAX,
    DISC,
    GROSS,
    RA,
    REG28,
    TFSA,
    WRAPPER_NAMES,
    WRAPPERS,
    Reg28Limits,
    TaxParameters,
    WrapperOutcome,
    check_reg28,
    compare_wrappers,
    project_to_reg28,
    simulate_wrapper,
)

PERCENTILES = (10, 25, 50, 75, 90)
PATH_BLOCK = 1000  # paths drawn per RNG block; fixes the random stream independent of n_paths


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimulationConfig:
    """A client plan plus simulation settings.

    ``target_real`` is the retirement nest-egg goal in *today's* rand. ``horizon_years`` must be
    between 5 and 40. ``wrapper`` selects the tax vehicle for :func:`simulate`.
    """

    initial_lump_sum: float = 250_000.0
    monthly_contribution: float = 5_000.0
    horizon_years: int = 25
    target_real: float = 5_000_000.0
    n_paths: int = 10_000
    seed: int = 42
    marginal_rate: float = 0.36
    wrapper: str = RA
    annual_income: float | None = None
    retirement_tax_rate: float | None = None
    tax: TaxParameters = DEFAULT_TAX
    rebalance: bool = True
    reinvest_refund: bool = True
    limits: Reg28Limits = REG28

    def __post_init__(self) -> None:
        if not 5 <= int(self.horizon_years) <= 40:
            raise ValueError("horizon_years must be between 5 and 40")
        if self.n_paths < 100:
            raise ValueError("n_paths must be at least 100")
        if self.initial_lump_sum < 0 or self.monthly_contribution < 0:
            raise ValueError("contributions must be non-negative")
        if not 0.0 <= self.marginal_rate < 0.6:
            raise ValueError("marginal_rate must be a fraction such as 0.36")
        if self.wrapper not in (RA, TFSA, DISC, GROSS):
            raise ValueError(f"Unknown wrapper '{self.wrapper}'")

    @property
    def n_months(self) -> int:
        return int(self.horizon_years) * 12


# --------------------------------------------------------------------------- #
# Stress scenarios
# --------------------------------------------------------------------------- #


def zar_shock(usd_return: float, usdzar_change: float) -> float:
    """ZAR return of a USD asset: ``(1 + r_usd)(1 + d_usdzar) - 1``.

    A -30% USD equity drawdown with a 18% Rand depreciation is only -17.4% in ZAR - the Rand hedge.
    """
    return (1.0 + usd_return) * (1.0 + usdzar_change) - 1.0


@dataclass(frozen=True)
class StressScenario:
    """A deterministic macro overlay on top of the stochastic market model.

    The overlay lives in a window of ``window_years`` starting ``start_year`` years from today
    (``anchor="start"``) or that many years before retirement (``anchor="end"``).

    ``shocks``      total ZAR return shock per asset, spread evenly (in logs) across the window
    ``drift_adj``   extra annual log-return per asset during the window
    ``vol_mult``    volatility multiplier per asset during the window
    ``inflation``   SA CPI inflation rate during the window (reverts to the base mean afterwards)
    """

    key: str
    name: str
    description: str
    anchor: str = "start"
    start_year: float = 0.0
    window_years: float = 1.0
    inflation: float | None = None
    shocks: Mapping[str, float] = field(default_factory=dict)
    drift_adj: Mapping[str, float] = field(default_factory=dict)
    vol_mult: Mapping[str, float] = field(default_factory=dict)

    def window(self, horizon_years: int) -> tuple[int, int]:
        t = horizon_years * 12
        start = self.start_year * 12 if self.anchor == "start" else t - self.start_year * 12
        start = int(round(max(0, min(start, t - 1))))
        end = int(round(min(t, start + self.window_years * 12)))
        return start, max(end, start + 1)

    def arrays(self, horizon_years: int, assets: Sequence[str] = ASSETS) -> dict[str, np.ndarray | None]:
        t, a = horizon_years * 12, len(assets)
        s, e = self.window(horizon_years)
        drift = np.zeros((t, a))
        vmult = np.ones((t, a))
        shock = np.zeros((t, a))
        for name, v in self.drift_adj.items():
            drift[s:e, assets.index(name)] = v / 12.0
        for name, v in self.vol_mult.items():
            vmult[s:e, assets.index(name)] = v
        for name, total in self.shocks.items():
            shock[s:e, assets.index(name)] = np.log1p(total) / (e - s)
        return {"drift_adj": drift, "vol_mult": vmult, "shock_log": shock}

    def inflation_path(self, base_mean: float, horizon_years: int) -> np.ndarray:
        path = np.full(horizon_years * 12, base_mean)
        if self.inflation is not None:
            s, e = self.window(horizon_years)
            path[s:e] = self.inflation
        return path


STAGFLATION = StressScenario(
    key="stagflation",
    name="SA Stagflation & Fiscal Slippage",
    description=(
        "CPI inflation of 8.5% for the first 7 years; JSE equities and the ALBI de-rate (-6% / -3.5% p.a. vs base), "
        "listed property -6% p.a., cash benefits from higher rates (+1.5% p.a.), and the Rand depreciates, lifting "
        "global equity (+4% p.a.) and global bonds (+5% p.a.) in ZAR. SA volatility rises 30%."
    ),
    anchor="start",
    start_year=0,
    window_years=7,
    inflation=0.085,
    drift_adj={"JSE_EQ": -0.06, "SA_BOND": -0.035, "SA_PROP": -0.06, "SA_CASH": 0.015, "GLB_EQ": 0.04, "GLB_BOND": 0.05},
    vol_mult={"JSE_EQ": 1.3, "SA_BOND": 1.3, "SA_PROP": 1.3, "GLB_EQ": 1.2, "GLB_BOND": 1.2},
)

RISK_OFF = StressScenario(
    key="risk_off",
    name="Global Risk-Off & Rand Spike",
    description=(
        "Ten years before retirement, global equities fall 30% in USD while USD/ZAR spikes 18%, so global equity "
        "is only -17.4% in ZAR and global bonds gain +20.4% in ZAR. JSE equities -22%, SA property -25%, ALBI -8%. "
        "Volatility is 50-80% higher for a year and Rand pass-through lifts inflation to 6.5%."
    ),
    anchor="end",
    start_year=10,
    window_years=1,
    inflation=0.065,
    shocks={
        "JSE_EQ": -0.22,
        "SA_PROP": -0.25,
        "SA_BOND": -0.08,
        "GLB_EQ": zar_shock(-0.30, 0.18),
        "GLB_BOND": zar_shock(0.02, 0.18),
    },
    vol_mult={"JSE_EQ": 1.5, "SA_BOND": 1.5, "SA_PROP": 1.5, "GLB_EQ": 1.8, "GLB_BOND": 1.8},
)

SEQUENCE_CRASH = StressScenario(
    key="sequence_crash",
    name="Sequence-of-Returns Crash",
    description=(
        "A -25% equity shock (JSE and global, in ZAR) with listed property -20% hits during the 12 months that "
        "begin 24 months before retirement, leaving only about a year to recover. Equity volatility is 50% higher."
    ),
    anchor="end",
    start_year=2,
    window_years=1,
    shocks={"JSE_EQ": -0.25, "GLB_EQ": -0.25, "SA_PROP": -0.20},
    vol_mult={"JSE_EQ": 1.5, "GLB_EQ": 1.5, "SA_PROP": 1.5},
)

DEFAULT_SCENARIOS: tuple[StressScenario, ...] = (STAGFLATION, RISK_OFF, SEQUENCE_CRASH)


# --------------------------------------------------------------------------- #
# Path generation
# --------------------------------------------------------------------------- #


@dataclass
class MarketPaths:
    """Simulated monthly market data. ``returns`` is ``(P, T, A)`` float32 simple total returns."""

    returns: np.ndarray
    cpi: np.ndarray  # (P, T + 1)
    horizon_years: int
    seed: int | None
    assumptions: MarketAssumptions
    scenario: StressScenario | None = None

    @property
    def n_paths(self) -> int:
        return self.returns.shape[0]

    @property
    def n_months(self) -> int:
        return self.returns.shape[1]

    def annualised_inflation(self) -> np.ndarray:
        return self.cpi[:, -1] ** (1.0 / self.horizon_years) - 1.0


def generate_paths(
    assumptions: MarketAssumptions,
    n_paths: int,
    horizon_years: int,
    seed: int | None = 42,
    scenario: StressScenario | None = None,
) -> MarketPaths:
    """Cholesky-correlated Monte Carlo of monthly ZAR returns and SA CPI.

    Monthly log returns are ``m_t + s_t * z`` with ``z = L @ eps`` (``L`` the Cholesky factor of the
    7x7 correlation of the six assets and the CPI shock). The log drift is chosen so that the annual
    *arithmetic* mean equals the assumed expected return. A stress ``scenario`` adds deterministic
    drift/shock/volatility overlays *after* the random draw, so a scenario run uses exactly the same
    random numbers as the base run (common random numbers) and differences are purely the scenario.
    """
    t = horizon_years * 12
    a = len(assumptions.assets)
    chol = np.linalg.cholesky(nearest_psd(assumptions.sim_corr) + 1e-12 * np.eye(a + 1))

    vol_mult = np.ones((t, a))
    drift_adj = np.zeros((t, a))
    shock_log = np.zeros((t, a))
    infl_mean = np.full(t, assumptions.inflation_mean)
    if scenario is not None:
        arr = scenario.arrays(horizon_years, assumptions.assets)
        vol_mult, drift_adj, shock_log = arr["vol_mult"], arr["drift_adj"], arr["shock_log"]
        infl_mean = scenario.inflation_path(assumptions.inflation_mean, horizon_years)
    sigma = assumptions.vol[None, :] * vol_mult  # (T, A) annual
    drift = (np.log1p(assumptions.mu)[None, :] - 0.5 * sigma**2) / 12.0 + drift_adj / 12.0 + shock_log
    sd = sigma / np.sqrt(12.0)

    returns = np.empty((n_paths, t, a), dtype=np.float32)
    cpi = np.empty((n_paths, t + 1))
    n_blocks = int(np.ceil(n_paths / PATH_BLOCK))
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_blocks)]
    for b, rng in enumerate(rngs):
        lo, hi = b * PATH_BLOCK, min((b + 1) * PATH_BLOCK, n_paths)
        eps = rng.standard_normal((hi - lo, t, a + 1))
        z = eps @ chol.T
        returns[lo:hi] = np.expm1(drift[None] + sd[None] * z[:, :, :a]).astype(np.float32)
        _, cpi[lo:hi] = simulate_inflation(z[:, :, a], assumptions, infl_mean)
    return MarketPaths(returns, cpi, horizon_years, seed, assumptions, scenario)


def expected_return_paths(assumptions: MarketAssumptions, horizon_years: int) -> MarketPaths:
    """A single deterministic path: every month earns exactly the expected return, CPI at its mean."""
    t = horizon_years * 12
    r = (1.0 + assumptions.mu) ** (1.0 / 12.0) - 1.0
    returns = np.tile(r, (1, t, 1))
    m = (1.0 + assumptions.inflation_mean) ** (1.0 / 12.0)
    cpi = (m ** np.arange(t + 1))[None, :]
    return MarketPaths(returns, cpi, horizon_years, None, assumptions)


def build_contributions(config: SimulationConfig, cpi: np.ndarray) -> np.ndarray:
    """``(P, T)`` nominal cash flow: monthly contribution stepped up each year with the realised CPI path;
    the initial lump sum is added to month 0."""
    t = config.n_months
    year_start = (np.arange(t) // 12) * 12
    c = config.monthly_contribution * cpi[:, year_start]
    c[:, 0] += config.initial_lump_sum
    return c


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def cvar(values: np.ndarray, alpha: float = 0.05) -> float:
    """Expected value of the worst ``alpha`` share of outcomes (lower tail)."""
    v = np.asarray(values, dtype=float)
    cutoff = np.quantile(v, alpha)
    return float(v[v <= cutoff].mean())


def percentile_frame(values: np.ndarray, years: np.ndarray | None = None) -> pd.DataFrame:
    """P10/P25/P50/P75/P90 across paths for every column (year) of a ``(P, Y+1)`` array."""
    q = np.percentile(values, PERCENTILES, axis=0)
    df = pd.DataFrame(q.T, columns=[f"P{p}" for p in PERCENTILES])
    df.index = np.arange(values.shape[1]) if years is None else years
    df.index.name = "year"
    return df


@dataclass
class SimulationResult:
    config: SimulationConfig
    wrapper: str
    weights_requested: np.ndarray
    weights_used: np.ndarray
    reg28_adjusted: bool
    years: np.ndarray
    nominal: np.ndarray  # (P, Y+1) after-tax value if liquidated / retired at each year end
    real: np.ndarray  # (P, Y+1) same, in today's rand
    terminal_nominal: np.ndarray  # net of exit / liquidation tax
    terminal_real: np.ndarray
    outcome: WrapperOutcome
    gross_terminal_nominal: np.ndarray | None
    gross_terminal_real: np.ndarray | None
    cpi_final: np.ndarray
    assets: tuple[str, ...] = ASSETS

    # -- trajectories --------------------------------------------------------- #
    def percentiles(self, real: bool = True) -> pd.DataFrame:
        """Percentile trajectories of after-tax wealth (value kept if liquidated / retired that year)."""
        return percentile_frame(self.real if real else self.nominal, self.years)

    # -- metrics ---------------------------------------------------------------- #
    @property
    def prob_goal(self) -> float:
        return float(np.mean(self.terminal_real >= self.config.target_real))

    @property
    def cvar5_real(self) -> float:
        return cvar(self.terminal_real, 0.05)

    @property
    def cvar5_nominal(self) -> float:
        return cvar(self.terminal_nominal, 0.05)

    @property
    def expected_shortfall_vs_goal(self) -> float:
        """Average real shortfall below the goal (zero when the goal is met), in today's rand."""
        return float(np.mean(np.maximum(self.config.target_real - self.terminal_real, 0.0)))

    @property
    def cumulative_tax_drag(self) -> float:
        """Mean nominal rand of tax paid over the plan (income tax, CGT and RA exit tax)."""
        return float(self.outcome.tax_paid.mean())

    @property
    def cumulative_tax_drag_real(self) -> float:
        return float(self.outcome.tax_paid_real.mean())

    def summary(self) -> dict:
        tr, tn = self.terminal_real, self.terminal_nominal
        out = {
            "wrapper": WRAPPER_NAMES[self.wrapper],
            "horizon_years": self.config.horizon_years,
            "n_paths": int(tr.size),
            "median_terminal_nominal": float(np.median(tn)),
            "median_terminal_real": float(np.median(tr)),
            "mean_terminal_real": float(tr.mean()),
            "p10_terminal_real": float(np.percentile(tr, 10)),
            "p90_terminal_real": float(np.percentile(tr, 90)),
            "target_real": self.config.target_real,
            "prob_goal": self.prob_goal,
            "var5_real": float(np.quantile(tr, 0.05)),
            "cvar5_real": self.cvar5_real,
            "cvar5_nominal": self.cvar5_nominal,
            "expected_shortfall_vs_goal_real": self.expected_shortfall_vs_goal,
            "cumulative_tax_drag_nominal": self.cumulative_tax_drag,
            "cumulative_tax_drag_real": self.cumulative_tax_drag_real,
            "ra_refunds_nominal": float(self.outcome.refunds.mean()),
            "mean_contributions_nominal": float(self.outcome.contributions_total.mean()),
            "mean_annual_inflation": float(np.mean(self.cpi_final ** (1.0 / self.config.horizon_years) - 1.0)),
        }
        if self.gross_terminal_real is not None:
            out["median_gross_terminal_real"] = float(np.median(self.gross_terminal_real))
            out["median_wealth_lost_to_tax_real"] = float(np.median(self.gross_terminal_real - self.terminal_real))
        return out


def _prepare_weights(
    weights: Sequence[float] | Mapping[str, float], wrapper: str, limits: Reg28Limits, assets: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, bool]:
    w = weights_from_mapping(weights, assets)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    w = w / w.sum()
    if wrapper == RA and not check_reg28(w, limits, assets).compliant:
        return w, project_to_reg28(w, limits, assets), True
    return w, w, False


def simulate(
    config: SimulationConfig,
    weights: Sequence[float] | Mapping[str, float],
    assumptions: MarketAssumptions | None = None,
    paths: MarketPaths | None = None,
    with_gross: bool = True,
) -> SimulationResult:
    """Run the Monte Carlo for one allocation and one wrapper.

    For ``wrapper="RA"`` a non-compliant allocation is replaced by its closest Regulation 28
    projection (``reg28_adjusted=True`` on the result). Pass pre-generated ``paths`` to reuse the
    same random numbers across several runs.
    """
    assumptions = assumptions or default_assumptions()
    w_req, w_used, adjusted = _prepare_weights(weights, config.wrapper, config.limits, assumptions.assets)
    if paths is None:
        paths = generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed)
    elif paths.horizon_years != config.horizon_years:
        raise ValueError("paths were generated for a different horizon")
    contrib = build_contributions(config, paths.cpi)
    income_class = tuple(INCOME_CLASS.get(a, "local_interest") for a in assumptions.assets)
    common = dict(
        returns=paths.returns,
        cpi=paths.cpi,
        contributions=contrib,
        income_yield=assumptions.income_yield,
        income_class=income_class,
        tax=config.tax,
        marginal_rate=config.marginal_rate,
        annual_income=config.annual_income,
        retirement_tax_rate=config.retirement_tax_rate,
        rebalance=config.rebalance,
        reinvest_refund=config.reinvest_refund,
        limits=config.limits,
        assets=assumptions.assets,
    )
    outcome = simulate_wrapper(weights=w_used, wrapper=config.wrapper, **common)
    gross = simulate_wrapper(weights=w_used, wrapper=GROSS, **common) if with_gross and config.wrapper != GROSS else None
    defl = paths.cpi[:, :: 12]
    return SimulationResult(
        config=config,
        wrapper=config.wrapper,
        weights_requested=w_req,
        weights_used=w_used,
        reg28_adjusted=adjusted,
        years=np.arange(config.horizon_years + 1),
        nominal=outcome.values_liq,
        real=outcome.values_liq / defl,
        terminal_nominal=outcome.terminal_net,
        terminal_real=outcome.terminal_net / paths.cpi[:, -1],
        outcome=outcome,
        gross_terminal_nominal=None if gross is None else gross.terminal_net,
        gross_terminal_real=None if gross is None else gross.terminal_net / paths.cpi[:, -1],
        cpi_final=paths.cpi[:, -1],
        assets=tuple(assumptions.assets),
    )


# --------------------------------------------------------------------------- #
# Tax wrapper comparison
# --------------------------------------------------------------------------- #


@dataclass
class TaxComparison:
    config: SimulationConfig
    outcomes: dict[str, WrapperOutcome]
    cpi: np.ndarray
    weights_unconstrained: np.ndarray
    weights_ra: np.ndarray
    summary: pd.DataFrame

    def curves(self, real: bool = True, q: float = 50.0) -> pd.DataFrame:
        """Percentile (default median) after-tax value of each wrapper by year (+ pre-tax reference)."""
        defl = self.cpi[:, :: 12] if real else 1.0
        cols = {}
        for key in (RA, TFSA, DISC, GROSS):
            cols[WRAPPER_NAMES[key]] = np.percentile(self.outcomes[key].values_liq / defl, q, axis=0)
        df = pd.DataFrame(cols)
        df.index = np.arange(self.config.horizon_years + 1)
        df.index.name = "year"
        return df


def _tax_summary(config: SimulationConfig, outcomes: dict[str, WrapperOutcome], cpi: np.ndarray) -> pd.DataFrame:
    cf = cpi[:, -1]
    disc = outcomes[DISC]
    disc_real = disc.terminal_net / cf
    rows = []
    for key in (RA, TFSA, DISC):
        o = outcomes[key]
        net_real = o.terminal_net / cf
        cost_real = o.tax_paid_real - o.refunds_real
        rows.append(
            {
                "wrapper": WRAPPER_NAMES[key],
                "key": key,
                "median_terminal_nominal": float(np.median(o.terminal_net)),
                "median_terminal_real": float(np.median(net_real)),
                "p10_terminal_real": float(np.percentile(net_real, 10)),
                "p90_terminal_real": float(np.percentile(net_real, 90)),
                "tax_paid_nominal": float(o.tax_paid.mean()),
                "tax_paid_real": float(o.tax_paid_real.mean()),
                "ra_refunds_real": float(o.refunds_real.mean()),
                "net_tax_cost_nominal": float(o.net_tax_cost.mean()),
                "net_tax_cost_real": float(cost_real.mean()),
                # "Rands saved in tax": extra after-tax terminal wealth versus the taxable account. Because
                # tax also erodes *compounding*, this is larger than the difference in cash taxes paid.
                "tax_saved_vs_disc_nominal": float((o.terminal_net - disc.terminal_net).mean()),
                "tax_saved_vs_disc_real": float((net_real - disc_real).mean()),
                "median_gain_vs_disc_real": float(np.median(net_real - disc_real)),
                "share_paths_beating_disc": float(np.mean(net_real > disc_real)),
            }
        )
    df = pd.DataFrame(rows).set_index("key")
    if outcomes[TFSA].tfsa_balance is not None:
        df.loc[TFSA, "tfsa_balance_median_real"] = float(np.median(outcomes[TFSA].tfsa_balance / cf))
        df.loc[TFSA, "overflow_balance_median_real"] = float(np.median(outcomes[TFSA].overflow_balance / cf))
        df.loc[TFSA, "tfsa_contributed_nominal"] = float(outcomes[TFSA].tfsa_contributed.mean())
    df["median_pretax_real"] = float(np.median(outcomes[GROSS].terminal_net / cf))
    return df


def run_tax_comparison(
    config: SimulationConfig,
    weights: Sequence[float] | Mapping[str, float],
    assumptions: MarketAssumptions | None = None,
    paths: MarketPaths | None = None,
) -> TaxComparison:
    """RA vs TFSA vs Discretionary on identical market paths.

    TFSA and Discretionary hold ``weights`` unconstrained; the RA holds ``weights`` if they comply with
    Regulation 28 and otherwise their closest compliant projection.
    """
    assumptions = assumptions or default_assumptions()
    w = weights_from_mapping(weights, assumptions.assets)
    w = w / w.sum()
    if paths is None:
        paths = generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed)
    contrib = build_contributions(config, paths.cpi)
    income_class = tuple(INCOME_CLASS.get(a, "local_interest") for a in assumptions.assets)
    outcomes = compare_wrappers(
        paths.returns,
        paths.cpi,
        w,
        contrib,
        assumptions.income_yield,
        income_class,
        tax=config.tax,
        marginal_rate=config.marginal_rate,
        annual_income=config.annual_income,
        retirement_tax_rate=config.retirement_tax_rate,
        limits=config.limits,
        rebalance=config.rebalance,
        reinvest_refund=config.reinvest_refund,
        assets=assumptions.assets,
    )
    return TaxComparison(
        config=config,
        outcomes=outcomes,
        cpi=paths.cpi,
        weights_unconstrained=w,
        weights_ra=outcomes[RA].weights,
        summary=_tax_summary(config, outcomes, paths.cpi),
    )


# --------------------------------------------------------------------------- #
# Stress tests
# --------------------------------------------------------------------------- #


@dataclass
class StressResult:
    table: pd.DataFrame
    results: dict[str, SimulationResult]
    scenarios: tuple[StressScenario, ...]


def run_stress(
    config: SimulationConfig,
    weights: Sequence[float] | Mapping[str, float],
    assumptions: MarketAssumptions | None = None,
    scenarios: Sequence[StressScenario] = DEFAULT_SCENARIOS,
) -> StressResult:
    """Terminal wealth under the SA macro stress scenarios versus the base case.

    Every scenario uses the same seed as the base run, so ``change_vs_base`` isolates the scenario.
    The ``table`` has one row per scenario with median / P10 real terminal wealth, probability of
    reaching the goal, CVaR 5% and the change in median wealth versus the base case.
    """
    assumptions = assumptions or default_assumptions()
    base_paths = generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed)
    base = simulate(config, weights, assumptions, base_paths)
    results: dict[str, SimulationResult] = {"base": base}
    for sc in scenarios:
        sp = generate_paths(assumptions, config.n_paths, config.horizon_years, config.seed, scenario=sc)
        results[sc.key] = simulate(config, weights, assumptions, sp, with_gross=False)
    base_med = float(np.median(base.terminal_real))
    rows = []
    names = {"base": "Base case (no stress)"} | {s.key: s.name for s in scenarios}
    for key, r in results.items():
        tr = r.terminal_real
        rows.append(
            {
                "key": key,
                "scenario": names[key],
                "median_real": float(np.median(tr)),
                "p10_real": float(np.percentile(tr, 10)),
                "mean_real": float(tr.mean()),
                "median_nominal": float(np.median(r.terminal_nominal)),
                "prob_goal": r.prob_goal,
                "cvar5_real": r.cvar5_real,
                "change_vs_base": float(np.median(tr)) / base_med - 1.0,
                "mean_annual_inflation": float(np.mean(r.cpi_final ** (1.0 / config.horizon_years) - 1.0)),
            }
        )
    return StressResult(pd.DataFrame(rows).set_index("key"), results, tuple(scenarios))


def retirement_horizon(current_age: int, retirement_age: int) -> int:
    """Years to retirement, validated to the simulator's 5-40 year range."""
    h = int(retirement_age) - int(current_age)
    if not 5 <= h <= 40:
        raise ValueError(f"Horizon of {h} years is outside the supported 5-40 year range")
    return h
