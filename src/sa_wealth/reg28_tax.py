"""Regulation 28 compliance, constrained optimisation and the SA tax-wrapper engine.

Part 1 - **Regulation 28** (Pension Funds Act): a validator and an SLSQP optimiser that
enforce the key prudential limits

* total equity (``JSE_EQ + GLB_EQ``)         <= 75%
* total offshore (``GLB_EQ + GLB_BOND``)     <= 45%
* listed property (``SA_PROP``)              <= 25%
* weights non-negative and summing to 1

and quantify what those limits cost compared with an unconstrained portfolio.

Part 2 - **tax wrappers**: a vectorised, path-wise simulator for a Retirement Annuity (RA),
a Tax-Free Savings Account (TFSA) and a taxable Discretionary account. See the second half
of this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .assets import (
    ASSETS,
    EQUITY_ASSETS,
    FOREIGN_DIVIDEND,
    FOREIGN_INTEREST,
    INCOME_CLASS,
    LOCAL_DIVIDEND,
    LOCAL_INTEREST,
    OFFSHORE_ASSETS,
    PROPERTY_ASSETS,
    PROPERTY_DISTRIBUTION,
    MarketAssumptions,
    weights_from_mapping,
)

# =========================================================================== #
# Part 1 - Regulation 28
# =========================================================================== #


@dataclass(frozen=True)
class Reg28Limits:
    """Regulation 28 asset-class ceilings (fractions of the portfolio).

    A limit set to ``None`` is switched off, which lets us isolate the cost of a single
    cap (e.g. "everything except the offshore cap").
    """

    max_equity: float | None = 0.75
    max_offshore: float | None = 0.45
    max_property: float | None = 0.25

    def replace(self, **kwargs) -> "Reg28Limits":
        return replace(self, **kwargs)


REG28 = Reg28Limits()
NO_LIMITS = Reg28Limits(None, None, None)


def _idx(names: Iterable[str], assets: Sequence[str] = ASSETS) -> list[int]:
    return [assets.index(n) for n in names]


def exposures(weights: Sequence[float], assets: Sequence[str] = ASSETS) -> dict[str, float]:
    """Regulation 28 bucket exposures of a weight vector."""
    w = np.asarray(weights, dtype=float)
    return {
        "equity": float(w[_idx(EQUITY_ASSETS, assets)].sum()),
        "offshore": float(w[_idx(OFFSHORE_ASSETS, assets)].sum()),
        "property": float(w[_idx(PROPERTY_ASSETS, assets)].sum()),
    }


@dataclass(frozen=True)
class Reg28Report:
    """Result of a Regulation 28 compliance check."""

    compliant: bool
    equity: float
    offshore: float
    property: float
    breaches: tuple[str, ...]
    headroom: dict = field(default_factory=dict)

    def badge(self) -> str:
        return "Regulation 28 COMPLIANT" if self.compliant else "NOT Regulation 28 compliant"


def check_reg28(
    weights: Sequence[float],
    limits: Reg28Limits = REG28,
    assets: Sequence[str] = ASSETS,
    tol: float = 1e-6,
) -> Reg28Report:
    """Validate a portfolio against the Regulation 28 limits.

    Also flags negative weights and weights that do not sum to 1 (within ``tol``).
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(assets),):
        raise ValueError(f"Expected {len(assets)} weights, got shape {w.shape}")
    ex = exposures(w, assets)
    breaches: list[str] = []
    if np.any(w < -tol):
        breaches.append("negative weights (short positions are not permitted)")
    if abs(w.sum() - 1.0) > tol:
        breaches.append(f"weights sum to {w.sum():.4f}, not 1.0")
    caps = {"equity": limits.max_equity, "offshore": limits.max_offshore, "property": limits.max_property}
    labels = {"equity": "total equity", "offshore": "total offshore", "property": "listed property"}
    headroom = {}
    for key, cap in caps.items():
        if cap is None:
            continue
        headroom[key] = cap - ex[key]
        if ex[key] > cap + tol:
            breaches.append(f"{labels[key]} {ex[key]:.1%} exceeds the {cap:.0%} limit")
    return Reg28Report(
        compliant=not breaches,
        equity=ex["equity"],
        offshore=ex["offshore"],
        property=ex["property"],
        breaches=tuple(breaches),
        headroom=headroom,
    )


def is_reg28_compliant(weights: Sequence[float], limits: Reg28Limits = REG28, **kw) -> bool:
    return check_reg28(weights, limits, **kw).compliant


def _scipy_constraints(limits: Reg28Limits, assets: Sequence[str] = ASSETS) -> list[dict]:
    """SLSQP constraint dictionaries: budget equality plus the active Reg 28 caps."""
    cons: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}]
    n = len(assets)
    for cap, names in (
        (limits.max_equity, EQUITY_ASSETS),
        (limits.max_offshore, OFFSHORE_ASSETS),
        (limits.max_property, PROPERTY_ASSETS),
    ):
        if cap is None:
            continue
        mask = np.zeros(n)
        mask[_idx(names, assets)] = 1.0
        cons.append(
            {
                "type": "ineq",
                "fun": (lambda w, m=mask, c=cap: c - m @ w),
                "jac": (lambda w, m=mask: -m),
            }
        )
    return cons


def _linear_ineq(limits: Reg28Limits, assets: Sequence[str] = ASSETS) -> tuple[np.ndarray, np.ndarray]:
    rows, rhs = [], []
    n = len(assets)
    for cap, names in (
        (limits.max_equity, EQUITY_ASSETS),
        (limits.max_offshore, OFFSHORE_ASSETS),
        (limits.max_property, PROPERTY_ASSETS),
    ):
        if cap is None:
            continue
        row = np.zeros(n)
        row[_idx(names, assets)] = 1.0
        rows.append(row)
        rhs.append(cap)
    if not rows:
        return np.zeros((0, n)), np.zeros(0)
    return np.vstack(rows), np.array(rhs)


@dataclass
class OptimisationResult:
    """Optimiser output with portfolio statistics and a Regulation 28 report."""

    label: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float
    success: bool
    report: Reg28Report
    limits: Reg28Limits | None
    diversification_ratio: float
    effective_n: float

    def as_row(self) -> dict:
        return {
            "portfolio": self.label,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "equity": self.report.equity,
            "offshore": self.report.offshore,
            "property": self.report.property,
            "diversification_ratio": self.diversification_ratio,
            "effective_n": self.effective_n,
            "reg28_compliant": self.report.compliant,
        }


def portfolio_stats(
    weights: Sequence[float], assumptions: MarketAssumptions, rf: float | None = None
) -> tuple[float, float, float]:
    """Return ``(expected_return, volatility, sharpe)`` of a weight vector."""
    w = np.asarray(weights, dtype=float)
    ret = float(assumptions.mu @ w)
    vol = float(np.sqrt(max(w @ assumptions.cov @ w, 0.0)))
    rf = _risk_free(assumptions, rf)
    return ret, vol, (ret - rf) / vol if vol > 0 else 0.0


def _risk_free(assumptions: MarketAssumptions, rf: float | None) -> float:
    if rf is not None:
        return float(rf)
    return float(assumptions.table.loc["SA_CASH", "expected_return"])


def _clean(w: np.ndarray) -> np.ndarray:
    w = np.where(np.abs(w) < 1e-7, 0.0, w)
    w = np.clip(w, 0.0, None)
    return w / w.sum()


def _make_result(
    label: str, w: np.ndarray, assumptions: MarketAssumptions, limits: Reg28Limits | None, rf: float | None, success: bool
) -> OptimisationResult:
    w = _clean(w)
    ret, vol, sharpe = portfolio_stats(w, assumptions, rf)
    sig = assumptions.vol
    div = float(sig @ w / vol) if vol > 0 else 1.0
    eff = float(1.0 / np.sum(w**2))
    report = check_reg28(w, limits or REG28, assumptions.assets)
    if limits is None:
        # report is informational: how would this portfolio fare against Regulation 28?
        pass
    return OptimisationResult(
        label=label,
        weights=pd.Series(w, index=assumptions.assets, name=label),
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        success=success,
        report=report,
        limits=limits,
        diversification_ratio=div,
        effective_n=eff,
    )


def _starts(n: int, n_starts: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [np.full(n, 1.0 / n)] + [rng.dirichlet(np.ones(n)) for _ in range(n_starts)]


def _feasible(w: np.ndarray, limits: Reg28Limits, assets: Sequence[str], tol: float = 1e-6) -> bool:
    return check_reg28(w, limits, assets, tol=tol).compliant


def _solve(fun, starts, constraints, bounds, limits, assets, jac=None) -> tuple[np.ndarray | None, float]:
    best_w, best_f = None, np.inf
    for w0 in starts:
        try:
            r = minimize(
                fun,
                w0,
                jac=jac,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-13},
            )
        except (ValueError, FloatingPointError):  # pragma: no cover - defensive
            continue
        if not np.all(np.isfinite(r.x)):
            continue
        w = np.clip(r.x, 0.0, 1.0)
        w = w / w.sum()
        if _feasible(w, limits, assets) and r.fun < best_f:
            best_w, best_f = w, float(r.fun)
    return best_w, best_f


def optimize_portfolio(
    assumptions: MarketAssumptions,
    objective: str = "max_sharpe",
    limits: Reg28Limits | None = REG28,
    rf: float | None = None,
    target_return: float | None = None,
    exclude: Sequence[str] = (),
    n_starts: int = 8,
    label: str | None = None,
) -> OptimisationResult:
    """Optimise a long-only portfolio with ``scipy.optimize.minimize(method="SLSQP")``.

    Parameters
    ----------
    objective:
        ``"max_sharpe"``, ``"min_variance"`` (optionally subject to ``target_return``),
        ``"risk_parity"`` (equal risk contribution over the non-excluded assets) or
        ``"max_return"``.
    limits:
        :class:`Reg28Limits` to enforce, or ``None`` for an unconstrained
        (long-only, fully invested) portfolio.
    exclude:
        Assets forced to zero weight (risk parity excludes cash by default - see below).
    """
    assets = assumptions.assets
    n = len(assets)
    mu, cov = assumptions.mu, assumptions.cov
    rf_val = _risk_free(assumptions, rf)
    lim = limits if limits is not None else NO_LIMITS
    cons = _scipy_constraints(lim, assets)
    ex_idx = set(_idx(exclude, assets)) if exclude else set()
    bounds = [(0.0, 0.0) if i in ex_idx else (0.0, 1.0) for i in range(n)]
    starts = _starts(n, n_starts)
    for k, s in enumerate(starts):
        s = s.copy()
        s[list(ex_idx)] = 0.0
        starts[k] = s / s.sum()
    name = label or (f"{objective} ({'Reg 28' if limits is not None else 'unconstrained'})")

    if objective == "max_return":
        a_ub, b_ub = _linear_ineq(lim, assets)
        res = linprog(
            -mu,
            A_ub=a_ub if len(b_ub) else None,
            b_ub=b_ub if len(b_ub) else None,
            A_eq=np.ones((1, n)),
            b_eq=[1.0],
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"max_return infeasible: {res.message}")
        return _make_result(name, res.x, assumptions, limits, rf, True)

    if objective == "max_sharpe":

        def fun(w):
            v = w @ cov @ w
            return -(mu @ w - rf_val) / np.sqrt(max(v, 1e-14))

        w, _ = _solve(fun, starts, cons, bounds, lim, assets)
    elif objective == "min_variance":
        c = list(cons)
        if target_return is not None:
            c.append({"type": "ineq", "fun": lambda w, t=target_return: mu @ w - t, "jac": lambda w: mu})
        w, _ = _solve(lambda w: 1e4 * (w @ cov @ w), starts, c, bounds, lim, assets, jac=lambda w: 2e4 * (cov @ w))
    elif objective == "risk_parity":
        active = [i for i in range(n) if i not in ex_idx]
        k = len(active)

        def fun(w):
            m = cov @ w
            var = w @ m
            rc = w * m / max(var, 1e-14)
            return 1e3 * float(np.sum((rc[active] - 1.0 / k) ** 2))

        w, _ = _solve(fun, starts, cons, bounds, lim, assets)
    else:
        raise ValueError(f"Unknown objective '{objective}'")

    if w is None:
        raise RuntimeError(f"Optimiser failed to find a feasible solution for objective '{objective}'")
    if target_return is not None and objective == "min_variance" and mu @ w < target_return - 1e-6:
        raise RuntimeError(f"target return {target_return:.2%} is not attainable under the given constraints")
    return _make_result(name, w, assumptions, limits, rf, True)


def risk_parity_weights(assumptions: MarketAssumptions, limits: Reg28Limits | None = None) -> OptimisationResult:
    """Equal-risk-contribution portfolio over the risky assets (cash, with ~0.7% volatility,
    would otherwise absorb the whole book, so it is excluded from the risk budget)."""
    return optimize_portfolio(
        assumptions, "risk_parity", limits, exclude=("SA_CASH",), label=f"Risk Parity ({'Reg 28' if limits else 'unconstrained'})"
    )


def project_to_reg28(
    weights: Sequence[float], limits: Reg28Limits = REG28, assets: Sequence[str] = ASSETS
) -> np.ndarray:
    """Closest Regulation-28-compliant portfolio (least squares) to ``weights``.

    Used when a client's preferred allocation breaches the limits but they still want to
    hold the retirement-annuity wrapper: it keeps the allocation as close as possible to
    their intent. Compliant inputs are returned unchanged (after normalisation).
    """
    w0 = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if w0.sum() <= 0:
        raise ValueError("weights must contain a positive allocation")
    w0 = w0 / w0.sum()
    if _feasible(w0, limits, assets):
        return w0
    n = len(assets)
    r = minimize(
        lambda w: np.sum((w - w0) ** 2),
        w0,
        jac=lambda w: 2 * (w - w0),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=_scipy_constraints(limits, assets),
        options={"maxiter": 500, "ftol": 1e-14},
    )
    w = _clean(r.x)
    if not _feasible(w, limits, assets, tol=1e-6):  # pragma: no cover - defensive
        raise RuntimeError("could not project weights onto the Regulation 28 region")
    return w


def efficient_frontier(
    assumptions: MarketAssumptions, limits: Reg28Limits | None = REG28, n_points: int = 40
) -> pd.DataFrame:
    """Long-only efficient frontier (minimum variance for each target return).

    Returns a DataFrame with ``return``, ``volatility``, ``sharpe`` and one weight column
    per asset, sorted by return.
    """
    lo = optimize_portfolio(assumptions, "min_variance", limits)
    hi = optimize_portfolio(assumptions, "max_return", limits)
    targets = np.linspace(lo.expected_return, hi.expected_return, n_points)
    rows = []
    for t in targets:
        try:
            r = optimize_portfolio(assumptions, "min_variance", limits, target_return=float(t) - 1e-9, n_starts=2)
        except RuntimeError:  # pragma: no cover - numerically at the edge
            continue
        row = {"return": r.expected_return, "volatility": r.volatility, "sharpe": r.sharpe}
        row.update(r.weights.to_dict())
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("return").reset_index(drop=True)
    # keep only the efficient (upper) branch: volatility must be non-decreasing in return
    df = df[df["volatility"] >= df["volatility"].cummax() - 1e-9].reset_index(drop=True)
    return df


def _nonneg(x: float) -> float:
    """Clip round-off noise below zero (NaN passes through)."""
    return x if np.isnan(x) else max(x, 0.0)


def reg28_tradeoff(
    assumptions: MarketAssumptions, limits: Reg28Limits = REG28, rf: float | None = None
) -> dict[str, pd.DataFrame]:
    """Quantify what Regulation 28 costs, and which cap is responsible.

    Returns a dictionary of DataFrames

    ``portfolios``  max-Sharpe and risk-parity portfolios: unconstrained vs Reg 28 vs each cap alone
    ``cap_by_cap``  max-Sharpe with one cap removed at a time (marginal cost of each cap)
    ``frontier_gap`` minimum volatility needed for a set of target returns, with and without Reg 28
    ``offshore_cap`` frontier gap attributable to the offshore cap alone
    """
    unc = optimize_portfolio(assumptions, "max_sharpe", None, rf, label="Max-Sharpe - unconstrained")
    reg = optimize_portfolio(assumptions, "max_sharpe", limits, rf, label="Max-Sharpe - Regulation 28")
    rp_unc = risk_parity_weights(assumptions, None)
    rp_reg = risk_parity_weights(assumptions, limits)
    portfolios = pd.DataFrame([r.as_row() for r in (unc, reg, rp_unc, rp_reg)]).set_index("portfolio")

    variants = {
        "Reg 28 (all caps)": limits,
        "Reg 28 without offshore cap": limits.replace(max_offshore=None),
        "Reg 28 without equity cap": limits.replace(max_equity=None),
        "Reg 28 without property cap": limits.replace(max_property=None),
        "Offshore cap only": Reg28Limits(None, limits.max_offshore, None),
        "Equity cap only": Reg28Limits(limits.max_equity, None, None),
    }
    rows = []
    for name, lim in variants.items():
        r = optimize_portfolio(assumptions, "max_sharpe", lim, rf, label=name)
        rows.append(
            {
                "variant": name,
                "sharpe": r.sharpe,
                "sharpe_cost_vs_unconstrained": max(unc.sharpe - r.sharpe, 0.0),
                "expected_return": r.expected_return,
                "volatility": r.volatility,
                "offshore": r.report.offshore,
                "equity": r.report.equity,
            }
        )
    cap_by_cap = pd.DataFrame(rows).set_index("variant")

    reg_hi = optimize_portfolio(assumptions, "max_return", limits)
    unc_hi = optimize_portfolio(assumptions, "max_return", None)
    reg_lo = optimize_portfolio(assumptions, "min_variance", limits)
    targets = np.linspace(max(reg_lo.expected_return, unc.expected_return), unc_hi.expected_return, 7)
    ex_off = limits.replace(max_offshore=None)
    gaps = []
    for t in targets:
        row = {"target_return": float(t)}
        for tag, lim in (("unconstrained", None), ("reg28", limits), ("reg28_no_offshore_cap", ex_off)):
            try:
                r = optimize_portfolio(assumptions, "min_variance", lim, target_return=float(t) - 1e-9, n_starts=3)
                row[f"vol_{tag}"] = r.volatility
                row[f"offshore_{tag}"] = r.report.offshore
            except RuntimeError:
                row[f"vol_{tag}"] = np.nan
                row[f"offshore_{tag}"] = np.nan
        row["vol_cost_reg28"] = _nonneg(row["vol_reg28"] - row["vol_unconstrained"])
        row["vol_cost_offshore_cap"] = _nonneg(row["vol_reg28"] - row["vol_reg28_no_offshore_cap"])
        gaps.append(row)
    frontier_gap = pd.DataFrame(gaps).set_index("target_return")
    frontier_gap.attrs["max_return_unconstrained"] = unc_hi.expected_return
    frontier_gap.attrs["max_return_reg28"] = reg_hi.expected_return
    return {"portfolios": portfolios, "cap_by_cap": cap_by_cap, "frontier_gap": frontier_gap}


def offshore_sensitivity(
    assumptions: MarketAssumptions,
    premiums: Sequence[float] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05),
    limits: Reg28Limits = REG28,
) -> pd.DataFrame:
    """When does the 45% offshore cap start to bite?

    Adds an expected-return ``premium`` to both global assets (a "weak Rand" view: ZAR
    depreciation beyond the base case) and re-solves max-Sharpe with and without Regulation 28.
    The Sharpe/return give-up and the unconstrained offshore weight show the point at which the
    cap turns from slack to binding.
    """
    rows = []
    gi = [assumptions.assets.index(a) for a in OFFSHORE_ASSETS]
    for prem in premiums:
        tbl = assumptions.table.copy()
        tbl.iloc[gi, tbl.columns.get_loc("expected_return")] += prem
        a2 = replace(assumptions, table=tbl)
        unc = optimize_portfolio(a2, "max_sharpe", None)
        reg = optimize_portfolio(a2, "max_sharpe", limits)
        rows.append(
            {
                "offshore_premium": prem,
                "offshore_unconstrained": unc.report.offshore,
                "offshore_reg28": reg.report.offshore,
                "sharpe_unconstrained": unc.sharpe,
                "sharpe_reg28": reg.sharpe,
                "sharpe_cost": max(unc.sharpe - reg.sharpe, 0.0),
                "return_cost": max(unc.expected_return - reg.expected_return, 0.0),
                "cap_binding": unc.report.offshore > (limits.max_offshore or 1.0) + 1e-6,
            }
        )
    return pd.DataFrame(rows).set_index("offshore_premium")


#: Static presets (all Regulation 28 compliant unless flagged otherwise).
STATIC_PRESETS: dict[str, dict[str, float]] = {
    "Conservative (Reg 28)": {"JSE_EQ": 0.15, "SA_BOND": 0.35, "SA_PROP": 0.05, "SA_CASH": 0.25, "GLB_EQ": 0.10, "GLB_BOND": 0.10},
    "Moderate (Reg 28)": {"JSE_EQ": 0.30, "SA_BOND": 0.25, "SA_PROP": 0.05, "SA_CASH": 0.05, "GLB_EQ": 0.25, "GLB_BOND": 0.10},
    "Aggressive (Reg 28)": {"JSE_EQ": 0.38, "SA_BOND": 0.10, "SA_PROP": 0.07, "SA_CASH": 0.00, "GLB_EQ": 0.37, "GLB_BOND": 0.08},
    "Offshore Growth (unconstrained)": {"JSE_EQ": 0.15, "SA_BOND": 0.05, "SA_PROP": 0.00, "SA_CASH": 0.00, "GLB_EQ": 0.75, "GLB_BOND": 0.05},
    "JSE Home Bias (unconstrained)": {"JSE_EQ": 0.90, "SA_BOND": 0.00, "SA_PROP": 0.05, "SA_CASH": 0.05, "GLB_EQ": 0.00, "GLB_BOND": 0.00},
}


def portfolio_presets(assumptions: MarketAssumptions, include_optimised: bool = True) -> dict[str, np.ndarray]:
    """Named portfolios: static risk-profile presets plus optimiser-derived ones."""
    out = {k: weights_from_mapping(v, assumptions.assets) for k, v in STATIC_PRESETS.items()}
    if include_optimised:
        out["Max-Sharpe (Reg 28)"] = optimize_portfolio(assumptions, "max_sharpe", REG28).weights.to_numpy()
        out["Risk Parity (Reg 28)"] = risk_parity_weights(assumptions, REG28).weights.to_numpy()
        out["Risk Parity (unconstrained)"] = risk_parity_weights(assumptions, None).weights.to_numpy()
    return out


# =========================================================================== #
# Part 2 - South African tax wrappers
# =========================================================================== #

RA = "RA"
TFSA = "TFSA"
DISC = "DISC"
GROSS = "GROSS"  # pre-tax counterfactual, no wrapper limits - used to measure tax drag
WRAPPERS = (RA, TFSA, DISC)
WRAPPER_NAMES = {
    RA: "Retirement Annuity (Reg 28)",
    TFSA: "Tax-Free Savings (TFSA)",
    DISC: "Taxable Discretionary",
    GROSS: "Pre-tax (no wrapper)",
}

MARGINAL_RATES = (0.18, 0.26, 0.31, 0.36, 0.39, 0.41, 0.45)

# Mid-bracket annual taxable incomes used to size the RA deduction when the user only
# supplies a marginal rate (override with an explicit annual income whenever you can).
_INCOME_2025_26 = ((0.18, 180_000), (0.26, 300_000), (0.31, 440_000), (0.36, 590_000), (0.39, 760_000), (0.41, 1_300_000), (0.45, 2_000_000))
_INCOME_2026_27 = ((0.18, 190_000), (0.26, 310_000), (0.31, 450_000), (0.36, 610_000), (0.39, 790_000), (0.41, 1_350_000), (0.45, 2_100_000))

# Retirement-fund lump sum benefits table (R1 - R550,000 at 0%, unchanged for 2025/26 - 2026/27):
# (lower bound, marginal rate above the bound, cumulative tax at the bound)
LUMP_SUM_TABLE: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.00, 0.0),
    (550_000.0, 0.18, 0.0),
    (770_000.0, 0.27, 39_600.0),
    (1_155_000.0, 0.36, 143_550.0),
)


@dataclass(frozen=True)
class TaxParameters:
    """Statutory parameters of the SA tax wrappers. All limits are nominal and held fixed."""

    label: str = "2025/26 (project specification)"
    dwt_rate: float = 0.20
    foreign_dividend_rate: float = 0.20
    interest_exemption: float = 23_800.0
    cgt_inclusion: float = 0.40
    cgt_annual_exclusion: float = 40_000.0
    tfsa_annual_limit: float = 36_000.0
    tfsa_lifetime_limit: float = 500_000.0
    ra_deduction_pct: float = 0.275
    ra_deduction_cap: float = 350_000.0
    ra_lump_sum_fraction: float = 1.0 / 3.0
    lump_sum_table: tuple = LUMP_SUM_TABLE
    representative_incomes: tuple = _INCOME_2025_26

    def effective_cgt_rate(self, marginal_rate: float) -> float:
        """Effective CGT rate = inclusion rate x marginal rate (40% x 45% = 18%)."""
        return self.cgt_inclusion * marginal_rate

    def income_for_bracket(self, marginal_rate: float) -> float:
        table = dict(self.representative_incomes)
        key = min(table, key=lambda r: abs(r - marginal_rate))
        return float(table[key])


TAX_2025_26 = TaxParameters()
#: Budget 2026 changes: TFSA R46,000 p.a. (lifetime R500,000 unchanged), RA deduction cap R430,000,
#: CGT annual exclusion R50,000. Interest exemption (R23,800) and DWT (20%) are unchanged.
TAX_2026_27 = TaxParameters(
    label="2026/27 (Budget 2026)",
    tfsa_annual_limit=46_000.0,
    ra_deduction_cap=430_000.0,
    cgt_annual_exclusion=50_000.0,
    representative_incomes=_INCOME_2026_27,
)
TAX_PRESETS = {"2025/26": TAX_2025_26, "2026/27": TAX_2026_27}
DEFAULT_TAX = TAX_2025_26


def lump_sum_tax(amount: np.ndarray | float, table: Sequence[Sequence[float]] = LUMP_SUM_TABLE) -> np.ndarray:
    """Tax on a retirement-fund lump sum (cumulative-table method, first R550,000 tax-free)."""
    x = np.asarray(amount, dtype=float)
    tax = np.zeros_like(x)
    for lower, rate, base in table:
        tax = np.where(x > lower, base + rate * (x - lower), tax)
    return tax


class Reg28Violation(ValueError):
    """Raised when a non-compliant allocation is placed in a Retirement Annuity."""


def apply_tfsa_limits(
    amounts: np.ndarray,
    annual_limit: float = TAX_2025_26.tfsa_annual_limit,
    lifetime_limit: float = TAX_2025_26.tfsa_lifetime_limit,
) -> tuple[np.ndarray, np.ndarray]:
    """Split contributions into the part a TFSA may accept and the overflow.

    ``amounts`` has shape ``(P, T)`` (or ``(T,)``): the cash invested each month. Tax years are
    modelled as blocks of 12 months. In every year at most ``annual_limit`` enters the TFSA and
    over the whole horizon at most ``lifetime_limit``; anything beyond that is returned as
    ``overflow`` (in practice it goes to a taxable account, since excess TFSA contributions are
    penalised at 40% by SARS).
    """
    a = np.atleast_2d(np.asarray(amounts, dtype=float))
    p, t = a.shape
    inside = np.zeros_like(a)
    life = np.zeros(p)
    year = np.zeros(p)
    for m in range(t):
        if m % 12 == 0:
            year[:] = 0.0
        room = np.maximum(np.minimum(annual_limit - year, lifetime_limit - life), 0.0)
        put = np.minimum(np.maximum(a[:, m], 0.0), room)
        inside[:, m] = put
        year += put
        life += put
    overflow = a - inside
    if np.asarray(amounts).ndim == 1:
        return inside[0], overflow[0]
    return inside, overflow


@dataclass
class WrapperOutcome:
    """Path-wise result of one wrapper (nominal ZAR unless stated)."""

    wrapper: str
    weights: np.ndarray
    values: np.ndarray  # (P, Y+1) portfolio market value at each year end (before exit / liquidation tax)
    values_liq: np.ndarray  # (P, Y+1) after-tax value if liquidated (RA: retired) at that year end
    terminal_pre_exit: np.ndarray
    terminal_net: np.ndarray
    tax_income: np.ndarray  # dividend withholding + interest + property-distribution tax
    tax_cgt: np.ndarray  # CGT on rebalancing and liquidation
    tax_exit: np.ndarray  # RA lump-sum and annuity tax at retirement
    tax_paid_real: np.ndarray  # all taxes paid, each deflated to today's rand at payment date
    refunds: np.ndarray  # RA upfront deduction refunds received
    refunds_real: np.ndarray
    contributions_total: np.ndarray
    tfsa_balance: np.ndarray | None = None
    overflow_balance: np.ndarray | None = None
    tfsa_contributed: np.ndarray | None = None
    reg28_report: Reg28Report | None = None

    @property
    def tax_paid(self) -> np.ndarray:
        return self.tax_income + self.tax_cgt + self.tax_exit

    @property
    def net_tax_cost(self) -> np.ndarray:
        """Taxes paid minus RA refunds received (nominal)."""
        return self.tax_paid - self.refunds


class _Account:
    """One investment account: per-asset market values and cost bases for every path."""

    def __init__(
        self,
        n_paths: int,
        weights: np.ndarray,
        income_yield: np.ndarray,
        income_class: Sequence[str],
        taxable: bool,
        tax: TaxParameters,
        marginal_rate: float,
    ) -> None:
        a = len(weights)
        self.w = weights
        self.taxable = taxable
        self.tax = tax
        self.marg = marginal_rate
        self.y_m = income_yield / 12.0
        self.V = np.zeros((n_paths, a))
        self.C = np.zeros((n_paths, a))
        self.ytd_int = np.zeros(n_paths)
        self.cap_loss_cf = np.zeros(n_paths)
        self.tax_income = np.zeros(n_paths)
        self.tax_cgt = np.zeros(n_paths)
        self.tax_real = np.zeros(n_paths)
        cls = np.array(income_class)
        self.div_idx = np.where((cls == LOCAL_DIVIDEND) | (cls == FOREIGN_DIVIDEND))[0]
        self.div_rate = np.array(
            [tax.dwt_rate if c == LOCAL_DIVIDEND else min(tax.foreign_dividend_rate, marginal_rate) for c in cls[self.div_idx]]
        )
        self.lint_idx = np.where(cls == LOCAL_INTEREST)[0]
        self.other_idx = np.where((cls == FOREIGN_INTEREST) | (cls == PROPERTY_DISTRIBUTION))[0]
        self.cgt_rate = tax.effective_cgt_rate(marginal_rate)

    def start_year(self) -> None:
        self.ytd_int[:] = 0.0

    def contribute(self, amount: np.ndarray) -> None:
        add = amount[:, None] * self.w[None, :]
        self.V += add
        if self.taxable:
            self.C += add

    def grow(self, r: np.ndarray, cpi_t: np.ndarray) -> None:
        if not self.taxable:
            self.V *= 1.0 + r
            return
        inc = self.V * self.y_m
        self.V *= 1.0 + r
        tax = np.zeros_like(inc)
        tax[:, self.div_idx] = inc[:, self.div_idx] * self.div_rate
        tax[:, self.other_idx] = inc[:, self.other_idx] * self.marg
        li = inc[:, self.lint_idx].sum(axis=1)
        ex = self.tax.interest_exemption
        before = np.maximum(self.ytd_int - ex, 0.0)
        self.ytd_int += li
        taxable_li = np.maximum(self.ytd_int - ex, 0.0) - before
        frac = np.divide(self.marg * taxable_li, li, out=np.zeros_like(li), where=li > 0)
        tax[:, self.lint_idx] = inc[:, self.lint_idx] * frac[:, None]
        self.V -= tax
        self.C += inc - tax  # net income is reinvested at cost
        t = tax.sum(axis=1)
        self.tax_income += t
        self.tax_real += t / cpi_t

    def _cgt(self, net_gain: np.ndarray) -> np.ndarray:
        """CGT on a year's net realised gain: annual exclusion, assessed-loss carry-forward, inclusion rate."""
        gain = np.maximum(net_gain, 0.0)
        loss = np.maximum(-net_gain, 0.0)
        after_excl = np.maximum(gain - self.tax.cgt_annual_exclusion, 0.0)
        used = np.minimum(self.cap_loss_cf, after_excl)
        self.cap_loss_cf = self.cap_loss_cf - used + loss
        return (after_excl - used) * self.cgt_rate

    def year_end(self, last: bool, rebalance: bool, cpi_t: np.ndarray) -> None:
        tot = self.V.sum(axis=1)
        if self.taxable:
            if last:  # liquidation at the end of the horizon realises every unrealised gain
                net_gain = (self.V - self.C).sum(axis=1)
            elif rebalance:
                target = tot[:, None] * self.w[None, :]
                trade = target - self.V
                sell = np.maximum(-trade, 0.0)
                frac = np.divide(sell, self.V, out=np.zeros_like(sell), where=self.V > 0)
                net_gain = (frac * (self.V - self.C)).sum(axis=1)
                self.C = self.C * (1.0 - frac) + np.maximum(trade, 0.0)
                self.V = target
            else:
                net_gain = np.zeros_like(tot)
            cgt = np.minimum(self._cgt(net_gain), tot)
            f = 1.0 - np.divide(cgt, tot, out=np.zeros_like(cgt), where=tot > 0)
            self.V *= f[:, None]
            self.C *= f[:, None]
            self.tax_cgt += cgt
            self.tax_real += cgt / cpi_t
        elif rebalance and not last:
            self.V = tot[:, None] * self.w[None, :]

    def liquidation_tax(self) -> np.ndarray:
        """CGT that would be due if the whole account were sold now (does not change any state)."""
        if not self.taxable:
            return np.zeros(self.V.shape[0])
        gain = np.maximum((self.V - self.C).sum(axis=1), 0.0)
        after_excl = np.maximum(gain - self.tax.cgt_annual_exclusion, 0.0)
        return np.maximum(after_excl - self.cap_loss_cf, 0.0) * self.cgt_rate

    @property
    def value(self) -> np.ndarray:
        return self.V.sum(axis=1)


def ra_exit_tax(
    value: np.ndarray, cpi_final: np.ndarray, tax: TaxParameters, retirement_tax_rate: float
) -> np.ndarray:
    """Tax at retirement on a Retirement Annuity (nominal ZAR).

    One third may be taken as a lump sum, taxed on the retirement lump-sum table (the table is
    expressed in today's rand, so the lump sum is deflated, taxed and re-inflated). The remaining two
    thirds must buy an annuity whose income is taxed at ``retirement_tax_rate``; the present value of
    that future income tax is approximated as ``rate x capital``.
    """
    lump = value * tax.ra_lump_sum_fraction
    lump_tax = lump_sum_tax(lump / cpi_final, tax.lump_sum_table) * cpi_final
    annuity_tax = value * (1.0 - tax.ra_lump_sum_fraction) * retirement_tax_rate
    return lump_tax + annuity_tax


def simulate_wrapper(
    returns: np.ndarray,
    cpi: np.ndarray,
    weights: Sequence[float],
    contributions: np.ndarray,
    wrapper: str,
    income_yield: np.ndarray,
    income_class: Sequence[str] = tuple(INCOME_CLASS[a] for a in ASSETS),
    tax: TaxParameters = DEFAULT_TAX,
    marginal_rate: float = 0.45,
    annual_income: float | None = None,
    retirement_tax_rate: float | None = None,
    rebalance: bool = True,
    reinvest_refund: bool = True,
    enforce_reg28: bool = True,
    limits: Reg28Limits = REG28,
    assets: Sequence[str] = ASSETS,
) -> WrapperOutcome:
    """Run one tax wrapper over pre-generated market paths.

    Parameters
    ----------
    returns:
        ``(P, T, A)`` simple monthly *total* returns in ZAR (T must be a multiple of 12).
    cpi:
        ``(P, T + 1)`` cumulative CPI index (1.0 today).
    contributions:
        ``(P, T)`` nominal cash invested at the start of each month; the initial lump sum is
        included in month 0.
    wrapper:
        ``"RA"``, ``"TFSA"``, ``"DISC"`` or ``"GROSS"`` (no tax, no limits).

    Income (dividends / interest / property distributions) is paid out monthly, taxed according to
    its character and reinvested; capital gains are only taxed in the Discretionary account, at each
    annual rebalance and on liquidation at the end of the horizon. The TFSA takes contributions up to
    its limits and any overflow is invested (same allocation) in a taxable account. The RA validates
    Regulation 28 and adds the upfront tax refund (reinvested by default).
    """
    if wrapper not in (RA, TFSA, DISC, GROSS):
        raise ValueError(f"Unknown wrapper '{wrapper}'")
    w = np.asarray(weights, dtype=float)
    if abs(w.sum() - 1.0) > 1e-6 or np.any(w < -1e-9):
        raise ValueError("weights must be non-negative and sum to 1")
    p, t, a = returns.shape
    if t % 12 != 0:
        raise ValueError("number of months must be a multiple of 12")
    years = t // 12
    report = None
    if wrapper == RA:
        report = check_reg28(w, limits, assets)
        if enforce_reg28 and not report.compliant:
            raise Reg28Violation("Retirement Annuity allocation breaches Regulation 28: " + "; ".join(report.breaches))
    if retirement_tax_rate is None:
        retirement_tax_rate = min(marginal_rate, 0.31)
    income = annual_income if annual_income is not None else tax.income_for_bracket(marginal_rate)

    def acct(taxable: bool) -> _Account:
        return _Account(p, w, income_yield, income_class, taxable, tax, marginal_rate)

    if wrapper == TFSA:
        main, overflow_acct = acct(False), acct(True)
        to_tfsa, to_overflow = apply_tfsa_limits(contributions, tax.tfsa_annual_limit, tax.tfsa_lifetime_limit)
        accounts = [main, overflow_acct]
    else:
        main = acct(wrapper == DISC)
        overflow_acct = None
        accounts = [main]

    values = np.zeros((p, years + 1))
    values_liq = np.zeros((p, years + 1))
    values[:, 0] = values_liq[:, 0] = contributions[:, 0]
    year_contrib = np.zeros(p)
    refunds = np.zeros(p)
    refunds_real = np.zeros(p)

    for m in range(t):
        k = m % 12
        if k == 0:
            for ac in accounts:
                ac.start_year()
            year_contrib[:] = 0.0
        c = contributions[:, m]
        year_contrib += c
        if wrapper == TFSA:
            main.contribute(to_tfsa[:, m])
            overflow_acct.contribute(to_overflow[:, m])
        else:
            main.contribute(c)
        r = returns[:, m, :].astype(float, copy=False)
        cpi_t = cpi[:, m + 1]
        for ac in accounts:
            ac.grow(r, cpi_t)
        if k == 11:
            last = m == t - 1
            for ac in accounts:
                ac.year_end(last, rebalance, cpi_t)
            if wrapper == RA:
                y = m // 12
                inc_y = income * cpi[:, 12 * y]
                deductible = np.minimum(np.minimum(year_contrib, tax.ra_deduction_pct * inc_y), tax.ra_deduction_cap)
                refund = deductible * marginal_rate
                refunds += refund
                refunds_real += refund / cpi_t
                if reinvest_refund:
                    main.contribute(refund)
            mv = sum(ac.value for ac in accounts)
            values[:, m // 12 + 1] = mv
            if not last:
                liq = mv - sum(ac.liquidation_tax() for ac in accounts)
                if wrapper == RA:
                    liq = liq - ra_exit_tax(mv, cpi[:, m + 1], tax, retirement_tax_rate)
                values_liq[:, m // 12 + 1] = liq

    pre_exit = sum(ac.value for ac in accounts)
    tax_income = sum(ac.tax_income for ac in accounts)
    tax_cgt = sum(ac.tax_cgt for ac in accounts)
    tax_real = sum(ac.tax_real for ac in accounts)
    tax_exit = np.zeros(p)
    if wrapper == RA:
        tax_exit = ra_exit_tax(pre_exit, cpi[:, -1], tax, retirement_tax_rate)
        tax_real = tax_real + tax_exit / cpi[:, -1]
    net = pre_exit - tax_exit
    values_liq[:, -1] = net
    return WrapperOutcome(
        wrapper=wrapper,
        weights=w,
        values=values,
        values_liq=values_liq,
        terminal_pre_exit=pre_exit,
        terminal_net=net,
        tax_income=tax_income,
        tax_cgt=tax_cgt,
        tax_exit=tax_exit,
        tax_paid_real=tax_real,
        refunds=refunds,
        refunds_real=refunds_real,
        contributions_total=contributions.sum(axis=1),
        tfsa_balance=main.value if wrapper == TFSA else None,
        overflow_balance=overflow_acct.value if wrapper == TFSA else None,
        tfsa_contributed=to_tfsa.sum(axis=1) if wrapper == TFSA else None,
        reg28_report=report,
    )


def compare_wrappers(
    returns: np.ndarray,
    cpi: np.ndarray,
    weights: Sequence[float],
    contributions: np.ndarray,
    income_yield: np.ndarray,
    income_class: Sequence[str] = tuple(INCOME_CLASS[a] for a in ASSETS),
    tax: TaxParameters = DEFAULT_TAX,
    marginal_rate: float = 0.45,
    annual_income: float | None = None,
    retirement_tax_rate: float | None = None,
    limits: Reg28Limits = REG28,
    rebalance: bool = True,
    reinvest_refund: bool = True,
    assets: Sequence[str] = ASSETS,
) -> dict[str, WrapperOutcome]:
    """RA vs TFSA vs Discretionary on identical market paths (common random numbers).

    The TFSA and Discretionary accounts may hold ``weights`` as is. The RA must be Regulation 28
    compliant, so it holds ``weights`` if they already comply, otherwise their closest compliant
    projection. A ``GROSS`` (pre-tax, unlimited) run of ``weights`` is returned for reference.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    w_ra = w if is_reg28_compliant(w, limits, assets=assets) else project_to_reg28(w, limits, assets)
    common = dict(
        returns=returns,
        cpi=cpi,
        contributions=contributions,
        income_yield=income_yield,
        income_class=income_class,
        tax=tax,
        marginal_rate=marginal_rate,
        annual_income=annual_income,
        retirement_tax_rate=retirement_tax_rate,
        rebalance=rebalance,
        reinvest_refund=reinvest_refund,
        limits=limits,
        assets=assets,
    )
    return {
        RA: simulate_wrapper(weights=w_ra, wrapper=RA, **common),
        TFSA: simulate_wrapper(weights=w, wrapper=TFSA, **common),
        DISC: simulate_wrapper(weights=w, wrapper=DISC, **common),
        GROSS: simulate_wrapper(weights=w, wrapper=GROSS, **common),
    }
