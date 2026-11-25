"""Regulation 28 validator, optimiser, frontier and trade-off analytics."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sa_wealth.assets import ASSETS
from sa_wealth.reg28_tax import (
    NO_LIMITS,
    REG28,
    STATIC_PRESETS,
    Reg28Limits,
    check_reg28,
    efficient_frontier,
    exposures,
    is_reg28_compliant,
    offshore_sensitivity,
    optimize_portfolio,
    portfolio_presets,
    portfolio_stats,
    project_to_reg28,
    reg28_tradeoff,
    risk_parity_weights,
)

TOL = 1e-6


def w(**kw):
    out = np.zeros(len(ASSETS))
    for k, v in kw.items():
        out[ASSETS.index(k)] = v
    return out


def assert_reg28(weights, limits=REG28):
    weights = np.asarray(weights)
    assert weights.min() >= -TOL
    assert weights.sum() == pytest.approx(1.0, abs=TOL)
    ex = exposures(weights)
    assert ex["equity"] <= limits.max_equity + TOL  # JSE_EQ + GLB_EQ <= 0.75
    assert ex["offshore"] <= limits.max_offshore + TOL  # GLB_EQ + GLB_BOND <= 0.45
    assert ex["property"] <= limits.max_property + TOL  # SA_PROP <= 0.25


@pytest.fixture(scope="module")
def weak_rand(assumptions):
    """A +4pp weak-Rand view on global assets: the unconstrained optimum breaches the offshore cap."""
    tbl = assumptions.table.copy()
    tbl.loc[["GLB_EQ", "GLB_BOND"], "expected_return"] += 0.04
    return replace(assumptions, table=tbl)


# ----------------------------------------------------------------------------- validator


def test_limits_are_the_regulation_28_values():
    assert (REG28.max_equity, REG28.max_offshore, REG28.max_property) == (0.75, 0.45, 0.25)


def test_exposure_buckets():
    ex = exposures(w(JSE_EQ=0.3, GLB_EQ=0.2, GLB_BOND=0.1, SA_PROP=0.1, SA_CASH=0.3))
    assert ex == pytest.approx({"equity": 0.5, "offshore": 0.3, "property": 0.1})


@pytest.mark.parametrize("name", list(STATIC_PRESETS))
def test_static_presets_flagged_correctly(name, assumptions):
    weights = np.array([STATIC_PRESETS[name][a] for a in ASSETS])
    assert weights.sum() == pytest.approx(1.0)
    assert is_reg28_compliant(weights) == ("(Reg 28)" in name)


def test_validator_accepts_boundary_portfolio():
    boundary = w(JSE_EQ=0.30, GLB_EQ=0.45, SA_PROP=0.25)  # 75% equity, 45% offshore, 25% property
    r = check_reg28(boundary)
    assert r.compliant and r.breaches == ()
    assert r.headroom["equity"] == pytest.approx(0.0, abs=1e-12)


def test_validator_flags_each_breach():
    r = check_reg28(w(JSE_EQ=0.8, SA_CASH=0.2))
    assert not r.compliant and any("equity" in b for b in r.breaches) and r.equity == pytest.approx(0.8)
    r = check_reg28(w(GLB_BOND=0.5, SA_CASH=0.5))
    assert not r.compliant and any("offshore" in b for b in r.breaches)
    r = check_reg28(w(SA_PROP=0.3, SA_CASH=0.7))
    assert not r.compliant and any("property" in b for b in r.breaches)


def test_validator_flags_negative_and_non_unit_weights():
    assert not check_reg28(w(JSE_EQ=1.2, SA_CASH=-0.2)).compliant
    assert any("sum" in b for b in check_reg28(w(JSE_EQ=0.4)).breaches)
    with pytest.raises(ValueError):
        check_reg28([0.5, 0.5])


def test_limits_can_be_switched_off():
    assert is_reg28_compliant(w(GLB_EQ=0.7, SA_CASH=0.3), REG28.replace(max_offshore=None, max_equity=None))
    assert is_reg28_compliant(w(GLB_EQ=1.0), NO_LIMITS)


# ----------------------------------------------------------------------------- optimiser


@pytest.mark.parametrize("objective", ["max_sharpe", "min_variance", "risk_parity", "max_return"])
def test_optimiser_enforces_every_constraint(objective, weak_rand):
    kw = {"exclude": ("SA_CASH",)} if objective == "risk_parity" else {}
    r = optimize_portfolio(weak_rand, objective, REG28, **kw)
    assert_reg28(r.weights.to_numpy())
    assert r.report.compliant


def test_optimiser_is_binding_when_unconstrained_would_breach(weak_rand):
    unc = optimize_portfolio(weak_rand, "max_sharpe", None)
    reg = optimize_portfolio(weak_rand, "max_sharpe", REG28)
    assert unc.report.offshore > 0.45 + 1e-3  # unconstrained portfolio breaches the offshore cap
    assert not unc.report.compliant
    assert reg.report.offshore == pytest.approx(0.45, abs=1e-5)  # the cap binds exactly
    assert reg.sharpe <= unc.sharpe + 1e-9  # constraints can never help
    assert reg.sharpe < unc.sharpe  # ... and here they strictly cost something


def test_max_sharpe_beats_every_preset_and_asset(assumptions):
    best = optimize_portfolio(assumptions, "max_sharpe", REG28)
    rng = np.random.default_rng(0)
    for _ in range(300):
        cand = rng.dirichlet(np.ones(6))
        if is_reg28_compliant(cand):
            assert portfolio_stats(cand, assumptions)[2] <= best.sharpe + 1e-7


def test_unconstrained_sharpe_at_least_reg28(assumptions):
    unc = optimize_portfolio(assumptions, "max_sharpe", None)
    reg = optimize_portfolio(assumptions, "max_sharpe", REG28)
    assert unc.sharpe >= reg.sharpe - 1e-9


def test_max_return_matches_hand_calculation(assumptions):
    # Reg 28: at most 75% in the best equity (JSE 12%) and 25% in the next best bucket (property 10%)
    reg = optimize_portfolio(assumptions, "max_return", REG28)
    assert reg.expected_return == pytest.approx(0.75 * 0.12 + 0.25 * 0.10, abs=1e-9)
    unc = optimize_portfolio(assumptions, "max_return", None)
    assert unc.expected_return == pytest.approx(0.12)
    assert unc.weights["JSE_EQ"] == pytest.approx(1.0)


def test_min_variance_with_return_target(assumptions):
    r = optimize_portfolio(assumptions, "min_variance", REG28, target_return=0.10)
    assert r.expected_return >= 0.10 - 1e-6
    assert_reg28(r.weights.to_numpy())
    with pytest.raises(RuntimeError):
        optimize_portfolio(assumptions, "min_variance", REG28, target_return=0.119)  # above the Reg 28 maximum


def test_unknown_objective(assumptions):
    with pytest.raises(ValueError):
        optimize_portfolio(assumptions, "max_fun")


def test_risk_parity_equalises_risk_contributions(assumptions):
    r = risk_parity_weights(assumptions, None)
    wt = r.weights.to_numpy()
    assert wt[ASSETS.index("SA_CASH")] == 0.0  # cash excluded from the risk budget
    rc = wt * (assumptions.cov @ wt)
    rc = rc[wt > 0] / rc.sum()
    assert np.allclose(rc, 1 / len(rc), atol=0.02)


def test_risk_parity_respects_reg28(weak_rand):
    r = risk_parity_weights(weak_rand, REG28)
    assert_reg28(r.weights.to_numpy())


def test_project_to_reg28(assumptions):
    ok = w(JSE_EQ=0.4, SA_BOND=0.4, GLB_EQ=0.2)
    assert np.allclose(project_to_reg28(ok), ok)  # compliant input returned unchanged
    bad = w(JSE_EQ=0.1, GLB_EQ=0.85, SA_BOND=0.05)  # 95% equity, 85% offshore
    p = project_to_reg28(bad)
    assert_reg28(p)
    # offshore equity is cut back to the cap rather than discarded entirely
    assert p[ASSETS.index("GLB_EQ")] == pytest.approx(0.45, abs=1e-3)
    with pytest.raises(ValueError):
        project_to_reg28(np.zeros(6))


def test_presets_include_optimised_and_are_valid(assumptions):
    presets = portfolio_presets(assumptions)
    for name, wt in presets.items():
        assert wt.sum() == pytest.approx(1.0, abs=1e-6)
        if "Reg 28" in name:
            assert is_reg28_compliant(wt), name
    assert "Max-Sharpe (Reg 28)" in presets and "Risk Parity (unconstrained)" in presets


# ----------------------------------------------------------------------------- frontier & trade-off


def test_frontiers_are_ordered_and_respect_limits(assumptions):
    fu = efficient_frontier(assumptions, None, 25)
    fr = efficient_frontier(assumptions, REG28, 25)
    for f in (fu, fr):
        assert f["return"].is_monotonic_increasing
        assert f["volatility"].is_monotonic_increasing
    for _, row in fr.iterrows():
        assert_reg28(row[list(ASSETS)].to_numpy(dtype=float))
    assert fu["return"].max() == pytest.approx(0.12, abs=1e-6)
    assert fr["return"].max() == pytest.approx(0.115, abs=1e-6)
    # at any return the Reg 28 frontier needs at least as much risk as the unconstrained one
    grid = np.linspace(fr["return"].min() + 1e-3, fr["return"].max() - 1e-3, 15)
    vol_u = np.interp(grid, fu["return"], fu["volatility"])
    vol_r = np.interp(grid, fr["return"], fr["volatility"])
    assert np.all(vol_r >= vol_u - 2e-4)


def test_tradeoff_tables(assumptions):
    tr = reg28_tradeoff(assumptions)
    assert set(tr) == {"portfolios", "cap_by_cap", "frontier_gap"}
    p = tr["portfolios"]
    assert p.loc["Max-Sharpe - unconstrained", "sharpe"] >= p.loc["Max-Sharpe - Regulation 28", "sharpe"] - 1e-9
    assert (tr["cap_by_cap"]["sharpe_cost_vs_unconstrained"] >= 0).all()
    g = tr["frontier_gap"]
    assert g.attrs["max_return_unconstrained"] > g.attrs["max_return_reg28"]
    assert g["vol_reg28"].isna().iloc[-1]  # the highest targets are infeasible under Regulation 28
    assert (g["vol_cost_reg28"].dropna() >= 0).all()


def test_offshore_cap_sensitivity(assumptions):
    s = offshore_sensitivity(assumptions, premiums=(0.0, 0.03, 0.06))
    assert not s["cap_binding"].iloc[0]  # at base assumptions the 45% cap is slack
    assert s["cap_binding"].iloc[-1]
    assert (s["sharpe_cost"] >= 0).all()
    assert s["sharpe_cost"].iloc[-1] > s["sharpe_cost"].iloc[0]
    assert s["offshore_reg28"].iloc[-1] == pytest.approx(0.45, abs=1e-5)
    assert s["offshore_unconstrained"].is_monotonic_increasing
