"""Monte Carlo engine, metrics and stress scenarios."""

from dataclasses import replace

import numpy as np
import pytest

from sa_wealth.assets import ASSETS, INCOME_CLASS
from sa_wealth.montecarlo import (
    DEFAULT_SCENARIOS,
    RISK_OFF,
    SEQUENCE_CRASH,
    STAGFLATION,
    SimulationConfig,
    build_contributions,
    cvar,
    expected_return_paths,
    generate_paths,
    retirement_horizon,
    run_stress,
    run_tax_comparison,
    simulate,
    zar_shock,
)
from sa_wealth.reg28_tax import DISC, GROSS, RA, TFSA, is_reg28_compliant, simulate_wrapper


# ----------------------------------------------------------------------------- path generation


def test_paths_are_reproducible_and_seed_dependent(assumptions):
    a = generate_paths(assumptions, 300, 5, seed=11)
    b = generate_paths(assumptions, 300, 5, seed=11)
    c = generate_paths(assumptions, 300, 5, seed=12)
    assert np.array_equal(a.returns, b.returns) and np.array_equal(a.cpi, b.cpi)
    assert not np.array_equal(a.returns, c.returns)


def test_path_shapes_and_validity(paths, config):
    assert paths.returns.shape == (config.n_paths, config.horizon_years * 12, 6)
    assert paths.cpi.shape == (config.n_paths, config.horizon_years * 12 + 1)
    assert np.isfinite(paths.returns).all() and (paths.returns > -1).all()
    assert (paths.cpi[:, 0] == 1).all()


def test_simulated_moments_match_assumptions(assumptions):
    p = generate_paths(assumptions, 5_000, 10, seed=3)
    r = p.returns.astype(float)
    # annual arithmetic return (first year) equals the assumed expected return
    annual = np.prod(1 + r[:, :12, :], axis=1) - 1
    assert np.allclose(annual.mean(axis=0), assumptions.mu, atol=0.012)
    # annualised volatility of the monthly series
    assert np.allclose(np.log1p(r).std(axis=(0, 1)) * np.sqrt(12), assumptions.vol, rtol=0.05)


def test_cholesky_reproduces_correlation_matrix(assumptions):
    p = generate_paths(assumptions, 5_000, 10, seed=3)
    pooled = np.log1p(p.returns.astype(float)).reshape(-1, 6)
    sample = np.corrcoef(pooled, rowvar=False)
    assert np.abs(sample - assumptions.asset_corr).max() < 0.03
    # Rand-hedge sign structure survives the simulation
    assert sample[ASSETS.index("GLB_EQ"), ASSETS.index("GLB_BOND")] > 0.3
    assert sample[ASSETS.index("SA_BOND"), ASSETS.index("GLB_BOND")] < 0


def test_inflation_is_stochastic_and_centred(assumptions):
    p = generate_paths(assumptions, 3_000, 10, seed=5)
    infl = p.annualised_inflation()
    assert infl.mean() == pytest.approx(0.055, abs=0.003)
    assert infl.std() > 0.002  # stochastic inflation shocks, not a flat 5.5%


def test_expected_return_paths_are_deterministic(assumptions):
    p = expected_return_paths(assumptions, 10)
    assert p.returns.shape == (1, 120, 6)
    assert np.allclose(np.prod(1 + p.returns[0], axis=0) ** (1 / 10) - 1, assumptions.mu)
    assert p.cpi[0, -1] == pytest.approx(1.055**10)


# ----------------------------------------------------------------------------- contributions


def test_contributions_escalate_with_cpi_each_year():
    cfg = SimulationConfig(initial_lump_sum=100_000, monthly_contribution=5_000, horizon_years=5, n_paths=100)
    cpi = np.cumprod(np.full((2, 61), 1.005), axis=1) / 1.005
    c = build_contributions(cfg, cpi)
    assert c.shape == (2, 60)
    assert c[0, 0] == pytest.approx(105_000)  # initial lump sum in month 0
    assert c[0, 5] == pytest.approx(5_000)  # year 1 is not escalated
    assert c[0, 12] == pytest.approx(5_000 * cpi[0, 12])  # year 2 steps up with the CPI index
    assert c[0, 23] == pytest.approx(c[0, 12])  # constant within a year
    assert c[0, 48] > c[0, 12]


# ----------------------------------------------------------------------------- engine correctness


def test_gross_engine_matches_hand_recursion(assumptions):
    """100% JSE, no tax, deterministic returns: engine terminal wealth equals a plain-Python recursion."""
    cfg = SimulationConfig(initial_lump_sum=50_000, monthly_contribution=2_000, horizon_years=5, n_paths=100, wrapper=GROSS)
    p = expected_return_paths(assumptions, 5)
    w = np.zeros(6)
    w[0] = 1.0
    res = simulate(cfg, w, assumptions, p, with_gross=False)
    r = (1 + assumptions.mu[0]) ** (1 / 12) - 1
    cpi = p.cpi[0]
    v = 0.0
    for m in range(60):
        v += 2_000 * cpi[(m // 12) * 12] + (50_000 if m == 0 else 0)
        v *= 1 + r
    assert res.terminal_nominal[0] == pytest.approx(v, rel=1e-9)
    assert res.terminal_real[0] == pytest.approx(v / cpi[-1], rel=1e-9)


# ----------------------------------------------------------------------------- simulate(): percentiles & metrics


@pytest.mark.parametrize("wrapper", [RA, TFSA, DISC])
def test_percentile_ordering(wrapper, assumptions, config, paths, moderate):
    res = simulate(replace(config, wrapper=wrapper), moderate, assumptions, paths)
    for real in (True, False):
        pct = res.percentiles(real=real)
        assert list(pct.columns) == ["P10", "P25", "P50", "P75", "P90"]
        after_start = pct.iloc[1:]
        assert (after_start["P10"] < after_start["P25"]).all()
        assert (after_start["P25"] < after_start["P50"]).all()
        assert (after_start["P50"] < after_start["P75"]).all()
        assert (after_start["P75"] < after_start["P90"]).all()
        assert pct["P10"].iloc[-1] < pct["P50"].iloc[-1] < pct["P90"].iloc[-1]


def test_median_wealth_grows_over_time(assumptions, config, paths, moderate):
    res = simulate(config, moderate, assumptions, paths)
    assert res.percentiles(real=True)["P50"].is_monotonic_increasing
    assert res.percentiles().index[0] == 0 and res.percentiles().index[-1] == config.horizon_years


def test_goal_probability_and_risk_metrics(assumptions, config, paths, moderate):
    res = simulate(config, moderate, assumptions, paths)
    s = res.summary()
    assert 0.0 <= s["prob_goal"] <= 1.0
    assert s["cvar5_real"] <= s["var5_real"] <= s["median_terminal_real"]  # CVaR is the mean of the worst tail
    assert s["p10_terminal_real"] < s["median_terminal_real"] < s["p90_terminal_real"]
    assert s["cumulative_tax_drag_nominal"] > 0
    assert s["expected_shortfall_vs_goal_real"] >= 0
    assert s["mean_annual_inflation"] == pytest.approx(0.055, abs=0.005)
    # goal probability is monotone in the target
    easy = simulate(replace(config, target_real=100_000), moderate, assumptions, paths)
    hard = simulate(replace(config, target_real=50_000_000), moderate, assumptions, paths)
    assert easy.prob_goal == 1.0 and hard.prob_goal == 0.0
    assert easy.prob_goal >= res.prob_goal >= hard.prob_goal


def test_cvar_helper():
    v = np.arange(1, 101, dtype=float)
    assert cvar(v, 0.05) == pytest.approx(3.0)  # mean of 1..5
    assert cvar(v, 0.10) == pytest.approx(5.5)


def test_more_savings_means_more_wealth(assumptions, config, paths, moderate):
    base = simulate(config, moderate, assumptions, paths)
    more = simulate(replace(config, monthly_contribution=10_000), moderate, assumptions, paths)
    assert np.median(more.terminal_real) > np.median(base.terminal_real)
    assert more.prob_goal >= base.prob_goal


def test_tax_drag_is_positive_and_gross_exceeds_net_without_refunds(assumptions, config, paths, moderate):
    for wrapper in (DISC, TFSA):
        res = simulate(replace(config, wrapper=wrapper), moderate, assumptions, paths)
        assert res.cumulative_tax_drag > 0
        assert np.all(res.gross_terminal_nominal >= res.terminal_nominal - 1e-6)
        assert res.summary()["median_wealth_lost_to_tax_real"] >= 0


def test_ra_adjusts_non_compliant_allocation(assumptions, config, paths):
    aggressive = np.array([0.1, 0.0, 0.0, 0.0, 0.8, 0.1])
    res = simulate(replace(config, wrapper=RA), aggressive, assumptions, paths, with_gross=False)
    assert res.reg28_adjusted and is_reg28_compliant(res.weights_used)
    assert np.allclose(res.weights_requested, aggressive)
    held = simulate(replace(config, wrapper=DISC), aggressive, assumptions, paths, with_gross=False)
    assert not held.reg28_adjusted and np.allclose(held.weights_used, aggressive)


def test_weights_accept_mappings_and_are_normalised(assumptions, config, paths):
    res = simulate(config, {"JSE_EQ": 30, "SA_BOND": 50, "SA_CASH": 20}, assumptions, paths, with_gross=False)
    assert res.weights_used.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        simulate(config, [-0.5, 1.5, 0, 0, 0, 0], assumptions, paths)


def test_config_validation():
    with pytest.raises(ValueError):
        SimulationConfig(horizon_years=4)
    with pytest.raises(ValueError):
        SimulationConfig(horizon_years=41)
    with pytest.raises(ValueError):
        SimulationConfig(n_paths=10)
    with pytest.raises(ValueError):
        SimulationConfig(wrapper="ISA")
    with pytest.raises(ValueError):
        SimulationConfig(monthly_contribution=-1)
    assert SimulationConfig(horizon_years=5).n_months == 60 and SimulationConfig(horizon_years=40).n_months == 480
    assert retirement_horizon(30, 65) == 35
    with pytest.raises(ValueError):
        retirement_horizon(30, 80)


def test_paths_must_match_horizon(assumptions, config, paths, moderate):
    with pytest.raises(ValueError):
        simulate(replace(config, horizon_years=12), moderate, assumptions, paths)


# ----------------------------------------------------------------------------- tax comparison


def test_run_tax_comparison(assumptions, config, paths, moderate):
    tc = run_tax_comparison(config, moderate, assumptions, paths)
    s = tc.summary
    assert list(s.index) == [RA, TFSA, DISC]
    assert s.loc[TFSA, "median_terminal_real"] > s.loc[DISC, "median_terminal_real"]
    assert s.loc[TFSA, "share_paths_beating_disc"] == 1.0
    assert s.loc[TFSA, "tax_saved_vs_disc_real"] > 0 and s.loc[RA, "tax_saved_vs_disc_real"] > 0
    assert s.loc[DISC, "tax_saved_vs_disc_real"] == 0
    assert s.loc[RA, "ra_refunds_real"] > 0 and s.loc[TFSA, "ra_refunds_real"] == 0
    assert s.loc[TFSA, "tfsa_contributed_nominal"] <= 10 * 36_000 + 1e-6
    curves = tc.curves(real=True)
    assert curves.shape == (config.horizon_years + 1, 4)
    assert (curves.iloc[-1][["Tax-Free Savings (TFSA)"]].to_numpy() > curves.iloc[-1][["Taxable Discretionary"]].to_numpy()).all()
    assert tc.curves(real=False).iloc[-1].min() > tc.curves(real=True).iloc[-1].min()


def test_tax_comparison_unconstrained_weights_for_tfsa_and_disc(assumptions, config, paths):
    w = np.array([0.1, 0.0, 0.0, 0.0, 0.8, 0.1])
    tc = run_tax_comparison(config, w, assumptions, paths)
    assert np.allclose(tc.weights_unconstrained, w)
    assert is_reg28_compliant(tc.weights_ra)


# ----------------------------------------------------------------------------- stress scenarios


def test_zar_shock_is_the_rand_hedge():
    assert zar_shock(-0.30, 0.18) == pytest.approx(-0.174)
    assert zar_shock(0.02, 0.18) == pytest.approx(0.2036)


def test_scenario_definitions():
    assert [s.key for s in DEFAULT_SCENARIOS] == ["stagflation", "risk_off", "sequence_crash"]
    assert STAGFLATION.inflation == pytest.approx(0.085)
    assert STAGFLATION.drift_adj["JSE_EQ"] < 0 and STAGFLATION.drift_adj["SA_BOND"] < 0
    assert STAGFLATION.drift_adj["GLB_EQ"] > 0  # ZAR depreciation lifts offshore assets in rand
    assert RISK_OFF.shocks["GLB_EQ"] == pytest.approx(-0.174)  # -30% USD equity cushioned by the weaker Rand
    assert SEQUENCE_CRASH.shocks["JSE_EQ"] == pytest.approx(-0.25) and SEQUENCE_CRASH.shocks["GLB_EQ"] == pytest.approx(-0.25)


def test_scenario_windows():
    assert STAGFLATION.window(30) == (0, 84)
    s, e = SEQUENCE_CRASH.window(30)
    assert (s, e) == (30 * 12 - 24, 30 * 12 - 12)  # inside the final two years before retirement
    s, e = RISK_OFF.window(30)
    assert s == 240 and e == 252
    assert RISK_OFF.window(5)[0] == 0  # clipped for short horizons


def test_sequence_crash_hits_equities_in_the_window(assumptions):
    base = generate_paths(assumptions, 500, 10, seed=9)
    crash = generate_paths(assumptions, 500, 10, seed=9, scenario=SEQUENCE_CRASH)
    s, e = SEQUENCE_CRASH.window(10)
    jse = ASSETS.index("JSE_EQ")
    cum = lambda p: np.prod(1 + p.returns[:, s:e, jse].astype(float), axis=1)
    assert cum(crash).mean() / cum(base).mean() == pytest.approx(0.75, rel=0.01)
    # outside the window the paths are identical (common random numbers)
    assert np.array_equal(base.returns[:, :s], crash.returns[:, :s])
    assert np.array_equal(base.returns[:, e:], crash.returns[:, e:])
    assert np.array_equal(base.cpi[:, : s + 1], crash.cpi[:, : s + 1])


def test_stagflation_lifts_inflation(assumptions):
    base = generate_paths(assumptions, 500, 10, seed=9)
    stag = generate_paths(assumptions, 500, 10, seed=9, scenario=STAGFLATION)
    assert stag.annualised_inflation().mean() > base.annualised_inflation().mean() + 0.01
    assert stag.cpi[:, 84].mean() ** (1 / 7) - 1 == pytest.approx(0.085, abs=0.004)


def test_run_stress_hurts_every_scenario(assumptions, config, moderate):
    st = run_stress(config, moderate, assumptions)
    t = st.table
    assert list(t.index) == ["base", "stagflation", "risk_off", "sequence_crash"]
    assert t.loc["base", "change_vs_base"] == 0
    for key in ("stagflation", "risk_off", "sequence_crash"):
        assert t.loc[key, "median_real"] < t.loc["base", "median_real"]
        assert t.loc[key, "change_vs_base"] < 0
        assert t.loc[key, "prob_goal"] <= t.loc["base", "prob_goal"]
        assert t.loc[key, "cvar5_real"] < t.loc["base", "cvar5_real"]
    assert t.loc["stagflation", "mean_annual_inflation"] > t.loc["base", "mean_annual_inflation"]
    assert set(st.results) == {"base", "stagflation", "risk_off", "sequence_crash"}


def test_crash_hurts_equity_heavy_portfolios_more(assumptions, config):
    safe = np.array([0.05, 0.4, 0.0, 0.45, 0.05, 0.05])
    risky = np.array([0.4, 0.1, 0.1, 0.0, 0.35, 0.05])
    cfg = replace(config, wrapper=DISC)
    s_safe = run_stress(cfg, safe, assumptions, scenarios=(SEQUENCE_CRASH,)).table.loc["sequence_crash", "change_vs_base"]
    s_risky = run_stress(cfg, risky, assumptions, scenarios=(SEQUENCE_CRASH,)).table.loc["sequence_crash", "change_vs_base"]
    assert s_risky < s_safe < 0
