"""Living Annuity drawdown & ruin engine."""

from dataclasses import replace

import numpy as np
import pytest

from sa_wealth.assets import ASSETS, default_assumptions
from sa_wealth.drawdown import (
    LA_MAX_RATE,
    LA_MIN_RATE,
    LA_PORTFOLIOS,
    RETIREMENT_CRASH,
    DrawdownConfig,
    clamp_drawdown_rate,
    resolve_la_portfolio,
    run_crash_impact,
    run_drawdown,
    withdrawal_rate_grid,
)
from sa_wealth.montecarlo import generate_paths
from sa_wealth.reg28_tax import check_reg28


def cfg_for(rate, capital=8_000_000.0, **kw):
    """A plan whose first-year desired income is ``rate`` of the starting capital."""
    kw.setdefault("n_paths", 1_000)
    kw.setdefault("seed", 11)
    return DrawdownConfig(initial_capital=capital, monthly_income_real=rate * capital / 12.0, **kw)


@pytest.fixture(scope="module")
def base_cfg():
    return DrawdownConfig(n_paths=1_000, seed=11)


@pytest.fixture(scope="module")
def base_run(base_cfg):
    return run_drawdown(base_cfg)


# --------------------------------------------------------------------------- #
# statutory band
# --------------------------------------------------------------------------- #


def test_statutory_band_constants():
    assert (LA_MIN_RATE, LA_MAX_RATE) == (0.025, 0.175)


@pytest.mark.parametrize(
    "desired,expected",
    [(0.0, 0.025), (0.01, 0.025), (0.025, 0.025), (0.06, 0.06), (0.175, 0.175), (0.20, 0.175), (5.0, 0.175), (float("inf"), 0.175)],
)
def test_clamp_drawdown_rate(desired, expected):
    assert clamp_drawdown_rate(desired) == pytest.approx(expected)


def test_clamp_works_on_arrays():
    out = clamp_drawdown_rate(np.array([0.0, 0.05, 0.5]))
    assert out == pytest.approx([0.025, 0.05, 0.175])


@pytest.mark.parametrize("rate", [0.005, 0.03, 0.06, 0.10, 0.20, 0.35])
def test_applied_drawdown_rate_always_within_band(rate):
    r = run_drawdown(cfg_for(rate, n_paths=400))
    assert r.rates.min() >= LA_MIN_RATE - 1e-12
    assert r.rates.max() <= LA_MAX_RATE + 1e-12
    assert np.all(np.isfinite(r.rates))
    # income paid is exactly rate x anniversary capital
    assert r.income == pytest.approx(r.rates * r.capital[:, :-1])


def test_rate_is_clamped_to_floor_and_ceiling_at_the_extremes():
    low = run_drawdown(cfg_for(0.005, n_paths=300))
    assert np.allclose(low.rates[:, 0], LA_MIN_RATE)  # forced to draw the 2.5% minimum
    assert np.all(low.income[:, 0] >= low.desired[:, 0])  # retiree receives more than wanted
    high = run_drawdown(cfg_for(0.30, n_paths=300))
    assert np.allclose(high.rates[:, 0], LA_MAX_RATE)  # cannot draw more than 17.5%
    assert np.allclose(high.income[:, 0], LA_MAX_RATE * 8_000_000)
    assert np.all(high.income[:, 0] < high.desired[:, 0])


def test_rate_follows_desired_income_inside_the_band(base_run):
    r = base_run
    inside = (r.desired[:, :-1] / r.capital[:, :-1] > LA_MIN_RATE) & (r.desired[:, :-1] / r.capital[:, :-1] < LA_MAX_RATE)
    assert inside.any()
    assert r.income[inside] == pytest.approx(r.desired[:, :-1][inside])


# --------------------------------------------------------------------------- #
# survival
# --------------------------------------------------------------------------- #


def test_lower_initial_withdrawal_rate_has_higher_survival_4_vs_8_percent():
    safe = run_drawdown(cfg_for(0.04))
    risky = run_drawdown(cfg_for(0.08))
    assert safe.survival_prob() > risky.survival_prob()
    assert safe.survival_prob() > 0.9
    assert risky.survival_prob() < 0.5
    assert safe.median_bequest_real > risky.median_bequest_real
    assert safe.cap_breach_prob < risky.cap_breach_prob


def test_survival_falls_monotonically_with_withdrawal_rate():
    grid = withdrawal_rate_grid(DrawdownConfig(n_paths=800, seed=3), rates=(0.03, 0.05, 0.07, 0.09))
    assert grid["survival_prob"].is_monotonic_decreasing
    assert grid["cap_breach_prob"].is_monotonic_increasing
    assert grid["median_bequest_real"].is_monotonic_decreasing


def test_survival_curve_is_non_increasing_and_starts_at_one(base_run):
    c = base_run.survival_curve()
    assert c.iloc[0] == 1.0
    assert c.is_monotonic_decreasing
    assert c.index[0] == 65 and c.index[-1] == 95
    assert base_run.survival_prob() == c.iloc[-1]
    assert base_run.survival_prob(80) == c.loc[80]


def test_survival_milestones_use_ages_inside_the_horizon():
    r = run_drawdown(DrawdownConfig(horizon_years=25, n_paths=200, seed=1))
    assert list(r.survival_milestones()) == [90]
    r = run_drawdown(DrawdownConfig(horizon_years=35, n_paths=200, seed=1))
    assert list(r.survival_milestones()) == [90, 95]
    assert r.survival_milestones()[95] <= r.survival_milestones()[90]


def test_survival_age_outside_horizon_is_rejected(base_run):
    with pytest.raises(ValueError):
        base_run.survival_prob(60)
    with pytest.raises(ValueError):
        base_run.survival_prob(101)


def test_more_capital_means_higher_survival_for_the_same_income():
    a = run_drawdown(DrawdownConfig(initial_capital=6_000_000, n_paths=600, seed=2))
    b = run_drawdown(DrawdownConfig(initial_capital=10_000_000, n_paths=600, seed=2))
    assert b.survival_prob() > a.survival_prob()


def test_ruin_means_capital_below_the_income_multiple(base_run):
    r = base_run
    ruined = ~np.isnan(r.ruin_year)
    assert ruined.any()
    idx = r.ruin_year[ruined].astype(int)
    rows = np.where(ruined)[0]
    assert np.all(r.capital[rows, idx] < r.config.ruin_multiple * r.desired[rows, idx])
    # and in every earlier anniversary the fund was not ruined
    assert np.all(r.capital[rows[idx > 0], idx[idx > 0] - 1] >= r.desired[rows[idx > 0], idx[idx > 0] - 1])


# --------------------------------------------------------------------------- #
# ceiling breach
# --------------------------------------------------------------------------- #


def test_income_above_ceiling_breaches_in_year_zero():
    r = run_drawdown(cfg_for(0.20, n_paths=300))
    assert r.cap_breach_prob == 1.0
    assert np.all(r.breach_year == 0)
    assert r.median_years_to_cap_breach == 0.0
    assert r.cap_breach_curve().iloc[0] == 1.0


def test_no_breach_for_a_small_income():
    r = run_drawdown(cfg_for(0.02, n_paths=300))
    assert r.cap_breach_prob == 0.0
    assert np.isnan(r.median_years_to_cap_breach)
    assert r.share_years_income_shortfall == 0.0


def test_breach_year_is_the_first_year_desired_needs_more_than_the_ceiling(base_run):
    r = base_run
    hit = ~np.isnan(r.breach_year)
    assert hit.any()
    rows = np.where(hit)[0][:50]
    for i in rows:
        y = int(r.breach_year[i])
        need = r.desired[i, :-1] / r.capital[i, :-1]
        assert need[y] > LA_MAX_RATE
        assert np.all(need[:y] <= LA_MAX_RATE)
    # after a breach the income is clamped at the ceiling
    y = int(r.breach_year[rows[0]])
    assert r.rates[rows[0], y] == pytest.approx(LA_MAX_RATE)


def test_breach_curve_is_cumulative(base_run):
    c = base_run.cap_breach_curve()
    assert c.is_monotonic_increasing
    assert c.iloc[-1] == pytest.approx(base_run.cap_breach_prob)


# --------------------------------------------------------------------------- #
# cash flows, real vs nominal, inflation
# --------------------------------------------------------------------------- #


def test_shapes_and_initial_capital(base_run, base_cfg):
    r = base_run
    assert r.capital.shape == (1_000, 31) and r.income.shape == (1_000, 30) and r.rates.shape == (1_000, 30)
    assert np.all(r.capital[:, 0] == base_cfg.initial_capital)
    assert np.all(r.capital >= 0) and np.all(np.isfinite(r.capital))
    assert list(r.years) == list(range(31))


def test_real_and_nominal_are_linked_by_realised_cpi(base_run):
    r = base_run
    cpi_year = r.cpi[:, 12 * np.arange(31)]
    assert r.capital_real * cpi_year == pytest.approx(r.capital)
    assert r.income_real * cpi_year[:, :-1] == pytest.approx(r.income)
    assert np.all(r.capital_real[:, 0] == r.capital[:, 0])


def test_desired_income_escalates_with_the_stochastic_cpi_path(base_run, base_cfg):
    r = base_run
    expected = 12 * base_cfg.monthly_income_real * r.cpi[:, 12 * np.arange(31)]
    assert r.desired == pytest.approx(expected)
    assert r.desired[:, 0] == pytest.approx(12 * base_cfg.monthly_income_real)
    # CPI really is stochastic: different paths escalate differently
    assert r.desired[:, 10].std() > 0


def test_ample_capital_pays_the_desired_real_income_whenever_the_band_allows():
    r = run_drawdown(cfg_for(0.03, capital=20_000_000.0, n_paths=300))
    wanted_rate = r.desired[:, :-1] / r.capital[:, :-1]
    in_band = (wanted_rate >= LA_MIN_RATE) & (wanted_rate <= LA_MAX_RATE)
    assert in_band.mean() > 0.5
    assert r.income_real[in_band] == pytest.approx(np.full(in_band.sum(), 12 * r.config.monthly_income_real), rel=1e-9)
    # a fund that has grown so large that the wish falls below the 2.5% floor must still pay the minimum
    below = wanted_rate < LA_MIN_RATE
    assert np.all(r.rates[below] == LA_MIN_RATE)
    assert np.all(r.income[below] >= r.desired[:, :-1][below])


def test_annual_fee_lowers_the_bequest():
    free = run_drawdown(DrawdownConfig(n_paths=500, seed=4))
    fee = run_drawdown(DrawdownConfig(n_paths=500, seed=4, annual_fee=0.015))
    assert fee.median_bequest_real < free.median_bequest_real
    assert fee.survival_prob() <= free.survival_prob()


def test_reproducible_with_same_seed(base_cfg):
    a, b = run_drawdown(base_cfg), run_drawdown(base_cfg)
    assert np.array_equal(a.capital, b.capital)
    assert a.summary() == b.summary()
    c = run_drawdown(replace(base_cfg, seed=12))
    assert not np.array_equal(a.capital, c.capital)


def test_reusing_paths_gives_the_same_result(base_cfg):
    paths = generate_paths(default_assumptions(), base_cfg.n_paths, base_cfg.horizon_years, base_cfg.seed)
    assert np.array_equal(run_drawdown(base_cfg, paths=paths).capital, run_drawdown(base_cfg).capital)
    with pytest.raises(ValueError):
        run_drawdown(replace(base_cfg, horizon_years=25), paths=paths)


def test_percentile_frames(base_run):
    p = base_run.percentiles(real=True)
    assert {"P10", "P25", "P50", "P75", "P90"} <= set(p.columns)
    assert (p["P10"] <= p["P50"]).all() and (p["P50"] <= p["P90"]).all()
    assert p["P50"].iloc[-1] == pytest.approx(base_run.median_bequest_real)
    inc = base_run.income_percentiles(real=True)
    assert len(inc) == 30


def test_summary_contents(base_run):
    s = base_run.summary()
    for key in ("survival_prob", "median_bequest_real", "cap_breach_prob", "median_years_to_cap_breach", "survival_age_90", "survival_age_95"):
        assert key in s
    assert s["initial_withdrawal_rate"] == pytest.approx(0.06)
    assert s["end_age"] == 95 and s["n_paths"] == 1_000


# --------------------------------------------------------------------------- #
# sequence-of-returns crash
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def impact():
    return run_crash_impact(DrawdownConfig(n_paths=1_000, seed=11))


def test_crash_scenario_definition():
    assert RETIREMENT_CRASH.shocks["JSE_EQ"] == -0.25 and RETIREMENT_CRASH.shocks["GLB_EQ"] == -0.25
    assert RETIREMENT_CRASH.anchor == "start" and RETIREMENT_CRASH.start_year == 0 and RETIREMENT_CRASH.window_years == 1


def test_year_one_crash_hurts_survival_and_bequest(impact):
    assert impact.crash.survival_prob() < impact.base.survival_prob()
    assert impact.survival_drop > 0.05
    assert impact.crash.median_bequest_real < impact.base.median_bequest_real
    assert impact.bequest_drop_pct < 0
    assert impact.crash.cap_breach_prob > impact.base.cap_breach_prob
    assert impact.crash.median_years_to_cap_breach <= impact.base.median_years_to_cap_breach


def test_crash_run_uses_the_same_random_numbers_and_leaves_base_untouched(impact):
    plain = run_drawdown(DrawdownConfig(n_paths=1_000, seed=11))
    assert np.array_equal(impact.base.capital, plain.capital)
    assert np.array_equal(impact.base.cpi, impact.crash.cpi)  # the shock does not touch inflation
    # the first-year hit is large, and the gap shows up immediately
    assert np.median(impact.crash.capital_real[:, 1]) < 0.9 * np.median(impact.base.capital_real[:, 1])
    assert np.all(impact.crash.rates >= LA_MIN_RATE - 1e-12) and np.all(impact.crash.rates <= LA_MAX_RATE + 1e-12)


def test_crash_table_and_formatting(impact):
    t = impact.table
    assert list(t.columns) == ["Base", RETIREMENT_CRASH.name, "Change"]
    assert t.loc["Survival probability (end of horizon)", "Change"] < 0
    shown = impact.formatted_table(lambda v: f"R{v:,.0f}")
    assert shown.loc["Survival probability (end of horizon)", "Base"].endswith("%")
    assert shown.loc["Survival probability (end of horizon)", "Change"].endswith("pp")
    assert shown.loc["Median real bequest (today's rand)", "Base"].startswith("R")


def test_safer_withdrawal_rate_is_more_robust_to_the_crash():
    safe = run_crash_impact(cfg_for(0.04, n_paths=600))
    risky = run_crash_impact(cfg_for(0.07, n_paths=600))
    assert safe.crash.survival_prob() > risky.crash.survival_prob()


# --------------------------------------------------------------------------- #
# portfolios & validation
# --------------------------------------------------------------------------- #


def test_aggressive_offshore_is_allowed_because_living_annuities_skip_reg28():
    assets = default_assumptions().assets
    _, w = resolve_la_portfolio("Aggressive Offshore (no Reg 28 cap)", assets)
    report = check_reg28(w, assets=assets)
    assert not report.compliant and report.offshore > 0.45
    r = run_drawdown(DrawdownConfig(n_paths=200, seed=1), portfolio="aggressive-offshore")
    assert not r.reg28_compliant and r.portfolio.startswith("Aggressive Offshore")
    assert np.all(np.isfinite(r.capital))


def test_the_two_conservative_portfolios_respect_reg28():
    assets = default_assumptions().assets
    for name in ("Conservative (Reg 28)", "Moderate (Reg 28)"):
        assert check_reg28(resolve_la_portfolio(name, assets)[1], assets=assets).compliant
    assert list(LA_PORTFOLIOS)[:2] == ["Conservative (Reg 28)", "Moderate (Reg 28)"]


def test_portfolio_resolution_by_slug_mapping_and_vector():
    assets = default_assumptions().assets
    assert resolve_la_portfolio("moderate", assets)[0] == "Moderate (Reg 28)"
    assert resolve_la_portfolio("conservative", assets)[0] == "Conservative (Reg 28)"
    assert resolve_la_portfolio("aggressive-offshore", assets)[0].startswith("Aggressive Offshore")
    label, w = resolve_la_portfolio({"JSE_EQ": 1, "SA_CASH": 1}, assets)
    assert label == "Custom" and w.sum() == pytest.approx(1.0) and w[ASSETS.index("JSE_EQ")] == pytest.approx(0.5)
    assert resolve_la_portfolio(np.ones(len(assets)), assets)[1].sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        resolve_la_portfolio("no-such-portfolio", assets)
    with pytest.raises(ValueError):
        resolve_la_portfolio(np.ones(3), assets)


def test_riskier_portfolio_changes_the_outcome_distribution():
    cons = run_drawdown(DrawdownConfig(n_paths=600, seed=5), portfolio="conservative")
    aggr = run_drawdown(DrawdownConfig(n_paths=600, seed=5), portfolio="aggressive-offshore")
    def spread(r):
        return np.subtract(*np.percentile(r.capital_real[:, -1], [90, 10]))

    assert spread(aggr) > spread(cons)


@pytest.mark.parametrize(
    "kwargs",
    [dict(initial_capital=0), dict(monthly_income_real=-1), dict(horizon_years=3), dict(horizon_years=50), dict(n_paths=10),
     dict(min_rate=0.2, max_rate=0.1), dict(annual_fee=0.2), dict(ruin_multiple=0)],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        DrawdownConfig(**kwargs)


def test_config_helpers():
    c = DrawdownConfig(initial_capital=8_000_000, monthly_income_real=40_000, horizon_years=30, retirement_age=65)
    assert c.initial_withdrawal_rate == pytest.approx(0.06)
    assert c.max_initial_monthly_income == pytest.approx(116_666.67, abs=0.01)
    assert c.end_age == 95 and c.n_months == 360
