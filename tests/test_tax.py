"""SA tax wrappers: RA, TFSA and taxable Discretionary."""

import numpy as np
import pytest

from sa_wealth.assets import ASSETS, INCOME_CLASS
from sa_wealth.montecarlo import build_contributions, expected_return_paths
from sa_wealth.reg28_tax import (
    DISC,
    GROSS,
    RA,
    TAX_2025_26,
    TAX_2026_27,
    TFSA,
    Reg28Violation,
    apply_tfsa_limits,
    compare_wrappers,
    is_reg28_compliant,
    lump_sum_tax,
    ra_exit_tax,
    simulate_wrapper,
)

A = len(ASSETS)
CLASSES = tuple(INCOME_CLASS[a] for a in ASSETS)


def wvec(**kw):
    out = np.zeros(A)
    for k, v in kw.items():
        out[ASSETS.index(k)] = v
    return out


def flat(months, ret=0.0, paths=1):
    return np.full((paths, months, A), ret), np.ones((paths, months + 1))


def run(wrapper, weights, returns, cpi, contributions, yields=None, **kw):
    yields = np.zeros(A) if yields is None else yields
    return simulate_wrapper(returns, cpi, weights, contributions, wrapper, yields, CLASSES, **kw)


def lump(months, amount, monthly=0.0):
    c = np.full((1, months), monthly, dtype=float)
    c[0, 0] += amount
    return c


# ----------------------------------------------------------------------------- statutory parameters


def test_spec_parameters():
    t = TAX_2025_26
    assert t.dwt_rate == 0.20
    assert t.interest_exemption == 23_800
    assert t.cgt_inclusion == 0.40 and t.cgt_annual_exclusion == 40_000
    assert t.tfsa_annual_limit == 36_000 and t.tfsa_lifetime_limit == 500_000
    assert t.ra_deduction_pct == 0.275 and t.ra_deduction_cap == 350_000
    assert t.effective_cgt_rate(0.45) == pytest.approx(0.18)  # 40% inclusion x 45% top marginal rate


def test_budget_2026_preset_differs_only_where_it_should():
    assert TAX_2026_27.tfsa_annual_limit == 46_000
    assert TAX_2026_27.ra_deduction_cap == 430_000
    assert TAX_2026_27.cgt_annual_exclusion == 50_000
    assert TAX_2026_27.tfsa_lifetime_limit == TAX_2025_26.tfsa_lifetime_limit
    assert TAX_2026_27.interest_exemption == TAX_2025_26.interest_exemption


def test_lump_sum_table():
    assert lump_sum_tax(550_000) == 0
    assert lump_sum_tax(770_000) == pytest.approx(39_600)
    assert lump_sum_tax(1_155_000) == pytest.approx(143_550)
    assert lump_sum_tax(1_500_000) == pytest.approx(143_550 + 0.36 * 345_000)
    assert np.allclose(lump_sum_tax(np.array([0.0, 660_000.0])), [0.0, 0.18 * 110_000])


def test_ra_exit_tax_is_indexed_to_inflation():
    rate = 0.31
    # R1.65m in today's rand: lump sum R550k is inside the tax-free band, annuity taxed at 31%
    assert ra_exit_tax(np.array([1_650_000.0]), np.array([1.0]), TAX_2025_26, rate)[0] == pytest.approx(1_100_000 * rate)
    # the same real wealth after 14 years of ~5.5% inflation (CPI index 2.0) is taxed identically in real terms
    assert ra_exit_tax(np.array([3_300_000.0]), np.array([2.0]), TAX_2025_26, rate)[0] == pytest.approx(2_200_000 * rate)


# ----------------------------------------------------------------------------- TFSA limits


def test_tfsa_annual_limit_enforced():
    amounts = np.full((1, 36), 10_000.0)  # R120k a year offered
    inside, overflow = apply_tfsa_limits(amounts)
    for y in range(3):
        assert inside[0, 12 * y : 12 * (y + 1)].sum() == pytest.approx(36_000)
    assert np.allclose(inside + overflow, amounts)
    assert inside.max() <= 10_000


def test_tfsa_lifetime_limit_enforced():
    amounts = np.full((2, 12 * 30), 5_000.0)  # R60k a year for 30 years
    inside, overflow = apply_tfsa_limits(amounts)
    assert np.allclose(inside.sum(axis=1), 500_000)  # R36k x 13.9 years hits the R500k ceiling
    assert np.allclose(overflow.sum(axis=1), amounts.sum(axis=1) - 500_000)
    # once the lifetime cap is reached nothing more enters the TFSA
    assert np.all(inside[:, -60:] == 0)
    for y in range(30):
        assert inside[:, 12 * y : 12 * (y + 1)].sum(axis=1).max() <= 36_000 + 1e-9


def test_tfsa_accepts_small_contributions_in_full():
    amounts = np.full((1, 24), 2_000.0)  # R24k a year, below both limits
    inside, overflow = apply_tfsa_limits(amounts)
    assert np.allclose(inside, amounts) and overflow.sum() == 0


def test_tfsa_limits_are_configurable_and_1d_input_supported():
    inside, overflow = apply_tfsa_limits(np.full(12, 10_000.0), TAX_2026_27.tfsa_annual_limit, 500_000)
    assert inside.sum() == pytest.approx(46_000) and inside.shape == (12,)


def test_tfsa_wrapper_respects_limits_end_to_end():
    months = 12 * 20
    r, cpi = flat(months, 0.004)
    out = run(TFSA, wvec(JSE_EQ=1.0), r, cpi, lump(months, 0, 10_000))
    assert out.tfsa_contributed[0] == pytest.approx(500_000)
    assert out.overflow_balance[0] > 0 and out.tfsa_balance[0] > 500_000
    # 5-year horizon: the annual limit is what binds
    r5, cpi5 = flat(60, 0.004)
    out5 = run(TFSA, wvec(JSE_EQ=1.0), r5, cpi5, lump(60, 0, 10_000))
    assert out5.tfsa_contributed[0] == pytest.approx(5 * 36_000)


# ----------------------------------------------------------------------------- Discretionary taxes


def test_cgt_on_liquidation_uses_inclusion_rate_and_exclusion():
    r, cpi = flat(12)
    r[:, 11, :] = 0.5  # +50% in the final month
    out = run(DISC, wvec(JSE_EQ=1.0), r, cpi, lump(12, 1_000_000), marginal_rate=0.45)
    expected = 0.40 * 0.45 * (500_000 - 40_000)  # R82,800
    assert out.tax_cgt[0] == pytest.approx(expected)
    assert out.terminal_net[0] == pytest.approx(1_500_000 - expected)
    assert out.tax_income[0] == 0


def test_cgt_annual_exclusion_wipes_out_small_gains():
    r, cpi = flat(12)
    r[:, 11, :] = 0.03  # R30k gain < R40k exclusion
    out = run(DISC, wvec(JSE_EQ=1.0), r, cpi, lump(12, 1_000_000), marginal_rate=0.45)
    assert out.tax_cgt[0] == 0


def test_cgt_realised_on_rebalancing_then_on_liquidation():
    months = 24
    r, cpi = flat(months)
    r[:, 11, ASSETS.index("JSE_EQ")] = 1.0  # JSE doubles in year 1
    w = wvec(JSE_EQ=0.5, SA_CASH=0.5)
    out = run(DISC, w, r, cpi, lump(months, 1_000_000), marginal_rate=0.45)
    # year-end 1: sell R250k of a R1m JSE holding with R500k cost -> R125k realised gain
    cgt1 = 0.18 * (125_000 - 40_000)
    f = 1 - cgt1 / 1_500_000  # tax is paid by pro-rata disinvestment
    cgt2 = 0.18 * (375_000 * f - 40_000)  # remaining unrealised gain realised on liquidation
    assert out.tax_cgt[0] == pytest.approx(cgt1 + cgt2, rel=1e-9)


def test_no_rebalancing_defers_cgt_until_liquidation():
    months = 24
    r, cpi = flat(months)
    r[:, 11, ASSETS.index("JSE_EQ")] = 1.0
    w = wvec(JSE_EQ=0.5, SA_CASH=0.5)
    out = run(DISC, w, r, cpi, lump(months, 1_000_000), marginal_rate=0.45, rebalance=False)
    assert out.tax_cgt[0] == pytest.approx(0.18 * (500_000 - 40_000))


def test_dividend_withholding_tax_is_20_percent():
    months = 12
    r, cpi = flat(months)
    y = wvec(JSE_EQ=0.03)
    out = run(DISC, wvec(JSE_EQ=1.0), r, cpi, lump(months, 1_000_000), yields=y, marginal_rate=0.45)
    assert out.tax_income[0] == pytest.approx(0.20 * 0.03 * 1_000_000, rel=0.01)  # independent of marginal rate


def test_interest_exemption():
    months = 12
    r, cpi = flat(months)
    y = wvec(SA_CASH=0.07)
    small = run(DISC, wvec(SA_CASH=1.0), r, cpi, lump(months, 200_000), yields=y, marginal_rate=0.45)
    assert small.tax_income[0] == 0  # R14k of interest is inside the R23,800 exemption
    big = run(DISC, wvec(SA_CASH=1.0), r, cpi, lump(months, 1_000_000), yields=y, marginal_rate=0.45)
    assert big.tax_income[0] == pytest.approx(0.45 * (70_000 - 23_800), rel=0.02)


def test_property_distributions_taxed_at_marginal_rate_without_exemption():
    r, cpi = flat(12)
    out = run(DISC, wvec(SA_PROP=1.0), r, cpi, lump(12, 1_000_000), yields=wvec(SA_PROP=0.075), marginal_rate=0.41)
    assert out.tax_income[0] == pytest.approx(0.41 * 75_000, rel=0.02)


# ----------------------------------------------------------------------------- RA


def test_ra_has_no_internal_tax_only_exit_tax(assumptions):
    months = 120
    a = assumptions
    p = expected_return_paths(a, 10)
    r, cpi = p.returns, p.cpi
    w = wvec(JSE_EQ=0.4, SA_BOND=0.3, GLB_EQ=0.3)
    c = lump(months, 100_000, 3_000)
    out = run(RA, w, r, cpi, c, yields=a.income_yield, marginal_rate=0.41)
    assert out.tax_income[0] == 0 and out.tax_cgt[0] == 0  # 0% on dividends, interest and CGT
    assert out.tax_exit[0] > 0
    assert out.terminal_net[0] == pytest.approx(out.terminal_pre_exit[0] - out.tax_exit[0])


def test_ra_refund_is_marginal_rate_times_deduction():
    r, cpi = flat(12)
    c = lump(12, 250_000, 5_000)  # R250k + 12 x R5k = R310k in the year
    out = run(RA, wvec(SA_CASH=1.0), r, cpi, c, marginal_rate=0.41, annual_income=1_300_000)  # 27.5% of income = R357.5k
    assert out.contributions_total[0] == pytest.approx(310_000)
    assert out.refunds[0] == pytest.approx(0.41 * 310_000)


def test_ra_deduction_capped_at_r350k():
    r, cpi = flat(12)
    c = lump(12, 600_000)  # R600k contributed
    out = run(RA, wvec(SA_CASH=1.0), r, cpi, c, marginal_rate=0.45, annual_income=2_000_000)  # 27.5% = R550k > cap
    assert out.refunds[0] == pytest.approx(0.45 * 350_000)


def test_ra_deduction_capped_at_27_5_percent_of_income():
    r, cpi = flat(12)
    c = lump(12, 300_000)
    out = run(RA, wvec(SA_CASH=1.0), r, cpi, c, marginal_rate=0.36, annual_income=400_000)  # 27.5% = R110k
    assert out.refunds[0] == pytest.approx(0.36 * 110_000)


def test_ra_refund_reinvestment_toggle():
    months = 60
    r, cpi = flat(months, 0.005)
    c = lump(months, 0, 10_000)
    on = run(RA, wvec(SA_BOND=1.0), r, cpi, c, marginal_rate=0.41, annual_income=1_300_000, reinvest_refund=True)
    off = run(RA, wvec(SA_BOND=1.0), r, cpi, c, marginal_rate=0.41, annual_income=1_300_000, reinvest_refund=False)
    assert on.refunds[0] == pytest.approx(off.refunds[0])
    assert on.terminal_pre_exit[0] > off.terminal_pre_exit[0] + 0.9 * on.refunds[0]


def test_ra_enforces_regulation_28():
    r, cpi = flat(12)
    c = lump(12, 100_000)
    with pytest.raises(Reg28Violation):
        run(RA, wvec(GLB_EQ=1.0), r, cpi, c)
    with pytest.raises(Reg28Violation):
        run(RA, wvec(JSE_EQ=0.6, GLB_EQ=0.2, SA_CASH=0.2), r, cpi, c)  # 80% equity
    ok = run(RA, wvec(JSE_EQ=0.3, GLB_EQ=0.4, SA_CASH=0.3), r, cpi, c)
    assert ok.reg28_report.compliant
    assert run(RA, wvec(GLB_EQ=1.0), r, cpi, c, enforce_reg28=False).reg28_report.compliant is False


# ----------------------------------------------------------------------------- compounding order


def test_tfsa_beats_discretionary_for_identical_returns_deterministic(assumptions, moderate):
    p = expected_return_paths(assumptions, 25)
    cfg_contrib = np.full((1, 300), 4_000.0)
    cfg_contrib[0, 0] += 100_000
    kw = dict(income_yield=assumptions.income_yield, income_class=CLASSES, marginal_rate=0.41)
    tfsa = simulate_wrapper(p.returns, p.cpi, moderate, cfg_contrib, TFSA, **kw)
    disc = simulate_wrapper(p.returns, p.cpi, moderate, cfg_contrib, DISC, **kw)
    gross = simulate_wrapper(p.returns, p.cpi, moderate, cfg_contrib, GROSS, **kw)
    assert gross.terminal_net[0] > tfsa.terminal_net[0] > disc.terminal_net[0]
    assert tfsa.tax_paid[0] < disc.tax_paid[0]
    assert tfsa.terminal_net[0] > disc.terminal_net[0] * 1.01  # a material, not rounding-level, gap


def test_tfsa_beats_discretionary_on_every_simulated_path(assumptions, config, paths, moderate):
    contrib = build_contributions(config, paths.cpi)
    kw = dict(income_yield=assumptions.income_yield, income_class=CLASSES, marginal_rate=0.36)
    tfsa = simulate_wrapper(paths.returns, paths.cpi, moderate, contrib, TFSA, **kw)
    disc = simulate_wrapper(paths.returns, paths.cpi, moderate, contrib, DISC, **kw)
    gross = simulate_wrapper(paths.returns, paths.cpi, moderate, contrib, GROSS, **kw)
    assert np.all(tfsa.terminal_net >= disc.terminal_net - 1e-6)
    assert tfsa.terminal_net.mean() > disc.terminal_net.mean()
    assert np.all(gross.terminal_net >= tfsa.terminal_net - 1e-6)  # tax can only reduce wealth
    assert np.all(disc.tax_paid >= tfsa.tax_paid - 1e-6)


def test_tfsa_fully_tax_free_inside_the_limits(assumptions, moderate):
    p = expected_return_paths(assumptions, 10)
    c = np.full((1, 120), 1_000.0)  # R12k a year: inside the R36k limit
    kw = dict(income_yield=assumptions.income_yield, income_class=CLASSES, marginal_rate=0.45)
    tfsa = simulate_wrapper(p.returns, p.cpi, moderate, c, TFSA, **kw)
    gross = simulate_wrapper(p.returns, p.cpi, moderate, c, GROSS, **kw)
    assert tfsa.tax_paid[0] == 0
    assert tfsa.terminal_net[0] == pytest.approx(gross.terminal_net[0])
    assert tfsa.overflow_balance[0] == 0


def test_compare_wrappers_projects_ra_allocation(assumptions, config, paths):
    contrib = build_contributions(config, paths.cpi)
    aggressive = wvec(JSE_EQ=0.15, GLB_EQ=0.8, SA_BOND=0.05)  # 95% equity, 80% offshore
    out = compare_wrappers(paths.returns[:200], paths.cpi[:200], aggressive, contrib[:200], assumptions.income_yield, CLASSES, marginal_rate=0.41)
    assert set(out) == {RA, TFSA, DISC, GROSS}
    assert is_reg28_compliant(out[RA].weights)
    assert np.allclose(out[TFSA].weights, aggressive) and np.allclose(out[DISC].weights, aggressive)


# ----------------------------------------------------------------------------- validation


def test_input_validation():
    r, cpi = flat(12)
    c = lump(12, 1000)
    with pytest.raises(ValueError):
        run("ISA", wvec(SA_CASH=1.0), r, cpi, c)
    with pytest.raises(ValueError):
        run(DISC, wvec(SA_CASH=0.5), r, cpi, c)  # does not sum to 1
    r7, cpi7 = flat(7)
    with pytest.raises(ValueError):
        run(DISC, wvec(SA_CASH=1.0), r7, cpi7, lump(7, 1000))  # not whole years
