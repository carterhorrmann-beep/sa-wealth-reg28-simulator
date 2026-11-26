"""SA tax wrappers (RA, TFSA, Discretionary), SARS income-tax tables and the Smart Contribution Allocator."""

import math

import numpy as np
import pytest

from sa_wealth.assets import ASSETS, INCOME_CLASS
from sa_wealth.montecarlo import build_contributions, expected_return_paths
from sa_wealth.reg28_tax import (
    DISC,
    GROSS,
    INCOME_TAX_TABLES,
    RA,
    SARS_2025_26,
    SARS_2026_27,
    TAX_2025_26,
    TAX_2026_27,
    TFSA,
    IncomeTaxTable,
    Reg28Violation,
    apply_tfsa_limits,
    compare_wrappers,
    income_tax,
    income_tax_profile,
    is_reg28_compliant,
    lump_sum_tax,
    marginal_tax_rate,
    optimize_wrapper_split,
    ra_arbitrage_threshold,
    ra_deduction_limit,
    ra_exit_tax,
    ra_tax_refund,
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


# --------------------------------------------------------------------------- #
# SARS individual income tax tables (hand-worked examples)
# --------------------------------------------------------------------------- #
def test_sars_2026_27_table_constants():
    assert [b for b, _ in SARS_2026_27.brackets[:-1]] == [245_100, 383_100, 530_200, 695_800, 887_000, 1_878_600]
    assert [r for _, r in SARS_2026_27.brackets] == [0.18, 0.26, 0.31, 0.36, 0.39, 0.41, 0.45]
    assert (SARS_2026_27.primary_rebate, SARS_2026_27.secondary_rebate, SARS_2026_27.tertiary_rebate) == (17_820, 9_765, 3_249)
    # the 'R x + y% above R z' constants published by SARS
    assert [round(x) for x in SARS_2026_27.cumulative_base] == [0, 44_118, 79_998, 125_599, 185_215, 259_783, 666_339]
    assert INCOME_TAX_TABLES["2026/27"] is SARS_2026_27 and INCOME_TAX_TABLES["2025/26"] is SARS_2025_26


def test_sars_2025_26_table_is_the_previous_year():
    assert SARS_2025_26.brackets[0] == (237_100, 0.18) and SARS_2025_26.brackets[-2][0] == 1_817_000
    assert (SARS_2025_26.primary_rebate, SARS_2025_26.secondary_rebate, SARS_2025_26.tertiary_rebate) == (17_235, 9_444, 3_145)
    assert [round(x) for x in SARS_2025_26.cumulative_base] == [0, 42_678, 77_362, 121_475, 179_147, 251_258, 644_489]


def test_tax_presets_carry_their_own_income_tax_table():
    assert TAX_2026_27.income_tax_table is SARS_2026_27
    assert TAX_2025_26.income_tax_table is SARS_2025_26


def test_income_tax_hand_worked_examples_2026_27():
    # R500,000: 79,998 + 31% x (500,000 - 383,100) = 116,237.00 before rebates
    assert SARS_2026_27.tax_before_rebates(500_000) == pytest.approx(116_237.0)
    assert income_tax(500_000, age=40) == pytest.approx(116_237.0 - 17_820.0)  # 98,417
    assert income_tax(500_000, age=66) == pytest.approx(116_237.0 - 17_820.0 - 9_765.0)  # 88,652
    assert income_tax(500_000, age=76) == pytest.approx(116_237.0 - 17_820.0 - 9_765.0 - 3_249.0)  # 85,403
    # R1,000,000: 259,783 + 41% x 113,000 = 306,113
    assert income_tax(1_000_000) == pytest.approx(306_113.0 - 17_820.0)
    # R2,000,000: 666,339 + 45% x 121,400 = 720,969
    assert income_tax(2_000_000) == pytest.approx(720_969.0 - 17_820.0)


def test_income_tax_hand_worked_example_2025_26_for_comparison():
    # R500,000: 77,362 + 31% x (500,000 - 370,500) = 117,507; less primary rebate 17,235
    assert income_tax(500_000, 40, SARS_2025_26) == pytest.approx(117_507.0 - 17_235.0)


@pytest.mark.parametrize("age,threshold", [(40, 99_000), (66, 153_250), (76, 171_300)])
def test_tax_thresholds_by_age(age, threshold):
    assert SARS_2026_27.tax_threshold(age) == pytest.approx(threshold)
    assert income_tax(threshold, age) == pytest.approx(0.0, abs=1e-6)
    assert income_tax(threshold + 10_000, age) == pytest.approx(1_800.0)
    assert income_tax(threshold - 1, age) == 0.0


def test_income_tax_never_negative_and_monotone_and_vectorised():
    inc = np.array([-5_000.0, 0.0, 50_000.0, 99_000.0, 300_000.0, 900_000.0, 3_000_000.0])
    tax = income_tax(inc)
    assert np.all(tax >= 0) and np.all(np.diff(tax) >= 0)
    assert tax[0] == tax[1] == 0.0
    assert tax[4] == pytest.approx(income_tax(300_000.0))


@pytest.mark.parametrize(
    "income,rate",
    [(0, 0.0), (100_000, 0.18), (245_100, 0.18), (245_101, 0.26), (383_100, 0.26), (500_000, 0.31), (650_000, 0.36),
     (800_000, 0.39), (1_000_000, 0.41), (1_878_600, 0.41), (1_878_601, 0.45), (10_000_000, 0.45)],
)
def test_marginal_rate_by_bracket(income, rate):
    assert marginal_tax_rate(income) == rate


def test_effective_rate_and_profile_fields():
    p = income_tax_profile(650_000, age=40)
    assert p.tax_year == "2026/27"
    # 125,599 + 36% x 119,800 - 17,820 = 150,907
    assert p.tax_payable == pytest.approx(150_907.0, abs=0.5)
    assert p.effective_rate == pytest.approx(p.tax_payable / 650_000)
    assert (p.marginal_rate, p.bracket_number, p.bracket_floor, p.bracket_ceiling) == (0.36, 4, 530_200, 695_800)
    assert p.ra_deduction_limit == pytest.approx(178_750.0)  # 27.5% x 650,000
    # deducting 178,750 takes income to 471,250: tax 89,504.50, so the refund is 61,402.50
    assert p.ra_refund == pytest.approx(61_402.5, abs=0.5)
    assert p.ra_refund_rate == pytest.approx(61_402.5 / 178_750, abs=1e-4)
    assert p.tax_after_ra == pytest.approx(89_504.5, abs=0.5)


def test_ra_refund_inside_one_bracket_is_marginal_rate_times_contribution():
    assert ra_tax_refund(100_000, 650_000) == pytest.approx(36_000.0)  # whole deduction taxed at 36%
    assert ra_tax_refund(0, 650_000) == 0.0


def test_ra_refund_spanning_two_brackets_is_exact():
    # income 400,000, deduct 100,000: 16,900 at 31% + 83,100 at 26% = 26,845 (not 100,000 x 31% = 31,000)
    assert ra_tax_refund(100_000, 400_000) == pytest.approx(26_845.0)


def test_ra_refund_respects_27_5_percent_limit():
    assert ra_deduction_limit(400_000) == pytest.approx(110_000.0)
    # a R200,000 contribution only deducts R110,000: refund = tax(400,000) - tax(290,000) = 29,445
    assert ra_tax_refund(200_000, 400_000) == pytest.approx(29_445.0)
    assert ra_tax_refund(200_000, 400_000) == pytest.approx(ra_tax_refund(110_000, 400_000))


def test_ra_refund_respects_r430k_cap():
    assert ra_deduction_limit(3_000_000) == 430_000.0
    assert ra_tax_refund(1_000_000, 3_000_000) == pytest.approx(0.45 * 430_000)  # all in the 45% bracket
    # 2025/26 cap was R350,000
    assert ra_deduction_limit(3_000_000, TAX_2025_26) == 350_000.0
    assert ra_tax_refund(1_000_000, 3_000_000, tax=TAX_2025_26) == pytest.approx(0.45 * 350_000)


def test_ra_refund_is_zero_below_the_tax_threshold():
    assert ra_tax_refund(20_000, 90_000) == 0.0
    assert income_tax_profile(90_000).tax_payable == 0.0


def test_ra_refund_accepts_arrays():
    out = ra_tax_refund(np.array([0.0, 100_000.0, 200_000.0]), 400_000)
    assert out == pytest.approx([0.0, 26_845.0, 29_445.0])


def test_income_tax_table_validation():
    with pytest.raises(ValueError):
        IncomeTaxTable("bad", ((200_000.0, 0.18), (100_000.0, 0.26), (math.inf, 0.31)), 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        IncomeTaxTable("bad", ((100_000.0, 0.18), (200_000.0, 0.26)), 1.0, 1.0, 1.0)  # no open top bracket


def test_arbitrage_threshold():
    # exit tax 31%: deducting is worthwhile while the bracket rate is >= 31%, i.e. down to R383,100; capped by the 27.5% limit
    assert ra_arbitrage_threshold(650_000, 0.31) == pytest.approx(178_750.0)
    assert ra_arbitrage_threshold(450_000, 0.31) == pytest.approx(450_000 - 383_100)
    assert ra_arbitrage_threshold(650_000, 0.40) == 0.0  # only the 41% bracket qualifies, income never reaches it
    assert ra_arbitrage_threshold(1_200_000, 0.40) == pytest.approx(min(1_200_000 - 887_000, 330_000))


# --------------------------------------------------------------------------- #
# Smart Contribution Allocator
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def split_36():
    return optimize_wrapper_split(10_000, 650_000, None, 20, n_paths=400, seed=3)


def test_blended_ra_tfsa_beats_discretionary_only(split_36):
    disc, blend = split_36.strategies["discretionary"], split_36.strategies["blended"]
    assert split_36.marginal_rate == 0.36
    assert np.median(blend.terminal_real) > np.median(disc.terminal_real)
    assert blend.terminal_real.mean() > disc.terminal_real.mean()
    assert np.mean(blend.terminal_net > disc.terminal_net) > 0.99
    gain = split_36.gain_vs("discretionary")
    assert gain["terminal_wealth_gain_real"] > 0
    assert gain["terminal_wealth_gain_nominal"] > gain["terminal_wealth_gain_real"]
    assert gain["lifetime_tax_saved_real"] > 0


def test_blended_beats_the_best_single_wrapper(split_36):
    gain = split_36.gain_vs(split_36.best_single)
    assert gain["terminal_wealth_gain_real"] > 0
    assert gain["share_paths_better"] > 0.9


def test_blended_never_worse_than_ra_only_when_refund_is_reinvested(split_36):
    assert split_36.gain_vs("ra_only")["terminal_wealth_gain_real"] > 0


def test_split_sums_to_budget_and_respects_statutory_limits(split_36):
    for st in split_36.strategies.values():
        assert sum(st.monthly_split.values()) == pytest.approx(10_000)
        assert st.monthly_split["TFSA"] <= TAX_2026_27.tfsa_annual_limit / 12 + 1e-9
        assert st.monthly_split["RA"] <= ra_deduction_limit(650_000) / 12 + 1e-9
    assert split_36.strategies["discretionary"].monthly_split == {"RA": 0.0, "TFSA": 0.0, "DISC": 10_000.0}
    assert split_36.strategies["ra_only"].monthly_split["TFSA"] == 0.0


def test_large_budget_uses_all_three_wrappers():
    r = optimize_wrapper_split(40_000, 1_000_000, None, 15, n_paths=300, seed=1)
    sp = r.optimal_split
    assert sp["RA"] == pytest.approx(275_000 / 12)  # statutory RA limit binds
    assert sp["TFSA"] == pytest.approx(46_000 / 12)  # TFSA limit binds
    assert sp["DISC"] == pytest.approx(40_000 - sp["RA"] - sp["TFSA"])
    assert r.profile.marginal_rate == 0.41


def test_naive_ra_only_is_capped_with_overflow_to_discretionary():
    r = optimize_wrapper_split(30_000, 300_000, None, 10, n_paths=200, seed=1)  # limit = 82,500 a year = 6,875 a month
    ra_only = r.strategies["ra_only"].monthly_split
    assert ra_only["RA"] == pytest.approx(82_500 / 12)
    assert ra_only["DISC"] == pytest.approx(30_000 - 82_500 / 12)


def test_no_ra_for_income_below_tax_threshold():
    r = optimize_wrapper_split(3_000, 90_000, None, 10, n_paths=200, seed=1)  # no tax payable, so no refund to chase
    assert r.optimal_split["RA"] == 0.0
    assert r.optimal_split["TFSA"] == pytest.approx(3_000)


def test_reinvesting_the_refund_raises_blended_wealth():
    on = optimize_wrapper_split(10_000, 650_000, None, 15, n_paths=300, seed=5, reinvest_refund=True)
    off = optimize_wrapper_split(10_000, 650_000, None, 15, n_paths=300, seed=5, reinvest_refund=False)
    assert np.median(on.blended.terminal_real) > np.median(off.blended.terminal_real)
    assert on.blended.refunds.mean() > 0
    assert on.blended.refunds_reinvested and not off.blended.refunds_reinvested
    # spent refunds never enter terminal wealth, so the optimiser has no reason to chase them: it uses less RA
    assert off.optimal_ra_monthly <= on.optimal_ra_monthly


def test_refunds_match_the_exact_sars_calculation_in_year_one():
    r = optimize_wrapper_split(5_000, 650_000, None, 5, n_paths=100, seed=2, reinvest_refund=False)
    # year-1 RA contribution is R60,000 -> refund = 36% x 60,000 = 21,600; later years are larger in nominal terms
    assert r.strategies["ra_only"].refunds.mean() > 5 * 21_600
    assert r.profile.ra_refund == pytest.approx(ra_tax_refund(r.optimal_ra_monthly * 12, 650_000))


def test_split_is_reproducible_and_marginal_rate_can_replace_income():
    a = optimize_wrapper_split(8_000, None, 0.31, 10, n_paths=200, seed=9)
    b = optimize_wrapper_split(8_000, None, 0.31, 10, n_paths=200, seed=9)
    assert a.gross_income == TAX_2026_27.income_for_bracket(0.31)
    assert np.array_equal(a.blended.terminal_net, b.blended.terminal_net)


def test_restrict_to_arbitrage_never_exceeds_threshold():
    r = optimize_wrapper_split(30_000, 300_000, None, 10, n_paths=200, seed=1, restrict_to_arbitrage=True)
    assert r.optimal_ra_monthly * 12 <= r.arbitrage_annual_ra + 1e-6


def test_split_summary_and_comparison_tables(split_36):
    s = split_36.summary()
    assert s["tax_year"] == "2026/27" and s["marginal_rate"] == 0.36
    assert s["terminal_wealth_gain_real_vs_disc"] > 0
    cmp = split_36.comparison()
    assert list(cmp.index) == [st.name for st in split_36.strategies.values()]
    assert cmp.loc[split_36.blended.name, "median after-tax wealth"] > cmp.iloc[0]["median after-tax wealth"]


@pytest.mark.parametrize(
    "kwargs",
    [dict(monthly_budget=0, gross_income=500_000), dict(monthly_budget=5_000, gross_income=-1),
     dict(monthly_budget=5_000, gross_income=None, marginal_rate=None), dict(monthly_budget=5_000, gross_income=500_000, horizon_years=2),
     dict(monthly_budget=5_000, gross_income=500_000, n_paths=10), dict(monthly_budget=5_000, gross_income=500_000, marginal_rate=0.9)],
)
def test_split_input_validation(kwargs):
    kwargs.setdefault("marginal_rate", None)
    kwargs.setdefault("horizon_years", 10)
    with pytest.raises(ValueError):
        optimize_wrapper_split(**kwargs)
