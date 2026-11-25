import numpy as np
import pandas as pd
import pytest

from sa_wealth.assets import (
    ASSETS,
    EQUITY_ASSETS,
    FACTORS,
    OFFSHORE_ASSETS,
    default_assumptions,
    load_assumptions,
    assumptions_from_prices,
    log_return_parameters,
    nearest_psd,
    sanitise_zac_series,
    simulate_inflation,
    weights_from_mapping,
)


def test_six_core_asset_classes():
    assert ASSETS == ("JSE_EQ", "SA_BOND", "SA_PROP", "SA_CASH", "GLB_EQ", "GLB_BOND")
    assert set(EQUITY_ASSETS) == {"JSE_EQ", "GLB_EQ"}
    assert set(OFFSHORE_ASSETS) == {"GLB_EQ", "GLB_BOND"}


def test_cma_table_is_sane(assumptions):
    t = assumptions.table
    assert list(t.index) == list(ASSETS)
    assert (t["volatility"] > 0).all()
    assert (t["income_yield"] >= 0).all()
    assert (t["income_yield"] <= t["expected_return"]).all()  # cash return is 100% income
    assert (t.drop("SA_CASH")["income_yield"] < t.drop("SA_CASH")["expected_return"]).all()
    # sensible SA ordering: cash < bonds < equities in expected return, cash has the least risk
    assert t.loc["SA_CASH", "expected_return"] < t.loc["SA_BOND", "expected_return"] < t.loc["JSE_EQ", "expected_return"]
    assert t["volatility"].idxmin() == "SA_CASH"
    # every asset beats cash in expected terms
    assert (t.drop("SA_CASH")["expected_return"] > t.loc["SA_CASH", "expected_return"]).all()


def test_correlation_matrix_is_valid(assumptions):
    c = assumptions.corr
    assert c.shape == (len(FACTORS), len(FACTORS))
    assert np.allclose(c, c.T)
    assert np.allclose(np.diag(c), 1.0)
    assert np.all(np.abs(c) <= 1.0)
    assert np.linalg.eigvalsh(c).min() > 0  # positive definite -> Cholesky works
    np.linalg.cholesky(assumptions.sim_corr)


def test_rand_hedge_dynamics(assumptions):
    """A stronger Rand lowers the ZAR value of global assets: negative correlation with ZAR strength."""
    d = assumptions.rand_hedge_diagnostics()
    assert d["GLB_EQ"] < 0
    assert d["GLB_BOND"] < d["GLB_EQ"] < 0  # global bonds are almost pure currency exposure in ZAR
    assert d["SA_BOND"] > 0 and d["JSE_EQ"] > 0
    assert d["CPI"] < 0  # weaker Rand -> higher inflation


def test_covariance_matches_vols_and_corr(assumptions):
    cov = assumptions.cov
    assert np.allclose(np.sqrt(np.diag(cov)), assumptions.vol)
    assert np.allclose(cov, cov.T)


def test_default_inflation_is_5_5_percent(assumptions):
    assert assumptions.inflation_mean == pytest.approx(0.055)
    assert assumptions.with_inflation(0.085).inflation_mean == pytest.approx(0.085)
    assert assumptions.inflation_mean == pytest.approx(0.055)  # original untouched


def test_inflation_simulation_is_stochastic_and_centred(assumptions):
    z = np.random.default_rng(1).standard_normal((4_000, 120))
    monthly, cpi = simulate_inflation(z, assumptions)
    assert cpi.shape == (4_000, 121)
    assert np.all(cpi[:, 0] == 1.0)
    annualised = cpi[:, -1] ** (1 / 10) - 1
    assert annualised.mean() == pytest.approx(0.055, abs=0.003)
    assert annualised.std() > 0.002  # genuine shocks, not a constant
    assert monthly.shape == (4_000, 120)


def test_inflation_mean_path_override(assumptions):
    z = np.zeros((10, 24))
    path = np.r_[np.full(12, 0.10), np.full(12, 0.02)]
    _, cpi = simulate_inflation(z, assumptions, path)
    assert cpi[0, 12] == pytest.approx(1.10, rel=1e-6)
    assert cpi[0, 24] == pytest.approx(1.10 * 1.02, rel=1e-6)


def test_log_return_drift_reproduces_arithmetic_mean(assumptions):
    drift, sd = log_return_parameters(assumptions)
    # annual log return ~ N(12 m, 12 s^2)  ->  E[exp] = exp(12 m + 6 s^2)
    expected = np.exp(12 * drift + 6 * sd**2) - 1
    assert np.allclose(expected, assumptions.mu)


def test_sanitise_zac_unit_flips_up_and_down():
    idx = pd.date_range("2024-01-31", periods=8, freq="30D")
    clean = pd.Series([60.0, 61.0, 59.5, 62.0, 63.0, 62.5, 64.0, 65.0], index=idx)
    # data vendor switches ZAR -> ZAc (x100) for the last 4 months
    flipped = clean.copy()
    flipped.iloc[4:] *= 100
    fixed = sanitise_zac_series(flipped)
    assert np.allclose(fixed.pct_change().dropna(), clean.pct_change().dropna())
    # and the opposite direction (ZAc -> ZAR)
    flipped2 = clean.copy()
    flipped2.iloc[:4] *= 100
    fixed2 = sanitise_zac_series(flipped2)
    assert np.allclose(fixed2.pct_change().dropna(), clean.pct_change().dropna())


def test_sanitise_zac_keeps_genuine_crashes():
    s = pd.Series([100.0, 98.0, 50.0, 55.0, 160.0, 150.0])  # -49% and +190% moves are real
    out = sanitise_zac_series(s)
    assert np.allclose(out.to_numpy(), s.to_numpy())


def test_sanitise_drops_bad_rows():
    s = pd.Series([100.0, np.nan, -5.0, 102.0, 101.0])
    assert list(sanitise_zac_series(s)) == [100.0, 102.0, 101.0]


def _synthetic_prices(n=60, flip=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-31", periods=n, freq="30D")
    base = rng.normal(0.008, 0.04, size=(n, 6))
    px = pd.DataFrame(
        100 * np.cumprod(1 + base, axis=0),
        index=idx,
        columns=["JSE_EQ", "SA_BOND", "SA_PROP", "USDZAR", "SPY", "AGG"],
    )
    if flip:
        px.loc[px.index[:20], "JSE_EQ"] *= 100  # ZAc / ZAR flip in the early history
    return px


def test_assumptions_from_prices_builds_valid_model():
    a = assumptions_from_prices(_synthetic_prices())
    assert a.source == "yfinance"
    assert np.linalg.eigvalsh(a.corr).min() > 0
    assert (a.table["volatility"] < 1.0).all()  # the 100x flip must not explode volatility
    assert (a.table["income_yield"] <= a.table["expected_return"]).all()
    # SA_CASH keeps its CMA value (no proxy supplied)
    assert a.table.loc["SA_CASH", "expected_return"] == pytest.approx(0.07)


def test_assumptions_from_prices_requires_columns_and_history():
    px = _synthetic_prices()
    with pytest.raises(KeyError):
        assumptions_from_prices(px.drop(columns=["SPY"]))
    with pytest.raises(ValueError):
        assumptions_from_prices(px.iloc[:10])


def test_nearest_psd_repairs_invalid_matrix():
    bad = np.array([[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < 0
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() > 0
    assert np.allclose(np.diag(fixed), 1.0)


def test_load_assumptions():
    assert load_assumptions("cma").source == "cma"
    with pytest.raises(ValueError):
        load_assumptions("bloomberg")


def test_weights_from_mapping():
    w = weights_from_mapping({"JSE_EQ": 0.6, "SA_CASH": 0.4})
    assert w.tolist() == [0.6, 0.0, 0.0, 0.4, 0.0, 0.0]
    with pytest.raises(KeyError):
        weights_from_mapping({"BITCOIN": 1.0})
    with pytest.raises(ValueError):
        weights_from_mapping([0.5, 0.5])
