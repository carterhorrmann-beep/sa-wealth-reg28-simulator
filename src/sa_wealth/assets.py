"""Multi-asset South African & global return engine (all returns in ZAR).

This module holds the *inputs* of the simulator:

* the six core asset classes and their long-run capital market assumptions (CMAs),
* a cross-asset correlation matrix that also contains two macro factors -
  South African CPI shocks and a "ZAR strength" factor that captures
  Rand-hedge dynamics (a stronger Rand *lowers* the ZAR value of global assets),
* a stochastic CPI model (AR(1) deviation around a mean inflation rate),
* an optional ``yfinance`` loader that re-estimates the CMAs from liquid ETF
  proxies, including a sanitiser for the classic ZAc/ZAR 100x unit flip.

Everything in here is deterministic, offline and unit-testable apart from
:func:`fetch_yfinance_prices`, which is the only function that touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.signal import lfilter

# --------------------------------------------------------------------------- #
# Universe definition
# --------------------------------------------------------------------------- #

ASSETS: tuple[str, ...] = (
    "JSE_EQ",
    "SA_BOND",
    "SA_PROP",
    "SA_CASH",
    "GLB_EQ",
    "GLB_BOND",
)

ASSET_NAMES: dict[str, str] = {
    "JSE_EQ": "JSE All Share / Top 40 Equities",
    "SA_BOND": "SA All Bond Index (ALBI / SAGBs)",
    "SA_PROP": "SA Listed Property (SAPY)",
    "SA_CASH": "SA Money Market / STeFI Cash",
    "GLB_EQ": "Global Developed Equities (ZAR)",
    "GLB_BOND": "Global Aggregate Bonds (ZAR)",
}

# Regulation 28 buckets
EQUITY_ASSETS: tuple[str, ...] = ("JSE_EQ", "GLB_EQ")
OFFSHORE_ASSETS: tuple[str, ...] = ("GLB_EQ", "GLB_BOND")
PROPERTY_ASSETS: tuple[str, ...] = ("SA_PROP",)

# Tax character of the income each asset throws off (used by the tax engine)
LOCAL_DIVIDEND = "local_dividend"  # 20% dividend withholding tax
FOREIGN_DIVIDEND = "foreign_dividend"  # effectively taxed at max 20% for individuals
LOCAL_INTEREST = "local_interest"  # marginal rate, after the annual interest exemption
FOREIGN_INTEREST = "foreign_interest"  # marginal rate, no exemption
PROPERTY_DISTRIBUTION = "property_distribution"  # REIT distributions: marginal rate, no DWT

INCOME_CLASS: dict[str, str] = {
    "JSE_EQ": LOCAL_DIVIDEND,
    "SA_BOND": LOCAL_INTEREST,
    "SA_PROP": PROPERTY_DISTRIBUTION,
    "SA_CASH": LOCAL_INTEREST,
    "GLB_EQ": FOREIGN_DIVIDEND,
    "GLB_BOND": FOREIGN_INTEREST,
}

# Factor order of the full correlation matrix: the six assets, then two macro factors.
FACTORS: tuple[str, ...] = ASSETS + ("CPI", "ZAR_STRENGTH")
N_ASSETS = len(ASSETS)

# --------------------------------------------------------------------------- #
# Capital market assumptions
# --------------------------------------------------------------------------- #

#: Long-run nominal ZAR assumptions. ``income_yield`` is the part of the total
#: return that is paid out as dividends / interest / distributions each year
#: (and is therefore taxed annually outside a tax-free wrapper); the remainder is
#: capital growth, taxed only when realised.
CMA_TABLE = pd.DataFrame(
    {
        "expected_return": [0.120, 0.095, 0.100, 0.070, 0.115, 0.080],
        "volatility": [0.170, 0.080, 0.200, 0.007, 0.170, 0.120],
        "income_yield": [0.030, 0.085, 0.075, 0.070, 0.016, 0.040],
    },
    index=pd.Index(ASSETS, name="asset"),
)

# Full factor correlation matrix (assets, CPI shocks, ZAR strength).
# ZAR_STRENGTH is the *negative* of USD/ZAR returns, so GLB_EQ and GLB_BOND (priced in
# USD, translated to ZAR) are negatively correlated with it, while SA assets that
# benefit from a firm Rand (SA bonds, listed property) are positively correlated.
_PAIRS: dict[tuple[str, str], float] = {
    ("JSE_EQ", "SA_BOND"): 0.35,
    ("JSE_EQ", "SA_PROP"): 0.65,
    ("JSE_EQ", "SA_CASH"): 0.00,
    ("JSE_EQ", "GLB_EQ"): 0.45,
    ("JSE_EQ", "GLB_BOND"): -0.10,
    ("JSE_EQ", "CPI"): -0.10,
    ("JSE_EQ", "ZAR_STRENGTH"): 0.25,
    ("SA_BOND", "SA_PROP"): 0.40,
    ("SA_BOND", "SA_CASH"): 0.15,
    ("SA_BOND", "GLB_EQ"): -0.05,
    ("SA_BOND", "GLB_BOND"): -0.30,
    ("SA_BOND", "CPI"): -0.30,
    ("SA_BOND", "ZAR_STRENGTH"): 0.35,
    ("SA_PROP", "SA_CASH"): 0.05,
    ("SA_PROP", "GLB_EQ"): 0.25,
    ("SA_PROP", "GLB_BOND"): -0.05,
    ("SA_PROP", "CPI"): -0.15,
    ("SA_PROP", "ZAR_STRENGTH"): 0.20,
    ("SA_CASH", "GLB_EQ"): -0.05,
    ("SA_CASH", "GLB_BOND"): -0.05,
    ("SA_CASH", "CPI"): 0.20,
    ("SA_CASH", "ZAR_STRENGTH"): 0.00,
    ("GLB_EQ", "GLB_BOND"): 0.45,
    ("GLB_EQ", "CPI"): 0.15,
    ("GLB_EQ", "ZAR_STRENGTH"): -0.55,
    ("GLB_BOND", "CPI"): 0.25,
    ("GLB_BOND", "ZAR_STRENGTH"): -0.85,
    ("CPI", "ZAR_STRENGTH"): -0.40,
}


def nearest_psd(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the correlation matrices (PSD, unit diagonal).

    Uses eigenvalue clipping followed by re-normalisation. A matrix that is already
    positive definite is returned unchanged (up to round-off).
    """
    a = np.asarray(corr, dtype=float)
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    if vals.min() >= eps:
        out = a.copy()
        np.fill_diagonal(out, 1.0)
        return out
    vals = np.clip(vals, eps, None)
    b = (vecs * vals) @ vecs.T
    d = 1.0 / np.sqrt(np.diag(b))
    b = b * d[:, None] * d[None, :]
    b = (b + b.T) / 2.0
    np.fill_diagonal(b, 1.0)
    return b


def _build_full_corr(pairs: Mapping[tuple[str, str], float] = _PAIRS) -> np.ndarray:
    n = len(FACTORS)
    m = np.eye(n)
    for (a, b), v in pairs.items():
        i, j = FACTORS.index(a), FACTORS.index(b)
        m[i, j] = m[j, i] = v
    return m


FULL_CORR = nearest_psd(_build_full_corr())


# --------------------------------------------------------------------------- #
# Market assumptions container
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class MarketAssumptions:
    """All inputs needed to simulate ZAR asset returns and SA CPI.

    Parameters
    ----------
    table:
        DataFrame indexed by :data:`ASSETS` with ``expected_return`` (annual, nominal,
        arithmetic), ``volatility`` (annual) and ``income_yield`` columns.
    corr:
        8x8 correlation matrix over :data:`FACTORS` (6 assets + CPI + ZAR strength).
    inflation_mean:
        Mean annual SA CPI inflation (default 5.5%).
    inflation_vol:
        Stationary standard deviation of the annual inflation rate (stochastic shocks).
    inflation_persistence:
        Monthly AR(1) coefficient of the inflation-rate deviation (0.97 -> ~2 year half-life).
    fx_vol:
        Annual USD/ZAR volatility, used only for the Rand-hedge diagnostics.
    """

    table: pd.DataFrame
    corr: np.ndarray
    inflation_mean: float = 0.055
    inflation_vol: float = 0.015
    inflation_persistence: float = 0.97
    fx_vol: float = 0.14
    source: str = "cma"

    # -- basic accessors ---------------------------------------------------- #
    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(self.table.index)

    @property
    def mu(self) -> np.ndarray:
        return self.table["expected_return"].to_numpy(dtype=float)

    @property
    def vol(self) -> np.ndarray:
        return self.table["volatility"].to_numpy(dtype=float)

    @property
    def income_yield(self) -> np.ndarray:
        return self.table["income_yield"].to_numpy(dtype=float)

    @property
    def asset_corr(self) -> np.ndarray:
        return self.corr[:N_ASSETS, :N_ASSETS]

    @property
    def cov(self) -> np.ndarray:
        return self.asset_corr * np.outer(self.vol, self.vol)

    @property
    def sim_corr(self) -> np.ndarray:
        """7x7 correlation of the factors that are actually simulated (assets + CPI)."""
        return self.corr[: N_ASSETS + 1, : N_ASSETS + 1]

    @property
    def real_expected_return(self) -> np.ndarray:
        return (1 + self.mu) / (1 + self.inflation_mean) - 1

    def with_inflation(self, mean: float | None = None, vol: float | None = None) -> "MarketAssumptions":
        return replace(
            self,
            inflation_mean=self.inflation_mean if mean is None else float(mean),
            inflation_vol=self.inflation_vol if vol is None else float(vol),
        )

    # -- reporting ---------------------------------------------------------- #
    def summary_table(self) -> pd.DataFrame:
        t = self.table.copy()
        t["capital_growth"] = t["expected_return"] - t["income_yield"]
        t["real_return"] = self.real_expected_return
        t["sharpe_vs_cash"] = (t["expected_return"] - t.loc["SA_CASH", "expected_return"]) / t["volatility"]
        t.insert(0, "name", [ASSET_NAMES.get(a, a) for a in t.index])
        return t

    def correlation_frame(self, include_macro: bool = False) -> pd.DataFrame:
        n = len(FACTORS) if include_macro else N_ASSETS
        labels = list(FACTORS[:n])
        return pd.DataFrame(self.corr[:n, :n], index=labels, columns=labels)

    def rand_hedge_diagnostics(self) -> dict[str, float]:
        """Correlation of each asset with ZAR strength (negative of USD/ZAR returns).

        A *negative* number means the asset behaves as a Rand hedge: it gains in ZAR
        when the Rand weakens, and loses when the Rand strengthens.
        """
        zs = FACTORS.index("ZAR_STRENGTH")
        out = {a: float(self.corr[i, zs]) for i, a in enumerate(ASSETS)}
        out["CPI"] = float(self.corr[FACTORS.index("CPI"), zs])
        return out


def default_assumptions() -> MarketAssumptions:
    """Calibrated long-run South African capital market assumptions."""
    return MarketAssumptions(table=CMA_TABLE.copy(), corr=FULL_CORR.copy())


# --------------------------------------------------------------------------- #
# Stochastic inflation
# --------------------------------------------------------------------------- #


def simulate_inflation(
    z: np.ndarray,
    assumptions: MarketAssumptions,
    mean_path: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate monthly SA CPI inflation and the cumulative CPI index.

    The *annualised* inflation rate follows ``a_t = mean_t + x_t`` where ``x_t`` is an
    AR(1) process driven by the supplied standard-normal shocks ``z`` (shape ``(P, T)``).
    These shocks are the CPI column of the correlated draw, so inflation surprises are
    correlated with asset returns (e.g. higher inflation hurts SA bonds).

    Returns
    -------
    monthly_inflation : ``(P, T)`` monthly CPI changes
    cpi_index : ``(P, T + 1)`` index with ``cpi_index[:, 0] == 1`` (today's money)
    """
    p, t = z.shape
    phi = assumptions.inflation_persistence
    sigma_eps = assumptions.inflation_vol * np.sqrt(1.0 - phi**2)
    x = lfilter([1.0], [1.0, -phi], z * sigma_eps, axis=1)
    mean = np.full(t, assumptions.inflation_mean) if mean_path is None else np.asarray(mean_path, dtype=float)
    annual = np.clip(mean[None, :] + x, -0.02, 0.35)
    monthly = (1.0 + annual) ** (1.0 / 12.0) - 1.0
    cpi = np.ones((p, t + 1))
    cpi[:, 1:] = np.cumprod(1.0 + monthly, axis=1)
    return monthly, cpi


def log_return_parameters(
    assumptions: MarketAssumptions,
    vol_mult: np.ndarray | float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Monthly log-return drift and volatility per asset.

    The drift is set so that the *arithmetic* annual return equals the assumption:
    ``E[1 + R_annual] = 1 + mu``  =>  ``m = (ln(1 + mu) - sigma^2 / 2) / 12``.
    ``vol_mult`` (scalar or broadcastable to ``(T, A)``) scales volatility for stress tests
    while keeping the arithmetic mean unchanged.
    """
    sigma = assumptions.vol * np.asarray(vol_mult)
    drift = (np.log1p(assumptions.mu) - 0.5 * sigma**2) / 12.0
    return drift, sigma / np.sqrt(12.0)


# --------------------------------------------------------------------------- #
# Optional market-data loader (yfinance proxies)
# --------------------------------------------------------------------------- #

PROXY_TICKERS: dict[str, str] = {
    "JSE_EQ": "STX40.JO",  # Satrix 40 ETF (JSE Top 40)
    "SA_BOND": "NFGOVI.JO",  # NewFunds GOVI ETF
    "SA_PROP": "STXPRO.JO",  # Satrix Property ETF
    "USDZAR": "USDZAR=X",
    "SPY": "SPY",
    "AGG": "AGG",
}


def sanitise_zac_series(prices: pd.Series, tol: float = 0.25) -> pd.Series:
    """Remove ZAc/ZAR 100x unit flips from a price series.

    JSE-listed instruments are quoted in South African cents (ZAc). Data vendors
    occasionally switch the quote unit mid-history, producing a day-on-day jump of
    ~100x (or ~1/100x). Every such jump is detected and the *earlier* segment is
    rescaled so the whole series is expressed in the unit of the most recent data.
    Genuine moves (even -50% crashes) are untouched.

    Parameters
    ----------
    tol:
        Allowed log-distance of a jump from exactly 100x / 0.01x to count as a unit flip.
    """
    s = pd.to_numeric(prices, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) < 2:
        return s
    ratio = (s / s.shift(1)).to_numpy()[1:]
    log_ratio = np.log(ratio)
    ln100 = np.log(100.0)
    scale = np.ones(len(s))
    for k, lr in enumerate(log_ratio, start=1):
        if abs(lr - ln100) < tol:
            scale[:k] *= 100.0
        elif abs(lr + ln100) < tol:
            scale[:k] /= 100.0
    return pd.Series(s.to_numpy() * scale, index=s.index, name=prices.name)


def fetch_yfinance_prices(years: int = 10, interval: str = "1mo") -> pd.DataFrame:  # pragma: no cover - network
    """Download adjusted closes for the liquid proxies (requires the optional ``yfinance``)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is not installed. Install it with: pip install 'sa-wealth[yf]'") from exc
    tickers = list(PROXY_TICKERS.values())
    raw = yf.download(tickers, period=f"{years}y", interval=interval, auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    inv = {v: k for k, v in PROXY_TICKERS.items()}
    close = close.rename(columns=inv)
    return close


def assumptions_from_prices(
    prices: pd.DataFrame,
    base: MarketAssumptions | None = None,
    shrinkage: float = 0.5,
    periods_per_year: int = 12,
) -> MarketAssumptions:
    """Blend sample statistics from ETF-proxy prices into the long-run CMAs.

    ``prices`` must hold columns from :data:`PROXY_TICKERS` keys (``JSE_EQ``, ``SA_BOND``,
    ``SA_PROP``, ``USDZAR``, ``SPY``, ``AGG``). JSE proxies are cleaned with
    :func:`sanitise_zac_series`; ``USDZAR`` is also sanitised in case of unit flips.
    Global assets are converted to ZAR by compounding the USD return with USD/ZAR.

    A short history is a noisy estimator of long-run returns, so the sample mean is
    blended with the CMA via ``shrinkage`` (0 = pure CMA, 1 = pure sample). Volatilities
    and correlations are taken from the sample (shrunk 50/50 with the CMA correlations),
    and the result is projected to a valid correlation matrix. ``SA_CASH`` and the CPI
    factor are not available from the proxies and keep their CMA values.
    """
    base = base or default_assumptions()
    cleaned = {}
    for col in ("JSE_EQ", "SA_BOND", "SA_PROP", "USDZAR", "SPY", "AGG"):
        if col not in prices.columns:
            raise KeyError(f"prices is missing required column '{col}'")
        cleaned[col] = sanitise_zac_series(prices[col])
    px = pd.DataFrame(cleaned).dropna()
    if len(px) < 24:
        raise ValueError("Need at least 24 overlapping observations to estimate assumptions")
    rets = px.pct_change().dropna()
    zar = pd.DataFrame(
        {
            "JSE_EQ": rets["JSE_EQ"],
            "SA_BOND": rets["SA_BOND"],
            "SA_PROP": rets["SA_PROP"],
            "GLB_EQ": (1 + rets["SPY"]) * (1 + rets["USDZAR"]) - 1,
            "GLB_BOND": (1 + rets["AGG"]) * (1 + rets["USDZAR"]) - 1,
        }
    )
    zs = -rets["USDZAR"]  # ZAR strength

    table = base.table.copy()
    corr = base.corr.copy()
    ann = periods_per_year
    for a in zar.columns:
        sample_mu = (1 + zar[a].mean()) ** ann - 1
        table.loc[a, "expected_return"] = (1 - shrinkage) * table.loc[a, "expected_return"] + shrinkage * sample_mu
        table.loc[a, "volatility"] = float(zar[a].std(ddof=1) * np.sqrt(ann))
    # keep the income yield within the total return (cash is all income; others keep some capital growth)
    cap = table["expected_return"] - 0.005
    cap["SA_CASH"] = table.loc["SA_CASH", "expected_return"]
    table["income_yield"] = np.minimum(table["income_yield"], cap)

    frame = zar.copy()
    frame["ZAR_STRENGTH"] = zs
    sample_corr = frame.corr()
    for a in frame.columns:
        for b in frame.columns:
            if a >= b:
                continue
            i, j = FACTORS.index(a), FACTORS.index(b)
            corr[i, j] = corr[j, i] = 0.5 * base.corr[i, j] + 0.5 * float(sample_corr.loc[a, b])
    fx_vol = float(rets["USDZAR"].std(ddof=1) * np.sqrt(ann))
    return replace(base, table=table, corr=nearest_psd(corr), fx_vol=fx_vol, source="yfinance")


def load_assumptions(source: str = "cma", years: int = 10) -> MarketAssumptions:
    """Entry point used by the CLI: ``source`` is ``"cma"`` (default, offline) or ``"yfinance"``."""
    source = source.lower()
    if source == "cma":
        return default_assumptions()
    if source == "yfinance":  # pragma: no cover - network
        return assumptions_from_prices(fetch_yfinance_prices(years=years))
    raise ValueError(f"Unknown source '{source}'. Use 'cma' or 'yfinance'.")


def weights_from_mapping(mapping: Mapping[str, float] | Sequence[float], assets: Sequence[str] = ASSETS) -> np.ndarray:
    """Convert ``{asset: weight}`` (missing assets = 0) or a sequence into a weight vector."""
    if isinstance(mapping, Mapping):
        unknown = set(mapping) - set(assets)
        if unknown:
            raise KeyError(f"Unknown assets: {sorted(unknown)}")
        return np.array([float(mapping.get(a, 0.0)) for a in assets])
    w = np.asarray(mapping, dtype=float)
    if w.shape != (len(assets),):
        raise ValueError(f"Expected {len(assets)} weights, got shape {w.shape}")
    return w
