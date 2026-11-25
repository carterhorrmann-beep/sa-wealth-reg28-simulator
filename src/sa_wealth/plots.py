"""High-DPI charts (matplotlib, object-oriented API so they work headless and inside Streamlit).

Every ``plot_*`` function returns a :class:`matplotlib.figure.Figure`. :func:`generate_all_charts`
writes the four documentation charts used in the README.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from matplotlib.ticker import FuncFormatter

from .assets import ASSETS, MarketAssumptions, default_assumptions
from .montecarlo import (
    SimulationConfig,
    SimulationResult,
    StressResult,
    TaxComparison,
    run_stress,
    run_tax_comparison,
    simulate,
)
from .reg28_tax import (
    DISC,
    GROSS,
    RA,
    REG28,
    STATIC_PRESETS,
    TFSA,
    WRAPPER_NAMES,
    efficient_frontier,
    offshore_sensitivity,
    optimize_portfolio,
)

DPI = 300
COLORS = {
    RA: "#00796b",
    TFSA: "#f9a825",
    DISC: "#c62828",
    GROSS: "#607d8b",
    "band_outer": "#b2dfdb",
    "band_inner": "#4db6ac",
    "median": "#004d40",
    "goal": "#c62828",
    "unc": "#3949ab",
    "reg28": "#00796b",
    "text": "#263238",
    "grid": "#cfd8dc",
}

_RC = {
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelcolor": COLORS["text"],
    "axes.edgecolor": "#90a4ae",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.6,
    "text.color": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def zar(x: float, _pos=None) -> str:
    """Format a rand amount compactly: R850k, R4.2m."""
    ax = abs(x)
    sign = "-" if x < 0 else ""
    if ax >= 1e9:
        return f"{sign}R{ax / 1e9:.1f}bn"
    if ax >= 1e6:
        return f"{sign}R{ax / 1e6:.1f}m"
    if ax >= 1e3:
        return f"{sign}R{ax / 1e3:.0f}k"
    return f"{sign}R{ax:.0f}"


ZAR_FMT = FuncFormatter(zar)


def _fig(width: float, height: float, **kw) -> Figure:
    with mpl.rc_context(_RC):
        return Figure(figsize=(width, height), **kw)


def _style(ax) -> None:
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")


def save_figure(fig: Figure, path: str | Path, dpi: int = DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


# --------------------------------------------------------------------------- #
# Monte Carlo fan chart & gauge
# --------------------------------------------------------------------------- #


def plot_fan_chart(result: SimulationResult, real: bool = True) -> Figure:
    """Percentile wealth fan (P10-P90 and P25-P75 bands, median line) with the goal."""
    pct = result.percentiles(real=real)
    years = pct.index.to_numpy()
    cfg = result.config
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(9.5, 5.4))
        ax = fig.subplots()
        ax.fill_between(years, pct["P10"], pct["P90"], color=COLORS["band_outer"], label="P10 - P90", lw=0)
        ax.fill_between(years, pct["P25"], pct["P75"], color=COLORS["band_inner"], alpha=0.85, label="P25 - P75", lw=0)
        ax.plot(years, pct["P50"], color=COLORS["median"], lw=2.4, label="Median (P50)")
        if real:
            ax.axhline(cfg.target_real, color=COLORS["goal"], ls="--", lw=1.6, label=f"Goal {zar(cfg.target_real)} (today's rand)")
        else:
            cpi_med = np.median(result.nominal[:, 1:] / np.maximum(result.real[:, 1:], 1e-9), axis=0)
            goal_curve = cfg.target_real * np.concatenate([[1.0], cpi_med])
            ax.plot(years, goal_curve, color=COLORS["goal"], ls="--", lw=1.6, label="Goal (inflated at median CPI)")
        ax.yaxis.set_major_formatter(ZAR_FMT)
        ax.set_xlabel("Years from today")
        ax.set_ylabel("Real wealth (today's purchasing power)" if real else "Nominal wealth (ZAR)")
        end = pct["P50"].iloc[-1]
        ax.annotate(
            f"Median {zar(end)}",
            xy=(years[-1], end),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            color=COLORS["median"],
            fontweight="bold",
        )
        ax.set_xlim(0, years[-1])
        ax.set_ylim(bottom=0)
        ax.set_title(
            f"Monte Carlo wealth projection - {WRAPPER_NAMES[result.wrapper]}\n"
            f"{result.outcome.values_liq.shape[0]:,} paths | P(goal) = {result.prob_goal:.0%} | CVaR 5% = {zar(result.cvar5_real)} (real)",
            loc="left",
        )
        ax.legend(loc="upper left")
        _style(ax)
        fig.text(0.01, -0.02, "Wealth shown is after tax: the value you would keep if you liquidated (RA: retired) in that year. Illustrative; not financial advice.", fontsize=8, color="#78909c")
    return fig


def plot_gauge(prob: float, title: str = "Probability of reaching goal") -> Figure:
    """Half-circle gauge for a probability in [0, 1]."""
    prob = float(np.clip(prob, 0.0, 1.0))
    color = "#c62828" if prob < 0.5 else "#f9a825" if prob < 0.75 else "#2e7d32"
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(4.2, 2.6))
        ax = fig.subplots()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Wedge((0, 0), 1.0, 0, 180, width=0.28, facecolor="#eceff1", edgecolor="none"))
        ax.add_patch(Wedge((0, 0), 1.0, 180 - 180 * prob, 180, width=0.28, facecolor=color, edgecolor="none"))
        ax.text(0, 0.12, f"{prob:.0%}", ha="center", va="center", fontsize=26, fontweight="bold", color=COLORS["text"])
        ax.text(0, -0.18, title, ha="center", va="center", fontsize=9, color="#546e7a")
        ax.text(-1.0, -0.06, "0%", ha="center", va="top", fontsize=8, color="#78909c")
        ax.text(1.0, -0.06, "100%", ha="center", va="top", fontsize=8, color="#78909c")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.3, 1.1)
    return fig


# --------------------------------------------------------------------------- #
# Regulation 28 frontier
# --------------------------------------------------------------------------- #


def plot_reg28_frontier(
    frontier_unc: pd.DataFrame,
    frontier_reg: pd.DataFrame,
    assumptions: MarketAssumptions,
    sensitivity: pd.DataFrame | None = None,
    highlight: Mapping[str, np.ndarray] | None = None,
) -> Figure:
    """Efficient frontiers (unconstrained vs Reg 28) and, optionally, the offshore-cap sensitivity."""
    two = sensitivity is not None
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(13.5 if two else 8.5, 5.6))
        axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.35, 1]}) if two else [fig.subplots()]
        ax = axes[0]
        ax.plot(frontier_unc["volatility"] * 100, frontier_unc["return"] * 100, color=COLORS["unc"], lw=2.2, label="Unconstrained frontier")
        ax.plot(frontier_reg["volatility"] * 100, frontier_reg["return"] * 100, color=COLORS["reg28"], lw=3.2, ls="-", alpha=0.9, label="Regulation 28 frontier")
        r_max = frontier_reg["return"].max()
        if frontier_unc["return"].max() > r_max + 1e-4:
            ax.fill_between(
                frontier_unc.loc[frontier_unc["return"] >= r_max - 1e-4, "volatility"] * 100,
                r_max * 100,
                frontier_unc.loc[frontier_unc["return"] >= r_max - 1e-4, "return"] * 100,
                color=COLORS["unc"],
                alpha=0.10,
                label="Return unreachable under Reg 28",
            )
        for i, a in enumerate(assumptions.assets):
            ax.scatter(assumptions.vol[i] * 100, assumptions.mu[i] * 100, color="#90a4ae", s=28, zorder=3)
            ax.annotate(a, (assumptions.vol[i] * 100, assumptions.mu[i] * 100), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#546e7a")
        for tag, df, col, mk in (("Max-Sharpe (unconstrained)", frontier_unc, COLORS["unc"], "*"), ("Max-Sharpe (Reg 28)", frontier_reg, COLORS["reg28"], "o")):
            row = df.loc[df["sharpe"].idxmax()]
            ax.scatter(row["volatility"] * 100, row["return"] * 100, marker=mk, s=170 if mk == "*" else 60, color=col, edgecolor="white", zorder=5, label=tag)
        if highlight:
            for name, w in highlight.items():
                ret, vol = float(assumptions.mu @ w), float(np.sqrt(w @ assumptions.cov @ w))
                ax.scatter(vol * 100, ret * 100, marker="D", s=50, color="#6a1b9a", zorder=6)
                ax.annotate(name, (vol * 100, ret * 100), xytext=(6, -10), textcoords="offset points", fontsize=8, color="#6a1b9a")
        ax.set_xlabel("Volatility (% p.a.)")
        ax.set_ylabel("Expected return (% p.a., nominal ZAR)")
        ax.set_title("Efficient frontier: unconstrained vs Regulation 28", loc="left")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True)
        if two:
            ax2 = axes[1]
            x = sensitivity.index.to_numpy() * 100
            ax2.bar(x, sensitivity["sharpe_cost"] * 1000, width=0.7, color=np.where(sensitivity["cap_binding"], COLORS[DISC], "#b0bec5"))
            ax2.set_xlabel("Extra expected return on global assets (pp, weak-Rand view)")
            ax2.set_ylabel("Sharpe ratio given up under Reg 28 (x 0.001)")
            ax2.set_title("When does the 45% offshore cap bite?", loc="left")
            ax3 = ax2.twinx()
            ax3.plot(x, sensitivity["offshore_unconstrained"] * 100, color=COLORS["unc"], marker="o", lw=2, label="Unconstrained offshore weight")
            ax3.axhline(45, color=COLORS["goal"], ls="--", lw=1.4)
            ax3.set_ylabel("Offshore weight (%)")
            ax3.spines["right"].set_visible(True)
            ax3.grid(False)
            ax3.legend(loc="upper left", fontsize=8)
            _style(ax2)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Tax wrappers
# --------------------------------------------------------------------------- #


def plot_tax_wrappers(comparison: TaxComparison, real: bool = True) -> Figure:
    """After-tax compounding curves and the Rands saved versus the taxable account."""
    curves = comparison.curves(real=real)
    s = comparison.summary
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(13.5, 5.4))
        ax, bx = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
        for key, ls, lw in ((RA, "-", 2.6), (TFSA, "-", 2.6), (DISC, "-", 2.6), (GROSS, ":", 1.8)):
            name = WRAPPER_NAMES[key]
            ax.plot(curves.index, curves[name], color=COLORS[key], ls=ls, lw=lw, label=name)
        ax.yaxis.set_major_formatter(ZAR_FMT)
        ax.set_xlabel("Years from today")
        ax.set_ylabel("Median after-tax wealth " + ("(today's rand)" if real else "(nominal ZAR)"))
        ax.set_title("After-tax compounding by wrapper (median path)", loc="left")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_ylim(bottom=0)
        ax.grid(True)
        col = "tax_saved_vs_disc_real" if real else "tax_saved_vs_disc_nominal"
        keys = [RA, TFSA]
        vals = [float(s.loc[k, col]) for k in keys]
        bars = bx.bar([WRAPPER_NAMES[k] for k in keys], vals, color=[COLORS[k] for k in keys], width=0.55)
        for b, v in zip(bars, vals):
            bx.annotate(zar(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")
        bx.yaxis.set_major_formatter(ZAR_FMT)
        bx.set_title("Rands saved in tax vs Discretionary", loc="left")
        bx.set_ylabel("Extra after-tax wealth at retirement " + ("(today's rand)" if real else "(nominal)"))
        bx.tick_params(axis="x", labelsize=8.5)
        _style(bx)
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Stress tests
# --------------------------------------------------------------------------- #


def plot_stress(stress: StressResult, target_real: float | None = None) -> Figure:
    """Median real terminal wealth by scenario, with P10 markers and probability of reaching the goal."""
    t = stress.table.iloc[::-1]
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(12.5, 4.8))
        ax = fig.subplots()
        colors = ["#00796b" if k == "base" else "#e57373" for k in t.index]
        y = np.arange(len(t))
        ax.barh(y, t["median_real"], color=colors, height=0.55, label="Median")
        ax.scatter(t["p10_real"], y, marker="|", s=420, color=COLORS["text"], zorder=4, label="P10")
        for yi, (_, r) in zip(y, t.iterrows()):
            change = "" if r.name == "base" else f"\n{r['change_vs_base']:+.0%} vs base"
            ax.annotate(
                f"{zar(r['median_real'])}  |  P(goal) {r['prob_goal']:.0%}{change}",
                xy=(1.01, yi),
                xycoords=("axes fraction", "data"),
                va="center",
                fontsize=9,
                annotation_clip=False,
            )
        if target_real:
            ax.axvline(target_real, color=COLORS["goal"], ls="--", lw=1.5, label=f"Goal {zar(target_real)}")
        ax.set_yticks(y)
        ax.set_yticklabels(t["scenario"])
        ax.xaxis.set_major_formatter(ZAR_FMT)
        ax.set_xlabel("Terminal wealth at retirement (today's rand, after tax)")
        ax.set_title("SA macro stress tests: terminal wealth", loc="left")
        ax.set_xlim(0, float(max(t["median_real"].max(), target_real or 0)) * 1.15)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.45, -0.04))
        fig.tight_layout(rect=(0, 0.05, 0.82, 1))
    return fig


# --------------------------------------------------------------------------- #
# Documentation charts
# --------------------------------------------------------------------------- #


def generate_all_charts(
    out_dir: str | Path = "docs",
    dpi: int = DPI,
    config: SimulationConfig | None = None,
    weights: Mapping[str, float] | np.ndarray | None = None,
    assumptions: MarketAssumptions | None = None,
) -> list[Path]:
    """Write ``fan_chart.png``, ``reg28_frontier.png``, ``tax_wrappers.png`` and ``stress_scenarios.png``."""
    assumptions = assumptions or default_assumptions()
    config = config or SimulationConfig(horizon_years=30, n_paths=5_000)
    weights = STATIC_PRESETS["Moderate (Reg 28)"] if weights is None else weights
    out = Path(out_dir)
    paths = []
    res = simulate(config, weights, assumptions)
    paths.append(save_figure(plot_fan_chart(res, real=True), out / "fan_chart.png", dpi))
    fu = efficient_frontier(assumptions, None, 45)
    fr = efficient_frontier(assumptions, REG28, 45)
    sens = offshore_sensitivity(assumptions, premiums=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05))
    paths.append(save_figure(plot_reg28_frontier(fu, fr, assumptions, sens), out / "reg28_frontier.png", dpi))
    tc = run_tax_comparison(config, weights, assumptions)
    paths.append(save_figure(plot_tax_wrappers(tc, real=True), out / "tax_wrappers.png", dpi))
    st = run_stress(config, weights, assumptions)
    paths.append(save_figure(plot_stress(st, config.target_real), out / "stress_scenarios.png", dpi))
    return paths
