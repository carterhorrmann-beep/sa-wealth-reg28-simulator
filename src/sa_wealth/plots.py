"""High-DPI charts (matplotlib, object-oriented API so they work headless and inside Streamlit).

Every ``plot_*`` function returns a :class:`matplotlib.figure.Figure`. :func:`generate_all_charts`
writes the six documentation charts used in the README.
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
from .drawdown import CrashImpact, DrawdownConfig, DrawdownResult, run_crash_impact
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
    STRATEGY_BLENDED,
    STRATEGY_DISC,
    STRATEGY_RA,
    TAX_2026_27,
    TFSA,
    WRAPPER_NAMES,
    SmartSplitResult,
    efficient_frontier,
    offshore_sensitivity,
    optimize_portfolio,
    optimize_wrapper_split,
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
    "crash": "#d84315",
    "income": "#3949ab",
    "cap": "#8e24aa",
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
# Living Annuity drawdown
# --------------------------------------------------------------------------- #


def _draw_fan(ax, result: DrawdownResult, crash: DrawdownResult | None = None, real: bool = True) -> None:
    pct = result.percentiles(real=real)
    ages = result.ages
    ax.fill_between(ages, pct["P10"], pct["P90"], color=COLORS["band_outer"], label="P10 - P90", lw=0)
    ax.fill_between(ages, pct["P25"], pct["P75"], color=COLORS["band_inner"], alpha=0.85, label="P25 - P75", lw=0)
    ax.plot(ages, pct["P50"], color=COLORS["median"], lw=2.4, label="Median (P50)")
    if crash is not None:
        cp = crash.percentiles(real=real)
        ax.plot(ages, cp["P50"], color=COLORS["crash"], lw=2.0, ls="--", label="Median after year-1 crash")
    ax.yaxis.set_major_formatter(ZAR_FMT)
    ax.set_xlabel("Age")
    ax.set_ylabel("Capital (today's rand)" if real else "Capital (nominal ZAR)")
    ax.set_xlim(ages[0], ages[-1])
    ax.set_ylim(bottom=0)
    ax.annotate(
        f"Median bequest {zar(pct['P50'].iloc[-1])}",
        xy=(ages[-1], pct["P50"].iloc[-1]),
        xytext=(-6, 16),
        textcoords="offset points",
        ha="right",
        color=COLORS["median"],
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True)


def _draw_income(ax, result: DrawdownResult, real: bool = True) -> None:
    inc = result.income_percentiles(real=real) / 12.0
    ages = result.ages[:-1]
    ax.fill_between(ages, inc["P10"], inc["P90"], color=COLORS["band_outer"], label="P10 - P90", lw=0)
    ax.plot(ages, inc["P50"], color=COLORS["income"], lw=2.4, label="Median income")
    if real:
        ax.axhline(
            result.config.monthly_income_real, color=COLORS["goal"], ls="--", lw=1.6,
            label=f"Desired {zar(result.config.monthly_income_real)} / month",
        )
    ax.yaxis.set_major_formatter(ZAR_FMT)
    ax.set_xlabel("Age")
    ax.set_ylabel("Monthly income (today's rand)" if real else "Monthly income (nominal ZAR)")
    ax.set_xlim(ages[0], ages[-1])
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True)


def _draw_survival(ax, result: DrawdownResult, crash: DrawdownResult | None = None) -> None:
    ax.plot(result.ages, result.survival_curve() * 100, color=COLORS["median"], lw=2.4, label="Survival probability")
    ax.plot(result.ages, result.cap_breach_curve() * 100, color=COLORS["cap"], lw=2.0, ls="-.", label="17.5% ceiling breached")
    if crash is not None:
        ax.plot(result.ages, crash.survival_curve() * 100, color=COLORS["crash"], lw=2.0, ls="--", label="Survival after year-1 crash")
    milestones = result.survival_milestones()
    for age in milestones:
        ax.axvline(age, color="#90a4ae", ls=":", lw=1.0)
    if milestones:
        ax.text(
            0.03, 0.55, "Survival probability\n" + "\n".join(f"to age {a}: {p:.0%}" for a, p in milestones.items()),
            transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=COLORS["median"], va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=COLORS["grid"]),
        )
    ax.set_xlabel("Age")
    ax.set_ylabel("Probability (%)")
    ax.set_xlim(result.ages[0], result.ages[-1])
    ax.set_ylim(0, 102)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True)


def plot_drawdown_fan(result: DrawdownResult, crash: DrawdownResult | None = None, real: bool = True) -> Figure:
    """Living Annuity capital fan (P10-P90, P25-P75, median) by age, optionally with the crash-scenario median."""
    cfg = result.config
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(9.5, 5.2))
        ax = fig.subplots()
        _draw_fan(ax, result, crash, real)
        ax.set_title(
            f"Living Annuity capital - {zar(cfg.initial_capital)} start, {zar(cfg.monthly_income_real)}/month "
            f"(today's rand)\n{result.capital.shape[0]:,} paths | survival to {cfg.end_age} = {result.survival_prob():.0%} | "
            f"{result.portfolio}",
            loc="left",
        )
        fig.tight_layout()
    return fig


def plot_drawdown_income(result: DrawdownResult, real: bool = True) -> Figure:
    """Monthly income paid (P10 / median / P90) against the income the retiree wants."""
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(9.5, 4.6))
        ax = fig.subplots()
        _draw_income(ax, result, real)
        ax.set_title("Income actually paid vs desired income", loc="left")
        fig.tight_layout()
    return fig


def plot_survival_curve(result: DrawdownResult, crash: DrawdownResult | None = None) -> Figure:
    """Probability the fund survives, and the cumulative probability that the 17.5% ceiling has been breached."""
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(9.5, 4.6))
        ax = fig.subplots()
        _draw_survival(ax, result, crash)
        ax.set_title("Portfolio survival and statutory-ceiling breach by age", loc="left")
        fig.tight_layout()
    return fig


def plot_living_annuity(result: DrawdownResult, impact: CrashImpact | None = None, real: bool = True) -> Figure:
    """Three-panel Living Annuity summary: capital fan, income paid vs desired, survival and ceiling breach."""
    cfg = result.config
    crash = None if impact is None else impact.crash
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(18, 5.6))
        ax, bx, cx = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.25, 1, 1]})
        _draw_fan(ax, result, crash, real)
        ax.set_title("Capital fan", loc="left")
        _draw_income(bx, result, real)
        bx.set_title("Income paid vs desired", loc="left")
        _draw_survival(cx, result, crash)
        cx.set_title("Survival & 17.5% ceiling", loc="left")
        breach = result.median_years_to_cap_breach
        breach_txt = "never" if np.isnan(breach) else f"median year {breach:.0f}"
        fig.suptitle(
            f"Living Annuity drawdown: {zar(cfg.initial_capital)} at age {cfg.retirement_age}, "
            f"{zar(cfg.monthly_income_real)}/month escalating with CPI ({cfg.initial_withdrawal_rate:.1%} initial rate), "
            f"{result.portfolio}\n"
            f"survival to {cfg.end_age}: {result.survival_prob():.0%} | ceiling breached in {result.cap_breach_prob:.0%} of paths ({breach_txt}) | "
            f"median real bequest {zar(result.median_bequest_real)}"
            + ("" if impact is None else f" | year-1 -25% equity crash: survival {impact.crash.survival_prob():.0%}"),
            x=0.01,
            y=0.995,
            ha="left",
            fontsize=11.5,
            fontweight="bold",
        )
        fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# SARS smart contribution split
# --------------------------------------------------------------------------- #

_SPLIT_LABELS = {STRATEGY_DISC: "Discretionary\nonly", STRATEGY_RA: "RA only\n(capped)", STRATEGY_BLENDED: "Tax-optimal\nblend"}
_SPLIT_COLORS = {"RA": COLORS[RA], "TFSA": COLORS[TFSA], "DISC": COLORS[DISC]}


def plot_smart_split(split: SmartSplitResult, real: bool = True) -> Figure:
    """Year-1 monthly split, after-tax terminal wealth and lifetime tax for the three contribution strategies."""
    keys = [STRATEGY_DISC, STRATEGY_RA, STRATEGY_BLENDED]
    labels = [_SPLIT_LABELS[k] for k in keys]
    sfx = "(today's rand)" if real else "(nominal ZAR)"
    with mpl.rc_context(_RC):
        fig = Figure(figsize=(17, 5.6))
        ax, bx, cx = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1.1, 1, 1.15]})

        bottom = np.zeros(len(keys))
        for part, name in (("RA", "RA"), ("TFSA", "TFSA"), ("DISC", "Discretionary")):
            vals = np.array([split.strategies[k].monthly_split[part] for k in keys])
            ax.bar(labels, vals, bottom=bottom, color=_SPLIT_COLORS[part], label=name, width=0.6)
            for i, v in enumerate(vals):
                if v >= 0.045 * split.monthly_budget:
                    ax.text(i, bottom[i] + v / 2, f"R{v:,.0f}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
            bottom += vals
        ax.set_ylabel("Year-1 monthly contribution (ZAR)")
        ax.yaxis.set_major_formatter(ZAR_FMT)
        ax.set_title(f"Monthly split of R{split.monthly_budget:,.0f}", loc="left")
        ax.legend(loc="upper left", fontsize=8.5, ncol=3, bbox_to_anchor=(0, -0.16))
        _style(ax)

        med = [float(np.median(split.strategies[k].terminal_real if real else split.strategies[k].terminal_net)) for k in keys]
        bars = bx.bar(labels, med, color=["#c62828", "#00796b", "#1565c0"], width=0.6)
        for b, v in zip(bars, med):
            bx.annotate(zar(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")
        gain = split.gain_vs(split.best_single)["terminal_wealth_gain_real" if real else "terminal_wealth_gain_nominal"]
        bx.set_title(f"Median after-tax wealth, {split.horizon_years} years", loc="left")
        bx.set_ylabel(f"After-tax terminal wealth {sfx}")
        bx.yaxis.set_major_formatter(ZAR_FMT)
        bx.set_ylim(0, max(med) * 1.3)
        bx.text(0.02, 0.98, f"Blend vs best single wrapper:\n{'+' if gain >= 0 else '-'}{zar(abs(gain))} on average", transform=bx.transAxes, va="top", fontsize=9.5, fontweight="bold", color="#1565c0")
        _style(bx)

        x = np.arange(len(keys))
        paid = [float((split.strategies[k].tax_paid_real if real else split.strategies[k].tax_paid).mean()) for k in keys]
        ref = [float((split.strategies[k].refunds_real if real else split.strategies[k].refunds).mean()) for k in keys]
        cx.bar(x - 0.19, paid, width=0.36, color="#e57373", label="Tax paid (income, CGT, RA exit)")
        cx.bar(x + 0.19, ref, width=0.36, color="#66bb6a", label="SARS RA refunds")
        for i, (a, b) in enumerate(zip(paid, ref)):
            cx.annotate(f"net {zar(a - b)}", (i, max(a, b)), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
        cx.set_xticks(x)
        cx.set_xticklabels(labels)
        cx.set_ylim(0, max(paid + ref) * 1.15)
        cx.set_ylabel(f"Lifetime tax {sfx}")
        cx.yaxis.set_major_formatter(ZAR_FMT)
        saved = split.gain_vs(split.best_single)["lifetime_tax_saved_real" if real else "lifetime_tax_saved_nominal"]
        cx.set_title(f"Lifetime tax (blend saves {zar(saved)} vs best single)", loc="left")
        cx.legend(loc="upper left", fontsize=8.5, bbox_to_anchor=(0, -0.16))
        _style(cx)

        p = split.profile
        fig.suptitle(
            f"SARS {p.tax_year} Smart Contribution Split: income R{split.gross_income:,.0f} | marginal {split.marginal_rate:.0%} | "
            f"effective {p.effective_rate:.1%} | {split.horizon_years} years | {split.n_paths:,} paths"
            + (" | refunds reinvested" if split.reinvest_refund else " | refunds spent"),
            x=0.01,
            y=0.995,
            ha="left",
            fontsize=11.5,
            fontweight="bold",
        )
        fig.tight_layout()
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
    drawdown_config: DrawdownConfig | None = None,
    split_budget: float = 40_000.0,
    split_income: float = 1_000_000.0,
    split_years: int = 25,
) -> list[Path]:
    """Write the six documentation charts.

    ``fan_chart.png``, ``reg28_frontier.png``, ``tax_wrappers.png``, ``stress_scenarios.png`` plus
    ``living_annuity.png`` (R8m Living Annuity, R40k/month, 30 years by default) and ``smart_split.png``
    (R40k/month budget on a R1m taxable income, SARS 2026/27 tables).
    """
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

    dcfg = drawdown_config or DrawdownConfig(n_paths=config.n_paths, seed=config.seed)
    impact = run_crash_impact(dcfg, assumptions, "Moderate (Reg 28)")
    paths.append(save_figure(plot_living_annuity(impact.base, impact), out / "living_annuity.png", dpi))
    split = optimize_wrapper_split(
        split_budget,
        split_income,
        None,
        split_years,
        weights=weights,
        assumptions=assumptions,
        tax=TAX_2026_27,
        n_paths=config.n_paths,
        seed=config.seed,
    )
    paths.append(save_figure(plot_smart_split(split), out / "smart_split.png", dpi))
    return paths
