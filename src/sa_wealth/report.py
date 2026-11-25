"""Client wealth plan: a Markdown proposal assembled from the simulation outputs."""

from __future__ import annotations

from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from .assets import ASSET_NAMES, MarketAssumptions, default_assumptions
from .montecarlo import SimulationResult, StressResult, TaxComparison
from .reg28_tax import REG28, WRAPPER_NAMES, check_reg28


def rand(x: float) -> str:
    """R1,234,567 style formatting."""
    sign = "-" if x < 0 else ""
    return f"{sign}R{abs(x):,.0f}"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in [df.index.name or ""] + cols) + " |"]
    lines.append("|" + "|".join(["---"] * (len(cols) + 1)) + "|")
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)


def build_client_plan(
    result: SimulationResult,
    comparison: TaxComparison | None = None,
    stress: StressResult | None = None,
    tradeoff: Mapping[str, pd.DataFrame] | None = None,
    assumptions: MarketAssumptions | None = None,
    client_name: str = "Client",
    current_age: int | None = None,
    retirement_age: int | None = None,
) -> str:
    """Return the *Client Wealth Plan* as Markdown."""
    a = assumptions or default_assumptions()
    cfg = result.config
    s = result.summary()
    rep = check_reg28(result.weights_used, cfg.limits, a.assets)
    L: list[str] = []
    L.append(f"# Client Wealth Plan - {client_name}")
    L.append(f"*Prepared {date.today():%d %B %Y} with sa-wealth-reg28-simulator. Illustrative modelling only - not financial advice.*\n")

    L.append("## 1. Your plan")
    ages = f"- Age today / retirement age: **{current_age} / {retirement_age}**\n" if current_age and retirement_age else ""
    L.append(
        ages
        + f"- Investment horizon: **{cfg.horizon_years} years**\n"
        + f"- Initial capital: **{rand(cfg.initial_lump_sum)}**\n"
        + f"- Monthly contribution (today's rand, escalating with inflation): **{rand(cfg.monthly_contribution)}**\n"
        + f"- Retirement goal: **{rand(cfg.target_real)}** in today's purchasing power\n"
        + f"- Marginal tax rate: **{cfg.marginal_rate:.0%}**   |   Tax tables: **{cfg.tax.label}**\n"
        + f"- Vehicle modelled: **{WRAPPER_NAMES[result.wrapper]}**   |   Inflation assumption: **{a.inflation_mean:.1%}** p.a. (stochastic)\n"
    )

    L.append("## 2. Recommended allocation")
    alloc = pd.DataFrame(
        {
            "Asset class": [ASSET_NAMES.get(x, x) for x in a.assets],
            "Weight": [f"{w:.1%}" for w in result.weights_used],
            "Exp. return": [f"{m:.1%}" for m in a.mu],
            "Volatility": [f"{v:.1%}" for v in a.vol],
        },
        index=pd.Index(a.assets, name="Code"),
    )
    L.append(_md_table(alloc))
    L.append(
        f"\n**{rep.badge()}** - equity {rep.equity:.1%} (limit {cfg.limits.max_equity:.0%}), "
        f"offshore {rep.offshore:.1%} (limit {cfg.limits.max_offshore:.0%}), "
        f"listed property {rep.property:.1%} (limit {cfg.limits.max_property:.0%})."
    )
    if result.reg28_adjusted:
        L.append("\n> Your requested allocation breached Regulation 28, so the retirement annuity holds the closest compliant allocation shown above.")
    L.append("")

    L.append(f"## 3. Monte Carlo projection ({s['n_paths']:,} simulated paths)")
    pn, pr = result.percentiles(real=False).iloc[-1], result.percentiles(real=True).iloc[-1]
    proj = pd.DataFrame(
        {"Nominal ZAR": [rand(pn[c]) for c in pn.index], "Today's rand (real)": [rand(pr[c]) for c in pr.index]},
        index=pd.Index(list(pn.index), name="Percentile"),
    )
    L.append(_md_table(proj))
    L.append(
        f"\n- **Probability of reaching the goal: {s['prob_goal']:.0%}**\n"
        f"- Shortfall risk (CVaR 5%, average of the worst 5% of outcomes): **{rand(s['cvar5_real'])}** real / {rand(s['cvar5_nominal'])} nominal\n"
        f"- Average shortfall versus goal across all paths: {rand(s['expected_shortfall_vs_goal_real'])} (today's rand)\n"
        f"- Cumulative tax paid over the plan (incl. exit tax): {rand(s['cumulative_tax_drag_nominal'])} nominal / {rand(s['cumulative_tax_drag_real'])} real"
    )
    if s["ra_refunds_nominal"] > 0:
        L.append(f"- Upfront RA tax refunds received (reinvested): {rand(s['ra_refunds_nominal'])} nominal")
    L.append("")

    if tradeoff is not None:
        L.append("## 4. Regulation 28: what do the limits cost?")
        p = tradeoff["portfolios"]
        t = pd.DataFrame(
            {
                "Exp. return": p["expected_return"].map("{:.2%}".format),
                "Volatility": p["volatility"].map("{:.2%}".format),
                "Sharpe": p["sharpe"].map("{:.3f}".format),
                "Offshore": p["offshore"].map("{:.1%}".format),
                "Equity": p["equity"].map("{:.1%}".format),
            }
        )
        L.append(_md_table(t))
        g = tradeoff["frontier_gap"]
        L.append(
            f"\nHighest reachable expected return: **{g.attrs.get('max_return_reg28', float('nan')):.2%}** under Regulation 28 vs "
            f"**{g.attrs.get('max_return_unconstrained', float('nan')):.2%}** unconstrained.\n"
        )

    if comparison is not None:
        L.append("## 5. Tax wrappers compared")
        c = comparison.summary
        t = pd.DataFrame(
            {
                "Median after-tax (real)": c["median_terminal_real"].map(rand),
                "Tax paid (real, mean)": c["tax_paid_real"].map(rand),
                "RA refunds (real)": c["ra_refunds_real"].map(rand),
                "Rands saved vs Discretionary (real)": c["tax_saved_vs_disc_real"].map(rand),
            }
        )
        t.index = [WRAPPER_NAMES[k] for k in c.index]
        t.index.name = "Wrapper"
        L.append(_md_table(t))
        L.append(
            "\n*TFSA contributions are capped at the annual and lifetime limits; any excess is invested in a taxable account. "
            "The RA is Regulation 28 constrained, the other two wrappers are not.*\n"
        )

    if stress is not None:
        L.append("## 6. SA macro stress tests")
        t = stress.table
        out = pd.DataFrame(
            {
                "Median (real)": t["median_real"].map(rand),
                "P10 (real)": t["p10_real"].map(rand),
                "P(goal)": t["prob_goal"].map("{:.0%}".format),
                "CVaR 5% (real)": t["cvar5_real"].map(rand),
                "Median vs base": t["change_vs_base"].map("{:+.1%}".format),
            }
        )
        out.index = t["scenario"]
        out.index.name = "Scenario"
        L.append(_md_table(out))
        L.append("")
        for sc in stress.scenarios:
            L.append(f"- **{sc.name}:** {sc.description}")
        L.append("")

    L.append("## Key assumptions & limitations")
    L.append(
        "- Returns are lognormal and correlated (Cholesky); expected returns are long-run capital market assumptions, not forecasts.\n"
        "- All limits (TFSA, RA deduction cap, CGT exclusion, interest exemption) are nominal and held fixed; income is taxed as dividends (20% DWT / foreign effective 20%), "
        "interest (marginal rate after the annual exemption) or property distributions (marginal rate).\n"
        "- RA exit: one third as a lump sum on the retirement table, two thirds as an annuity taxed at an assumed retirement marginal rate.\n"
        "- The discretionary account realises gains when it rebalances annually and at the end of the horizon; capital losses are carried forward.\n"
        "- Past performance does not predict future returns. This document is not financial advice; consult a licensed financial adviser."
    )
    return "\n".join(L) + "\n"
