"""Streamlit dashboard for sa-wealth-reg28-simulator.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        Streamlit Community Cloud, main file path ``app/streamlit_app.py``
               (requirements.txt installs the package via ``-e .[app]``).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # works when the package is installed (Streamlit Cloud: -e .[app])
    import sa_wealth  # noqa: F401
except ImportError:  # fall back to the repo's src/ folder when run from a bare checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from sa_wealth.assets import ASSET_NAMES, ASSETS, default_assumptions
from sa_wealth.montecarlo import SimulationConfig, run_stress, run_tax_comparison, simulate
from sa_wealth.plots import plot_fan_chart, plot_gauge, plot_reg28_frontier, plot_stress, plot_tax_wrappers
from sa_wealth.report import build_client_plan, rand
from sa_wealth.reg28_tax import (
    MARGINAL_RATES,
    REG28,
    TAX_PRESETS,
    WRAPPER_NAMES,
    check_reg28,
    efficient_frontier,
    offshore_sensitivity,
    portfolio_presets,
    reg28_tradeoff,
)

st.set_page_config(page_title="SA Wealth | Regulation 28 Simulator", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# cached computations (all arguments are plain, hashable values)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def cached_presets(infl: float) -> dict[str, tuple[float, ...]]:
    a = default_assumptions().with_inflation(infl)
    return {k: tuple(float(x) for x in v) for k, v in portfolio_presets(a).items()}


def make_config(p: dict) -> SimulationConfig:
    return SimulationConfig(
        initial_lump_sum=p["initial"],
        monthly_contribution=p["monthly"],
        horizon_years=p["years"],
        target_real=p["target"],
        n_paths=p["paths"],
        seed=p["seed"],
        marginal_rate=p["marginal"],
        annual_income=p["income"] or None,
        wrapper=p["wrapper"],
        tax=TAX_PRESETS[p["tax_year"]],
    )


@st.cache_data(show_spinner="Running Monte Carlo simulation...")
def cached_simulation(p: dict, weights: tuple[float, ...]):
    a = default_assumptions().with_inflation(p["infl"])
    return simulate(make_config(p), np.array(weights), a)


@st.cache_data(show_spinner="Comparing tax wrappers...")
def cached_tax(p: dict, weights: tuple[float, ...]):
    a = default_assumptions().with_inflation(p["infl"])
    return run_tax_comparison(make_config(p), np.array(weights), a)


@st.cache_data(show_spinner="Running macro stress scenarios...")
def cached_stress(p: dict, weights: tuple[float, ...]):
    a = default_assumptions().with_inflation(p["infl"])
    return run_stress(make_config(p), np.array(weights), a)


@st.cache_data(show_spinner="Optimising portfolios...")
def cached_optimiser(infl: float):
    a = default_assumptions().with_inflation(infl)
    return {
        "unc": efficient_frontier(a, None, 45),
        "reg": efficient_frontier(a, REG28, 45),
        "sens": offshore_sensitivity(a, premiums=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05)),
        "tradeoff": reg28_tradeoff(a),
    }


def badge(ok: bool, text: str) -> None:
    color = "#1b5e20" if ok else "#b71c1c"
    mark = "&#10004;" if ok else "&#10008;"
    st.markdown(
        f"<span style='background:{color};color:#fff;padding:6px 16px;border-radius:18px;font-weight:600;"
        f"font-size:0.95rem'>{mark}&nbsp; {text}</span>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #

st.title("SA Wealth: Regulation 28 & Tax-Aware Monte Carlo Allocator")
st.caption("Illustrative modelling of a South African investor's plan in ZAR. Not financial advice.")

sb = st.sidebar
sb.header("Client")
age = sb.slider("Current age", 18, 70, 30)
ret_age = sb.slider("Retirement age", 45, 75, 65)
years = ret_age - age
if not 5 <= years <= 40:
    sb.error(f"Horizon is {years} years - choose ages 5 to 40 years apart.")
    st.stop()
sb.caption(f"Investment horizon: **{years} years**")
initial = sb.slider("Initial capital (ZAR)", 0, 5_000_000, 250_000, step=10_000, format="R%d")
monthly = sb.slider("Monthly contribution (ZAR, today's money)", 0, 50_000, 5_000, step=500, format="R%d")
target = sb.slider("Target real retirement capital (today's ZAR)", 500_000, 30_000_000, 5_000_000, step=250_000, format="R%d")
marginal = sb.select_slider("Marginal tax bracket", options=list(MARGINAL_RATES), value=0.36, format_func=lambda x: f"{x:.0%}")

sb.header("Portfolio")
with sb.expander("Market & model settings", expanded=False):
    infl = st.slider("Mean SA CPI inflation (%)", 3.0, 10.0, 5.5, step=0.1) / 100.0
    paths = st.select_slider("Monte Carlo paths", options=[2_000, 5_000, 10_000], value=5_000)
    seed = int(st.number_input("Random seed", 0, 10_000, 42))
    tax_year = st.selectbox(
        "Statutory tax parameters",
        list(TAX_PRESETS),
        index=0,
        help="2025/26 uses TFSA R36k, RA cap R350k, CGT exclusion R40k. 2026/27 reflects Budget 2026 (R46k / R430k / R50k).",
    )
    income_in = st.number_input("Annual taxable income (ZAR, 0 = typical for bracket)", 0, 10_000_000, 0, step=10_000)
    wrapper = st.selectbox("Wrapper for projection & stress tabs", ["RA", "TFSA", "DISC"], format_func=lambda k: WRAPPER_NAMES[k])

presets = cached_presets(infl)
choice = sb.selectbox("Portfolio preset", list(presets) + ["Custom weights"], index=list(presets).index("Moderate (Reg 28)"))
if choice == "Custom weights":
    base = presets["Moderate (Reg 28)"]
    raw = [
        sb.slider(ASSET_NAMES[a], 0, 100, int(round(base[i] * 100)), key=f"w_{a}")
        for i, a in enumerate(ASSETS)
    ]
    if sum(raw) == 0:
        sb.error("Set at least one weight above zero.")
        st.stop()
    weights = tuple(x / sum(raw) for x in raw)
    sb.caption("Weights are normalised to 100%: " + ", ".join(f"{a} {w:.0%}" for a, w in zip(ASSETS, weights) if w > 0))
else:
    weights = presets[choice]

params = dict(
    initial=float(initial),
    monthly=float(monthly),
    years=int(years),
    target=float(target),
    paths=int(paths),
    seed=seed,
    marginal=float(marginal),
    income=float(income_in),
    wrapper=wrapper,
    tax_year=tax_year,
    infl=float(infl),
)
w_arr = np.array(weights)
report = check_reg28(w_arr)

tab1, tab2, tab3, tab4 = st.tabs(
    ["1 | Wealth projection", "2 | Reg 28 optimiser & offshore cap", "3 | Tax wrappers", "4 | Stress tests & client plan"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - Monte Carlo projection
# --------------------------------------------------------------------------- #
with tab1:
    res = cached_simulation(params, weights)
    mode = st.radio("Show wealth in", ["Real (today's purchasing power)", "Nominal ZAR"], horizontal=True)
    real = mode.startswith("Real")
    s = res.summary()
    st.caption(f"{WRAPPER_NAMES[res.wrapper]}  |  {years} years  |  {s['n_paths']:,} paths  |  after-tax wealth")
    if res.reg28_adjusted:
        st.info(
            "Your allocation is not Regulation 28 compliant, so the retirement annuity uses the closest compliant "
            "allocation. Pick TFSA or Discretionary in the model settings to hold it as is."
        )
    c1, c2 = st.columns([1, 2])
    with c1:
        st.pyplot(plot_gauge(s["prob_goal"]))
        st.caption(f"Probability of reaching {rand(target)} (today's rand)")
    with c2:
        m1, m2, m3 = st.columns(3)
        pct = res.percentiles(real=real).iloc[-1]
        m1.metric("Median at retirement", rand(pct["P50"]))
        m2.metric(
            "Shortfall risk: CVaR 5%",
            rand(s["cvar5_real"] if real else s["cvar5_nominal"]),
            help="Average terminal wealth in the worst 5% of simulated paths.",
        )
        m3.metric(
            "Cumulative tax drag",
            rand(s["cumulative_tax_drag_real"] if real else s["cumulative_tax_drag_nominal"]),
            help="Mean tax paid over the plan including exit tax (RA refunds are shown separately).",
        )
        m4, m5, m6 = st.columns(3)
        m4.metric("P10 (bad case)", rand(pct["P10"]))
        m5.metric("P90 (good case)", rand(pct["P90"]))
        m6.metric(
            "Avg shortfall vs goal",
            rand(s["expected_shortfall_vs_goal_real"]),
            help="Average amount below the goal across all paths, in today's rand.",
        )
    st.pyplot(plot_fan_chart(res, real=real))
    st.subheader("Percentile trajectories")
    traj = res.percentiles(real=real)
    step = max(1, len(traj) // 8)
    rows = sorted(set(list(range(0, len(traj), step)) + [len(traj) - 1]))
    st.dataframe(traj.iloc[rows].map(rand))

# --------------------------------------------------------------------------- #
# Tab 2 - Regulation 28
# --------------------------------------------------------------------------- #
with tab2:
    st.subheader("Regulation 28 compliance of your allocation")
    badge(report.compliant, report.badge())
    cols = st.columns(3)
    for col, (label, val, cap) in zip(
        cols,
        (
            ("Total equity", report.equity, REG28.max_equity),
            ("Total offshore", report.offshore, REG28.max_offshore),
            ("Listed property", report.property, REG28.max_property),
        ),
    ):
        col.metric(label, f"{val:.1%}", delta=f"{cap - val:+.1%} headroom to {cap:.0%}", delta_color="normal")
    for b in report.breaches:
        st.warning(b)
    opt = cached_optimiser(infl)
    st.subheader("Efficient frontier: unconstrained vs Regulation 28")
    st.pyplot(
        plot_reg28_frontier(
            opt["unc"], opt["reg"], default_assumptions().with_inflation(infl), opt["sens"], highlight={"Your portfolio": w_arr}
        )
    )
    tr = opt["tradeoff"]
    st.subheader("Risk-return table")
    p = tr["portfolios"]
    table = pd.DataFrame(
        {
            "Expected return": p["expected_return"].map("{:.2%}".format),
            "Volatility": p["volatility"].map("{:.2%}".format),
            "Sharpe": p["sharpe"].map("{:.3f}".format),
            "Equity": p["equity"].map("{:.1%}".format),
            "Offshore": p["offshore"].map("{:.1%}".format),
            "Property": p["property"].map("{:.1%}".format),
            "Diversification ratio": p["diversification_ratio"].map("{:.2f}".format),
        }
    )
    st.dataframe(table)
    g = tr["frontier_gap"]
    st.markdown(
        f"**What the limits cost.** At the base assumptions the Sharpe-optimal portfolio is already inside the limits "
        f"(about 30% offshore), so Regulation 28 is almost free there. The limits bite at the growth end of the frontier: "
        f"the highest reachable return is **{g.attrs['max_return_reg28']:.2%}** under Regulation 28 versus "
        f"**{g.attrs['max_return_unconstrained']:.2%}** unconstrained. The right-hand chart shows how strong a weak-Rand view on "
        f"global assets must be before the 45% offshore cap binds."
    )
    gap = g[["vol_unconstrained", "vol_reg28", "vol_cost_reg28", "offshore_unconstrained", "offshore_reg28"]].copy()
    gap.index = [f"{x:.2%}" for x in gap.index]
    gap.index.name = "Target return"
    st.caption("Minimum volatility needed to reach each target return (%)")
    st.dataframe((gap * 100).round(2).astype(object).where(gap.notna(), "infeasible"))
    st.caption("Marginal cost of each cap (max-Sharpe with one cap removed)")
    st.dataframe(tr["cap_by_cap"].round(4))

# --------------------------------------------------------------------------- #
# Tab 3 - Tax wrappers
# --------------------------------------------------------------------------- #
with tab3:
    tc = cached_tax(params, weights)
    sm = tc.summary
    col_saved = "tax_saved_vs_disc_real" if real else "tax_saved_vs_disc_nominal"
    st.subheader("RA vs TFSA vs Taxable Discretionary")
    k1, k2, k3 = st.columns(3)
    k1.metric("RA: Rands saved vs discretionary", rand(sm.loc["RA", col_saved]))
    k2.metric("TFSA: Rands saved vs discretionary", rand(sm.loc["TFSA", col_saved]))
    pretax = float(sm["median_pretax_real"].iloc[0]) if real else float(np.median(tc.outcomes["GROSS"].terminal_net))
    k3.metric("Median pre-tax wealth (no wrapper)", rand(pretax))
    st.pyplot(plot_tax_wrappers(tc, real=real))
    out = pd.DataFrame(
        {
            "Median after-tax wealth": sm["median_terminal_real" if real else "median_terminal_nominal"].map(rand),
            "Tax paid (mean)": sm["tax_paid_real" if real else "tax_paid_nominal"].map(rand),
            "RA refunds (mean, real)": sm["ra_refunds_real"].map(rand),
            "Rands saved vs Discretionary": sm[col_saved].map(rand),
            "Paths beating Discretionary": sm["share_paths_beating_disc"].map("{:.0%}".format),
        }
    )
    out.index = [WRAPPER_NAMES[k] for k in sm.index]
    st.dataframe(out)
    cfg = make_config(params)
    st.caption(
        f"TFSA limits: R{cfg.tax.tfsa_annual_limit:,.0f} per year and R{cfg.tax.tfsa_lifetime_limit:,.0f} lifetime "
        f"(mean {rand(sm.loc['TFSA', 'tfsa_contributed_nominal'])} contributed; the excess is invested in a taxable account). "
        f"Discretionary: 20% dividend withholding tax, R{cfg.tax.interest_exemption:,.0f} interest exemption, effective CGT "
        f"{cfg.tax.effective_cgt_rate(marginal):.1%} ({cfg.tax.cgt_inclusion:.0%} inclusion x {marginal:.0%}) after a "
        f"R{cfg.tax.cgt_annual_exclusion:,.0f} annual exclusion. Tax tables: {cfg.tax.label}."
    )
    if not np.allclose(tc.weights_ra, tc.weights_unconstrained, atol=1e-6):
        st.info(
            "The RA holds the closest Regulation 28 compliant allocation: "
            + ", ".join(f"{a} {x:.0%}" for a, x in zip(ASSETS, tc.weights_ra) if x > 1e-6)
        )

# --------------------------------------------------------------------------- #
# Tab 4 - Stress tests & export
# --------------------------------------------------------------------------- #
with tab4:
    stress = cached_stress(params, weights)
    st.subheader("SA macro stress tests")
    st.pyplot(plot_stress(stress, target))
    t = stress.table
    out = pd.DataFrame(
        {
            "Median (real)": t["median_real"].map(rand),
            "P10 (real)": t["p10_real"].map(rand),
            "P(goal)": t["prob_goal"].map("{:.0%}".format),
            "CVaR 5% (real)": t["cvar5_real"].map(rand),
            "Median vs base": t["change_vs_base"].map("{:+.1%}".format),
            "Mean CPI": t["mean_annual_inflation"].map("{:.2%}".format),
        }
    )
    out.index = t["scenario"]
    st.dataframe(out)
    for sc in stress.scenarios:
        with st.expander(sc.name):
            st.write(sc.description)
    st.subheader("Client proposal")
    name = st.text_input("Client name", "Client")
    plan_md = build_client_plan(
        res,
        tc,
        stress,
        opt["tradeoff"],
        default_assumptions().with_inflation(infl),
        client_name=name,
        current_age=age,
        retirement_age=ret_age,
    )
    st.download_button(
        "Download Client Wealth Plan (Markdown)", data=plan_md, file_name="client_wealth_plan.md", mime="text/markdown"
    )
    with st.expander("Preview"):
        st.markdown(plan_md)
