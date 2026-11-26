"""Command line interface: ``sa-wealth simulate | optimize | tax-compare | stress | smart-split | drawdown | charts``."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import __version__
from .assets import ASSETS, MarketAssumptions, load_assumptions, weights_from_mapping
from .drawdown import DrawdownConfig, resolve_la_portfolio, run_crash_impact
from .montecarlo import SimulationConfig, run_stress, run_tax_comparison, simulate
from .reg28_tax import (
    MARGINAL_RATES,
    REG28,
    TAX_PRESETS,
    WRAPPER_NAMES,
    check_reg28,
    efficient_frontier,
    offshore_sensitivity,
    optimize_portfolio,
    optimize_wrapper_split,
    portfolio_presets,
    ra_tax_refund,
    reg28_tradeoff,
)


def rand(x: float) -> str:
    return f"R{x:,.0f}" if x >= 0 else f"-R{abs(x):,.0f}"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def parse_weights(text: str) -> dict[str, float]:
    """``"JSE_EQ=0.3,SA_BOND=0.7"`` -> dict (values may be fractions or percentages that sum to 100)."""
    out: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k = k.strip().upper()
        if k not in ASSETS:
            raise argparse.ArgumentTypeError(f"unknown asset '{k}'. Choose from {', '.join(ASSETS)}")
        out[k] = float(v)
    total = sum(out.values())
    if total <= 0:
        raise argparse.ArgumentTypeError("weights must sum to a positive number")
    return {k: v / total for k, v in out.items()}


def resolve_weights(preset: str | None, weights: str | None, assumptions: MarketAssumptions) -> np.ndarray:
    if weights:
        return weights_from_mapping(parse_weights(weights), assumptions.assets)
    presets = portfolio_presets(assumptions)
    key = _slug(preset or "moderate")
    matches = [n for n in presets if _slug(n).startswith(key) or key in _slug(n)]
    if not matches:
        raise SystemExit(f"Unknown preset '{preset}'. Available: " + ", ".join(_slug(n) for n in presets))
    return presets[matches[0]]


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("client plan")
    g.add_argument("--initial", type=float, default=250_000, help="initial lump sum in ZAR (default 250000)")
    g.add_argument("--monthly", type=float, default=5_000, help="monthly contribution in today's ZAR (default 5000)")
    g.add_argument("--years", type=int, default=25, help="investment horizon, 5-40 years (default 25)")
    g.add_argument("--target", type=float, default=5_000_000, help="retirement goal in today's ZAR (default 5000000)")
    g.add_argument("--marginal", type=float, default=0.36, help=f"marginal tax rate, one of {MARGINAL_RATES} (default 0.36)")
    g.add_argument("--income", type=float, default=None, help="annual taxable income in ZAR (sizes the RA deduction)")
    g.add_argument("--preset", default="moderate", help="portfolio preset slug (e.g. conservative, moderate, aggressive, max-sharpe, risk-parity)")
    g.add_argument("--weights", default=None, help='custom weights, e.g. "JSE_EQ=30,SA_BOND=25,GLB_EQ=25,SA_CASH=10,GLB_BOND=10"')
    m = p.add_argument_group("market model")
    m.add_argument("--source", choices=("cma", "yfinance"), default="cma", help="assumption source (default: built-in CMAs)")
    m.add_argument("--inflation", type=float, default=5.5, help="mean SA CPI inflation in percent (default 5.5)")
    m.add_argument("--paths", type=int, default=10_000, help="Monte Carlo paths (default 10000)")
    m.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    m.add_argument("--tax-year", choices=tuple(TAX_PRESETS), default="2025/26", help="statutory tax parameters (default 2025/26 as specified)")


def _setup(args: argparse.Namespace, wrapper: str = "RA"):
    assumptions = load_assumptions(args.source).with_inflation(args.inflation / 100.0)
    cfg = SimulationConfig(
        initial_lump_sum=args.initial,
        monthly_contribution=args.monthly,
        horizon_years=args.years,
        target_real=args.target,
        n_paths=args.paths,
        seed=args.seed,
        marginal_rate=args.marginal,
        annual_income=args.income,
        wrapper=wrapper,
        tax=TAX_PRESETS[args.tax_year],
    )
    w = resolve_weights(args.preset, args.weights, assumptions)
    return assumptions, cfg, w


def _print_weights(w: np.ndarray, assumptions: MarketAssumptions) -> None:
    rep = check_reg28(w, assets=assumptions.assets)
    print("Allocation: " + ", ".join(f"{a} {x:.1%}" for a, x in zip(assumptions.assets, w) if x > 1e-9))
    print(f"Regulation 28: {rep.badge()} (equity {rep.equity:.1%}, offshore {rep.offshore:.1%}, property {rep.property:.1%})")
    for b in rep.breaches:
        print(f"  ! {b}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_simulate(args: argparse.Namespace) -> int:
    assumptions, cfg, w = _setup(args, args.wrapper)
    res = simulate(cfg, w, assumptions)
    s = res.summary()
    print(f"\n=== Monte Carlo: {s['wrapper']} | {cfg.horizon_years} years | {s['n_paths']:,} paths (seed {cfg.seed}) ===")
    _print_weights(res.weights_requested, assumptions)
    if res.reg28_adjusted:
        print("Retirement annuity must be Regulation 28 compliant -> using the closest compliant allocation:")
        _print_weights(res.weights_used, assumptions)
    pct = pd.concat({"nominal ZAR": res.percentiles(False).iloc[-1], "today's rand": res.percentiles(True).iloc[-1]}, axis=1)
    print("\nAfter-tax wealth at retirement")
    print(pct.map(rand).to_string())
    print(f"\nProbability of reaching {rand(cfg.target_real)} (today's rand): {s['prob_goal']:.1%}")
    print(f"Shortfall risk, CVaR 5% terminal wealth:  {rand(s['cvar5_real'])} real | {rand(s['cvar5_nominal'])} nominal")
    print(f"Average shortfall vs goal (real):         {rand(s['expected_shortfall_vs_goal_real'])}")
    print(f"Cumulative tax drag (mean):               {rand(s['cumulative_tax_drag_nominal'])} nominal | {rand(s['cumulative_tax_drag_real'])} real")
    if s["ra_refunds_nominal"]:
        print(f"RA upfront tax refunds (mean, reinvested):{rand(s['ra_refunds_nominal'])} nominal")
    print(f"Mean annual SA CPI over horizon:          {s['mean_annual_inflation']:.2%}")
    if args.report:
        from .report import build_client_plan

        comp = run_tax_comparison(cfg, w, assumptions)
        stress = run_stress(cfg, w, assumptions)
        tradeoff = reg28_tradeoff(assumptions)
        md = build_client_plan(res, comp, stress, tradeoff, assumptions)
        Path(args.report).write_text(md, encoding="utf-8")
        print(f"\nClient wealth plan written to {args.report}")
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    assumptions = load_assumptions(args.source).with_inflation(args.inflation / 100.0)
    rf = None if args.rf is None else args.rf / 100.0
    tr = reg28_tradeoff(assumptions, rf=rf)
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.width = 200
    print("\n=== Unconstrained vs Regulation 28 portfolios ===")
    p = tr["portfolios"]
    print(p.assign(**{c: p[c].map("{:.1%}".format) for c in ("expected_return", "volatility", "equity", "offshore", "property")}).to_string())
    w = pd.DataFrame(
        {
            "Max-Sharpe unconstrained": optimize_portfolio(assumptions, "max_sharpe", None, rf).weights,
            "Max-Sharpe Reg 28": optimize_portfolio(assumptions, "max_sharpe", REG28, rf).weights,
            "Risk parity unconstrained": optimize_portfolio(assumptions, "risk_parity", None, exclude=("SA_CASH",)).weights,
            "Risk parity Reg 28": optimize_portfolio(assumptions, "risk_parity", REG28, exclude=("SA_CASH",)).weights,
        }
    )
    print("\nWeights")
    print((w * 100).round(1).to_string())
    print("\n=== Marginal cost of each cap (max-Sharpe with one cap removed) ===")
    print(tr["cap_by_cap"].round(4).to_string())
    g = tr["frontier_gap"]
    print("\n=== Volatility needed per target return (min-variance) ===")
    show = g[["vol_unconstrained", "vol_reg28", "vol_cost_reg28", "offshore_unconstrained", "offshore_reg28"]].copy()
    show.index = [f"{x:.2%}" for x in show.index]
    print((show * 100).round(2).rename(columns=lambda c: c + " (%)").to_string(na_rep="infeasible"))
    print(
        f"\nHighest reachable return: unconstrained {g.attrs['max_return_unconstrained']:.2%} | "
        f"Regulation 28 {g.attrs['max_return_reg28']:.2%}"
    )
    print("\n=== When does the 45% offshore cap bite? (extra expected return on global assets) ===")
    sens = offshore_sensitivity(assumptions)
    sens.index = [f"+{x:.0%}" for x in sens.index]
    print(sens.round(4).to_string())
    if args.target_return is not None:
        t = args.target_return / 100.0
        print(f"\n=== Minimum-variance portfolio for a {t:.2%} return target ===")
        for label, lim in (("unconstrained", None), ("Regulation 28", REG28)):
            try:
                r = optimize_portfolio(assumptions, "min_variance", lim, target_return=t)
                print(f"{label:>14}: vol {r.volatility:.2%}, " + ", ".join(f"{a} {x:.1%}" for a, x in r.weights.items() if x > 1e-4))
            except RuntimeError:
                print(f"{label:>14}: target not attainable")
    return 0


def cmd_tax(args: argparse.Namespace) -> int:
    assumptions, cfg, w = _setup(args)
    tc = run_tax_comparison(cfg, w, assumptions)
    print(f"\n=== Tax wrappers over {cfg.horizon_years} years | marginal rate {cfg.marginal_rate:.0%} | tax tables {cfg.tax.label} ===")
    _print_weights(w, assumptions)
    if not np.allclose(tc.weights_ra, tc.weights_unconstrained, atol=1e-6):
        print("RA holds the closest Regulation 28 compliant allocation: " + ", ".join(f"{a} {x:.1%}" for a, x in zip(assumptions.assets, tc.weights_ra) if x > 1e-9))
    s = tc.summary
    out = pd.DataFrame(
        {
            "median after-tax (real)": s["median_terminal_real"].map(rand),
            "median after-tax (nominal)": s["median_terminal_nominal"].map(rand),
            "mean tax paid (real)": s["tax_paid_real"].map(rand),
            "RA refunds (real)": s["ra_refunds_real"].map(rand),
            "saved vs discretionary (real)": s["tax_saved_vs_disc_real"].map(rand),
            "saved vs discretionary (nominal)": s["tax_saved_vs_disc_nominal"].map(rand),
            "paths beating discretionary": s["share_paths_beating_disc"].map("{:.0%}".format),
        }
    )
    out.index = [WRAPPER_NAMES[k] for k in s.index]
    print(out.T.to_string())
    t = tc.outcomes["TFSA"]
    print(
        f"\nTFSA: mean {rand(float(t.tfsa_contributed.mean()))} contributed (limits R{cfg.tax.tfsa_annual_limit:,.0f}/yr, R{cfg.tax.tfsa_lifetime_limit:,.0f} lifetime); "
        f"excess is invested in a taxable account."
    )
    print(f"Pre-tax (no wrapper) median in today's rand: {rand(float(s['median_pretax_real'].iloc[0]))}")
    return 0


def cmd_stress(args: argparse.Namespace) -> int:
    assumptions, cfg, w = _setup(args, args.wrapper)
    st = run_stress(cfg, w, assumptions)
    t = st.table
    out = pd.DataFrame(
        {
            "median (real)": t["median_real"].map(rand),
            "P10 (real)": t["p10_real"].map(rand),
            "P(goal)": t["prob_goal"].map("{:.0%}".format),
            "CVaR 5% (real)": t["cvar5_real"].map(rand),
            "vs base": t["change_vs_base"].map("{:+.1%}".format),
            "mean CPI": t["mean_annual_inflation"].map("{:.2%}".format),
        }
    )
    out.index = t["scenario"]
    print(f"\n=== SA macro stress tests | {WRAPPER_NAMES[cfg.wrapper]} | goal {rand(cfg.target_real)} (today's rand) ===")
    print(out.to_string())
    for sc in st.scenarios:
        print(f"\n{sc.name}: {sc.description}")
    return 0


def cmd_smart_split(args: argparse.Namespace) -> int:
    assumptions = load_assumptions(args.source).with_inflation(args.inflation / 100.0)
    w = resolve_weights(args.preset, args.weights, assumptions)
    tax = TAX_PRESETS[args.tax_year]
    res = optimize_wrapper_split(
        args.budget,
        args.income,
        args.marginal,
        args.years,
        age=args.age,
        weights=w,
        assumptions=assumptions,
        tax=tax,
        n_paths=args.paths,
        seed=args.seed,
        reinvest_refund=not args.no_reinvest,
    )
    p = res.profile
    print(f"\n=== SARS {p.tax_year} Smart Contribution Split | {rand(res.monthly_budget)}/month | {res.horizon_years} years | {res.n_paths:,} paths (seed {res.seed}) ===")
    _print_weights(w, assumptions)
    print(f"\nTaxable income {rand(p.taxable_income)} (age {args.age:g}): tax before rebates {rand(p.tax_before_rebates)} - rebate {rand(p.rebate)} = {rand(p.tax_payable)} payable")
    print(f"Marginal rate {p.marginal_rate:.0%} (bracket {p.bracket_number}) | effective rate {p.effective_rate:.1%} | tax-free below {rand(p.tax_threshold)}")
    full = p.ra_deduction_limit
    full_refund = ra_tax_refund(full, p.taxable_income, p.age, tax)
    print(f"RA deduction limit {rand(full)} a year (27.5% of income, max {rand(tax.ra_deduction_cap)}); the full deduction refunds {rand(full_refund)}")
    sp = res.optimal_split
    print("\nTax-optimal monthly split (year 1)")
    for k, name in (("RA", "Retirement Annuity"), ("TFSA", "Tax-Free Savings (TFSA)"), ("DISC", "Taxable Discretionary")):
        print(f"  {name:<26}{rand(sp[k]):>12}  ({sp[k] / res.monthly_budget:.0%})")
    print(f"RA refund-arbitrage threshold: {rand(res.arbitrage_annual_ra)} a year (expected RA exit tax {res.arbitrage_exit_rate:.1%}); " + ("RA refunds are reinvested in the TFSA / taxable sleeve." if res.reinvest_refund else "RA refunds are spent (not reinvested)."))
    cmp = res.comparison(real=True)
    show = pd.DataFrame(
        {
            "RA / month": cmp["RA / month"].map(rand),
            "TFSA / month": cmp["TFSA / month"].map(rand),
            "Disc / month": cmp["Discretionary / month"].map(rand),
            "median after-tax (real)": cmp["median after-tax wealth"].map(rand),
            "lifetime tax paid (real)": cmp["lifetime tax paid"].map(rand),
            "SARS refunds (real)": cmp["SARS refunds"].map(rand),
            "net lifetime tax (real)": cmp["net lifetime tax"].map(rand),
        }
    )
    show.index = [n.split(" (")[0] for n in cmp.index]
    print("\n" + show.T.to_string())
    s = res.summary()
    for label, key in (("Discretionary only", "disc"), ("RA only", "ra")):
        print(
            f"\nBlended vs {label}: terminal wealth +{rand(s[f'terminal_wealth_gain_real_vs_{key}'])} (today's rand) | "
            f"+{rand(s[f'terminal_wealth_gain_nominal_vs_{key}'])} nominal | lifetime tax saved {rand(s[f'lifetime_tax_saved_real_vs_{key}'])} (today's rand) | "
            f"beats it on {s[f'share_paths_better_vs_{key}']:.0%} of paths"
        )
    print("\nLifetime tax = income tax + CGT + RA exit tax, minus SARS refunds; shown in today's rand because rand paid decades apart are not comparable.")
    return 0


def cmd_drawdown(args: argparse.Namespace) -> int:
    assumptions = load_assumptions(args.source).with_inflation(args.inflation / 100.0)
    cfg = DrawdownConfig(
        initial_capital=args.capital,
        monthly_income_real=args.monthly_income,
        horizon_years=args.years,
        retirement_age=args.retire_age,
        n_paths=args.paths,
        seed=args.seed,
        annual_fee=args.fee / 100.0,
    )
    if args.weights:
        portfolio = parse_weights(args.weights)
    else:
        portfolio = args.portfolio
    resolve_la_portfolio(portfolio, assumptions.assets)  # fail early on a bad name
    impact = run_crash_impact(cfg, assumptions, portfolio)
    r = impact.base
    s = r.summary()
    print(f"\n=== Living Annuity drawdown | {rand(cfg.initial_capital)} at age {cfg.retirement_age} | {cfg.horizon_years} years | {s['n_paths']:,} paths (seed {cfg.seed}) ===")
    rep = check_reg28(r.weights, assets=assumptions.assets)
    print("Allocation: " + ", ".join(f"{a} {x:.1%}" for a, x in zip(assumptions.assets, r.weights) if x > 1e-9))
    note = "" if r.reg28_compliant else " (above the Regulation 28 limits, which do not apply to a Living Annuity)"
    print(f"Equity {rep.equity:.1%}, offshore {rep.offshore:.1%}, property {rep.property:.1%}{note}")
    print(f"Desired income {rand(cfg.monthly_income_real)}/month in today's rand, escalating with simulated SA CPI = {cfg.initial_withdrawal_rate:.2%} of starting capital (statutory band {cfg.min_rate:.1%} to {cfg.max_rate:.1%}).")
    if cfg.initial_withdrawal_rate > cfg.max_rate:
        print(f"  ! this exceeds the {cfg.max_rate:.1%} ceiling: the most you can draw in year 1 is {rand(cfg.max_initial_monthly_income)}/month.")
    print(f"\nPortfolio survival to age {cfg.end_age}:          {s['survival_prob']:.1%}")
    for age in (90, 95):
        if f"survival_age_{age}" in s and age != cfg.end_age:
            print(f"Portfolio survival to age {age}:          {s[f'survival_age_{age}']:.1%}")
    be = "never" if s["median_years_to_cap_breach"] != s["median_years_to_cap_breach"] else f"year {s['median_years_to_cap_breach']:.0f} (age {cfg.retirement_age + s['median_years_to_cap_breach']:.0f})"
    print(f"Median terminal bequest (today's rand): {rand(s['median_bequest_real'])}  (P10 {rand(s['p10_bequest_real'])})")
    print(f"Income hits the 17.5% ceiling in {s['cap_breach_prob']:.1%} of paths; median years-to-cap-breach among those: {be}")
    print(f"Average share of years with income below the desired level: {s['share_years_income_shortfall']:.1%}")
    print(f"\n=== Sequence-of-returns test: {impact.scenario.name} ===")
    print(impact.scenario.description)
    print(impact.formatted_table(rand).to_string())
    return 0


def cmd_charts(args: argparse.Namespace) -> int:
    from .plots import generate_all_charts

    assumptions, cfg, w = _setup(args)
    cfg = replace(cfg, n_paths=args.paths)
    paths = generate_all_charts(args.out, dpi=args.dpi, config=cfg, weights=w, assumptions=assumptions)
    for p in paths:
        print(f"wrote {p}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sa-wealth", description="South African Regulation 28 & tax-aware Monte Carlo allocator")
    p.add_argument("--version", action="version", version=f"sa-wealth {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="Monte Carlo wealth projection")
    _common(s)
    s.add_argument("--wrapper", choices=("RA", "TFSA", "DISC"), default="RA", help="tax wrapper (default RA)")
    s.add_argument("--report", metavar="PATH", help="also write the Markdown client wealth plan to PATH")
    s.set_defaults(func=cmd_simulate)

    o = sub.add_parser("optimize", help="Regulation 28 optimiser vs unconstrained portfolios")
    o.add_argument("--source", choices=("cma", "yfinance"), default="cma")
    o.add_argument("--inflation", type=float, default=5.5)
    o.add_argument("--rf", type=float, default=None, help="risk-free rate in percent (default: SA cash expected return)")
    o.add_argument("--target-return", type=float, default=None, help="also solve min-variance for this return, in percent")
    o.set_defaults(func=cmd_optimize)

    t = sub.add_parser("tax-compare", help="RA vs TFSA vs Discretionary")
    _common(t)
    t.set_defaults(func=cmd_tax)

    x = sub.add_parser("stress", help="SA macro stress scenarios")
    _common(x)
    x.add_argument("--wrapper", choices=("RA", "TFSA", "DISC"), default="RA")
    x.set_defaults(func=cmd_stress)

    ss = sub.add_parser("smart-split", help="SARS-aware split of a monthly budget across RA, TFSA and Discretionary")
    ss.add_argument("--budget", type=float, default=15_000, help="monthly budget in today's ZAR (default 15000)")
    ss.add_argument("--income", type=float, default=650_000, help="gross annual taxable income in ZAR (default 650000)")
    ss.add_argument("--marginal", type=float, default=None, help="override the marginal rate (default: derived from --income)")
    ss.add_argument("--age", type=float, default=40, help="age, selects the SARS rebates (default 40)")
    ss.add_argument("--years", type=int, default=25, help="investment horizon, 5-40 years (default 25)")
    ss.add_argument("--preset", default="moderate", help="portfolio preset slug (default moderate)")
    ss.add_argument("--weights", default=None, help='custom weights, e.g. "JSE_EQ=30,SA_BOND=25,GLB_EQ=25,SA_CASH=10,GLB_BOND=10"')
    ss.add_argument("--tax-year", choices=tuple(TAX_PRESETS), default="2026/27", help="SARS tables and statutory limits (default 2026/27)")
    ss.add_argument("--no-reinvest", action="store_true", help="spend the RA refund instead of reinvesting it")
    ss.add_argument("--source", choices=("cma", "yfinance"), default="cma")
    ss.add_argument("--inflation", type=float, default=5.5, help="mean SA CPI inflation in percent (default 5.5)")
    ss.add_argument("--paths", type=int, default=2_000, help="Monte Carlo paths (default 2000)")
    ss.add_argument("--seed", type=int, default=42)
    ss.set_defaults(func=cmd_smart_split)

    d = sub.add_parser("drawdown", help="Living Annuity drawdown: survival, bequest, 17.5%% ceiling breach, year-1 crash")
    d.add_argument("--capital", type=float, default=8_000_000, help="initial retirement capital in ZAR (default 8000000)")
    d.add_argument("--monthly-income", type=float, default=40_000, help="desired monthly income in today's ZAR (default 40000)")
    d.add_argument("--years", type=int, default=30, help="retirement horizon, 5-40 years (default 30)")
    d.add_argument("--retire-age", type=int, default=65, help="retirement age (default 65)")
    d.add_argument("--portfolio", default="moderate", help="conservative | moderate | aggressive-offshore (default moderate)")
    d.add_argument("--weights", default=None, help="custom weights; Living Annuities are exempt from Regulation 28 caps")
    d.add_argument("--fee", type=float, default=0.0, help="all-in annual fee in percent (default 0)")
    d.add_argument("--source", choices=("cma", "yfinance"), default="cma")
    d.add_argument("--inflation", type=float, default=5.5, help="mean SA CPI inflation in percent (default 5.5)")
    d.add_argument("--paths", type=int, default=5_000, help="Monte Carlo paths (default 5000)")
    d.add_argument("--seed", type=int, default=42)
    d.set_defaults(func=cmd_drawdown)

    c = sub.add_parser("charts", help="write docs/*.png (fan chart, Reg 28 frontier, tax wrappers, stress, living annuity, smart split)")
    _common(c)
    c.set_defaults(paths=5_000, years=30)
    c.add_argument("--out", default="docs", help="output directory (default docs)")
    c.add_argument("--dpi", type=int, default=300)
    c.set_defaults(func=cmd_charts)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
