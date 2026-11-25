# sa-wealth-reg28-simulator

**South African Regulation 28 Private Wealth & Tax-Aware Monte Carlo Allocator**

https://sa-wealth-reg28-simulator-t9gykeq27yzyjmejcrfgwj.streamlit.app/

[![CI](https://github.com/carterhorrmann-beep/sa-wealth-reg28-simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/carterhorrmann-beep/sa-wealth-reg28-simulator/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A quantitative wealth-management toolkit for a South African investor. It answers four questions:

1. **What does Regulation 28 actually cost me?** Constrained (SLSQP) vs unconstrained portfolios, efficient frontiers and a cap-by-cap breakdown.
2. **Will I reach my retirement goal?** A 5,000-10,000 path, seeded, Cholesky-correlated Monte Carlo of six ZAR asset classes with stochastic SA CPI, reported in nominal and today's rand.
3. **Which wrapper should hold the money?** Retirement Annuity vs Tax-Free Savings Account vs taxable Discretionary account, run on identical market paths.
4. **What if South Africa has a bad decade?** Stagflation, a global risk-off with a Rand spike, and a sequence-of-returns crash just before retirement.

It ships as a Python package, a `sa-wealth` CLI, high-DPI charts and a 4-tab Streamlit dashboard that exports a client wealth plan as Markdown.

> **Disclaimer:** an educational modelling project, not financial advice. Capital market assumptions are long-run illustrations, not forecasts, and tax law changes every Budget.

## Gallery

| Monte Carlo fan chart | Regulation 28 frontier & offshore cap |
|---|---|
| ![fan chart](docs/fan_chart.png) | ![frontier](docs/reg28_frontier.png) |

| Tax wrappers | Macro stress scenarios |
|---|---|
| ![tax wrappers](docs/tax_wrappers.png) | ![stress](docs/stress_scenarios.png) |

*(Default plan: 30-year horizon, R250k lump sum, R5,000/month escalating with CPI, R5m goal in today's rand, "Moderate (Reg 28)" portfolio, 36% marginal rate, 5,000 paths, seed 42.)*

## Quick start

```bash
git clone https://github.com/carterhorrmann-beep/sa-wealth-reg28-simulator.git
cd sa-wealth-reg28-simulator

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[app,dev]"          # add ,yf for the optional yfinance loader

pytest -q                            # offline test-suite
sa-wealth simulate                   # Monte Carlo with the default plan
streamlit run app/streamlit_app.py   # interactive dashboard
```

### Deploy on Streamlit Community Cloud

1. Push the repository to `github.com/carterhorrmann-beep/sa-wealth-reg28-simulator`.
2. On [share.streamlit.io](https://share.streamlit.io) choose **New app**, pick the repository and set the main file path to `app/streamlit_app.py`.
3. `requirements.txt` contains `scipy>=1.11` and `-e .[app]`, so the package and Streamlit install without further configuration.

## CLI

```text
sa-wealth simulate     Monte Carlo wealth projection (+ --report plan.md for the client plan)
sa-wealth optimize     Regulation 28 optimiser vs unconstrained, cap-by-cap cost, offshore sensitivity
sa-wealth tax-compare  RA vs TFSA vs Discretionary on identical paths
sa-wealth stress       SA macro stress scenarios
sa-wealth charts       writes docs/fan_chart.png, reg28_frontier.png, tax_wrappers.png, stress_scenarios.png
```

```bash
sa-wealth simulate --initial 500000 --monthly 8000 --years 28 --target 9000000 \
                   --marginal 0.41 --preset aggressive --paths 10000 --wrapper RA --report plan.md
sa-wealth simulate --weights "JSE_EQ=20,GLB_EQ=70,SA_BOND=10" --wrapper TFSA
sa-wealth optimize --target-return 11
sa-wealth tax-compare --tax-year 2026/27
sa-wealth charts --out docs --dpi 300
```

Presets: `conservative`, `moderate`, `aggressive`, `offshore-growth`, `jse-home-bias`, `max-sharpe`, `risk-parity`.
`--source yfinance` re-estimates the assumptions from market data (see below).

## Python API

```python
from sa_wealth import SimulationConfig, simulate, run_tax_comparison, run_stress, default_assumptions

a = default_assumptions()
cfg = SimulationConfig(initial_lump_sum=250_000, monthly_contribution=5_000,
                       horizon_years=30, target_real=5_000_000, n_paths=10_000, marginal_rate=0.36)
w = {"JSE_EQ": .30, "SA_BOND": .25, "SA_PROP": .05, "SA_CASH": .05, "GLB_EQ": .25, "GLB_BOND": .10}

res = simulate(cfg, w, a)
res.prob_goal, res.cvar5_real, res.cumulative_tax_drag
res.percentiles(real=True)            # P10 / P25 / P50 / P75 / P90 by year
run_tax_comparison(cfg, w, a).summary
run_stress(cfg, w, a).table
```

## Repository layout

```text
src/sa_wealth/
  assets.py      6 ZAR asset classes, CMA table, correlation matrix (incl. CPI & Rand-strength factors),
                 stochastic CPI, ZAc/ZAR sanitiser and optional yfinance loader
  reg28_tax.py   Regulation 28 validator + SLSQP optimiser, frontiers, trade-off analytics,
                 RA / TFSA / Discretionary tax engine
  montecarlo.py  Cholesky Monte Carlo, metrics (P(goal), CVaR 5%, tax drag), tax comparison, stress tests
  plots.py       high-DPI matplotlib charts
  report.py      Markdown client wealth plan
  cli.py         the `sa-wealth` command
app/streamlit_app.py   4-tab dashboard
tests/                 test_assets / test_reg28 / test_tax / test_montecarlo / test_cli (offline)
docs/                  generated charts
```

## Methodology

### Assets and returns (`assets.py`)

| Code | Asset | E[return] | Vol | Income yield | Taxed as |
|---|---|---|---|---|---|
| `JSE_EQ` | JSE All Share / Top 40 | 12.0% | 17.0% | 3.0% | local dividend (20% DWT) |
| `SA_BOND` | ALBI / SAGBs | 9.5% | 8.0% | 8.5% | interest (marginal rate, after exemption) |
| `SA_PROP` | SA listed property | 10.0% | 20.0% | 7.5% | property distribution (marginal rate) |
| `SA_CASH` | STeFI / money market | 7.0% | 0.7% | 7.0% | interest |
| `GLB_EQ` | Global developed equity in ZAR | 11.5% | 17.0% | 1.6% | foreign dividend (effective 20%) |
| `GLB_BOND` | Global aggregate bonds in ZAR | 8.0% | 12.0% | 4.0% | foreign interest (marginal rate) |

* Monthly **log** returns with the drift set so the annual *arithmetic* mean equals the assumption, correlated via the Cholesky factor of an 8x8 correlation matrix: six assets, an SA **CPI shock** and a **ZAR-strength** factor (negative of USD/ZAR). A stronger Rand lowers the ZAR value of global assets, so `GLB_EQ` and `GLB_BOND` are negatively correlated with ZAR strength (-0.55 / -0.85) while SA bonds and property are positively correlated. The matrix is validated as positive definite in the tests.
* **SA CPI**: mean 5.5% (adjustable). The annualised inflation rate follows an AR(1) deviation (persistence 0.97/month, stationary volatility 1.5%) so shocks are persistent and correlated with asset returns. Contributions escalate with the *realised* CPI path and every result can be shown nominal or in today's rand.
* **`--source yfinance`** uses the liquid proxies `STX40.JO`, `NFGOVI.JO`, `STXPRO.JO`, `USDZAR=X`, `SPY`, `AGG`. `SPY` and `AGG` are converted to ZAR via USD/ZAR. JSE ETFs are quoted in cents (ZAc), and vendors occasionally flip the unit mid-history, so every series goes through `sanitise_zac_series`, which detects 100x / 0.01x day-on-day jumps and rescales the earlier segment (real crashes are untouched). Sample statistics are blended 50/50 with the CMAs, because a short ETF history is a noisy estimate of long-run returns. `SA_CASH` and the CPI factor keep their CMA values.

### Regulation 28 (`reg28_tax.py`)

Limits: equity (`JSE_EQ + GLB_EQ`) <= 75%, offshore (`GLB_EQ + GLB_BOND`) <= 45%, listed property <= 25%, long-only, fully invested. `optimize_portfolio` uses `scipy.optimize.minimize(method="SLSQP")` with multi-start for max-Sharpe, min-variance (with an optional return target) and equal-risk-contribution (risk parity); `max_return` uses an exact LP. `project_to_reg28` returns the closest compliant allocation, which is what the RA holds when a client's preferred portfolio breaches the limits.

### What Regulation 28 costs: findings at the base assumptions

* The unconstrained max-Sharpe portfolio (about 60% SA bonds, 30% offshore, 26% equity) is **already compliant**, so at the Sharpe optimum the limits cost nothing.
* The limits bind on the **growth end** of the frontier: the highest reachable expected return is **11.5%** under Regulation 28 vs **12.0%** unconstrained, and a Reg 28 investor targeting about 11.3% needs roughly 0.3 percentage points more volatility.
* The **45% offshore cap is slack** at the base assumptions. `offshore_sensitivity` adds a "weak-Rand" premium to global assets: the cap starts to bind at roughly +3 percentage points, and even at +5 points the Sharpe given up is about 0.0005, because other assets substitute for the extra offshore weight.

These are properties of the chosen CMAs. Change `CMA_TABLE` and the picture changes, which is the point of the tool.

### Tax wrappers (`reg28_tax.py`)

| | RA | TFSA | Discretionary |
|---|---|---|---|
| Allocation | Regulation 28 | unconstrained | unconstrained |
| Tax on dividends / interest / CGT | 0% | 0% | 20% DWT; interest above the R23,800 exemption at the marginal rate; CGT at 40% x marginal rate (18% at 45%) after a R40,000 annual exclusion |
| Contributions | upfront deduction: 27.5% of income, max R350,000 a year, refund reinvested | R36,000 a year, R500,000 lifetime | unlimited |
| At the end | 1/3 lump sum on the retirement table (first R550,000 free), 2/3 annuity at an assumed retirement marginal rate (min(your rate, 31%)) | none | CGT on all unrealised gains |

Implementation notes:

* All wrappers run on **identical market paths** (common random numbers), so differences are purely tax.
* Income is paid monthly, taxed by character, and reinvested. In the Discretionary account gains are realised when the portfolio rebalances back to target each year and at liquidation, with an annual exclusion and capital-loss carry-forward.
* **TFSA overflow:** contributions above the annual / lifetime limit cannot enter the TFSA (excess contributions are penalised at 40%), so the overflow is invested in a taxable account with the same allocation. The TFSA result is the *strategy*, not the TFSA alone.
* "Rands saved in tax" is the extra **after-tax terminal wealth** versus the Discretionary account, which includes the compounding effect of tax and is therefore larger than the difference in cash taxes paid. For the RA part of the gap is the refund reinvested and, if your preferred allocation breaches Regulation 28, a different allocation.
* **Tax year:** the default parameters follow the original project specification (**2025/26**: TFSA R36,000, RA cap R350,000, CGT exclusion R40,000). Budget 2026 raised these to R46,000 / R430,000 / R50,000 for **2026/27** (lifetime TFSA R500,000, interest exemption R23,800 and DWT 20% unchanged). Use `--tax-year 2026/27` or the dashboard's model settings. All limits are held nominally fixed over the horizon, which is conservative. Verify against SARS before relying on any number.
* Income-tax brackets are not modelled: you supply the marginal rate. When no income is given, a representative mid-bracket income is used to size the RA deduction.

### Monte Carlo and stress tests (`montecarlo.py`)

Outputs: percentile trajectories (P10/P25/P50/P75/P90), **Probability of Reaching Goal**, **shortfall risk** (CVaR 5% of terminal wealth = mean of the worst 5% of paths, plus VaR and the average shortfall versus the goal) and **cumulative tax drag** (mean rands of tax paid including RA exit tax; RA refunds are reported separately). Wealth is after tax: the value you would keep if you liquidated (RA: retired) in that year.

Stress scenarios are deterministic overlays on the *same* random numbers as the base run:

| Scenario | Design |
|---|---|
| SA Stagflation & Fiscal Slippage | CPI 8.5% for 7 years; JSE -6% p.a., ALBI -3.5% p.a., property -6% p.a., cash +1.5% p.a., global equity +4% / global bonds +5% p.a. in ZAR (Rand depreciation); SA vol +30% |
| Global Risk-Off & Rand Spike | 10 years before retirement: global equity -30% in USD with USD/ZAR +18% (only -17.4% in ZAR), global bonds +20.4% in ZAR, JSE -22%, property -25%, ALBI -8%; vol +50-80%; CPI 6.5% |
| Sequence-of-Returns Crash | -25% shock to JSE and global equity (property -20%) in the 12 months beginning 24 months before retirement |

Default-plan results (30 years, RA, Moderate portfolio, 5,000 paths): median R4.19m in today's rand and P(goal R5m) = 32%; the stress scenarios cut the median by 6.5%, 10.3% and 13.9% and P(goal) to 26%, 23% and 20%.

## Testing

```bash
pytest -q
```

111 offline tests cover: constraint enforcement (<= 75% equity, <= 45% offshore, <= 25% property, long-only, sum to 1) across every optimiser objective; TFSA R36k/yr and R500k lifetime caps (unit-level and end-to-end); CGT, DWT, interest-exemption and RA-deduction arithmetic against hand calculations; TFSA > Discretionary for identical pre-tax returns (deterministically and on every simulated path); Monte Carlo moments, correlation and reproducibility; P10 < P25 < P50 < P75 < P90; a hand-written recursion that validates the wealth engine; stress-scenario windows and direction; ZAc/ZAR sanitising; and the CLI.

## Limitations

* Lognormal returns: no fat tails, regime switching or mean reversion beyond the stress overlays.
* Accumulation phase only: no drawdown / living-annuity modelling. The RA exit uses a flat retirement tax rate.
* Statutory limits are nominal and fixed; income-tax brackets, medical credits, estate duty, trusts and the extra 10% African allowance within the offshore limit are not modelled.
* Single investor, annual rebalance, no fees or platform costs.

## License

MIT, Copyright (c) 2026 Carter Horrmann. See [LICENSE](LICENSE).
