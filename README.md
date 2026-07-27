# Portfolio Optimizer

This Portfolio Optimizer provides robust asset allocation suggestions for multi-asset investments. Standard textbook models often fail in live markets because mean-variance optimization (MVO) is highly sensitive to estimation error in *both* its inputs. This engine addresses that on both sides — **Ledoit-Wolf Covariance Shrinkage** for the risk side and **Bayes-Stein Return Shrinkage** for the return side — and validates the resulting weights with a **walk-forward, out-of-sample backtest** rather than an in-sample calculation alone.

## Core Quantitative Upgrades

Standard MVO is highly sensitive to estimation errors, often resulting in extreme weights and poor out-of-sample performance. This project addresses that on every input that feeds the optimizer, and separately validates the output:

* **Ledoit-Wolf Covariance Shrinkage:** Replaces the noisy sample covariance matrix with a mathematically shrunk estimator, reducing statistical noise in the risk (Σ) side of the optimization.
* **Bayes-Stein Return Shrinkage:** Mean-variance weights are typically far *more* sensitive to errors in expected returns than to errors in the covariance matrix (Best & Grauer, 1991; Chopra & Ziemba, 1993) — shrinking Σ alone leaves the dominant source of instability untouched. Following Jorion (1986), each asset's historical mean return is shrunk toward the precision-weighted grand mean (the expected return of the global minimum-variance portfolio), computed over the **same** price window used for the covariance matrix so the two inputs describe the same period.
* **Exact Max-Sharpe (Tangency Portfolio) Optimization:** The "Max Sharpe" mode solves the exact convex reformulation of the tangency portfolio (Cornuejols & Tütüncü), incorporating the live risk-free rate directly — not an approximation at a fixed, arbitrary risk-aversion level.
* **Friction-Aware Convex Optimization ($L_1$ Penalty):** Every objective includes an $L_1$ norm penalty on weight changes from a starting portfolio, so a nonzero transaction-cost assumption forces sparsity and discourages marginal, unprofitable rebalancing. (The shipped default assumes 0% cost — see `transaction_cost_assump` in `version3.py` to enable it.)
* **Walk-Forward Out-of-Sample Backtest:** The headline "Portfolio Stats" comparison against an equal-weight baseline is an **in-sample** calculation — it evaluates the optimizer's own inputs at its own optimum, so it is expected to look favorable by construction and is not evidence of forward performance. To actually validate the weighting scheme, the tool separately re-estimates Σ and shrunk μ using only data available at each point in history, forms weights with those estimates, and scores the **realized** returns over the following holding period — repeated on a rolling basis across the full available history — before ever touching the data used to score it.

## Key Features

* **Advanced Optimization Modes:** Choose between Max Sharpe (exact tangency portfolio), Minimum Volatility, or a specific Target Return.
* **Walk-Forward Backtesting:** Rolling out-of-sample validation (quarterly rebalance, 2-year rolling estimation window) reports realized annualized return, volatility, Sharpe, and max drawdown — not just an in-sample estimate.
* **Full-History Data Pull:** Pulls each ticker's full available price history (rather than a fixed short window) so the backtest can span multiple market regimes instead of just the last few years.
* **Dynamic Risk-Free Rate Fetching:** Automatically pulls the live U.S. 10-Year Treasury yield (`^TNX`) to serve as the baseline risk-free rate, with a robust fallback for weekends and API outages.
* **Data Visualization:** Generates visual plots of your selected stocks' closing prices overlaid with a moving average (MA).

## Dependencies

* `yfinance` - Market data ingestion
* `pandas` & `numpy` - Data manipulation and numerical operations
* `matplotlib` - Financial plotting
* `scipy` - Standard mathematical functions
* `PyPortfolioOpt` - Historical mean-return estimation and Ledoit-Wolf covariance shrinkage
* `cvxpy` - Solves the convex optimization equations (tangency portfolio, min-variance, target-return, all with $L_1$ turnover penalties)

## Get Started

**1. Clone the Repository**
```bash
git clone https://github.com/1-xxxx/Portfolio-Optimizer.git
cd Portfolio-Optimizer
```

**2. Install Requirements**
- Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```
**3. Run the Engine**
```bash
python version3.py
```

## Demo
As an example, we are creating a permanent portfolio including U.S. Equities (`SPY`), Chinese Equities (`FXI`), Long/Short Treasuries (`TLT`, `SHY`), Precious Metals (`GLD`, `SLV`), Managed Futures (`WTMF`), and Cash Equivalents (`BIL`). Output below is real, live output from `version3.py` (run 2026-07-27) — not illustrative placeholder numbers. "Model-Implied Stats" are in-sample (they reflect the estimator's own inputs, not a validated forecast); the "Walk-Forward Backtest" section is the genuine out-of-sample evidence.

```text
Using 3911 overlapping trading days (2011-01-05 to 2026-07-27) for return & risk estimation.

Expected Returns (Bayes-Stein shrunk, annualized):
SPY      : 14.50%
FXI      :  4.58%
TLT      :  3.37%
SHY      :  1.33%
GLD      :  7.98%
SLV      :  9.11%
WTMF     :  1.34%
BIL      :  1.40%
```

### Max Sharpe Optimization Output

```text
Optimal Portfolio Weights
------------------------------
SPY      : 74.35%
FXI      :  0.00%
TLT      :  4.90%
SHY      :  0.00%
GLD      : 20.76%
SLV      :  0.00%
WTMF     :  0.00%
BIL      :  0.00%
Sum      : 100.00%

Optimization Model: Convex Optimization with L1 Turnover Penalty (Assumed Cost: 0.00%)
Covariance Estimator: Ledoit-Wolf Shrinkage
Return Estimator: Bayes-Stein Shrinkage (Jorion, 1986)

Model-Implied Stats (In-Sample -- reflects the estimator's own inputs, NOT a validated forecast):
Expected Return : 12.60%
Volatility      : 13.23%
Sharpe (Rf=4.65%) : 0.601

Equal-Weight Baseline (In-Sample):
Expected Return : 5.45%
Volatility      : 8.70%
Sharpe (Rf=4.65%) : 0.092

==================================================
WALK-FORWARD BACKTEST (Out-of-Sample)
==================================================
Lookback window : 504 trading days (~2.0y)
Rebalanced every: 63 trading days (~quarterly)
Out-of-sample   : 2013-01-09 to 2026-07-21 (3402 trading days, 54 rebalances)

Strategy (Realized):
Annualized Return : 7.29%
Annualized Vol    : 12.79%
Sharpe (Rf=4.65%)  : 0.206
Max Drawdown      : -32.53%

Equal-Weight Baseline (Realized):
Annualized Return : 5.53%
Annualized Vol    : 8.66%
Sharpe (Rf=4.65%)  : 0.101
Max Drawdown      : -19.10%
```

Note the realized out-of-sample Sharpe (0.206) beats equal-weight (0.101), but with a materially larger realized drawdown (-32.53% vs -19.10%) — a genuine, mixed result from held-out data, not a guaranteed win on every metric the way the in-sample comparison above it is.

### Min Variance Optimization Output
```text
Optimal Portfolio Weights
------------------------------
SPY      :  0.64%
FXI      :  0.00%
TLT      :  0.00%
SHY      : 35.37%
GLD      :  0.00%
SLV      :  0.00%
WTMF     :  2.06%
BIL      : 61.94%
Sum      : 100.00%

Optimization Model: Convex Optimization with L1 Turnover Penalty (Assumed Cost: 0.00%)
Covariance Estimator: Ledoit-Wolf Shrinkage
Return Estimator: Bayes-Stein Shrinkage (Jorion, 1986)

Model-Implied Stats (In-Sample -- reflects the estimator's own inputs, NOT a validated forecast):
Expected Return : 1.46%
Volatility      : 1.20%
Sharpe (Rf=4.65%) : -2.652

Equal-Weight Baseline (In-Sample):
Expected Return : 5.45%
Volatility      : 8.70%
Sharpe (Rf=4.65%) : 0.092

==================================================
WALK-FORWARD BACKTEST (Out-of-Sample)
==================================================
Lookback window : 504 trading days (~2.0y)
Rebalanced every: 63 trading days (~quarterly)
Out-of-sample   : 2013-01-09 to 2026-07-21 (3402 trading days, 54 rebalances)

Strategy (Realized):
Annualized Return : 1.94%
Annualized Vol    : 1.05%
Sharpe (Rf=4.65%)  : -2.576
Max Drawdown      : -4.06%

Equal-Weight Baseline (Realized):
Annualized Return : 5.53%
Annualized Vol    : 8.66%
Sharpe (Rf=4.65%)  : 0.101
Max Drawdown      : -19.10%
```

Min-variance mode is mathematically guaranteed to show lower in-sample volatility than equal-weight (equal-weight is always a feasible point of the same convex program the optimizer is minimizing over), which is why the in-sample comparison isn't itself evidence of anything — but the realized out-of-sample volatility (1.05%) genuinely was far lower than equal-weight's (8.66%) too, at the cost of a much lower realized return and a deeply negative realized Sharpe once the live risk-free rate is subtracted.

### Target Return (10% Annually) Optimization Output
```text
Optimal Portfolio Weights
------------------------------
SPY      : 53.13%
FXI      :  0.00%
TLT      : 21.51%
SHY      :  0.00%
GLD      : 18.51%
SLV      :  0.00%
WTMF     :  0.00%
BIL      :  6.85%
Sum      : 100.00%

Optimization Model: Convex Optimization with L1 Turnover Penalty (Assumed Cost: 0.00%)
Covariance Estimator: Ledoit-Wolf Shrinkage
Return Estimator: Bayes-Stein Shrinkage (Jorion, 1986)

Model-Implied Stats (In-Sample -- reflects the estimator's own inputs, NOT a validated forecast):
Expected Return : 10.00%
Volatility      : 9.70%
Sharpe (Rf=4.65%) : 0.551

Equal-Weight Baseline (In-Sample):
Expected Return : 5.45%
Volatility      : 8.70%
Sharpe (Rf=4.65%) : 0.092

==================================================
WALK-FORWARD BACKTEST (Out-of-Sample)
==================================================
Lookback window : 504 trading days (~2.0y)
Rebalanced every: 63 trading days (~quarterly)
Out-of-sample   : 2013-01-09 to 2026-07-21 (3402 trading days, 54 rebalances)

Strategy (Realized):
Annualized Return : 6.68%
Annualized Vol    : 8.05%
Sharpe (Rf=4.65%)  : 0.252
Max Drawdown      : -16.32%

Equal-Weight Baseline (Realized):
Annualized Return : 5.53%
Annualized Vol    : 8.66%
Sharpe (Rf=4.65%)  : 0.101
Max Drawdown      : -19.10%
```

In several early rolling windows the 10% target was infeasible for that window's estimated frontier (not every 2-year slice of this history could reach 10% long-only); the optimizer detects this and falls back to equal weights for that rebalance rather than returning a nonsensical result, which is why the realized numbers above reflect a mix of target-seeking and equal-weight periods.
