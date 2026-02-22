# Portfolio Optimizer (Institutional-Grade)

This Portfolio Optimizer provides robust asset allocation suggestions for multi-asset investments. While standard textbook models often fail in live markets, this engine bridges the gap between theory and production by implementing **Covariance Shrinkage** and **Turnover Constraints**, resulting in highly stable, friction-aware portfolio weights. 

## Core Quantitative Upgrades

Standard Markowitz Mean-Variance Optimization (MVO) is highly sensitive to estimation errors, often resulting in extreme weights and poor out-of-sample performance. This project solves these theoretical flaws through two advanced mechanisms:

* **Ledoit-Wolf Covariance Shrinkage:** Replaces the noisy sample covariance matrix with a mathematically shrunk estimator, drastically reducing statistical noise and preventing the optimizer from chasing "ghost" correlations.
* **Friction-Aware Convex Optimization ($L_1$ Penalty):** Simulates real-world trading by introducing transaction cost penalties (e.g., bid-ask spread, commissions). Using an $L_1$ norm penalty on weight changes, it forces sparsity—preventing the algorithm from making marginal, unprofitable trades just to achieve a theoretically higher Sharpe ratio.

## Key Features

* **Advanced Optimization Modes:** Choose between Max Sharpe (Risk-Adjusted), Minimum Volatility, or a specific Target Return.
* **Dynamic Risk-Free Rate Fetching:** Automatically pulls the live U.S. 10-Year Treasury yield (`^TNX`) to serve as the baseline risk-free rate, with a robust fallback for weekends and API outages.
* **Data Visualization:** Generates visual plots of your selected stocks' 5-year closing prices overlaid with a 20-day Moving Average (MA).

## Dependencies

* `yfinance` - Market data ingestion
* `pandas` & `numpy` - Data manipulation and numerical operations
* `statsmodels` - OLS regression for CAPM Beta calculation
* `matplotlib` - Financial plotting
* `scipy` - Standard mathematical functions
* `PyPortfolioOpt` - Handles Ledoit-Wolf Covariance Shrinkage
* `cvxpy` - Solves the convex optimization equations with $L_1$ turnover penalties

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
python version2.py
```

## Demo
As an example, we are creating a permanent portfolio including U.S. Equities (`SPY`), Chinese Equities (`FXI`), Long/Short Treasuries (`TLT`, `SHY`), Precious Metals (`GLD`, `SLV`), Managed Futures (`WTMF`), and Cash Equivalents (`BIL`)

### Max Sharpe Optimization Output

```text
Optimal Portfolio Weights
------------------------------
SPY      : 40.77%
FXI      :  0.00%
TLT      :  0.00%
SHY      :  0.00%
GLD      : 59.23%
SLV      :  0.00%
WTMF     :  0.00%
BIL      :  0.00%
Sum      : 100.00%

Optimization Model: Convex Optimization with L1 Turnover Penalty (Assumed Cost: 0.50%)
Covariance Estimator: Ledoit-Wolf Shrinkage

Portfolio Stats (annualized):
Expected Return : 14.62%
Volatility      : 13.01%
Sharpe (Rf=4.09%) : 0.810

Equal-Weight Baseline:
Expected Return : 6.90%
Volatility      : 9.98%
Sharpe (Rf=4.09%) : 0.282
```

### Min Variance Optimization Output
```text
Optimal Portfolio Weights
------------------------------
SPY      : 12.50%
FXI      :  0.48%
TLT      : 12.50%
SHY      : 12.50%
GLD      : 12.46%
SLV      :  0.00%
WTMF     : 12.50%
BIL      : 37.06%
Sum      : 100.00%

Optimization Model: Convex Optimization with L1 Turnover Penalty (Assumed Cost: 0.50%)
Covariance Estimator: Ledoit-Wolf Shrinkage

Portfolio Stats (annualized):
Expected Return : 4.92%
Volatility      : 4.92%
Sharpe (Rf=4.09%) : 0.170

Equal-Weight Baseline:
Expected Return : 6.90%
Volatility      : 9.98%
Sharpe (Rf=4.09%) : 0.282
```
### Target Return (10% Annually) Optimization Output
```text
Optimal Portfolio Weights
------------------------------
SPY      : 27.21%
FXI      :  0.00%
TLT      :  0.00%
SHY      : 12.50%
GLD      : 35.76%
SLV      :  0.00%
WTMF     : 12.03%
BIL      : 12.50%
Sum      : 100.00%

Optimization Model: Convex Optimization with L1 Turnover Penalty (Assumed Cost: 0.50%)
Covariance Estimator: Ledoit-Wolf Shrinkage

Portfolio Stats (annualized):
Expected Return : 10.00%
Volatility      : 8.71%
Sharpe (Rf=4.09%) : 0.679

Equal-Weight Baseline:
Expected Return : 6.90%
Volatility      : 9.98%
Sharpe (Rf=4.09%) : 0.282
```

