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
