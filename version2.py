import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
import cvxpy as cp
from pypfopt import risk_models

# -------------------- Helpers --------------------
def build_returns_aligned(stock_close: pd.Series, market_close: pd.Series) -> pd.DataFrame:
    """Align stock & market by date (tz-naive, date-only), then compute % changes."""
    s1 = stock_close.dropna().copy()
    s2 = market_close.dropna().copy()

    if getattr(s1.index, "tz", None) is not None:
        s1.index = s1.index.tz_localize(None)
    if getattr(s2.index, "tz", None) is not None:
        s2.index = s2.index.tz_localize(None)

    s1.index = s1.index.normalize()
    s2.index = s2.index.normalize()

    s1.name, s2.name = "stock", "market"
    both = pd.concat([s1, s2], axis=1, join="inner").dropna()
    return both.pct_change().dropna()


def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma_ann: np.ndarray, rf: float) -> dict:
    port_ret = float(w @ mu)
    port_var = float(w @ Sigma_ann @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    return {"expected_return": port_ret, "volatility": port_vol, "sharpe": sharpe}


def optimize_with_friction(mu: np.ndarray, Sigma_ann: np.ndarray, w0: np.ndarray, 
                           rf: float, allow_short: bool, target_ret: float = None, 
                           mode: str = "max_sharpe", tc: float = 0.005) -> np.ndarray:
    """
    Solves MVO with an L1 turnover penalty using cvxpy.
    tc: Transaction cost assumption (default 0.5% or 0.005)
    w0: Current portfolio weights
    """
    n = len(mu)
    w = cp.Variable(n)
    
    # Turnover constraint: L1 norm of weight changes
    turnover = cp.norm(w - w0, 1)
    
    # Portfolio Return & Variance
    ret = mu.T @ w
    var = cp.quad_form(w, Sigma_ann)
    
    # Constraints
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)
        
    if mode == "min_variance":
        objective = cp.Minimize(var + tc * turnover)
        
    elif mode == "target_return" and target_ret is not None:
        constraints.append(ret >= target_ret)
        objective = cp.Minimize(var + tc * turnover)
        
    else: # max_sharpe approximation via risk aversion
        # Max Sharpe is not directly convex. We approximate by maximizing 
        # risk-adjusted return with a risk aversion parameter (gamma).
        # We assume gamma = 2 for a balanced investor.
        gamma = cp.Parameter(nonneg=True, value=2.0)
        objective = cp.Maximize(ret - gamma * var - tc * turnover)

    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.SCS) # SCS is robust for these types of problems
        if w.value is None:
            raise ValueError("Solver failed.")
        return w.value
    except Exception as e:
        print(f"Optimization failed: {e}. Returning equal weights.")
        return np.ones(n) / n


def print_weights(tickers, w):
    print("\nOptimal Portfolio Weights")
    print("-" * 30)
    for t, wi in zip(tickers, w):
        print(f"{t:<8s} : {wi:6.2%}")
    print(f"Sum      : {w.sum():6.2%}")


def prompt_valid_ticker(existing: set[str], max_tries: int = 5) -> str:
    for attempt in range(1, max_tries + 1):
        choice = input("Enter your choice of stock (Yahoo ticker): ").strip().upper()

        if not choice:
            print("Ticker cannot be empty. Try again.")
            continue

        if choice in existing:
            print(f"'{choice}' is already included. Enter a different ticker.")
            continue

        try:
            tmp = yf.download(choice, period="5d", interval="1d", progress=False, auto_adjust=False)
            if tmp is None or tmp.empty:
                raise ValueError("Empty download")

            close = tmp.get("Close", None)
            if close is None or close.dropna().empty:
                raise ValueError("No valid Close prices")

            return choice  
        except Exception:
            print(f"'{choice}' is not a valid/available ticker (attempt {attempt}/{max_tries}). Please try again.")

    raise RuntimeError("Too many invalid ticker attempts.")

def get_risk_free_rate(fallback_rate=0.042):
    """
    Fetches the current 10-Year U.S. Treasury yield as the risk-free rate.
    Includes a fallback for weekends/holidays if the yfinance API fails.
    """
    try:
        tnx = yf.Ticker("^TNX")
        # Pull 1 month of data to ensure we catch the most recent trading day
        hist = tnx.history(period="1mo")
        
        if hist.empty:
            print(f"[Warning] ^TNX data is empty. Using fallback Rf = {fallback_rate:.2%}")
            return fallback_rate
            
        # Forward fill any missing weekend data, then grab the last valid close
        last_close = hist["Close"].ffill().iloc[-1]
        
        if pd.isna(last_close) or last_close <= 0:
            print(f"[Warning] Invalid ^TNX yield. Using fallback Rf = {fallback_rate:.2%}")
            return fallback_rate
            
        # ^TNX is quoted in percent (e.g., 4.20 means 4.20%). Convert to decimal.
        return float(last_close / 100.0)
        
    except Exception as e:
        print(f"[Warning] Failed to fetch risk-free rate ({e}). Using fallback Rf = {fallback_rate:.2%}")
        return fallback_rate

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    sp500 = yf.Ticker("^GSPC")
    hist_mkt = sp500.history(period="10y")

    sp_year = hist_mkt["Close"].resample("YE").last()
    annual_returns = sp_year.pct_change().dropna()
    Rm = float(annual_returns.mean())

    tnx = yf.Ticker("^TNX")
    Rf = get_risk_free_rate(fallback_rate=0.042)

    print("Here is how current market looks like: ")
    print(f"Expected market return (S&P 500 Index): {Rm:.2%}")
    print(f"Risk-free Rate (10-Year Treasury Yield): {Rf:.2%}")

    print("How many stocks are you considering?")
    try:
        i = int(input().strip())
    except ValueError:
        print("Invalid number. Exiting.")
        exit()

    market_close_5y = hist_mkt["Close"].dropna()

    tickers: list[str] = []
    expected_returns_capm: dict[str, float] = {}  
    daily_price_matrix: list[pd.Series] = []   

    for _ in range(i):
        existing = set(tickers)
        try:
            choice = prompt_valid_ticker(existing, max_tries=5)
        except RuntimeError as e:
            print(str(e))
            print("Stopping input early.")
            break

        tickers.append(choice)
        stock_close = yf.Ticker(choice).history(period="5y")["Close"].dropna()
        returns = build_returns_aligned(stock_close, market_close_5y)

        if returns.empty:
            stock_d = yf.download(choice, period="5y", interval="1d", progress=True)["Close"]
            market_d = yf.download("^GSPC", period="5y", interval="1d", progress=True)["Close"]
            returns = build_returns_aligned(stock_d, market_d)

        if returns.empty:
            print(f"Not enough observations for {choice}. Skipping.")
            continue

        X = sm.add_constant(returns["market"])
        model = sm.OLS(returns["stock"], X).fit()
        beta = float(model.params.get("market", np.nan))
        print(f"Beta of {choice}: {beta:.2f}")

        # Calculate CAPM purely for informational display, not for the optimizer
        stock_return_capm = Rf + beta * (Rm - Rf)
        print(f"Expected annual return (CAPM - Info Only): {stock_return_capm:.2%}")

        # Save price data for shrinkage covariance matrix
        stock_daily = stock_close.copy()
        stock_daily.name = choice
        daily_price_matrix.append(stock_daily)

        t = yf.Ticker(choice)
        hist_stock = t.history(start="2016-01-01", end="2026-02-20") 
        prices = hist_stock.get("Close", pd.Series(dtype=float)).dropna()

        year_end_prices = prices.resample("YE").last()
        price_dict = {d.year: float(p) for d, p in year_end_prices.items()}

        years = sorted(price_dict.keys())
        if len(years) >= 2:
            buy_price = price_dict[years[0]]
            sell_price = price_dict[years[-1]]
            n_years = max(1, years[-1] - years[0])
            if buy_price > 0:
                cagr = (sell_price / buy_price) ** (1 / n_years) - 1
                
                # OVERWRITE: Strictly use CAGR for the optimization's expected return
                expected_returns_capm[choice] = float(cagr) 
                
                print(f"Actual CAGR (Used for Optimization): {cagr:.2%}\n")
        else:
            # Fallback for recent IPOs (less than 1 year of data)
            daily_mean = returns["stock"].mean() * 252
            expected_returns_capm[choice] = float(daily_mean)
            print(f"Not enough years for CAGR. Using annualized daily mean: {daily_mean:.2%}\n")

    # -------------------- Optimization Execution --------------------
    if len(daily_price_matrix) >= 2:
        stock_prices_df = pd.concat(daily_price_matrix, axis=1, join="inner").dropna()
    else:
        print("Need at least 2 valid stocks to optimize.")
        exit()

    mu_series = pd.Series(index=stock_prices_df.columns, dtype=float)
    for col in stock_prices_df.columns:
        if col in expected_returns_capm:
            mu_series[col] = expected_returns_capm[col]

    if mu_series.isna().all() or stock_prices_df.empty:
        print("\n[Portfolio] Not enough data to optimize.")
    else:
        # === SHRINKAGE IMPLEMENTATION ===
        # Replaces raw sample covariance with Ledoit-Wolf Shrinkage
        Sigma_ann = risk_models.CovarianceShrinkage(stock_prices_df).ledoit_wolf().values
        mu = mu_series.values
        used_tickers = list(mu_series.index)
        n_assets = len(used_tickers)

        print("\nChoose optimizer mode: (1) max_sharpe  (2) min_variance  (3) target_return")
        mode_input = input().strip()
        
        allow_short = False 
        
        # Current weights (Assuming equal weight starting point for turnover calculation)
        w0_current = np.ones(n_assets) / n_assets
        transaction_cost_assump = 0.005 # 0.5% per trade, this is where you can adjust the transaction cost based on your trading policy

        if mode_input == "2":
            w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, mode="min_variance", tc=transaction_cost_assump)
        elif mode_input == "3":
            print("Enter target annual return in decimal (e.g., 0.12 for 12%):")
            try:
                target_ret = float(input().strip())
            except Exception:
                target_ret = float(np.nan)

            if not np.isfinite(target_ret):
                print("Invalid target; defaulting to max_sharpe.")
                w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, mode="max_sharpe", tc=transaction_cost_assump)
            else:
                w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, target_ret=target_ret, mode="target_return", tc=transaction_cost_assump)
        else:
             w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, mode="max_sharpe", tc=transaction_cost_assump)


        # Clean weights
        w = np.array(w, dtype=float)
        w = np.maximum(w, 0) if not allow_short else w
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            w = np.ones_like(w) / len(w)

        print_weights(used_tickers, w)
        stats = portfolio_stats(w, mu, Sigma_ann, Rf)
        
        print(f"\nOptimization Model: Convex Optimization with L1 Turnover Penalty (Assumed Cost: {transaction_cost_assump:.2%})")
        print("Covariance Estimator: Ledoit-Wolf Shrinkage")
        print("\nPortfolio Stats (annualized):")
        print(f"Expected Return : {stats['expected_return']:.2%}")
        print(f"Volatility      : {stats['volatility']:.2%}")
        print(f"Sharpe (Rf={Rf:.2%}) : {stats['sharpe']:.3f}")

        # Baseline Comparison
        stats_eq = portfolio_stats(w0_current, mu, Sigma_ann, Rf)
        print("\nEqual-Weight Baseline:")
        print(f"Expected Return : {stats_eq['expected_return']:.2%}")
        print(f"Volatility      : {stats_eq['volatility']:.2%}")
        print(f"Sharpe (Rf={Rf:.2%}) : {stats_eq['sharpe']:.3f}")
