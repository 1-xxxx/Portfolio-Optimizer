import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import risk_models, expected_returns

# -------------------- Backtest Config --------------------
BACKTEST_LOOKBACK_DAYS = 504   # ~2y of daily prices used to estimate mu/Sigma at each rebalance
BACKTEST_REBALANCE_DAYS = 63   # ~1 quarter between rebalances

# Pull each ticker's full available history rather than a fixed 5y window.
# A short pull (e.g. 5y) leaves only ~2-3y for out-of-sample testing after the
# lookback window is consumed -- too short/recent to span more than one market
# regime, and not representative of a long-horizon holding period. The joint
# window across all chosen tickers is still naturally capped by whichever
# ticker has the shortest listed history (via the inner join below).
PRICE_HISTORY_PERIOD = "max"


# -------------------- Helpers --------------------
def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma_ann: np.ndarray, rf: float) -> dict:
    port_ret = float(w @ mu)
    port_var = float(w @ Sigma_ann @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    return {"expected_return": port_ret, "volatility": port_vol, "sharpe": sharpe}


def bayes_stein_shrinkage(mu: np.ndarray, Sigma: np.ndarray, T: int) -> np.ndarray:
    """
    Shrinks the sample mean vector toward the precision-weighted grand mean
    (the expected return of the global minimum-variance portfolio), following
    Jorion (1986) "Bayes-Stein Estimation for Portfolio Analysis".

    Ledoit-Wolf shrinkage only stabilizes Sigma. MVO weights are far more
    sensitive to errors in mu than in Sigma (Best & Grauer, 1991; Chopra &
    Ziemba, 1993), so mu needs shrinkage too, or the optimizer will keep
    chasing noise in the mean vector even with a clean covariance matrix.
    """
    n = len(mu)
    ones = np.ones(n)
    Sigma_inv = np.linalg.pinv(Sigma)

    mu_min_var = float(ones @ Sigma_inv @ mu) / float(ones @ Sigma_inv @ ones)
    diff = mu - mu_min_var * ones
    quad = float(diff @ Sigma_inv @ diff)

    phi = (n + 2) / ((n + 2) + T * quad) if quad > 0 else 1.0
    phi = min(max(phi, 0.0), 1.0)

    return (1 - phi) * mu + phi * mu_min_var * ones


def optimize_with_friction(mu: np.ndarray, Sigma_ann: np.ndarray, w0: np.ndarray,
                           rf: float, allow_short: bool, target_ret: float = None,
                           mode: str = "max_sharpe", tc: float = 0.005) -> np.ndarray:
    """
    Solves MVO with an L1 turnover penalty using cvxpy.
    tc: Transaction cost assumption (default 0.5% or 0.005)
    w0: Current portfolio weights
    """
    n = len(mu)

    if mode == "max_sharpe":
        # Exact convex reformulation of the tangency (max-Sharpe) portfolio
        # for the long-only case (Cornuejols & Tutuncu change-of-variables:
        # w = y / kappa). This actually maximizes (mu'w - rf)/sqrt(w'Sigma w),
        # unlike a fixed risk-aversion MVO point, which has no guaranteed
        # relationship to the true tangency portfolio. The turnover penalty
        # isn't convex under this substitution, so it's omitted here --
        # harmless while transaction_cost_assump == 0.
        y = cp.Variable(n)
        kappa = cp.Variable(nonneg=True)
        excess = mu - rf

        constraints = [excess @ y == 1, cp.sum(y) == kappa, kappa >= 1e-6]
        if not allow_short:
            constraints.append(y >= 0)

        objective = cp.Minimize(cp.quad_form(y, Sigma_ann))
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.SCS)
            if y.value is None or kappa.value is None or kappa.value <= 1e-8:
                raise ValueError("No feasible tangency portfolio (no asset beats the risk-free rate).")
            return y.value / kappa.value
        except Exception as e:
            print(f"Max Sharpe optimization failed: {e}. Returning equal weights.")
            return np.ones(n) / n

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

    else:
        # Fallback for an unrecognized mode: a balanced risk-aversion MVO
        # point (NOT max Sharpe -- see the mode == "max_sharpe" branch above).
        gamma = cp.Parameter(nonneg=True, value=2.0)
        objective = cp.Maximize(ret - gamma * var - tc * turnover)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS)  # SCS is robust for these types of problems
        if w.value is None:
            raise ValueError("Solver failed.")
        return w.value
    except Exception as e:
        print(f"Optimization failed: {e}. Returning equal weights.")
        return np.ones(n) / n


def realized_stats(daily_returns: np.ndarray, rf: float) -> dict:
    """Annualized return/vol/Sharpe/max-drawdown computed from a REALIZED daily return series."""
    ann_return = float(np.mean(daily_returns) * 252)
    ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else np.nan

    equity = np.cumprod(1 + daily_returns)
    running_max = np.maximum.accumulate(equity)
    max_drawdown = float(((equity - running_max) / running_max).min())

    return {"annualized_return": ann_return, "annualized_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_drawdown}


def walk_forward_backtest(prices_df: pd.DataFrame, rf: float, allow_short: bool, mode: str,
                           target_ret, tc: float,
                           lookback: int = BACKTEST_LOOKBACK_DAYS,
                           rebalance: int = BACKTEST_REBALANCE_DAYS) -> dict:
    """
    Rolls forward in time: at each rebalance point, estimates mu/Sigma using
    ONLY price data up to that point, forms weights with the same optimizer
    used live, then evaluates REALIZED returns over the following holding
    period using price data that was never used to estimate that period's
    weights. This is what actually tells you whether the optimizer would
    have added value, as opposed to the in-sample "Portfolio Stats" that
    just re-evaluates the optimizer's own inputs at its own optimum.
    """
    daily_rets = prices_df.pct_change().dropna()
    n = len(daily_rets)
    n_assets = prices_df.shape[1]

    if n < lookback + rebalance:
        return None

    opt_returns, eq_returns, oos_dates = [], [], []
    w_prev = np.ones(n_assets) / n_assets
    num_rebalances = 0

    pos = lookback
    while pos + rebalance <= n:
        train_prices = prices_df.iloc[pos - lookback: pos + 1]
        test_rets = daily_rets.iloc[pos: pos + rebalance]

        mu_train = expected_returns.mean_historical_return(train_prices, frequency=252, compounding=False).values
        Sigma_train = risk_models.CovarianceShrinkage(train_prices).ledoit_wolf().values
        mu_train = bayes_stein_shrinkage(mu_train, Sigma_train, T=len(train_prices))

        w = optimize_with_friction(mu_train, Sigma_train, w_prev, rf, allow_short,
                                    target_ret=target_ret, mode=mode, tc=tc)
        w = np.maximum(w, 0) if not allow_short else w
        s = w.sum()
        w = w / s if s > 0 else np.ones(n_assets) / n_assets

        w_eq = np.ones(n_assets) / n_assets

        opt_returns.extend((test_rets.values @ w).tolist())
        eq_returns.extend((test_rets.values @ w_eq).tolist())
        oos_dates.extend(test_rets.index.tolist())

        w_prev = w
        num_rebalances += 1
        pos += rebalance

    if not opt_returns:
        return None

    return {
        "dates": oos_dates,
        "opt_returns": np.array(opt_returns),
        "eq_returns": np.array(eq_returns),
        "num_rebalances": num_rebalances,
    }


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
    Rf = get_risk_free_rate(fallback_rate=0.042)

    print("Here is how current market looks like: ")
    print(f"Risk-free Rate (10-Year Treasury Yield): {Rf:.2%}")

    print("How many stocks are you considering?")
    try:
        i = int(input().strip())
    except ValueError:
        print("Invalid number. Exiting.")
        exit()

    tickers: list[str] = []
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
        stock_close = yf.Ticker(choice).history(period=PRICE_HISTORY_PERIOD)["Close"].dropna()

        if stock_close.empty:
            stock_close = yf.download(choice, period=PRICE_HISTORY_PERIOD, interval="1d", progress=True)["Close"].dropna()

        if stock_close.empty:
            print(f"Not enough price history for {choice}. Skipping.")
            continue

        stock_close = stock_close.copy()
        stock_close.name = choice
        daily_price_matrix.append(stock_close)
        print(f"Loaded {len(stock_close)} daily prices for {choice}.\n")

    # -------------------- Optimization Execution --------------------
    if len(daily_price_matrix) >= 2:
        stock_prices_df = pd.concat(daily_price_matrix, axis=1, join="inner").dropna()
    else:
        print("Need at least 2 valid stocks to optimize.")
        exit()

    if stock_prices_df.empty or len(stock_prices_df) < 60:
        print("\n[Portfolio] Not enough overlapping price history to optimize.")
    else:
        print(f"\nUsing {len(stock_prices_df)} overlapping trading days "
              f"({stock_prices_df.index.min().date()} to {stock_prices_df.index.max().date()}) "
              f"for return & risk estimation.")

        # === RETURN & RISK ESTIMATION ===
        # mu and Sigma are estimated from the SAME price panel/window, so they
        # describe the same joint return distribution (rather than splicing a
        # long-horizon return estimate onto a short-horizon covariance matrix).
        used_tickers = list(stock_prices_df.columns)
        n_assets = len(used_tickers)

        mu_sample = expected_returns.mean_historical_return(stock_prices_df, frequency=252, compounding=False)
        Sigma_ann_df = risk_models.CovarianceShrinkage(stock_prices_df).ledoit_wolf()

        # Bayes-Stein shrinkage of mu (Jorion, 1986) -- Ledoit-Wolf shrinkage
        # above only stabilizes Sigma; mu is the dominant source of MVO
        # instability and needs shrinkage of its own.
        mu = bayes_stein_shrinkage(mu_sample.values, Sigma_ann_df.values, T=len(stock_prices_df))
        Sigma_ann = Sigma_ann_df.values

        print("\nExpected Returns (Bayes-Stein shrunk, annualized):")
        for t, m in zip(used_tickers, mu):
            print(f"{t:<8s} : {m:6.2%}")

        print("\nChoose optimizer mode: (1) max_sharpe  (2) min_variance  (3) target_return")
        mode_input = input().strip()

        allow_short = False
        w0_current = np.ones(n_assets) / n_assets
        transaction_cost_assump = 0.0 # There is 0% transaction cost for my current broakerage account, but you can adjust this if needed.

        target_ret = None
        if mode_input == "2":
            mode = "min_variance"
            w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, mode=mode, tc=transaction_cost_assump)
        elif mode_input == "3":
            mode = "target_return"
            print("Enter target annual return in decimal (e.g., 0.12 for 12%):")
            try:
                target_ret = float(input().strip())
            except Exception:
                target_ret = float(np.nan)

            if not np.isfinite(target_ret):
                print("Invalid target; defaulting to max_sharpe.")
                mode = "max_sharpe"
                w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, mode=mode, tc=transaction_cost_assump)
            else:
                w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, target_ret=target_ret, mode=mode, tc=transaction_cost_assump)
        else:
            mode = "max_sharpe"
            w = optimize_with_friction(mu, Sigma_ann, w0_current, Rf, allow_short, mode=mode, tc=transaction_cost_assump)


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
        print("Return Estimator: Bayes-Stein Shrinkage (Jorion, 1986)")
        print("\nModel-Implied Stats (In-Sample -- reflects the estimator's own inputs, NOT a validated forecast):")
        print(f"Expected Return : {stats['expected_return']:.2%}")
        print(f"Volatility      : {stats['volatility']:.2%}")
        print(f"Sharpe (Rf={Rf:.2%}) : {stats['sharpe']:.3f}")

        # Baseline Comparison (also in-sample -- see the walk-forward backtest
        # below for genuine out-of-sample evidence)
        stats_eq = portfolio_stats(w0_current, mu, Sigma_ann, Rf)
        print("\nEqual-Weight Baseline (In-Sample):")
        print(f"Expected Return : {stats_eq['expected_return']:.2%}")
        print(f"Volatility      : {stats_eq['volatility']:.2%}")
        print(f"Sharpe (Rf={Rf:.2%}) : {stats_eq['sharpe']:.3f}")

        # -------------------- Walk-Forward Backtest (Out-of-Sample) --------------------
        print("\n" + "=" * 50)
        print("WALK-FORWARD BACKTEST (Out-of-Sample)")
        print("=" * 50)

        bt = walk_forward_backtest(stock_prices_df, Rf, allow_short, mode, target_ret, transaction_cost_assump)

        if bt is None:
            print(f"Not enough history for a walk-forward test "
                  f"(need at least {BACKTEST_LOOKBACK_DAYS + BACKTEST_REBALANCE_DAYS} trading days, "
                  f"have {len(stock_prices_df)}).")
        else:
            opt_stats = realized_stats(bt["opt_returns"], Rf)
            eq_stats = realized_stats(bt["eq_returns"], Rf)

            print(f"Lookback window : {BACKTEST_LOOKBACK_DAYS} trading days (~{BACKTEST_LOOKBACK_DAYS / 252:.1f}y)")
            print(f"Rebalanced every: {BACKTEST_REBALANCE_DAYS} trading days (~quarterly)")
            print(f"Out-of-sample   : {bt['dates'][0].date()} to {bt['dates'][-1].date()} "
                  f"({len(bt['dates'])} trading days, {bt['num_rebalances']} rebalances)")

            print("\nStrategy (Realized):")
            print(f"Annualized Return : {opt_stats['annualized_return']:.2%}")
            print(f"Annualized Vol    : {opt_stats['annualized_vol']:.2%}")
            print(f"Sharpe (Rf={Rf:.2%})  : {opt_stats['sharpe']:.3f}")
            print(f"Max Drawdown      : {opt_stats['max_drawdown']:.2%}")

            print("\nEqual-Weight Baseline (Realized):")
            print(f"Annualized Return : {eq_stats['annualized_return']:.2%}")
            print(f"Annualized Vol    : {eq_stats['annualized_vol']:.2%}")
            print(f"Sharpe (Rf={Rf:.2%})  : {eq_stats['sharpe']:.3f}")
            print(f"Max Drawdown      : {eq_stats['max_drawdown']:.2%}")
