import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# ---- Optional: SciPy for constrained optimization ----
try:
    from scipy.optimize import minimize  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


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


def annualize_cov(cov_daily: np.ndarray, periods: int = 252) -> np.ndarray:
    return cov_daily * periods


def portfolio_stats(w: np.ndarray, mu: np.ndarray, Sigma_ann: np.ndarray, rf: float) -> dict:
    port_ret = float(w @ mu)
    port_var = float(w @ Sigma_ann @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    return {"expected_return": port_ret, "volatility": port_vol, "sharpe": sharpe}


def optimize_max_sharpe(mu: np.ndarray, Sigma_ann: np.ndarray, rf: float, allow_short: bool) -> np.ndarray:
    n = len(mu)
    if not _HAVE_SCIPY:
        # Fallback: heuristic proportional to (mu - rf) / diag(Sigma)
        risk = np.diag(Sigma_ann).clip(min=1e-12)
        scores = (mu - rf) / np.sqrt(risk)
        scores = np.maximum(scores, 0) if not allow_short else scores
        if np.all(scores == 0):
            return np.ones(n) / n
        w = scores / scores.sum()
        return w

    def neg_sharpe(w):
        r = w @ mu
        v = w @ Sigma_ann @ w
        vol = np.sqrt(max(v, 1e-18))
        return - (r - rf) / vol

    w0 = np.ones(n) / n
    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000}
    )
    return (res.x if res.success else w0)


def optimize_min_variance(mu: np.ndarray, Sigma_ann: np.ndarray, allow_short: bool) -> np.ndarray:
    n = len(mu)
    if not _HAVE_SCIPY:
        # Fallback: inverse-variance weights (no short)
        iv = 1 / np.diag(Sigma_ann).clip(min=1e-12)
        if not allow_short:
            iv = np.maximum(iv, 0)
        return iv / iv.sum()

    def var_obj(w):
        return w @ Sigma_ann @ w

    w0 = np.ones(n) / n
    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000})
    return (res.x if res.success else w0)


def optimize_target_return(mu: np.ndarray, Sigma_ann: np.ndarray, target_ret: float, allow_short: bool) -> np.ndarray:
    n = len(mu)
    if not _HAVE_SCIPY:
        # Fallback: naive tilt to higher mu
        w = np.ones(n) / n
        up = np.maximum(mu, 0)
        if up.sum() > 0:
            w = up / up.sum()
        return w

    def var_obj(w):
        return w @ Sigma_ann @ w

    w0 = np.ones(n) / n
    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, mu=mu, t=target_ret: w @ mu - t},
    ]
    res = minimize(var_obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000})
    return (res.x if res.success else w0)


def print_weights(tickers, w):
    print("\nOptimal Portfolio Weights")
    print("-" * 30)
    for t, wi in zip(tickers, w):
        print(f"{t:<8s} : {wi:6.2%}")
    print(f"Sum      : {w.sum():6.2%}")


def prompt_valid_ticker(existing: set[str], max_tries: int = 5) -> str:
    """
    Prompt until the user enters a valid Yahoo ticker.
    Valid means: yfinance can download non-empty Close data.
    Also prevents duplicates.
    """
    for attempt in range(1, max_tries + 1):
        choice = input("Enter your choice of stock (Yahoo ticker): ").strip().upper()

        if not choice:
            print("Ticker cannot be empty. Try again.")
            continue

        if choice in existing:
            print(f"'{choice}' is already included. Enter a different ticker.")
            continue

        # Quick validation download (fast). Avoids wasting time later.
        try:
            tmp = yf.download(choice, period="5d", interval="1d", progress=False)
            if tmp is None or tmp.empty:
                raise ValueError("Empty download")

            close = tmp.get("Close", None)
            if close is None or close.dropna().empty:
                raise ValueError("No valid Close prices")

            return choice  # ✅ valid ticker
        except Exception:
            print(f"'{choice}' is not a valid/available ticker (attempt {attempt}/{max_tries}). Please try again.")

    raise RuntimeError("Too many invalid ticker attempts.")


# -------------------- Existing market setup --------------------
sp500 = yf.Ticker("^GSPC")
hist_mkt = sp500.history(period="5y")  # last 5 years

# Fix FutureWarning: use "YE" instead of deprecated "Y"
sp_year = hist_mkt["Close"].resample("YE").last()
annual_returns = sp_year.pct_change().dropna()
Rm = float(annual_returns.mean())

tnx = yf.Ticker("^TNX")
# NOTE: Your output looks reasonable (~4.2%). Keep your chosen scaling.
# If your TNX Close is like 4.24 for 4.24%, use /100.
# If your TNX Close is like 42.4 for 4.24%, use /1000.
# Your current output suggests /100 is working for you, so we keep /100.
Rf = float(tnx.history(period="1mo")["Close"].iloc[-1] / 100)

print("Here is how current market looks like: ")
print("Expected market return (S&P 500 Index)", Rm)
print("Risk-free Rate (U.S. 10 Years Treasury): ", Rf)

print("How many stocks are you considering?")
i = int(input().strip())

# Pre-download market close once (avoid repeated downloads inside loop)
market_close_5y = hist_mkt["Close"].dropna()

# ---- Collect per-stock artifacts for later optimization ----
tickers: list[str] = []
expected_returns_capm: dict[str, float] = {}  # annualized, CAPM
daily_returns_matrix: list[pd.Series] = []   # list of Series; will align into DataFrame

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

    # ---- Robust, aligned returns ----
    returns = build_returns_aligned(stock_close, market_close_5y)

    # If something weird happens, retry with yf.download
    if returns.empty:
        stock_d = yf.download(choice, period="5y", interval="1d", progress=False)["Close"]
        market_d = yf.download("^GSPC", period="5y", interval="1d", progress=False)["Close"]
        returns = build_returns_aligned(stock_d, market_d)

    if returns.empty:
        print(f"Not enough overlapping observations for {choice}. Skipping CAPM beta.")
        # Still try to carry daily returns of stock alone for covariance later
        single = stock_close.pct_change().dropna()
        single.name = choice
        daily_returns_matrix.append(single)
        continue

    # CAPM beta on daily data (as in your original approach)
    X = sm.add_constant(returns["market"])
    model = sm.OLS(returns["stock"], X).fit()
    beta = float(model.params.get("market", np.nan))
    print("Beta of the stock, sensitivity to market: ", beta)

    stock_return = Rf + beta * (Rm - Rf)
    expected_returns_capm[choice] = float(stock_return)
    print("The expected annual return of your stock is:", f"{stock_return:.2%}")

    # Save stock daily returns (for covariance)
    stock_daily = returns["stock"].copy()
    stock_daily.name = choice
    daily_returns_matrix.append(stock_daily)

    # ---- Actual CAGR & plot ----
    t = yf.Ticker(choice)
    hist_stock = t.history(start="2020-01-01", end="2025-12-31")
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
            print("The actual annual return (CAGR) of your stock in this window is:", f"{cagr:.2%}")

    # Plot (per stock) with 20-day MA
    if not prices.empty:
        ma20 = prices.rolling(window=20).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(prices.index, prices.values, label=f"{choice} Close")
        plt.plot(ma20.index, ma20.values, label="20-day MA")
        plt.title(f"{choice} closing price (with 20-day MA)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# -------------------- PORTFOLIO OPTIMIZER --------------------
# Build a single DataFrame of daily returns aligned across all chosen stocks
if len(daily_returns_matrix) >= 2:
    stock_returns_df = pd.concat(daily_returns_matrix, axis=1, join="inner").dropna()
elif len(daily_returns_matrix) == 1:
    stock_returns_df = pd.DataFrame(daily_returns_matrix[0]).dropna()
else:
    stock_returns_df = pd.DataFrame()

# Keep tickers that actually have returns
valid_cols = list(stock_returns_df.columns)
stock_returns_df = stock_returns_df[valid_cols] if not stock_returns_df.empty else stock_returns_df

# Expected returns vector (annual, CAPM). If some tickers missed CAPM,
# fall back to their historical mean return annualized.
mu_series = pd.Series(index=stock_returns_df.columns, dtype=float)
for col in stock_returns_df.columns:
    if col in expected_returns_capm:
        mu_series[col] = expected_returns_capm[col]
    else:
        mu_series[col] = float(stock_returns_df[col].mean() * 252.0)

if mu_series.isna().all() or stock_returns_df.empty:
    print("\n[Portfolio] Not enough data to optimize (no returns or expected returns).")
else:
    Sigma_daily = stock_returns_df.cov().values
    Sigma_ann = annualize_cov(Sigma_daily, periods=252)
    mu = mu_series.values
    used_tickers = list(mu_series.index)

    print("\nChoose optimizer mode: (1) max_sharpe  (2) min_variance  (3) target_return (Enter an integer)")
    mode_input = input().strip()
    if mode_input not in {"1", "2", "3"}:
        mode_input = "1"

    allow_short = False  # set True if you want to allow shorting

    if mode_input == "1":
        w = optimize_max_sharpe(mu, Sigma_ann, Rf, allow_short)
    elif mode_input == "2":
        w = optimize_min_variance(mu, Sigma_ann, allow_short)
    else:
        print("Enter target annual return in decimal (e.g., 0.12 for 12%):")
        try:
            target_ret = float(input().strip())
        except Exception:
            target_ret = float(np.nan)

        if not np.isfinite(target_ret):
            print("Invalid target; defaulting to max_sharpe.")
            w = optimize_max_sharpe(mu, Sigma_ann, Rf, allow_short)
        else:
            w = optimize_target_return(mu, Sigma_ann, target_ret, allow_short)

    # Normalize minor numerical drift
    w = np.array(w, dtype=float)
    if w.sum() != 0:
        w = np.maximum(w, 0) if not allow_short else w
        s = w.sum()
        if s == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
    else:
        w = np.ones_like(w) / len(w)

    print_weights(used_tickers, w)
    stats = portfolio_stats(w, mu, Sigma_ann, Rf)
    print("\nPortfolio Stats (annualized):")
    print(f"Expected Return : {stats['expected_return']:.2%}")
    print(f"Volatility      : {stats['volatility']:.2%}")
    print(f"Sharpe (Rf={Rf:.2%}) : {stats['sharpe']:.3f}")

    # Compare to equal-weighted baseline
    weq = np.ones_like(w) / len(w)
    stats_eq = portfolio_stats(weq, mu, Sigma_ann, Rf)
    print("\nEqual-Weight Baseline:")
    print(f"Expected Return : {stats_eq['expected_return']:.2%}")
    print(f"Volatility      : {stats_eq['volatility']:.2%}")
    print(f"Sharpe (Rf={Rf:.2%}) : {stats_eq['sharpe']:.3f}")
