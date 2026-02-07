# -------------------------------------------------------------
# Data loader for the trading agent.
# Fetches stock price data (Yahoo Finance) and preprocesses it into returns
# and normalized features. Supports train/test split by time (no future leakage).
# -------------------------------------------------------------
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Fetch stock data from Yahoo Finance
# symbol: ticker (e.g. AAPL, MSFT)
# period: "1y", "2y", "5y", etc.
# Returns DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume.
# ---------------------------------------------------------------------------
def fetch_stock_data(symbol: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)
    except Exception as e:
        print(f"Failed to fetch data for {symbol}: {e}")
        print("Check your network connection and try again.")
        raise
    if df.empty or len(df) < 30:
        raise ValueError(f"Not enough data for {symbol}. Try another symbol or period.")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    
# ---------------------------------------------------------------------------
# Fetch stock data from Yahoo Finance and preprocess. Raises on fetch failure.
# Compute returns and features from OHLCV DataFrame and return the DataFrame.
# symbol: ticker (e.g. AAPL, MSFT). 
# period: "1y", "2y", "5y", etc.
# Returns DataFrame with DatetimeIndex and columns: Return, CloseNorm.
# ---------------------------------------------------------------------------
def load_data(symbol: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    df = fetch_stock_data(symbol=symbol, period=period)
    # Compute returns and features from OHLCV DataFrame and return the DataFrame.
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["Return"] = out["Return"].fillna(0.0)
    # Normalize close to [0,1] over full history for consistent scale (we use returns mainly)
    out["CloseNorm"] = (out["Close"] - out["Close"].min()) / (out["Close"].max() - out["Close"].min() + 1e-8)
    return out.dropna()

# ---------------------------------------------------------------------------
# Time-based split: first train_ratio for training, rest for test.
# No shuffling — we must not use future data in training.
# df: DataFrame
# train_ratio: the ratio of the training set
# Returns tuple of training and test DataFrames.
# ---------------------------------------------------------------------------
def train_test_split(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * train_ratio)
    return df.iloc[:split], df.iloc[split:]

# ---------------------------------------------------------------------------
# Main - quick test (requires network)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data(symbol="AAPL", period="2y")
    print("Data shape:", df.shape)
    print(df.head())
    train, test = train_test_split(df, train_ratio=0.7)
    print("Train len:", len(train), "Test len:", len(test))
