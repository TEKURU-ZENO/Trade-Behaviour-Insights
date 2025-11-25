import pandas as pd
import numpy as np

def add_volatility(df: pd.DataFrame, window=50) -> pd.DataFrame:
    """
    Computes rolling price volatility per symbol based on execution_price.
    Volatility is defined as rolling std of pct returns.
    """
    df = df.copy().sort_values(["symbol", "time"])

    # Ensure correct dtypes
    df["time"] = pd.to_datetime(df["time"])

    # Per-symbol returns
    df["price_ret"] = df.groupby("symbol")["execution_price"].pct_change()

    # Rolling volatility per symbol
    df["volatility"] = (
        df.groupby("symbol")["price_ret"]
          .transform(lambda s: s.rolling(window, min_periods=1).std())
    )

    # Replace NaN with 0 for first few entries
    df["volatility"] = df["volatility"].fillna(0)

    return df
