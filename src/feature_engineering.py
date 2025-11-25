import pandas as pd
import numpy as np
from .data_utils import load_raw_fg
from .volatility import add_volatility


def basic_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core financial/trade-level engineered features.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    df["notional"] = (df["execution_price"].abs() * df["size"].abs()).fillna(0.0)
    df["return_pct"] = df["closedpnl"] / (df["notional"] + 1e-9)
    df["win"] = (df["closedpnl"] > 0).astype(int)
    df["trade_date"] = df["time"].dt.date

    # Standardized time-of-day feature (required by Notebook-3)
    df["time_of_day"] = df["time"].dt.hour + df["time"].dt.minute / 60

    df["weekday"] = df["time"].dt.weekday
    df["weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # SAFE Leverage handling (this fixes your error)
    if "leverage" in df.columns:
        df["leverage"] = pd.to_numeric(df["leverage"], errors="coerce").fillna(1)
    else:
        df["leverage"] = 1

    return df


def merge_sentiment(trades: pd.DataFrame, fg: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge sentiment and create multi-horizon sentiment regimes.
    """
    trades = trades.copy()
    if fg is None:
        fg = load_raw_fg()

    fg_small = fg[["date", "score", "classification"]].copy()

    # Direct merge on same-day
    trades = trades.merge(fg_small, left_on="trade_date", right_on="date", how="left")

    # Fill missing sentiment with ±1 day sentiment
    if trades["score"].isna().sum() > 0:
        trades["date_minus1"] = (pd.to_datetime(trades["trade_date"]) - pd.Timedelta(days=1)).dt.date
        trades["date_plus1"] = (pd.to_datetime(trades["trade_date"]) + pd.Timedelta(days=1)).dt.date

        fg_minus = fg_small.rename(columns={
            "date": "date_minus1", "score": "score_minus1", "classification": "classification_minus1"
        })
        fg_plus = fg_small.rename(columns={
            "date": "date_plus1", "score": "score_plus1", "classification": "classification_plus1"
        })

        trades = trades.merge(fg_minus, on="date_minus1", how="left")
        trades = trades.merge(fg_plus, on="date_plus1", how="left")

        trades["score"] = trades["score"].fillna(trades["score_minus1"]).fillna(trades["score_plus1"])
        trades["classification"] = trades["classification"].fillna(
            trades["classification_minus1"]
        ).fillna(trades["classification_plus1"])

    # Rolling sentiment features
    fg_sorted = fg_small.set_index("date").sort_index()
    fg_sorted["score_3d"] = fg_sorted["score"].rolling(3, min_periods=1).mean()
    fg_sorted["score_7d"] = fg_sorted["score"].rolling(7, min_periods=1).mean()
    fg_sorted["sentiment_shift"] = fg_sorted["score_3d"] - fg_sorted["score_7d"]
    fg_sorted = fg_sorted.reset_index().rename(columns={"date": "date_full"})

    trades = trades.merge(
        fg_sorted[["date_full", "score_3d", "score_7d", "sentiment_shift"]]
        .rename(columns={"date_full": "trade_date"}),
        on="trade_date",
        how="left"
    )

    return trades


def account_rolling_features(df: pd.DataFrame, windows=(10, 30, 100)) -> pd.DataFrame:
    """
    Capture trader behavior and skill progression.
    """
    df = df.copy().sort_values(["account", "time"])

    for w in windows:
        df[f"winrate_{w}"] = df.groupby("account")["win"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"avg_return_{w}"] = df.groupby("account")["return_pct"].transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"pnl_stability_{w}"] = df.groupby("account")["return_pct"].transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )

    df["conviction"] = df["size"] / df.groupby("account")["size"].transform(
        lambda s: s.rolling(30, min_periods=1).mean()
    )

    return df


def feature_pipeline(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Full engineered dataset including sentiment, volatility, and behavioral features.
    """
    trades = basic_trade_features(trades)
    trades = merge_sentiment(trades)

    if "account" in trades.columns:
        trades = account_rolling_features(trades)

    # Log Notional for numerical stability
    trades["log_notional"] = np.log1p(trades["notional"])

    # Volatility via execution_price (market risk)
    trades = add_volatility(trades)
    trades["volatility_bucket"] = pd.qcut(trades["volatility"], 4, labels=[0, 1, 2, 3])

    # Interaction alpha
    trades["sentiment_vol_interaction"] = trades["score"] * trades["volatility"]

    # Risk-based feature
    trades["risk_per_trade"] = trades["notional"] / (
        1 + trades.groupby("account")["notional"].transform(lambda s: s.rolling(50, min_periods=1).mean())
    )

    return trades
