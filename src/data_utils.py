from pathlib import Path
import pandas as pd
import numpy as np
from functools import lru_cache

# Anchor paths to project root (parent of src/)
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def save_parquet(df: pd.DataFrame, name: str = "trades_processed.parquet"):
    out = PROCESSED / name
    df.to_parquet(out, index=False)
    return out


def read_parquet(name: str = "trades_processed.parquet"):
    p = PROCESSED / name
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")
    return pd.read_parquet(p)


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def robust_datetime(series):
    s = pd.to_datetime(series, errors="coerce", utc=True)
    return s


@lru_cache(maxsize=1)
def load_raw_trades(fn: str = "historical.csv") -> pd.DataFrame:
    """
    Load Hyperliquid historical trader CSV and normalize to:
    time, execution_price, size, closedpnl, account, symbol, side
    """
    p = RAW / fn
    df = load_csv(p)
    df = clean_cols(df)

    # Column map for your real dataset
    rename_map = {
        "execution_price": "execution_price",
        "execution price": "execution_price",
        "size_tokens": "size",
        "closed_pnl": "closedpnl",
        "closed pnl": "closedpnl",
        "coin": "symbol",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # ---- TIME FIX (final) ----
    # Prefer timestamp_ist > timestamp
    if "timestamp_ist" in df.columns:
        df["time"] = robust_datetime(df["timestamp_ist"])
        df.drop(columns=["timestamp_ist"], inplace=True)
    elif "timestamp" in df.columns:
        df["time"] = robust_datetime(df["timestamp"])
        df.drop(columns=["timestamp"], inplace=True)
    else:
        raise KeyError("No timestamp column found (expected 'Timestamp' or 'Timestamp IST').")
    # ---- END FIX ----

    # Ensure numeric
    for col in ["execution_price", "size", "closedpnl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["time", "execution_price", "size", "closedpnl"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    if "account" not in df.columns:
        raise KeyError("Missing column: account")
    if "symbol" not in df.columns:
        raise KeyError("Missing column: symbol")

    return df


@lru_cache(maxsize=1)
def load_raw_fg(fn: str = "fear_greed_index.csv") -> pd.DataFrame:
    p = RAW / fn
    df = load_csv(p)
    df = clean_cols(df)

    if "date" not in df.columns:
        raise KeyError("Sentiment file must include column 'date'")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["score"] = pd.to_numeric(df["value"], errors="coerce") if "value" in df.columns else np.nan

    return df
