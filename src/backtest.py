import pandas as pd
import numpy as np

def apply_size_scaling(df: pd.DataFrame, cond_col: str, scale: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    df['closedpnl_scaled'] = df['closedpnl']
    mask = df[cond_col]
    df.loc[mask, 'closedpnl_scaled'] = df.loc[mask, 'closedpnl'] * scale
    return df

def metrics_from_daily_series(series: pd.Series) -> dict:
    daily = series.copy().fillna(0)
    total = float(daily.sum())
    if not isinstance(daily.index, pd.DatetimeIndex):
        return {'total_pnl': total}
    daily_sum = daily.resample('D').sum()
    cum = daily_sum.cumsum()
    peak = cum.cummax()
    drawdown = cum - peak
    maxdd = float(drawdown.min())
    return {
        'total_pnl': total,
        'max_drawdown': maxdd,
        'daily_mean': float(daily_sum.mean()),
        'daily_vol': float(daily_sum.std())
    }
