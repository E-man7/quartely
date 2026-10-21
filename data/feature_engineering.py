"""Technical indicators and HMM feature computation.

Every feature at row t uses only information available at the close of bar t,
so features can be fed to a filtered (forward-only) HMM without look-ahead.
HMM inputs are standardized with *rolling* z-scores (default 252 bars), which
are causal by construction: no full-sample or training-window statistics leak
into earlier rows.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RETURN_COL = "ret_1"   # raw 1-bar log return; used for regime return/vol statistics
Z_PREFIX = "z_"

# Raw observables standardized for the HMM (names without the z_ prefix)
FEATURE_COLUMNS: List[str] = [
    "ret_1", "ret_5", "ret_20",          # returns
    "rvol_20", "vol_ratio",              # volatility
    "volume_norm", "volume_trend",       # volume
    "adx_14", "sma50_slope",             # trend
    "rsi_14", "dist_sma200",             # mean reversion
    "roc_10", "roc_20",                  # momentum
    "natr_14",                           # range
]
HMM_FEATURES: List[str] = [Z_PREFIX + c for c in FEATURE_COLUMNS]


# ------------------------------------------------------------- indicators
def log_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    return np.log(close / close.shift(horizon))


def realized_vol(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Rate of change over `window` bars."""
    return close / close.shift(window) - 1.0


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def sma_slope(series: pd.Series, window: int) -> pd.Series:
    """One-bar change of the `window` SMA relative to its level (scale-free)."""
    avg = sma(series, window)
    return avg.diff() / avg.abs().replace(0, np.nan)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's Average Directional Index."""
    up, down = df["high"].diff(), -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smooth = dict(alpha=1 / window, adjust=False, min_periods=window)
    atr_w = true_range(df).ewm(**smooth).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(**smooth).mean() / atr_w
    minus_di = 100 * minus_dm.ewm(**smooth).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(**smooth).mean()


def drawdown(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1.0


def rolling_zscore(series: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    """(x - rolling mean) / rolling std over the trailing `window` bars (full window required)."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    flat = std <= 1e-10 * (mean.abs() + 1.0)   # constant window: deviation is zero, not undefined
    z = (series - mean) / std.mask(flat)
    return z.mask(flat & std.notna(), 0.0)


# --------------------------------------------------------------- features
def feature_warmup(cfg: Optional[Dict] = None) -> int:
    """Bars consumed before the first fully-populated standardized feature row."""
    cfg = cfg or {}
    return cfg.get("long_sma", 200) + cfg.get("zscore_window", TRADING_DAYS)


def build_features(df: pd.DataFrame, cfg: Optional[Dict] = None) -> pd.DataFrame:
    """Compute raw features and their rolling z-scores from an OHLCV DataFrame.

    Raw columns are kept (e.g. `rvol_*` for vol targeting, `ret_1` for regime
    statistics); HMM inputs are the `z_*` columns. Volume features are skipped
    if there is no volume column, range/trend features if there is no high/low.
    """
    cfg = cfg or {}
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close

    for h in cfg.get("return_horizons", [1, 5, 20]):
        out[f"ret_{h}"] = log_returns(close, h)

    one_bar = log_returns(close, 1)
    for w in cfg.get("vol_windows", [5, 20]):
        out[f"rvol_{w}"] = realized_vol(one_bar, w)
    short, long = cfg.get("vol_ratio_windows", [5, 20])
    out["vol_ratio"] = realized_vol(one_bar, short) / realized_vol(one_bar, long).replace(0, np.nan)

    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        w = cfg.get("volume_window", 50)
        out["volume_norm"] = rolling_zscore(vol, w)
        out["volume_trend"] = sma_slope(vol, cfg.get("volume_trend_window", 10))

    if {"high", "low"}.issubset(df.columns):
        out["adx_14"] = adx(df, cfg.get("adx_window", 14))
        out["natr_14"] = atr(df, cfg.get("atr_window", 14)) / close
    out["sma50_slope"] = sma_slope(close, cfg.get("trend_sma", 50))

    out["rsi_14"] = rsi(close, cfg.get("rsi_window", 14))
    long_sma = sma(close, cfg.get("long_sma", 200))
    out["dist_sma200"] = (close - long_sma) / close

    for w in cfg.get("roc_windows", [10, 20]):
        out[f"roc_{w}"] = momentum(close, w)
    out["drawdown"] = drawdown(close)

    out = out.replace([np.inf, -np.inf], np.nan)
    z_window = cfg.get("zscore_window", TRADING_DAYS)
    for col in FEATURE_COLUMNS:
        if col in out.columns:
            out[Z_PREFIX + col] = rolling_zscore(out[col], z_window)
    return out
