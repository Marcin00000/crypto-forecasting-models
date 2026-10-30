"""
Temporal flattening of feature windows for xgb_model_package.

Standalone copy of xgb_model/common/flattening.py: the package must stay
copyable without the training project, so no import from it is allowed.
Every method shifts features by one candle, so the row at index t only
sees information up to t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _create_aggregated_features(df: pd.DataFrame, features: list, lookback: int):
    """Aggregate temporal features (mean, std, min, max, last) over lookback window."""
    agg_list = []
    names = []
    for feat in features:
        rolled = df[feat].shift(1).rolling(window=lookback, min_periods=1)
        agg_list += [rolled.mean(), rolled.std(), rolled.min(), rolled.max(), df[feat].shift(1)]
        names += [
            f'{feat}_mean_{lookback}', f'{feat}_std_{lookback}',
            f'{feat}_min_{lookback}',  f'{feat}_max_{lookback}',
            f'{feat}_last',
        ]
    flat = pd.concat(agg_list, axis=1)
    flat.columns = names
    return flat, names


def _create_lag_features(df: pd.DataFrame, features: list, lookback: int):
    """Create simple lag features t-1 ... t-lookback."""
    lag_list = []
    names = []
    for feat in features:
        for lag in range(1, lookback + 1):
            lag_list.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
    flat = pd.concat(lag_list, axis=1)
    flat.columns = names
    return flat, names


def _create_hybrid_features(df: pd.DataFrame, features: list, lookback: int,
                            lag_points=None, windows=None):
    """Strategic lags + multi-window rolling stats + diffs."""
    if lag_points is None:
        lag_points = [1, 2, 3, 5, 10, 24]
    if windows is None:
        windows = [6, 12, 24, 48]
    lag_points = [l for l in lag_points if l <= lookback]
    windows = [w for w in windows if w <= lookback]

    parts = []
    names = []
    for feat in features:
        for lag in lag_points:
            parts.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
        for w in windows:
            rolled = df[feat].shift(1).rolling(window=w, min_periods=1)
            parts += [rolled.mean(), rolled.std()]
            names += [f'{feat}_mean_{w}', f'{feat}_std_{w}']
        parts.append(df[feat].shift(1).diff(1))
        names.append(f'{feat}_diff_1')
        if lookback >= 5:
            parts.append(df[feat].shift(1).diff(5))
            names.append(f'{feat}_diff_5')

    flat = pd.concat(parts, axis=1)
    flat.columns = names
    return flat, names


def _create_signal_features(df: pd.DataFrame, features: list, lookback: int,
                            short_windows=None, long_windows=None, diff_lags=None):
    """Signal-oriented features: trend, momentum, sign fraction, diffs."""
    if short_windows is None:
        short_windows = [3, 6, 12]
    if long_windows is None:
        long_windows = [24, 48]
    if diff_lags is None:
        diff_lags = [1, 3, 5, 12]
    short_windows = [w for w in short_windows if w <= lookback]
    long_windows = [w for w in long_windows if w <= lookback]
    diff_lags = [l for l in diff_lags if l <= lookback]

    parts = []
    names = []
    for feat in features:
        col = df[feat].shift(1)
        short_means = {sw: col.rolling(window=sw, min_periods=1).mean()
                       for sw in short_windows}
        long_means = {lw: col.rolling(window=lw, min_periods=1).mean()
                      for lw in long_windows}
        for sw, sm in short_means.items():
            for lw, lm in long_means.items():
                parts.append(sm - lm)
                names.append(f'{feat}_trend_{sw}_{lw}')
        full_mean = col.rolling(window=lookback, min_periods=1).mean()
        parts.append(col - full_mean)
        names.append(f'{feat}_last_vs_mean')
        sign_frac = col.rolling(window=lookback, min_periods=1).apply(
            lambda x: float(np.sum(x > 0)) / max(len(x), 1), raw=True
        )
        parts.append(sign_frac)
        names.append(f'{feat}_sign_frac')
        diffs = {}
        for lag in diff_lags:
            d = col.diff(lag)
            diffs[lag] = d
            parts.append(d)
            names.append(f'{feat}_diff_{lag}')
        if 1 in diffs and 5 in diffs:
            parts.append(diffs[1] - diffs[5])
            names.append(f'{feat}_accel_1_5')
    flat = pd.concat(parts, axis=1)
    flat.columns = names
    return flat, names


def _create_none_features(df: pd.DataFrame, features: list):
    """No temporal flattening -- pass only the current (t-1) feature values."""
    flat_df = pd.DataFrame(
        {feat: df[feat].shift(1) for feat in features}
    )
    return flat_df, list(features)


def _flatten_features(df: pd.DataFrame, features: list, lookback: int, method: str):
    """Dispatch to the correct flattening method."""
    if method == 'agg':
        return _create_aggregated_features(df, features, lookback)
    elif method == 'lag':
        return _create_lag_features(df, features, lookback)
    elif method == 'hybrid':
        return _create_hybrid_features(df, features, lookback)
    elif method == 'signal':
        return _create_signal_features(df, features, lookback)
    elif method == 'none':
        return _create_none_features(df, features)
    else:
        raise ValueError(f"Unknown flattening method: '{method}'. "
                         f"Expected 'agg', 'lag', 'hybrid', 'signal', or 'none'.")
