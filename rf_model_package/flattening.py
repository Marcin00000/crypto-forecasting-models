"""
Self-contained flattening of per-candle features into tabular rows for the Random Forest.

Mirrors rf_model/common/flattening.py (all methods shift by one candle so that row t only
sees data up to t-1). The package must stay copyable on its own, so the logic is
duplicated here on purpose; tests/test_rf_model_package.py guards against drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _create_aggregated_features(df: pd.DataFrame, features: list, lookback: int):
    """Aggregate temporal features (mean, std, min, max, last) over lookback window."""
    agg_list = []
    names    = []
    for feat in features:
        rolled = df[feat].shift(1).rolling(window=lookback, min_periods=1)
        agg_list += [
            rolled.mean(), rolled.std(), rolled.min(), rolled.max(),
            df[feat].shift(1),
        ]
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
    names    = []
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
    windows    = [w for w in windows    if w <= lookback]

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
                            short_windows=None, long_windows=None,
                            diff_lags=None):
    """
    Create signal-oriented features focused on momentum and trend direction.

    Generated features per input feature:
      trend_S_L       -- mean(S) - mean(L) for each (short, long) window pair
                         positive = short-term above long-term (uptrend inside window)
      last_vs_mean    -- last value - mean(lookback)
                         how far the most recent observation sits from window mean
      sign_frac       -- fraction of window values > 0
                         most informative for return_* features
      diff_<lag>      -- first difference at strategic lags (momentum pulses)
      accel_1_5       -- diff(1) - diff(5): short-term acceleration vs medium-term
                         (only generated when both lag 1 and lag 5 are present)

    All inputs are shifted by 1 before aggregation to prevent look-ahead leakage.
    Windows and lags exceeding lookback are silently skipped.
    """
    if short_windows is None:
        short_windows = [3, 6, 12]
    if long_windows is None:
        long_windows = [24, 48]
    if diff_lags is None:
        diff_lags = [1, 3, 5, 12]

    short_windows = [w for w in short_windows if w <= lookback]
    long_windows  = [w for w in long_windows  if w <= lookback]
    diff_lags     = [l for l in diff_lags     if l <= lookback]

    parts = []
    names = []

    for feat in features:
        col = df[feat].shift(1)   # shift-1: prevent look-ahead leakage

        # --- trend: mean(short) - mean(long) for every (short, long) pair ---
        short_means = {}
        for sw in short_windows:
            m = col.rolling(window=sw, min_periods=1).mean()
            short_means[sw] = m

        long_means = {}
        for lw in long_windows:
            m = col.rolling(window=lw, min_periods=1).mean()
            long_means[lw] = m

        for sw, sm in short_means.items():
            for lw, lm in long_means.items():
                parts.append(sm - lm)
                names.append(f'{feat}_trend_{sw}_{lw}')

        # --- last value vs full-window mean ---
        full_mean = col.rolling(window=lookback, min_periods=1).mean()
        parts.append(col - full_mean)
        names.append(f'{feat}_last_vs_mean')

        # --- sign fraction: fraction of window values > 0 ---
        sign_frac = col.rolling(window=lookback, min_periods=1).apply(
            lambda x: float(np.sum(x > 0)) / max(len(x), 1), raw=True
        )
        parts.append(sign_frac)
        names.append(f'{feat}_sign_frac')

        # --- diffs at strategic lags ---
        diffs = {}
        for lag in diff_lags:
            d = col.diff(lag)
            diffs[lag] = d
            parts.append(d)
            names.append(f'{feat}_diff_{lag}')

        # --- acceleration: diff(1) - diff(5) when both lags are present ---
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


def _flatten_features(df: pd.DataFrame, features: list, lookback: int,
                      method: str):
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
        raise ValueError(
            f"Unknown flattening method: '{method}'. "
            f"Expected 'agg', 'lag', 'hybrid', 'signal', or 'none'."
        )
