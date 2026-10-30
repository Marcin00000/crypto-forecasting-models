"""
Temporal flattening and feature-target alignment for Random Forest pipeline.

Converts sequential 2D/3D time-series features into flat 2D tabular data
and ensures mathematically exact causal alignment between X and y.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    'create_lag_features',
    'create_aggregated_features',
    'create_hybrid_features',
    'create_signal_features',
    'create_none_features',
    'flatten_features',
    'align_features_and_target',
]


def create_lag_features(df: pd.DataFrame, features: list[str], lookback: int) -> tuple[pd.DataFrame, list[str]]:
    """Create simple lag features (t-1, t-2, ..., t-lookback)."""
    lag_features = []
    feature_names = []
    for feature in features:
        for lag in range(1, lookback + 1):
            lag_features.append(df[feature].shift(lag))
            feature_names.append(f'{feature}_lag_{lag}')
    flat_df = pd.concat(lag_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def create_aggregated_features(df: pd.DataFrame, features: list[str], lookback: int) -> tuple[pd.DataFrame, list[str]]:
    """Create aggregated temporal features (mean, std, min, max, last)."""
    agg_features = []
    feature_names = []
    for feature in features:
        rolled = df[feature].shift(1).rolling(window=lookback, min_periods=1)
        agg_features.append(rolled.mean());  feature_names.append(f'{feature}_mean_{lookback}')
        agg_features.append(rolled.std());   feature_names.append(f'{feature}_std_{lookback}')
        agg_features.append(rolled.min());   feature_names.append(f'{feature}_min_{lookback}')
        agg_features.append(rolled.max());   feature_names.append(f'{feature}_max_{lookback}')
        agg_features.append(df[feature].shift(1)); feature_names.append(f'{feature}_last')
    flat_df = pd.concat(agg_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def create_hybrid_features(
    df: pd.DataFrame,
    features: list[str],
    lookback: int,
    lag_points: list[int] | None = None,
    windows: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create hybrid features: strategic lags + multi-window aggregations."""
    if lag_points is None:
        lag_points = [1, 2, 3, 5, 10, 24]
    if windows is None:
        windows = [6, 12, 24, 48]

    lag_points = [l for l in lag_points if l <= lookback]
    windows = [w for w in windows if w <= lookback]

    hybrid_features = []
    feature_names = []
    for feature in features:
        for lag in lag_points:
            hybrid_features.append(df[feature].shift(lag))
            feature_names.append(f'{feature}_lag_{lag}')
        for window in windows:
            rolled = df[feature].shift(1).rolling(window=window, min_periods=1)
            hybrid_features.append(rolled.mean()); feature_names.append(f'{feature}_mean_{window}')
            hybrid_features.append(rolled.std());  feature_names.append(f'{feature}_std_{window}')
        hybrid_features.append(df[feature].shift(1).diff(1))
        feature_names.append(f'{feature}_diff_1')
        if lookback >= 5:
            hybrid_features.append(df[feature].shift(1).diff(5))
            feature_names.append(f'{feature}_diff_5')
    flat_df = pd.concat(hybrid_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def _rolling_sign_fraction(col: pd.Series, window: int) -> pd.Series:
    """
    Fraction of values > 0 inside each trailing window (NaN stays in the denominator).

    Equivalent to rolling(window, min_periods=1).apply(count(x > 0) / len(x)), but built
    from two rolling sums (exact for integer counts) instead of one Python call per row.
    """
    positive = (col > 0).astype(np.float64).rolling(window=window, min_periods=1).sum()
    observed = col.notna().astype(np.float64).rolling(window=window, min_periods=1).sum()
    length = np.minimum(np.arange(1, len(col) + 1), window).astype(np.float64)
    return (positive / length).where(observed >= 1)


def create_signal_features(
    df: pd.DataFrame,
    features: list[str],
    lookback: int,
    short_windows: list[int] | None = None,
    long_windows: list[int] | None = None,
    diff_lags: list[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create signal-oriented features focused on momentum and trend direction."""
    if short_windows is None:
        short_windows = [3, 6, 12]
    if long_windows is None:
        long_windows = [24, 48]
    if diff_lags is None:
        diff_lags = [1, 3, 5, 12]

    short_windows = [w for w in short_windows if w <= lookback]
    long_windows = [w for w in long_windows if w <= lookback]
    diff_lags = [l for l in diff_lags if l <= lookback]

    signal_features = []
    feature_names = []

    for feature in features:
        col = df[feature].shift(1)

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
                signal_features.append(sm - lm)
                feature_names.append(f'{feature}_trend_{sw}_{lw}')

        full_mean = col.rolling(window=lookback, min_periods=1).mean()
        signal_features.append(col - full_mean)
        feature_names.append(f'{feature}_last_vs_mean')

        sign_frac = _rolling_sign_fraction(col, lookback)
        signal_features.append(sign_frac)
        feature_names.append(f'{feature}_sign_frac')

        diffs = {}
        for lag in diff_lags:
            d = col.diff(lag)
            diffs[lag] = d
            signal_features.append(d)
            feature_names.append(f'{feature}_diff_{lag}')

        if 1 in diffs and 5 in diffs:
            signal_features.append(diffs[1] - diffs[5])
            feature_names.append(f'{feature}_accel_1_5')

    flat_df = pd.concat(signal_features, axis=1)
    flat_df.columns = feature_names
    return flat_df, feature_names


def create_none_features(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """No temporal flattening -- pass only the current (t-1) feature values."""
    flat_df = pd.DataFrame(
        {feat: df[feat].shift(1) for feat in features}
    )
    return flat_df, list(features)


def flatten_features(
    df: pd.DataFrame,
    features: list[str],
    lookback: int,
    method: str = 'hybrid',
    **kwargs,
) -> tuple[pd.DataFrame, list[str]]:
    """Flatten sequential features into tabular format for Random Forest."""
    if method == 'lag':
        return create_lag_features(df, features, lookback)
    elif method == 'agg':
        return create_aggregated_features(df, features, lookback)
    elif method == 'hybrid':
        return create_hybrid_features(
            df, features, lookback,
            kwargs.get('lag_points'), kwargs.get('windows')
        )
    elif method == 'signal':
        return create_signal_features(
            df, features, lookback,
            kwargs.get('short_windows'), kwargs.get('long_windows'), kwargs.get('diff_lags')
        )
    elif method == 'none':
        return create_none_features(df, features)
    else:
        raise ValueError(f"Unknown flattening method: {method}")


def align_features_and_target(
    X_flat: pd.DataFrame,
    y_all: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """
    Centralised alignment between flattened feature matrix X and target y.

    Steps:
      1. Trim start of X by lookback (row 0 of X maps to candle lookback).
         Target y already starts at candle lookback, so y is NOT trimmed by lookback!
      2. Trim tail of X by horizon (no future price available for those steps).
      3. Verify exact length match: len(X) == len(y).
      4. Detect and drop common leading NaN prefix from feature warmup.
      5. Verify continuity (no NaNs inside series after warmup).
      6. Final invariant check: len(X) == len(y).

    Returns:
      (X_clean, y_clean, first_valid_idx)
    """
    lookback = int(lookback)
    horizon = int(horizon)

    # Step 1: Trim start of X by lookback
    X_flat = X_flat.iloc[lookback:].reset_index(drop=True)

    # Step 2: Trim tail of X by horizon
    if horizon > 0:
        X_flat = X_flat.iloc[:-horizon].reset_index(drop=True)

    # Step 3: Exact length equality before warmup
    if len(X_flat) != len(y_all):
        raise ValueError(f"Shape mismatch before warmup: X={len(X_flat)} vs y={len(y_all)}")

    # Step 4: Drop leading warmup NaNs simultaneously from X and y
    valid_rows = ~X_flat.isna().any(axis=1)
    if not valid_rows.any():
        raise ValueError("All rows contain NaN values in feature matrix.")
    first_valid_idx = int(valid_rows.idxmax())

    if first_valid_idx > 0:
        X_flat = X_flat.iloc[first_valid_idx:].reset_index(drop=True)
        y_all = y_all[first_valid_idx:]

    # Step 5: Check continuity after warmup
    if X_flat.isna().any().any():
        raise ValueError("Discontinuous NaN values detected inside feature matrix after warmup period.")

    # Step 6: Hard invariant
    if len(X_flat) != len(y_all):
        raise ValueError(f"Shape mismatch after warmup trimming: X={len(X_flat)} vs y={len(y_all)}")

    return X_flat, y_all, first_valid_idx
