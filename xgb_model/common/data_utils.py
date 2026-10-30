"""
Data utilities for XGBoost model
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler


# Target scaling & winsorisation delegated to target_utils
from .target_utils import TargetScaler, winsorize_targets  # noqa: F401


# =============================================================================
# FLATTENING (delegated to common.flattening)
# =============================================================================

from .flattening import flatten_features, align_features_and_target  # noqa: F401 — re-export for backward compat


def create_log_return_target(df, lookback, horizon):
    """
    Create log return target: log(close[t+horizon] / close[t])
    Features use window [t-lookback, t-1], current price at t, future at t+horizon.
    """
    prices = df['close'].values
    n_samples = len(prices) - lookback - horizon

    if n_samples <= 0:
        raise ValueError(f"Insufficient data: need at least {lookback + horizon} samples")

    current = prices[lookback: lookback + n_samples]
    future  = prices[lookback + horizon: lookback + horizon + n_samples]
    y = np.log(future / current).astype(np.float32)

    return y


# =============================================================================
# FEATURE SELECTION HELPER
# =============================================================================

def _strict_feature_error(what, data_id, dataset_feature_set, missing):
    """
    Error for a requested feature list that the dataset cannot provide.

    Silently training on the intersection would store a model under the name of a
    feature set it does not use, so the request is rejected with a pointer to the
    feature set the dataset was generated with.
    """
    hint = dataset_feature_set or '<metadata.feature_set>'
    return ValueError(
        f"{what} wymaga kolumn spoza datasetu {data_id or 'UNKNOWN'} "
        f"(feature_set datasetu: '{dataset_feature_set}', brakuje: {missing}). "
        f"Użyj --feature_set {hint}."
    )


def _resolve_features(base_features, feature_set, custom_features,
                       include_buy_pressure, data_id=None, dataset_feature_set=None):
    """
    Resolve the feature list to use from prepared-data metadata (strict policy).

    Args:
        base_features:       Features stored in metadata['features']
                             (already-computed, available in the CSV).
        feature_set:         Requested feature set name or None.
        custom_features:     Explicit feature list or None.
        include_buy_pressure: Whether the dataset really contains buy_pressure.
        data_id:             Dataset ID, used only in error messages.
        dataset_feature_set: metadata['feature_set'], used only in error messages.

    Returns:
        list of feature names, all of which exist in base_features.

    Raises:
        ValueError: the requested set / list is not a subset of base_features.
    """
    # Custom list overrides everything
    if custom_features is not None:
        missing = [f for f in custom_features if f not in base_features]
        if missing:
            raise _strict_feature_error(
                'custom_features', data_id, dataset_feature_set, missing)
        return list(custom_features)

    # Named feature set — resolve via universal FE_
    if feature_set is not None:
        from feature_engineering import get_feature_sets
        sets = get_feature_sets(include_buy_pressure=include_buy_pressure)

        # Legacy alias mapping
        _LEGACY_ALIAS = {
            'momentum':           'xgb_momentum',
            'volatility_focused': 'xgb_volatility',
        }
        resolved_set = _LEGACY_ALIAS.get(feature_set, feature_set)

        if resolved_set not in sets:
            raise ValueError(
                f"Unknown feature set: '{feature_set}'. "
                f"Available: {sorted(sets.keys())}"
            )
        requested = sets[resolved_set] or []
        missing = [f for f in requested if f not in base_features]
        if missing:
            raise _strict_feature_error(
                f"feature_set '{feature_set}'", data_id, dataset_feature_set, missing)
        return list(requested)

    # Default: use all features stored in prepared data
    return list(base_features)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_prepared_data(data_path, flattening_method=None, feature_set=None,
                       custom_features=None, lookback=None, horizon=None,
                       scale_target=True, target_scaler_type='standard',
                       winsorize_target=False, winsorize_percentile=99.0):
    """
    Load train/val data from a prepared directory (universal format).

    The prepared directory is produced by data_preparation/prepare_data.py
    and contains:
        train.csv              — features + OHLCV (scaled, NaN-dropped)
        scaler_features.pkl    — fitted feature scaler
        metadata.json          — data ID, features list, lookback, horizon, etc.

    buy_pressure handling:
        Derived from the column list, not from metadata['has_buy_pressure']:
        E114/E115 carry the flag as true although they have no buy_pressure
        column, which would add it to named feature sets and then fail.

    A feature_set / custom_features that is not a subset of metadata['features']
    raises ValueError (strict policy) instead of being trimmed silently.
    """
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    train_csv = os.path.join(data_path, 'train.csv')
    if not os.path.exists(train_csv):
        raise ValueError(f"Missing train.csv in {data_path}")

    train_df = pd.read_csv(train_csv)

    # Support both naming conventions for the feature scaler
    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Feature scaler not found in {data_path} "
            f"(tried scaler_features.pkl and feature_scaler.pkl)"
        )
    feature_scaler = joblib.load(scaler_path)

    base_features = metadata['features']

    if lookback is None:
        lookback = metadata['lookback']
    if horizon is None:
        horizon = metadata['horizon']

    has_buy_pressure = 'buy_pressure' in base_features

    features = _resolve_features(
        base_features, feature_set, custom_features, has_buy_pressure,
        data_id=metadata.get('data_id'),
        dataset_feature_set=metadata.get('feature_set'),
    )

    if not features:
        raise ValueError(
            f"No features selected (feature_set='{feature_set}', "
            f"base has {len(base_features)} features)"
        )

    if flattening_method is None:
        flattening_method = metadata.get('flattening_method', 'hybrid')

    print(f"Features: {len(features)}, Lookback: {lookback}, "
          f"Horizon: {horizon}, Flattening: {flattening_method}")
    if has_buy_pressure and 'buy_pressure' in features:
        print("  buy_pressure: included")

    X_flat, flat_features = flatten_features(
        train_df, features, lookback, method=flattening_method
    )
    y_all = create_log_return_target(train_df, lookback, horizon)
    # y[k] = log_return at time (lookback+k), X[k] must use data up to time (lookback+k-1);
    # NaN rows are trimmed from X and y together (interior NaN raises).
    X_flat, y_all, _ = align_features_and_target(X_flat, y_all, lookback, horizon)
    X_all = X_flat.values
    n = len(y_all)

    # Chronological train/val split
    train_size = int(n * 0.8)
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_val   = X_all[train_size:]
    y_val   = y_all[train_size:]

    # Winsorisation (before scaling — clips raw log-returns)
    clip_bounds = None
    if winsorize_target:
        y_train, y_val, clip_bounds = winsorize_targets(
            y_train, y_val, percentile=winsorize_percentile
        )

    # Target scaling
    target_scaler = None
    if scale_target:
        print(f"Scaling targets ({target_scaler_type}): "
              f"mean={y_train.mean():.6f}, std={y_train.std():.6f} -> 0, 1")
        target_scaler = TargetScaler(method=target_scaler_type)
        y_train = target_scaler.fit_transform(y_train)
        y_val   = target_scaler.transform(y_val)

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}")

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'feature_scaler': feature_scaler,
        'target_scaler':  target_scaler,
        'clip_bounds':    clip_bounds,
    }

    metadata.update({
        'flat_features':      flat_features,
        'n_flat_features':    len(flat_features),
        'n_features':         len(features),
        'features':           features,
        'lookback':           lookback,
        'horizon':            horizon,
        'flattening_method':  flattening_method,
        'has_buy_pressure':   has_buy_pressure,
        'scale_target':       scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target':   winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds':        clip_bounds,
    })

    return data, metadata


# Test data loading delegated to test_data_utils
from .test_data_utils import load_test_data_only  # noqa: F401


def prepare_raw_data_on_the_fly(df, features=None, lookback=48, horizon=12,
                                flattening_method='hybrid',
                                feature_set=None,
                                custom_features=None,
                                feature_scaler=None,
                                scale_target=True,
                                target_scaler_type='standard',
                                winsorize_target=False,
                                winsorize_percentile=99.0):
    """
    Prepare raw CSV data on-the-fly for training (train/val only, no test split).

    Calls universal prepare_features() → select_features() → flatten.
    Feature scaler is fit ONLY on the training split to prevent leakage.

    buy_pressure is included automatically when the raw DataFrame contains
    the 'taker_buy_base_asset_volume' column.
    """
    from feature_engineering import prepare_features, select_features

    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After feature engineering: {len(df)} samples")

    if len(df) < lookback + horizon + 100:
        raise ValueError(
            f"Insufficient data after feature engineering: {len(df)} samples. "
            f"Need at least {lookback + horizon + 100}."
        )

    # Detect buy_pressure availability from the engineered DataFrame
    has_buy_pressure = 'buy_pressure' in df.columns

    if custom_features is not None:
        selected = [f for f in custom_features if f in df.columns]
    elif feature_set is not None:
        selected = select_features(
            df,
            feature_set=feature_set,
            validate=False,
            include_buy_pressure=has_buy_pressure,
        )
    elif features is not None:
        # Explicit list passed by caller (legacy path)
        selected = [f for f in features if f in df.columns]
    else:
        selected = select_features(
            df,
            validate=False,
            include_buy_pressure=has_buy_pressure,
        )

    selected = [f for f in selected if f != 'close']

    if len(selected) < 3:
        raise ValueError(f"Too few features selected: {len(selected)}")

    print(f"Features: {len(selected)}, Flattening: {flattening_method}")

    X_flat_raw, flat_feature_names = flatten_features(
        df, selected, lookback, method=flattening_method
    )
    y_all = create_log_return_target(df, lookback, horizon)
    # Same alignment rule as load_prepared_data
    X_all, y_all, _ = align_features_and_target(X_flat_raw, y_all, lookback, horizon)
    n = len(y_all)

    train_size = int(n * 0.8)
    X_train_df = X_all.iloc[:train_size]
    X_val_df   = X_all.iloc[train_size:]
    y_train    = y_all[:train_size]
    y_val      = y_all[train_size:]

    if feature_scaler is None:
        feature_scaler = StandardScaler()
        feature_scaler.fit(X_train_df)   # fit on train only

    X_train = feature_scaler.transform(X_train_df)
    X_val   = feature_scaler.transform(X_val_df)

    # Winsorisation (before target scaling)
    clip_bounds = None
    if winsorize_target:
        y_train, y_val, clip_bounds = winsorize_targets(
            y_train, y_val, percentile=winsorize_percentile
        )

    # Target scaling
    target_scaler = None
    if scale_target:
        print(f"Scaling targets ({target_scaler_type}): "
              f"mean={y_train.mean():.6f}, std={y_train.std():.6f} → 0, 1")
        target_scaler = TargetScaler(method=target_scaler_type)
        y_train = target_scaler.fit_transform(y_train)
        y_val   = target_scaler.transform(y_val)

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}")

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'feature_scaler': feature_scaler,
        'target_scaler':  target_scaler,
        'clip_bounds':    clip_bounds,
    }

    metadata = {
        'model_type':         'xgboost',
        'target_type':        'log_return',
        'features':           selected,
        'flat_features':      flat_feature_names,
        'n_features':         len(selected),
        'n_flat_features':    len(flat_feature_names),
        'lookback':           lookback,
        'horizon':            horizon,
        'flattening_method':  flattening_method,
        'has_buy_pressure':   has_buy_pressure,
        'scale_target':       scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target':   winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds':        clip_bounds,
    }

    return data, metadata


def load_data(data_path, scale_target=True, target_scaler_type='standard',
              winsorize_target=False, winsorize_percentile=99.0, **kwargs):
    """
    Unified entry point — load data from prepared directory or raw CSV.

    Prepared directory (universal format, from data_preparation/prepare_data.py):
        Expects train.csv + scaler_features.pkl + metadata.json.

    Raw CSV:
        Calls prepare_raw_data_on_the_fly() which runs prepare_features()
        internally.

    kwargs forwarded to the underlying loader:
        flattening_method, feature_set, custom_features, lookback, horizon
    """
    if os.path.isdir(data_path):
        if not os.path.exists(os.path.join(data_path, 'metadata.json')):
            raise ValueError(f"Directory without metadata.json: {data_path}")
        data, metadata = load_prepared_data(
            data_path,
            flattening_method=kwargs.get('flattening_method'),
            feature_set=kwargs.get('feature_set'),
            custom_features=kwargs.get('custom_features'),
            lookback=kwargs.get('lookback'),
            horizon=kwargs.get('horizon'),
            scale_target=scale_target,
            target_scaler_type=target_scaler_type,
            winsorize_target=winsorize_target,
            winsorize_percentile=winsorize_percentile,
        )
        return data, True, metadata

    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from CSV")
        data, metadata = prepare_raw_data_on_the_fly(
            df,
            feature_set=kwargs.get('feature_set'),
            custom_features=kwargs.get('custom_features'),
            lookback=kwargs.get('lookback', 48),
            horizon=kwargs.get('horizon', 12),
            flattening_method=kwargs.get('flattening_method', 'hybrid'),
            scale_target=scale_target,
            target_scaler_type=target_scaler_type,
            winsorize_target=winsorize_target,
            winsorize_percentile=winsorize_percentile,
        )
        return data, False, metadata

    else:
        raise ValueError(
            f"Invalid data path: {data_path}. "
            f"Expected a prepared directory or a .csv file."
        )
