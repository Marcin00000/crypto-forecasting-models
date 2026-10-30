"""
Data loading and preparation utilities for Random Forest log-return pipeline.

Re-exports flattening and target scaling utilities.
"""
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Re-exports for complete backward compatibility
from .target_utils import TargetScaler, winsorize_targets
from .flattening import (
    create_lag_features,
    create_aggregated_features,
    create_hybrid_features,
    create_signal_features,
    create_none_features,
    flatten_features,
    align_features_and_target,
)

__all__ = [
    'TargetScaler',
    'winsorize_targets',
    'create_lag_features',
    'create_aggregated_features',
    'create_hybrid_features',
    'create_signal_features',
    'create_none_features',
    'flatten_features',
    'align_features_and_target',
    'create_log_return_target',
    'load_prepared_data',
    'prepare_raw_data_on_the_fly',
    'load_data',
]


def create_log_return_target(df: pd.DataFrame, lookback: int, horizon: int) -> np.ndarray:
    """
    Create log return target: log(close[t+horizon] / close[t])
    Features use window [t-lookback, t-1], current price at t, future at t+horizon.
    """
    prices = df['close'].values
    n_samples = len(prices) - int(lookback) - int(horizon)
    if n_samples <= 0:
        raise ValueError(f"Insufficient data: need at least {lookback + horizon} samples")
    current = prices[lookback: lookback + n_samples]
    future = prices[lookback + horizon: lookback + horizon + n_samples]
    return np.log(future / current).astype(np.float32)


def strict_feature_set_message(
    feature_set: str | None, data_id: str | None,
    dataset_feature_set: str | None, missing: list[str],
) -> str:
    """Error text for a feature selection that is not a subset of the dataset columns.

    Silently trimming the selection would change the model's input width while keeping
    the requested name in model_config.json, so the user is pointed at the set the
    dataset was actually built with.
    """
    target = dataset_feature_set or '<metadata.feature_set>'
    return (
        f"feature_set '{feature_set}' requires columns absent from dataset '{data_id}' "
        f"(dataset feature_set: '{dataset_feature_set}', missing: {list(missing)}). "
        f"Use --feature_set {target}."
    )


def _resolve_features(
    base_features: list[str],
    feature_set: str | None,
    custom_features: list[str] | None,
    include_buy_pressure: bool,
    data_id: str | None = None,
    dataset_feature_set: str | None = None,
) -> list[str]:
    """Resolve the feature list from prepared-data metadata (strict: no silent trimming)."""
    if custom_features is not None:
        requested, label = list(custom_features), 'custom'
    elif feature_set is not None:
        from feature_engineering import get_feature_sets
        sets = get_feature_sets(include_buy_pressure=include_buy_pressure)
        _LEGACY_ALIAS = {
            'momentum': 'xgb_momentum',
            'volatility': 'xgb_volatility',
            'volatility_focused': 'xgb_volatility',
            'trend_vol': 'trend_momentum',
        }
        resolved_set = _LEGACY_ALIAS.get(feature_set, feature_set)
        if resolved_set not in sets:
            raise ValueError(
                f"Unknown feature set: '{feature_set}'. "
                f"Available: {sorted(sets.keys())}"
            )
        requested, label = sets[resolved_set] or [], feature_set
    else:
        return base_features

    missing = [f for f in requested if f not in base_features]
    if missing:
        raise ValueError(
            strict_feature_set_message(label, data_id, dataset_feature_set, missing)
        )
    return list(requested)


def load_prepared_data(
    data_path: str,
    flattening_method: str | None = None,
    feature_set: str | None = None,
    custom_features: list[str] | None = None,
    lookback: int | None = None,
    horizon: int | None = None,
    scale_target: bool = True,
    target_scaler_type: str = 'robust',
    winsorize_target: bool = False,
    winsorize_percentile: float = 99.0,
    data_source: str = 'train',
) -> tuple[dict, dict]:
    """
    Load pre-engineered data and apply flattening, alignment, and target scaling.
    """
    metadata_path = os.path.join(data_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata.json in {data_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    csv_name = f"{data_source}.csv"
    train_csv = os.path.join(data_path, csv_name)
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing {csv_name} in {data_path}")
    train_df = pd.read_csv(train_csv)

    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Feature scaler not found in {data_path}")
    feature_scaler = joblib.load(scaler_path)

    base_features = metadata['features']
    lookback = int(lookback if lookback is not None else metadata['lookback'])
    horizon = int(horizon if horizon is not None else metadata['horizon'])

    # Derived from the columns: E114/E115 metadata claims buy_pressure without the column.
    has_buy_pressure = 'buy_pressure' in base_features
    features = _resolve_features(
        base_features, feature_set, custom_features, has_buy_pressure,
        data_id=metadata.get('data_id'),
        dataset_feature_set=metadata.get('feature_set'),
    )
    if not features:
        raise ValueError(
            f"No features selected (feature_set='{feature_set}', base has {len(base_features)} features)"
        )

    if flattening_method is None:
        flattening_method = metadata.get('flattening_method', 'hybrid')

    print(f"Features: {len(features)}, Lookback: {lookback}, Horizon: {horizon}, Flattening: {flattening_method}")

    X_flat, flat_features = flatten_features(train_df, features, lookback, method=flattening_method)
    y_all_raw = create_log_return_target(train_df, lookback, horizon)

    # Centralised 4-step alignment
    X_clean, y_all_clean, first_valid_idx = align_features_and_target(
        X_flat, y_all_raw, lookback, horizon
    )

    n = len(X_clean)
    train_size = int(n * 0.8)
    X_all = X_clean.values
    X_train = X_all[:train_size]
    y_train = y_all_clean[:train_size]
    X_val = X_all[train_size:]
    y_val = y_all_clean[train_size:]

    clip_bounds = None
    if winsorize_target:
        y_train, y_val, clip_bounds = winsorize_targets(
            y_train, y_val, percentile=winsorize_percentile
        )

    target_scaler = None
    if scale_target:
        print(f"Scaling targets ({target_scaler_type}): "
              f"mean={y_train.mean():.6f}, std={y_train.std():.6f} -> 0, 1")
        target_scaler = TargetScaler(method=target_scaler_type)
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}")

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'clip_bounds': clip_bounds,
        'first_valid_idx': first_valid_idx,
    }
    metadata.update({
        'flat_features': flat_features,
        'n_flat_features': len(flat_features),
        'n_features': len(features),
        'features': features,
        'lookback': lookback,
        'horizon': horizon,
        'flattening_method': flattening_method,
        'has_buy_pressure': has_buy_pressure,
        'scale_target': scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target': winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds': clip_bounds,
        'data_source': data_source,
        'data_alignment_version': 2,
    })
    return data, metadata


def prepare_raw_data_on_the_fly(
    df: pd.DataFrame,
    features: list[str] | None = None,
    lookback: int = 48,
    horizon: int = 12,
    flattening_method: str = 'hybrid',
    feature_set: str | None = None,
    custom_features: list[str] | None = None,
    feature_scaler=None,
    scale_target: bool = True,
    target_scaler_type: str = 'standard',
    winsorize_target: bool = False,
    winsorize_percentile: float = 99.0,
) -> tuple[dict, dict]:
    """Prepare raw CSV data on-the-fly for training."""
    from feature_engineering import prepare_features, select_features

    df_feats = prepare_features(df)
    # Replace infinite values with NaN without dropping rows to preserve candle index alignment
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan)
    print(f"After feature engineering: {len(df_feats)} samples")

    if len(df_feats) < lookback + horizon + 100:
        raise ValueError(
            f"Insufficient data after feature engineering: {len(df_feats)} samples. "
            f"Need at least {lookback + horizon + 100}."
        )

    has_buy_pressure = 'taker_buy_base_asset_volume' in df.columns
    if features is not None:
        selected = [f for f in features if f in df_feats.columns]
    elif custom_features is not None:
        selected = [f for f in custom_features if f in df_feats.columns]
    elif feature_set is not None:
        selected = select_features(
            df_feats, feature_set=feature_set, include_buy_pressure=has_buy_pressure
        )
    else:
        selected = select_features(df_feats, include_buy_pressure=has_buy_pressure)

    selected = [f for f in selected if f != 'close']
    if len(selected) < 3:
        raise ValueError(f"Too few features selected: {len(selected)}")

    print(f"Features: {len(selected)}, Flattening: {flattening_method}")

    X_flat_raw, flat_feature_names = flatten_features(
        df_feats, selected, lookback, method=flattening_method
    )
    y_raw = create_log_return_target(df_feats, lookback, horizon)

    # Centralised 4-step alignment
    X_clean, y_clean, first_valid_idx = align_features_and_target(
        X_flat_raw, y_raw, lookback, horizon
    )

    n = len(X_clean)
    train_size = int(n * 0.8)
    X_train_df = X_clean.iloc[:train_size]
    X_val_df = X_clean.iloc[train_size:]
    y_train = y_clean[:train_size]
    y_val = y_clean[train_size:]

    if feature_scaler is None:
        feature_scaler = StandardScaler()
        feature_scaler.fit(X_train_df)

    X_train = feature_scaler.transform(X_train_df)
    X_val = feature_scaler.transform(X_val_df)

    clip_bounds = None
    if winsorize_target:
        y_train, y_val, clip_bounds = winsorize_targets(
            y_train, y_val, percentile=winsorize_percentile
        )

    target_scaler = None
    if scale_target:
        print(f"Scaling targets ({target_scaler_type}): "
              f"mean={y_train.mean():.6f}, std={y_train.std():.6f} -> 0, 1")
        target_scaler = TargetScaler(method=target_scaler_type)
        y_train = target_scaler.fit_transform(y_train)
        y_val = target_scaler.transform(y_val)

    print(f"Shapes: train={X_train.shape}, val={X_val.shape}")

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'clip_bounds': clip_bounds,
        'first_valid_idx': first_valid_idx,
    }
    metadata = {
        'flat_features': flat_feature_names,
        'n_flat_features': len(flat_feature_names),
        'n_features': len(selected),
        'features': selected,
        'lookback': lookback,
        'horizon': horizon,
        'flattening_method': flattening_method,
        'has_buy_pressure': has_buy_pressure,
        'scale_target': scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target': winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds': clip_bounds,
        'data_alignment_version': 2,
    }
    return data, metadata


def load_data(
    data_path: str,
    scale_target: bool = True,
    target_scaler_type: str = 'robust',
    winsorize_target: bool = False,
    winsorize_percentile: float = 99.0,
    **kwargs,
) -> tuple[dict, bool, dict]:
    """
    Unified entry point for loading training data from prepared directory or raw CSV.
    Returns (data, is_prepared, metadata).
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
            data_source=kwargs.get('data_source', 'train'),
        )
        return data, True, metadata

    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from CSV")
        data, metadata = prepare_raw_data_on_the_fly(
            df,
            features=kwargs.get('features'),
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
