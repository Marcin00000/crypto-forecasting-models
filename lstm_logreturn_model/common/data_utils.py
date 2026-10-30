"""
Data utilities for LSTM Log-Return pipeline.

Public API:
  - load_data()        — unified entry point (auto-detects format)
  - create_sequences() — 3D sequence creation for LSTM

Format-specific loaders live in data_loaders.py.
"""
import json
import os

import numpy as np
import pandas as pd

from .target_utils import TargetScaler, winsorize_targets


# =========================================================================
# 3D SEQUENCE CREATION
# =========================================================================

def create_sequences(df_or_array, features, lookback):
    """
    Create 3D sequences for LSTM from DataFrame or 2D array.

    X[i] = features[i : i+lookback]  (lookback timesteps)

    Args:
        df_or_array: DataFrame with feature columns, or 2D np.ndarray
        features:    list of feature column names (used if DataFrame)
        lookback:    int — sequence length

    Returns:
        np.ndarray of shape (n_samples, lookback, n_features)
    """
    if isinstance(df_or_array, pd.DataFrame):
        data_array = df_or_array[features].values.astype(np.float32)
    else:
        data_array = np.asarray(df_or_array, dtype=np.float32)

    n_total = len(data_array)
    n_samples = n_total - lookback
    if n_samples <= 0:
        raise ValueError(
            f"Insufficient data for sequences: {n_total} rows, "
            f"lookback={lookback}, need at least {lookback + 1}"
        )

    n_features = data_array.shape[1]
    X = np.zeros((n_samples, lookback, n_features), dtype=np.float32)
    for i in range(n_samples):
        X[i] = data_array[i:i + lookback]

    return X


# =========================================================================
# FEATURE RESOLUTION
# =========================================================================

def _coverage_info(requested_set, n_requested, resolved, missing):
    return {
        'requested_set': requested_set,
        'n_requested': n_requested,
        'n_resolved': len(resolved),
        'coverage': (len(resolved) / n_requested) if n_requested else 1.0,
        'missing': list(missing),
    }


def strict_feature_set_message(feature_set, data_id, dataset_feature_set, missing):
    """Error text for a feature_set that is not a subset of the dataset columns.

    Silently trimming the set would change input_size and mislabel the model, so the
    caller is pointed at the set the dataset was actually built with.
    """
    target = dataset_feature_set or '<metadata.feature_set>'
    return (
        f"feature_set '{feature_set}' requires columns absent from dataset '{data_id}' "
        f"(dataset feature_set: '{dataset_feature_set}', missing: {list(missing)}). "
        f"Use --feature_set {target}."
    )


def _resolve_features(base_features, feature_set=None, custom_features=None,
                      include_buy_pressure=True, *, min_coverage=1.0,
                      allow_partial=False, data_id='UNKNOWN',
                      dataset_feature_set=None):
    """
    Resolve the feature list to use from prepared-data metadata.

    Args:
        base_features:        Features stored in metadata['features']
        feature_set:          Requested feature set name or None
        custom_features:      Explicit feature list or None
        include_buy_pressure: Whether buy_pressure is available
        min_coverage:         Minimum fraction of requested features required (default 1.0)
        allow_partial:        Downgrade coverage errors to warnings
        data_id:              Identifier of dataset for diagnostic logging

    Returns:
        list of feature names
    """
    import logging
    log = logging.getLogger('lr_data_utils')

    if custom_features is not None:
        resolved = [f for f in custom_features if f in base_features]
        missing = [f for f in custom_features if f not in base_features]
        if missing:
            msg = (
                f"{len(missing)} of {len(custom_features)} saved features are absent "
                f"from dataset '{data_id}': {missing}. Loading fewer features would "
                f"change input_size. Use a dataset containing these features, or pass "
                f"allow_partial=True to reproduce the old truncating behaviour."
            )
            if not allow_partial:
                raise ValueError(msg)
            log.warning(msg)
        return resolved

    if feature_set is not None:
        from feature_engineering import get_feature_sets
        sets = get_feature_sets(include_buy_pressure=include_buy_pressure)
        if feature_set not in sets:
            raise ValueError(
                f"Unknown feature set: '{feature_set}'. "
                f"Available: {sorted(sets.keys())}"
            )
        requested = sets[feature_set] or []
        resolved = [f for f in requested if f in base_features]
        missing = [f for f in requested if f not in base_features]
        n_req = len(requested)
        cov = (len(resolved) / n_req) if n_req else 1.0

        if cov < min_coverage:
            msg = (
                f"Feature set '{feature_set}' requested {n_req} features but "
                f"dataset '{data_id}' provides only {len(resolved)} "
                f"(coverage {cov:.1%} < required {min_coverage:.1%}). "
                + strict_feature_set_message(
                    feature_set, data_id, dataset_feature_set, missing)
            )
            if not allow_partial:
                raise ValueError(msg)
            log.warning(msg)

        if not resolved:
            raise ValueError(
                f"Feature set '{feature_set}' resolved to zero features on dataset '{data_id}'."
            )
        return resolved

    return list(base_features)



def _align_X_y(X, y):
    """Align X and y arrays to the same length (min of both)."""
    n = min(len(X), len(y))
    return X[:n], y[:n]


# =========================================================================
# POST-PROCESSING (shared by all loaders)
# =========================================================================

def _postprocess_targets(y_train, y_val, scale_target, target_scaler_type,
                         winsorize_target, winsorize_percentile):
    """
    Shared target postprocessing: winsorise then scale.

    Args:
        y_train, y_val:       Raw log-return targets
        scale_target:         bool — enable target scaling
        target_scaler_type:   'standard' | 'robust'
        winsorize_target:     bool — enable winsorisation
        winsorize_percentile: float — percentile for winsorisation

    Returns:
        y_train, y_val, target_scaler (or None), clip_bounds (or None)
    """
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

    return y_train, y_val, target_scaler, clip_bounds


def _build_result(X_train, y_train, X_val, y_val, feature_scaler,
                  target_scaler, clip_bounds, features, lookback, horizon,
                  metadata, *, val_close=None, y_val_raw=None,
                  y_train_raw=None, sigma_train=None, sigma_val=None,
                  sigma_val_raw=None,
                  val_close_prices=None,
                  **extra_meta):
    """
    Build standardised data_dict + metadata. Shared by all loaders.

    Returns:
        data_dict, metadata
    """
    print(f"Features: {len(features)}, Lookback: {lookback}, Horizon: {horizon}")
    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}")

    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'clip_bounds': clip_bounds,
    }
    if val_close is not None:
        data['val_close'] = val_close
    if y_val_raw is not None:
        data['y_val_raw'] = y_val_raw
    if y_train_raw is not None:
        data['y_train_raw'] = y_train_raw
    if sigma_train is not None:
        data['sigma_train'] = sigma_train
    if sigma_val is not None:
        data['sigma_val'] = sigma_val
    if sigma_val_raw is not None:
        data['sigma_val_raw'] = sigma_val_raw
    if val_close_prices is not None:
        data['val_close_prices'] = val_close_prices

    metadata.update({
        'features': features,
        'n_features': len(features),
        'lookback': lookback,
        'horizon': horizon,
        'scale_target': extra_meta.get('scale_target'),
        'target_scaler_type': extra_meta.get('target_scaler_type'),
        'clip_bounds': clip_bounds,
        **extra_meta,
    })

    return data, metadata


# =========================================================================
# UNIFIED ENTRY POINT
# =========================================================================

def load_data(data_path, preferred_format='csv', **kwargs):
    """
    Unified data loading — auto-detects format and delegates.

    Args:
        data_path:        Path to data (directory or .csv file)
        preferred_format: 'csv' or 'npy' — when directory has both formats

    Accepted kwargs (routed to the correct loader):
        feature_set, custom_features, lookback, horizon,
        scale_target, target_scaler_type,
        winsorize_target, winsorize_percentile,
        val_source

    Returns:
        data_dict, is_prepared (bool), metadata
    """
    # Extract params
    lookback = kwargs.get('lookback')
    horizon = kwargs.get('horizon')
    scale_target = kwargs.get('scale_target', False)
    target_scaler_type = kwargs.get('target_scaler_type', 'robust')
    winsorize_target = kwargs.get('winsorize_target', False)
    winsorize_percentile = kwargs.get('winsorize_percentile', None)
    if winsorize_percentile is None:
        from lstm_config import LSTMConfig as _cfg
        winsorize_percentile = _cfg.WINSORIZE_PERCENTILE
    val_source = kwargs.get('val_source', 'split')
    feature_set = kwargs.get('feature_set')
    custom_features = kwargs.get('custom_features')

    # Lazy import to break circular dependency (data_loaders → data_utils)
    from .data_loaders import (
        load_prepared_csv, load_prepared_npy, prepare_raw_csv,
    )

    # --- Select and call the appropriate loader ---
    if os.path.isdir(data_path):
        metadata_path = os.path.join(data_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Directory without metadata.json: {data_path}")

        has_csv = (os.path.exists(os.path.join(data_path, 'train.csv')) or
                   os.path.exists(os.path.join(data_path, 'agent.csv')))
        has_npy = os.path.exists(os.path.join(data_path, 'X_train.npy'))
        use_npy = ((preferred_format == 'npy' and has_npy)
                    or (not has_csv and has_npy))

        # NPY arrays are frozen to metadata['features']; a different feature_set can
        # only be honoured by the CSV loader, so switch format instead of ignoring it.
        if use_npy and has_csv:
            with open(metadata_path, 'r') as f:
                meta_set = json.load(f).get('feature_set')
            if custom_features is not None or (
                    feature_set is not None and feature_set != meta_set):
                import warnings
                warnings.warn(
                    f"feature_set '{feature_set}' differs from dataset feature_set "
                    f"'{meta_set}': NPY cannot select features, loading CSV instead.",
                    stacklevel=2,
                )
                use_npy = False

        if use_npy:
            if feature_set is not None or custom_features is not None:
                import warnings
                warnings.warn(
                    "feature_set/custom_features ignored for NPY format. "
                    "NPY uses all features from metadata.",
                    stacklevel=2,
                )
            result = load_prepared_npy(
                data_path, lookback=lookback, horizon=horizon,
                val_source=val_source,
            )
        elif has_csv:
            result = load_prepared_csv(
                data_path, feature_set=feature_set,
                custom_features=custom_features,
                lookback=lookback, horizon=horizon,
                val_source=val_source,
                data_source=kwargs.get('data_source', 'train'),
                min_coverage=kwargs.get('min_feature_coverage', 1.0),
                allow_partial=kwargs.get('allow_partial_feature_set', False),
            )
        else:
            raise ValueError(
                f"No train.csv/agent.csv or X_train.npy found in {data_path}"
            )
        is_prepared = True

    elif data_path.endswith('.csv'):
        parent = os.path.dirname(data_path) or '.'
        # train.csv of a prepared dataset must not be treated as raw OHLCV: that would
        # rebuild features and fit a new scaler instead of using metadata.json.
        if (os.path.basename(data_path) in ('train.csv', 'agent.csv')
                and os.path.isfile(os.path.join(parent, 'metadata.json'))):
            return load_data(parent, preferred_format=preferred_format, **kwargs)
        result = prepare_raw_csv(
            data_path, feature_set=feature_set,
            custom_features=custom_features,
            lookback=lookback or 48, horizon=horizon or 12,
            val_source=val_source,
        )
        is_prepared = False

    else:
        raise ValueError(
            f"Invalid data path: {data_path}. "
            f"Expected a prepared directory or a .csv file."
        )

    # --- Unpack loader result ---
    val_close = y_val_raw = train_close_prices = val_close_prices = None
    if len(result) >= 11:
        (X_train, y_train, X_val, y_val, feature_scaler, features, metadata,
         val_close, y_val_raw, train_close_prices, val_close_prices) = result
    else:
        X_train, y_train, X_val, y_val, feature_scaler, features, metadata = result

    # Resolve lookback/horizon from metadata if not set
    lookback = lookback or metadata.get('lookback', 48)
    horizon = horizon or metadata.get('horizon', 12)

    y_train_raw = np.asarray(y_train, dtype=np.float32).copy()
    if y_val_raw is None:
        y_val_raw = np.asarray(y_val, dtype=np.float32).copy()

    sigma_train = sigma_val = sigma_val_raw = None
    if train_close_prices is not None and val_close_prices is not None:
        from lstm_config import LSTMConfig as _cfg
        from .simulation import compute_sigma_series, align_sigma
        window = getattr(_cfg, 'SIGNAL_WINDOW_SIGMA', 100)
        sigma_train_series = compute_sigma_series(train_close_prices, window)
        sigma_val_series = compute_sigma_series(val_close_prices, window)
        sigma_train = align_sigma(
            sigma_train_series, lookback, len(y_train_raw)).astype(np.float32)
        sigma_val = align_sigma(
            sigma_val_series, lookback, len(y_val_raw)).astype(np.float32)
        sigma_val_raw = sigma_val.copy()
        finite = sigma_train[np.isfinite(sigma_train)]
        if len(finite):
            fill = float(np.median(finite))
            sigma_train = np.where(np.isfinite(sigma_train), sigma_train, fill)
            sigma_val = np.where(np.isfinite(sigma_val), sigma_val, fill)

    # --- Target volatility normalisation (CP-175) ---
    target_mode = kwargs.get('target_mode', 'raw')
    if target_mode == 'vol_norm' and sigma_train is not None and sigma_val is not None:
        from .target_utils import normalize_target_by_volatility
        print(f"Normalising targets by volatility (target_mode='vol_norm')")
        y_train = normalize_target_by_volatility(y_train, sigma_train)
        y_val = normalize_target_by_volatility(y_val, sigma_val)

    # --- Shared postprocessing ---
    y_train, y_val, target_scaler, clip_bounds = _postprocess_targets(
        y_train, y_val, scale_target, target_scaler_type,
        winsorize_target, winsorize_percentile,
    )

    # F1.I4b: pre-training rounding (after scaling, before sequences/gradients)
    feature_round = kwargs.get('feature_round_decimals', 0)
    target_round = kwargs.get('target_round_decimals', 0)
    if feature_round and feature_round > 0:
        X_train = np.round(X_train, decimals=int(feature_round))
        X_val = np.round(X_val, decimals=int(feature_round))
    if target_round and target_round > 0:
        y_train = np.round(y_train, decimals=int(target_round))
        y_val = np.round(y_val, decimals=int(target_round))

    data, metadata = _build_result(
        X_train, y_train, X_val, y_val, feature_scaler,
        target_scaler, clip_bounds, features, lookback, horizon,
        metadata,
        val_close=val_close,
        y_val_raw=y_val_raw,
        y_train_raw=y_train_raw,
        sigma_train=sigma_train,
        sigma_val=sigma_val,
        sigma_val_raw=sigma_val_raw,
        val_close_prices=val_close_prices,
        scale_target=scale_target,
        target_scaler_type=target_scaler_type if scale_target else None,
        winsorize_target=winsorize_target,
        target_mode=target_mode,
    )

    return data, is_prepared, metadata
