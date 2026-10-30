"""
Data loaders for LSTM Log-Return pipeline.

Three format-specific loaders, each returning raw (X_train, y_train,
X_val, y_val, feature_scaler, metadata). Post-processing (winsorisation,
target scaling) is handled by the caller (load_data in data_utils.py).

Formats:
  1. Prepared CSV directory (train.csv + metadata.json)
  2. Prepared NPY directory (X_train.npy + metadata.json)
  3. Raw CSV file (feature engineering on-the-fly)
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from .data_utils import create_sequences, _resolve_features, _align_X_y
from .target_utils import create_log_return_target, load_precomputed_target


# =========================================================================
# PREPARED CSV
# =========================================================================

def load_prepared_csv(data_path, feature_set=None, custom_features=None,
                      lookback=None, horizon=None, val_source='split',
                      data_source='train',
                      min_coverage=1.0, allow_partial=False):
    """
    Load from prepared CSV directory (universal format from data_preparation).

    Expected: train.csv, test.csv, metadata.json, scaler_features.pkl.
    Target is created on-the-fly from close prices (log-return).

    Args:
        val_source: 'split' -- training data split 80/20 -> train/val
                    'test'  -- test.csv used as val
                    'agent' -- agent.csv used as val
        data_source: 'train' (default) -- load train.csv
                     'agent'           -- load agent.csv
        min_coverage: minimum fraction of requested features required
        allow_partial: downgrade missing feature error to warning

    Returns:
        X_train, y_train, X_val, y_val, feature_scaler, features, metadata,
        val_close, y_val_raw, train_close_prices, val_close_prices
        (val_close aligned to val sequences at i+lookback-1; y_val_raw
         before scaling; full close series for sigma computation)
    """
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    base_features = metadata['features']
    if lookback is None:
        lookback = metadata['lookback']
    if horizon is None:
        horizon = metadata['horizon']

    has_buy_pressure = metadata.get(
        'has_buy_pressure', 'buy_pressure' in base_features
    )
    features = _resolve_features(
        base_features, feature_set, custom_features, has_buy_pressure,
        min_coverage=min_coverage, allow_partial=allow_partial,
        data_id=metadata.get('data_id', 'UNKNOWN'),
        dataset_feature_set=metadata.get('feature_set'),
    )
    if not features:
        raise ValueError(
            f"No features selected (feature_set='{feature_set}', "
            f"base has {len(base_features)} features)"
        )

    # Load scaler
    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    feature_scaler = joblib.load(scaler_path)

    # Load training CSV (train.csv or agent split (agent.csv))
    data_source = data_source or 'train'
    train_file = 'agent.csv' if data_source == 'agent' else 'train.csv'
    train_csv_path = os.path.join(data_path, train_file)
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"{train_file} not found in {data_path}")
    train_df = pd.read_csv(train_csv_path)

    if val_source == 'test':
        test_csv = os.path.join(data_path, 'test.csv')
        if not os.path.exists(test_csv):
            raise FileNotFoundError(
                f"VAL_SOURCE='test' but test.csv not found in {data_path}"
            )
        val_df = pd.read_csv(test_csv)
    elif val_source == 'agent':
        agent_csv = os.path.join(data_path, 'agent.csv')
        if not os.path.exists(agent_csv):
            raise FileNotFoundError(
                f"VAL_SOURCE='agent' but agent.csv not found in {data_path}"
            )
        val_df = pd.read_csv(agent_csv)
    else:
        # Chronological 80/20 split
        split_idx = int(len(train_df) * 0.8)
        val_df = train_df.iloc[split_idx:].copy().reset_index(drop=True)
        train_df = train_df.iloc[:split_idx].copy().reset_index(drop=True)

    # Create log-return targets from close prices
    y_train = create_log_return_target(train_df, lookback, horizon)
    y_val = create_log_return_target(val_df, lookback, horizon)

    # Create 3D sequences (features only, no close price)
    X_train = create_sequences(train_df, features, lookback)
    X_val = create_sequences(val_df, features, lookback)
    X_train, y_train = _align_X_y(X_train, y_train)
    X_val, y_val = _align_X_y(X_val, y_val)

    metadata.update({
        'has_buy_pressure': has_buy_pressure,
        'val_source': val_source,
        'data_source': data_source,
    })

    n_val = len(y_val)
    if 'close' in val_df.columns and 'close' in train_df.columns:
        val_idx = np.arange(n_val) + (lookback - 1)
        val_idx = np.clip(val_idx, 0, len(val_df) - 1)
        val_close = val_df['close'].values[val_idx].astype(np.float64)
        train_close_prices = train_df['close'].values.astype(np.float64)
        val_close_prices = val_df['close'].values.astype(np.float64)
    else:
        # Graceful fallback: signal-aware training (val_signal checkpoint,
        # sigma-based losses) is simply disabled downstream when these are
        # None; close-based target creation above already requires 'close'
        # for y_train/y_val, so this branch only guards against unexpected
        # column layouts feeding sigma/simulation-only consumers.
        val_close = None
        train_close_prices = None
        val_close_prices = None
    y_val_raw = np.asarray(y_val, dtype=np.float32).copy()

    return (X_train, y_train, X_val, y_val, feature_scaler, features, metadata,
            val_close, y_val_raw, train_close_prices, val_close_prices)


# =========================================================================
# PREPARED NPY
# =========================================================================

def load_prepared_npy(data_path, lookback=None, horizon=None,
                      val_source='split'):
    """
    Load from prepared NPY directory.

    Target is created from OHLCV data (NOT from old saved y_train.npy which
    contains price targets). Placeholder exists for future pre-computed targets.

    Expected: X_train.npy, metadata.json, and train.csv for target computation.

    Returns:
        X_train, y_train, X_val, y_val, feature_scaler, features, metadata
    """
    if val_source not in ('split', 'test'):
        raise ValueError(
            f"NPY loader supports val_source 'split' or 'test', got {val_source!r}. "
            "Use the CSV loader for 'agent'."
        )

    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    features = metadata['features']
    if lookback is None:
        lookback = metadata['lookback']
    if horizon is None:
        horizon = metadata['horizon']

    # --- Attempt to load pre-computed log-return target (placeholder) ---
    precomputed_y_train = load_precomputed_target(data_path, 'train')

    if precomputed_y_train is not None:
        # Reserved path: use the saved log-return target once it is available.
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        y_train = precomputed_y_train

        # Validation data
        if val_source == 'test':
            test_csv = os.path.join(data_path, 'test.csv')
            if os.path.exists(test_csv):
                val_df = pd.read_csv(test_csv)
                X_val = create_sequences(val_df, features, lookback)
                y_val = create_log_return_target(val_df, lookback, horizon)
                X_val, y_val = _align_X_y(X_val, y_val)
            else:
                raise FileNotFoundError(
                    f"VAL_SOURCE='test' but test.csv not found in {data_path}"
                )
        else:
            # Split precomputed sequences 80/20
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
    else:
        # Create target from CSV (OHLCV close prices)
        train_csv = os.path.join(data_path, 'train.csv')
        if os.path.exists(train_csv):
            train_df = pd.read_csv(train_csv)

            if val_source == 'test':
                test_csv = os.path.join(data_path, 'test.csv')
                if os.path.exists(test_csv):
                    val_df = pd.read_csv(test_csv)
                else:
                    raise FileNotFoundError(
                        f"VAL_SOURCE='test' but test.csv not found in {data_path}"
                    )
            else:
                # Chronological 80/20 split on rows BEFORE creating sequences
                split_idx = int(len(train_df) * 0.8)
                val_df = train_df.iloc[split_idx:].copy().reset_index(drop=True)
                train_df = train_df.iloc[:split_idx].copy().reset_index(drop=True)

            y_train = create_log_return_target(train_df, lookback, horizon)
            X_train = create_sequences(train_df, features, lookback)
            X_train, y_train = _align_X_y(X_train, y_train)

            y_val = create_log_return_target(val_df, lookback, horizon)
            X_val = create_sequences(val_df, features, lookback)
            X_val, y_val = _align_X_y(X_val, y_val)
        else:
            raise FileNotFoundError(
                f"Cannot create log-return target: no train.csv in {data_path}.\n"
                "NPY target (y_train.npy) contains price predictions from old LSTM.\n"
                "Re-run data_preparation with CSV output, or wait for log-return "
                "target support in data_preparation."
            )

    # Load feature scaler
    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    feature_scaler = (joblib.load(scaler_path)
                      if os.path.exists(scaler_path) else None)

    metadata.update({'val_source': val_source})

    return X_train, y_train, X_val, y_val, feature_scaler, features, metadata


# =========================================================================
# RAW CSV
# =========================================================================

def prepare_raw_csv(csv_path, feature_set=None, custom_features=None,
                    lookback=48, horizon=12, val_source='split'):
    """
    Load raw CSV and prepare data on-the-fly (feature engineering + scaling).

    Always uses chronological 80/20 split for train/val — a raw CSV has no
    separate test.csv, so val_source='test' is not applicable. The parameter
    is accepted for signature parity with the prepared loaders; a warning is
    emitted when a non-'split' value is passed and the actual mode used is
    recorded in metadata['val_source'].

    Args:
        csv_path:        Path to raw .csv file
        feature_set:     Name of feature set (from feature_engineering)
        custom_features: Explicit feature list (overrides feature_set)
        lookback:        Sequence length
        horizon:         Prediction horizon
        val_source:      Accepted for API parity; effective mode is always
                         'split' for raw CSV.

    Returns:
        X_train, y_train, X_val, y_val, feature_scaler, features, metadata
    """
    from feature_engineering import prepare_features, select_features

    if val_source not in ('split', 'test'):
        raise ValueError(
            f"val_source must be 'split' or 'test', got: {val_source!r}"
        )
    if val_source != 'split':
        print(
            f"Warning: val_source={val_source!r} is not applicable for raw "
            f"CSV (no separate test.csv); falling back to chronological "
            f"80/20 split."
        )
    effective_val_source = 'split'

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from CSV")

    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"After feature engineering: {len(df)} samples")

    if len(df) < lookback + horizon + 100:
        raise ValueError(
            f"Insufficient data after FE: {len(df)} samples. "
            f"Need at least {lookback + horizon + 100}."
        )

    has_buy_pressure = 'buy_pressure' in df.columns
    if custom_features is not None:
        features = [f for f in custom_features if f in df.columns]
    elif feature_set is not None:
        features = select_features(
            df, feature_set=feature_set, validate=False,
            include_buy_pressure=has_buy_pressure,
        )
    else:
        features = select_features(
            df, validate=False, include_buy_pressure=has_buy_pressure,
        )
    features = [f for f in features if f != 'close']

    if len(features) < 3:
        raise ValueError(f"Too few features selected: {len(features)}")

    # Chronological split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Fit scaler on train only
    feature_scaler = StandardScaler()
    train_df[features] = feature_scaler.fit_transform(train_df[features])
    val_df[features] = feature_scaler.transform(val_df[features])

    # Create targets and sequences
    y_train = create_log_return_target(train_df, lookback, horizon)
    y_val = create_log_return_target(val_df, lookback, horizon)
    X_train = create_sequences(train_df, features, lookback)
    X_val = create_sequences(val_df, features, lookback)
    X_train, y_train = _align_X_y(X_train, y_train)
    X_val, y_val = _align_X_y(X_val, y_val)

    metadata = {
        'model_type': 'lstm_logreturn',
        'target_type': 'log_return',
        'has_buy_pressure': has_buy_pressure,
        'data_source': 'raw_csv',
        'source_file': os.path.basename(csv_path),
        'val_source': effective_val_source,
    }

    return X_train, y_train, X_val, y_val, feature_scaler, features, metadata
