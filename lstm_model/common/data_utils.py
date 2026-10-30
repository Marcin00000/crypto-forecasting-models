"""
Common data loading and sequence generation utilities for the LSTM price model.

Enforces strict sequence length geometry (n = len(df) - lookback - horizon),
eliminating the legacy +1 offset, and ensures no test data leakage occurs into validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .id_utils import resolve_dataset_path

__all__ = [
    'create_sequences',
    'load_data',
    'prepare_raw_data',
    'detect_symbol',
]


def _resolve_prepared_dir(data_path: str) -> str:
    """Resolve prepared data directory from name, container, or path."""
    if os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, 'metadata.json')):
        return data_path

    if os.path.sep not in data_path and not os.path.exists(data_path):
        try:
            cand = str(resolve_dataset_path(data_path))
        except FileNotFoundError:
            cand = None
        if cand and os.path.exists(os.path.join(cand, 'metadata.json')):
            return cand

    if os.path.isdir(data_path):
        sub_dirs = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
            and os.path.exists(os.path.join(data_path, d, 'metadata.json'))
        ])
        if sub_dirs:
            return os.path.join(data_path, sub_dirs[-1])

    raise FileNotFoundError(f"Prepared-data directory not found: {data_path}")


def create_sequences(
    data: pd.DataFrame,
    features: list[str],
    lookback: int = 24,
    horizon: int = 1,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Create 3-D sequences for LSTM price prediction.

    Invariants:
      1. Formula for number of samples:
         n_samples = len(data) - lookback - horizon
         (Matches LR create_log_return_target, ending strictly at close[N-2]).
      2. For sample i with feature window [i, i + lookback):
         current_prices[i] = prices[i + lookback - 1] (decision candle t)
         y[i]              = prices[i + lookback + horizon - 1] (target candle t + horizon)
    """
    data_array = data[features].values.astype(np.float32)
    prices = data['close'].values.astype(np.float32)
    n_samples = len(data) - lookback - horizon

    if n_samples <= 0:
        return None, None, None

    X = np.zeros((n_samples, lookback, len(features)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    current_prices = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        X[i] = data_array[i:i + lookback]
        current_prices[i] = prices[i + lookback - 1]
        y[i] = prices[i + lookback + horizon - 1]

    return X, y, current_prices


def load_data(data_path: str, use_subset: bool = False, subset_size: int = 100000,
              feature_set: str | None = None,
              custom_features: list[str] | None = None) -> tuple[Any, bool, Any]:
    """
    Load data from prepared directory or direct CSV file.

    For prepared directories with train.csv this delegates to
    load_and_prepare_training_data() (the production loader that splits table rows
    before generating sequences); ``feature_set`` defaults to the dataset's own set.
    """
    if os.path.isdir(data_path) or (
        os.path.sep not in data_path
        and not data_path.endswith('.csv')
        and not os.path.exists(data_path)
    ):
        data_path = _resolve_prepared_dir(data_path)
        with open(os.path.join(data_path, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Check CSV priority first
        train_csv = os.path.join(data_path, 'train.csv')
        if os.path.exists(train_csv):
            from .training_utils import load_and_prepare_training_data
            data = load_and_prepare_training_data(
                data_path=data_path,
                feature_set_name=feature_set or metadata.get('feature_set', 'max'),
                lookback=int(metadata.get('lookback', 48)),
                horizon=int(metadata.get('horizon', 12)),
                use_log_prices=bool(metadata.get('scalers', {}).get('use_log_prices', True)),
                custom_features=custom_features,
            )
            return data, True, metadata

        # Legacy fallback: Train sequences from NPY (deprecated legacy format)
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))

        # Validation sequences: strictly split train (no test data in validation)
        split_idx = int(len(X_train) * 0.8)
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

        def _load_scaler(new_name: str, legacy_name: str):
            p = os.path.join(data_path, new_name)
            if not os.path.exists(p):
                p = os.path.join(data_path, legacy_name)
            return joblib.load(p)

        feature_scaler = _load_scaler('scaler_features.pkl', 'feature_scaler.pkl')
        price_scaler = _load_scaler('scaler_price.pkl', 'price_scaler.pkl')

        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_scaler': feature_scaler,
            'price_scaler': price_scaler,
        }
        return data, True, metadata

    elif str(data_path).endswith('.csv'):
        df = pd.read_csv(data_path)
        if use_subset:
            df = df.tail(subset_size).reset_index(drop=True)
        return df, False, None

    else:
        raise ValueError(
            f"Invalid data path: {data_path}\n"
            "Expected: prepared-data directory or .csv file."
        )


def prepare_raw_data(
    df: pd.DataFrame,
    features: dict | list | None,
    feature_scaler,
    price_scaler,
    use_log_prices: bool,
    lookback: int,
    horizon: int,
    val_split: float = 0.2,
):
    """Prepare raw OHLCV CSV data on-the-fly for legacy workflows."""
    from feature_engineering import prepare_features, select_features

    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    feature_set = features.get('set') if isinstance(features, dict) else None
    custom_features = features.get('custom') if isinstance(features, dict) else None
    validate = features.get('validate', True) if isinstance(features, dict) else True
    has_buy_pressure = 'buy_pressure' in df.columns

    features_list = select_features(
        df,
        feature_set=feature_set,
        custom_features=custom_features,
        validate=validate,
        include_buy_pressure=has_buy_pressure,
    )
    features_list = [f for f in features_list if f != 'close']

    if len(features_list) < 3:
        raise ValueError(f"Too few features: {len(features_list)}")

    split_idx = int(len(df) * (1.0 - val_split))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    if feature_scaler is None:
        feature_scaler = StandardScaler()
        train_df[features_list] = feature_scaler.fit_transform(train_df[features_list])
    else:
        train_df[features_list] = feature_scaler.transform(train_df[features_list])
    val_df[features_list] = feature_scaler.transform(val_df[features_list])

    if use_log_prices:
        train_df['close'] = np.log(np.maximum(train_df['close'].values, 1e-12))
        val_df['close'] = np.log(np.maximum(val_df['close'].values, 1e-12))

    if price_scaler is None:
        price_scaler = StandardScaler()
        train_df['close'] = price_scaler.fit_transform(train_df['close'].values.reshape(-1, 1)).flatten()
    else:
        train_df['close'] = price_scaler.transform(train_df['close'].values.reshape(-1, 1)).flatten()
    val_df['close'] = price_scaler.transform(val_df['close'].values.reshape(-1, 1)).flatten()

    X_train, y_train, _ = create_sequences(train_df, features_list, lookback, horizon)
    X_val, y_val, _ = create_sequences(val_df, features_list, lookback, horizon)

    return X_train, y_train, X_val, y_val, features_list, feature_scaler, price_scaler


def detect_symbol(filename: str) -> str:
    """Detect cryptocurrency symbol from filename."""
    filename_lower = filename.lower()
    for sym in ['btc', 'eth', 'bnb', 'sol', 'ada', 'xrp', 'doge', 'dot', 'matic', 'avax']:
        if sym in filename_lower:
            return sym
    return 'crypto'
