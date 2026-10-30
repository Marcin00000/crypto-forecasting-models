"""
Training utilities and data loaders for the LSTM price prediction pipeline.

Provides leak-free chronologic 80/20 train/validation splits, in-sample price scaler
fitting, sequence generation, PyTorch Dataset definitions, and reproducibility helpers.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import torch
from torch.utils.data import Dataset

from .data_utils import create_sequences
from .id_utils import resolve_dataset_path

__all__ = [
    'set_seed',
    'setup_logger',
    'CryptoDataset',
    'select_training_features',
    'load_and_prepare_training_data',
]


def set_seed(seed: int = 42) -> None:
    """Set global seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_file: str | Path | None = None) -> logging.Logger:
    """Configure standard training logger."""
    logger = logging.getLogger("LSTMPriceTrainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is not None:
        os.makedirs(os.path.dirname(str(log_file)), exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class CryptoDataset(Dataset):
    """PyTorch Dataset wrapping 3D sequences and 1D targets."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _resolve_prepared_dir(data_path: str) -> str:
    """Resolve prepared data directory from name or path (shared resolver, cwd-independent).

    A path to ``train.csv`` still belongs to the prepared directory: ``metadata.json``
    and the saved scalers live next to the file. Treating the file itself as the
    directory skips both and fits the price scaler with the wrong type.
    """
    if os.path.isfile(data_path):
        parent = os.path.dirname(data_path) or "."
        if os.path.isfile(os.path.join(parent, "metadata.json")):
            return parent
        return data_path
    if os.path.isdir(data_path):
        return data_path
    try:
        return str(resolve_dataset_path(data_path))
    except FileNotFoundError:
        return data_path


def select_training_features(
    df: pd.DataFrame,
    metadata: dict,
    feature_set_name: str,
    custom_features: list[str] | None = None,
) -> list[str]:
    """
    Strict feature selection: the requested set must be fully covered by the dataset.

    - the name equals ``metadata['feature_set']`` -> the dataset's own feature list
      (keeps the column order the dataset scalers were fitted with);
    - otherwise the set is taken from the root feature registry and every column must
      exist in the CSV. Silent truncation to the available subset would train a model
      on a different feature set than the one recorded in model_config.json.
    """
    from feature_engineering import get_feature_sets

    dataset_set = metadata.get('feature_set')
    dataset_features = [f for f in metadata.get('features', []) if f != 'close']

    if custom_features:
        wanted = [f for f in custom_features if f != 'close']
    elif feature_set_name == 'custom':
        raise ValueError("feature_set 'custom' requires --custom_features <name> [<name> ...].")
    elif dataset_features and feature_set_name == dataset_set:
        return dataset_features
    else:
        sets = get_feature_sets(include_buy_pressure='buy_pressure' in df.columns)
        if feature_set_name not in sets:
            raise ValueError(f"Unknown feature_set '{feature_set_name}'. Available: {sorted(sets)}")
        wanted = [f for f in sets[feature_set_name] if f != 'close']

    missing = [f for f in wanted if f not in df.columns]
    if missing:
        raise ValueError(
            f"feature_set '{feature_set_name}' requires columns missing from dataset "
            f"{metadata.get('data_id', 'UNKNOWN')} (dataset feature_set: '{dataset_set}', "
            f"missing: {missing}). Use --feature_set {dataset_set}."
        )
    return wanted


def load_and_prepare_training_data(
    data_path: str,
    feature_set_name: str,
    lookback: int,
    horizon: int,
    val_split: float = 0.2,
    use_log_prices: bool = True,
    seed: int = 42,
    custom_features: list[str] | None = None,
) -> dict[str, Any]:
    """
    Load training data from prepared directory CSVs and perform strict in-sample preparation.

    Enforces:
      1. Chronological row split 80% train / 20% val BEFORE creating sequences.
      2. No test data used during validation (test.csv left untouched).
      3. price_scaler.fit executed strictly on train_df['close'].
      4. Features from CSV not double-scaled when scale_csv=True.
      5. create_sequences called independently on train_df and val_df with n = len(df) - L - H.
    """
    resolved_path = _resolve_prepared_dir(data_path)
    metadata_path = os.path.join(resolved_path, 'metadata.json')
    metadata: dict = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    # 1. Load raw or prepared train.csv
    train_csv = os.path.join(resolved_path, 'train.csv')
    if os.path.exists(train_csv):
        df = pd.read_csv(train_csv)
    elif data_path.endswith('.csv') and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"Cannot find train.csv in {resolved_path} or as direct file.")

    # 2. Select features (strict: no silent truncation to the dataset's columns)
    features_list = select_training_features(df, metadata, feature_set_name, custom_features)

    if len(features_list) < 3:
        raise ValueError(f"Too few features selected ({len(features_list)}): {features_list}")

    # 3. Chronological split BEFORE sequence generation
    split_idx = int(len(df) * (1.0 - val_split))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    val_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # 4. Feature scaling
    scalers_meta = metadata.get('scalers', {})
    scale_csv = scalers_meta.get('scale_csv', True)
    feature_scaler = None
    if not scale_csv:
        # 'none' datasets (E112) are meant to be trained on unscaled features.
        if scalers_meta.get('features') != 'none':
            feature_scaler = StandardScaler()
            train_df[features_list] = feature_scaler.fit_transform(train_df[features_list])
            val_df[features_list] = feature_scaler.transform(val_df[features_list])
    else:
        # Load pre-saved feature scaler if available for packaging
        for s_name in ['scaler_features.pkl', 'feature_scaler.pkl']:
            s_path = os.path.join(resolved_path, s_name)
            if os.path.exists(s_path):
                feature_scaler = joblib.load(s_path)
                break

    # 5. Price log transformation
    if use_log_prices:
        train_df['close'] = np.log(np.maximum(train_df['close'].values, 1e-12))
        val_df['close'] = np.log(np.maximum(val_df['close'].values, 1e-12))

    # 6. Strict In-Sample Price Scaler Fit
    price_scaler_type = scalers_meta.get('price', 'robust')
    price_scaler = None
    if price_scaler_type not in ('none', 'raw'):
        if price_scaler_type == 'standard':
            price_scaler = StandardScaler()
        elif price_scaler_type == 'minmax':
            price_scaler = MinMaxScaler()
        else:
            price_scaler = RobustScaler()
        price_scaler.fit(train_df['close'].values.reshape(-1, 1))
        train_df['close'] = price_scaler.transform(train_df['close'].values.reshape(-1, 1)).flatten()
        val_df['close'] = price_scaler.transform(val_df['close'].values.reshape(-1, 1)).flatten()

    # 7. Independent sequence generation
    X_train, y_train, current_prices_train = create_sequences(train_df, features_list, lookback, horizon)
    X_val, y_val, current_prices_val = create_sequences(val_df, features_list, lookback, horizon)

    if X_train is None or len(X_train) == 0:
        raise ValueError("Insufficient training data to form sequences.")
    if X_val is None or len(X_val) == 0:
        raise ValueError("Insufficient validation data to form sequences.")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'current_prices_train': current_prices_train,
        'X_val': X_val,
        'y_val': y_val,
        'current_prices_val': current_prices_val,
        'features_list': features_list,
        'feature_scaler': feature_scaler,
        'price_scaler': price_scaler,
        'metadata': metadata,
        'resolved_path': resolved_path,
        'use_log_prices': use_log_prices,
    }
