"""
LSTM Price Prediction Training Script

Trains PyTorch LSTM neural networks to predict future asset prices using
leak-free in-sample sequence generation and modular training utilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.id_utils import get_next_model_id, resolve_data_id, to_repo_relative
from common.training_utils import (
    CryptoDataset,
    load_and_prepare_training_data,
    set_seed,
    setup_logger,
)
from common.visualization import plot_training_curves
from lstm_config import ALL_FEATURE_SETS, DEFAULT_EPOCHS, MODELS_DIR

warnings.filterwarnings('ignore')


class LSTMPriceModel(nn.Module):
    """LSTM neural network for continuous price target prediction."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super(LSTMPriceModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def _train_single_model(
    train_path: str,
    lookback: int,
    horizon: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    feature_set: str,
    use_log_prices: bool,
    early_stopping_patience: int,
    seed: int,
    optuna_config_name: str | None = None,
    custom_features: list[str] | None = None,
) -> dict:
    set_seed(seed)
    prep = load_and_prepare_training_data(
        data_path=train_path,
        feature_set_name=feature_set,
        lookback=lookback,
        horizon=horizon,
        val_split=0.2,
        use_log_prices=use_log_prices,
        seed=seed,
        custom_features=custom_features,
    )

    metadata = prep['metadata']
    symbol = metadata.get('symbol', 'eth')
    resolved_path = Path(prep['resolved_path'])
    # Directory name first: some archived datasets carry a stale ID in metadata (E101 -> 'E001').
    data_id = resolve_data_id(resolved_path, metadata)
    if custom_features:
        feature_set = 'custom'
    model_id = get_next_model_id(MODELS_DIR)
    date_short = datetime.now().strftime('%d%m')

    if optuna_config_name:
        run_name = f"{model_id}_{symbol}_model_{data_id}_{optuna_config_name}_{date_short}"
    else:
        run_name = f"{model_id}_{symbol}_model_{data_id}_{date_short}"

    base_dir = Path(MODELS_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / 'training.log')
    logger.info("=" * 70)
    logger.info(f"LSTM PRICE TRAINING: {model_id} | Data: {data_id} | Feature Set: {feature_set}")
    logger.info("=" * 70)
    logger.info(f"Lookback: {lookback} | Horizon: {horizon} | Epochs: {epochs} | Batch: {batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    X_train, y_train = prep['X_train'], prep['y_train']
    X_val, y_val = prep['X_val'], prep['y_val']
    features_list = prep['features_list']
    feature_scaler = prep['feature_scaler']
    price_scaler = prep['price_scaler']

    logger.info(f"Train samples: {len(X_train):,} | Val samples: {len(X_val):,}")
    logger.info(f"Input features ({len(features_list)}): {features_list[:5]}...")

    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMPriceModel(
        input_size=len(features_list),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * len(bx)

        train_loss = running_train_loss / len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device).unsqueeze(1)
                pred = model(bx)
                loss = criterion(pred, by)
                running_val_loss += loss.item() * len(bx)

        val_loss = running_val_loss / len(val_dataset)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1 or val_loss < best_val_loss:
            logger.info(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    # Save artifacts and model configs
    model_config = {
        'model_id': model_id,
        'data_id': data_id,
        'symbol': symbol,
        'lookback': int(lookback),
        'horizon': int(horizon),
        'hidden_size': int(hidden_size),
        'num_layers': int(num_layers),
        'dropout': float(dropout),
        'feature_set': feature_set,
        'features': features_list,
        # Repo-relative so that lstm_test_price.py can find the dataset on any machine.
        'train_path': to_repo_relative(resolved_path),
        'prepared_dir_name': resolved_path.name,
        'use_log_prices': bool(use_log_prices),
        # Dataset CSV features are pre-scaled when true; inference must then not scale them again.
        'scale_csv': bool(metadata.get('scalers', {}).get('scale_csv', True)),
        'epochs_requested': int(epochs),
        'epochs_trained': int(len(train_losses)),
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'batch_size': int(batch_size),
        'lr': float(lr),
        'weight_decay': float(weight_decay),
        'seed': int(seed),
    }

    with open(run_dir / 'model_config.json', 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)
    joblib.dump(model_config, run_dir / 'model_config.pkl')

    if price_scaler is not None:
        joblib.dump(price_scaler, run_dir / 'price_scaler.pkl')
    if feature_scaler is not None:
        joblib.dump(feature_scaler, run_dir / 'feature_scaler.pkl')

    plot_training_curves(train_losses, val_losses, best_val_loss, str(run_dir / 'loss_curve.png'))
    logger.info(f"Training completed successfully. Model artifacts saved to: {run_dir}")

    return {
        'model_id': model_id,
        'run_dir': str(run_dir),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'epochs_trained': len(train_losses),
    }


def load_optuna_training_params(path: str) -> dict:
    """Map a compatible price-study parameter file to canonical training arguments."""
    with open(path, encoding="utf-8") as stream:
        payload = json.load(stream)
    params = payload.get("params", payload)
    if params.get("use_batch_norm", True) is not True:
        raise ValueError("This study uses an unsupported price-model architecture.")
    if params.get("scheduler_type", "none") != "none":
        raise ValueError("This study uses a scheduler unsupported by final price training.")
    supported = {"hidden_size", "num_layers", "dropout", "batch_size", "weight_decay",
                 "lookback", "horizon", "lr", "seed", "use_log_prices", "early_stopping_patience"}
    result = {key: value for key, value in params.items() if key in supported}
    if "learning_rate" in params:
        result["lr"] = params["learning_rate"]
    metadata = payload.get("_metadata", {})
    features = metadata.get("custom_features")
    if features:
        result["custom_features"] = list(features)
        result["feature_set"] = "custom"
    elif metadata.get("feature_set"):
        result["feature_set"] = metadata["feature_set"]
    return result


def train_model(**kwargs) -> dict:
    """Wrapper entry point for training single or batch models."""
    parameter_file = kwargs.pop("load_optuna_params", None)
    if parameter_file:
        kwargs = {**load_optuna_training_params(parameter_file), **kwargs}
    return _train_single_model(
        train_path=kwargs.get('train_path', 'data_preparation/prepared_data/E109_prep_eth_universal_max_1m'),
        lookback=kwargs.get('lookback', 48),
        horizon=kwargs.get('horizon', 12),
        hidden_size=kwargs.get('hidden_size', 64),
        num_layers=kwargs.get('num_layers', 1),
        dropout=kwargs.get('dropout', 0.1),
        epochs=kwargs.get('epochs', DEFAULT_EPOCHS),
        batch_size=kwargs.get('batch_size', 256),
        lr=kwargs.get('lr', 0.001),
        weight_decay=kwargs.get('weight_decay', 1e-5),
        feature_set=kwargs.get('feature_set', 'max'),
        use_log_prices=kwargs.get('use_log_prices', True),
        early_stopping_patience=kwargs.get('early_stopping_patience', 25),
        seed=kwargs.get('seed', 42),
        custom_features=kwargs.get('custom_features'),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM Price Prediction model.")
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--load_optuna_params', type=str, default=None)
    parser.add_argument('--lookback', type=int, default=48)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--feature_set', type=str, default='max', choices=ALL_FEATURE_SETS)
    parser.add_argument('--use_log_prices', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--custom_features', type=str, nargs='+', default=None,
                        help="Explicit feature columns; implies --feature_set custom.")
    parser.add_argument('--early_stopping_patience', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    values = vars(args)
    if args.load_optuna_params:
        import sys
        explicit = {part.split("=", 1)[0] for part in sys.argv[1:] if part.startswith("--")}
        imported = load_optuna_training_params(args.load_optuna_params)
        values.update({key: value for key, value in imported.items()
                       if "--" + key not in explicit
                       and not (key == "use_log_prices" and "--no-use_log_prices" in explicit)})
        values.pop("load_optuna_params")
    train_model(**values)


if __name__ == "__main__":
    main()
