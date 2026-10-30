"""
Embedded inference for Price-LSTM submodels of the Signal fusion model.

The channel carries the predicted price P_hat in quote currency (the quantity the price
model predicts), not an implied log-return. Architecture and inversion are copies of
lstm_price_model_package/lstm_arch.py and target_inversion.py; layer names must match the
training class because load_state_dict matches parameters by name.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn


class _PriceLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lstm(x)[0][:, -1, :]
        return self.fc2(self.dropout(self.relu(self.bn1(self.fc1(out)))))


def _predicted_price(y_scaled: np.ndarray, price_scaler, use_log_prices: bool) -> np.ndarray:
    """Undo target scaling, then exp() when the network predicts log-prices."""
    y = np.asarray(y_scaled, dtype=np.float64).reshape(-1, 1)
    if price_scaler is not None:
        y = price_scaler.inverse_transform(y)
    y = y.flatten()
    return np.exp(y) if use_log_prices else y


class _PriceSubmodel:
    """Price LSTM producing P_hat for each Signal sample (window ending at the same candle)."""

    def __init__(self, model_dir: str, cfg: dict, device: torch.device):
        self.device = device
        self.features = list(cfg['features'])
        self.lookback = int(cfg['lookback'])
        self.use_log_prices = bool(cfg.get('use_log_prices', True))
        self.model = _PriceLSTM(
            input_size=len(self.features),
            hidden_size=int(cfg.get('hidden_size', 64)),
            num_layers=int(cfg.get('num_layers', 1)),
            dropout=float(cfg.get('dropout', 0.1)),
        )
        weights = next((os.path.join(model_dir, n) for n in ('best_model.pth', 'lstm_price_model.pth')
                        if os.path.exists(os.path.join(model_dir, n))), None)
        if weights is None:
            raise FileNotFoundError(
                f"Price submodel weights (best_model.pth / lstm_price_model.pth) not found in: {model_dir}")
        self.model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        self.model.to(device).eval()

        import joblib
        self.price_scaler = next(
            (joblib.load(os.path.join(model_dir, n)) for n in ('price_scaler.pkl', 'scaler_price.pkl')
             if os.path.exists(os.path.join(model_dir, n))), None)

    def predict(self, series: np.ndarray, n_samples: int, window: int,
                batch_size: int = 4096) -> np.ndarray:
        """P_hat for n_samples windows; series has n_samples + window - 1 rows."""
        if window < self.lookback:
            raise ValueError(
                f"Signal lookback ({window}) is shorter than the price submodel "
                f"lookback ({self.lookback}); the first samples would lack history.")
        if not np.isfinite(series).all():
            raise ValueError("price submodel input contains NaN/inf values")
        data = torch.from_numpy(np.ascontiguousarray(series[window - self.lookback:], dtype=np.float32))
        windows = data.unfold(0, self.lookback, 1).transpose(1, 2)
        if windows.shape[0] != n_samples:
            raise ValueError(f"price submodel window count {windows.shape[0]} != {n_samples} samples")
        preds = []
        with torch.inference_mode():
            for start in range(0, windows.shape[0], batch_size):
                x_b = windows[start:start + batch_size].to(self.device)
                preds.append(np.atleast_1d(self.model(x_b).squeeze(-1).cpu().numpy()))
        return _predicted_price(np.concatenate(preds), self.price_scaler, self.use_log_prices)
