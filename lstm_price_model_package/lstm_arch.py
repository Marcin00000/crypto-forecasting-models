"""
Standalone price-LSTM architecture for the inference package.

Faithful copy of ``LSTMPriceModel`` from ``lstm_model/lstm_train_price.py`` so that the
package can rebuild a trained model from ``model_config.json`` + ``best_model.pth``
without importing the training project. Layer names and shapes must stay identical to
the training class: ``load_state_dict`` matches parameters by name, and any drift would
silently corrupt predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


def build_model_from_config(config: dict) -> LSTMPriceModel:
    """Rebuild the network from model_config.json (same defaults as lstm_test_price.py)."""
    return LSTMPriceModel(
        input_size=len(config['features']),
        hidden_size=int(config.get('hidden_size', 64)),
        num_layers=int(config.get('num_layers', 1)),
        dropout=float(config.get('dropout', 0.1)),
    )
