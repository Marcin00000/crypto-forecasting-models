"""Optional output heads and FC building blocks for LSTM variants."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name):
    """Get activation module by name."""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.01),
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation: '{name}'. "
            f"Available: {list(activations.keys())}"
        )
    return activations[name]


def build_norm_layer(norm_type, size):
    """Build normalization layer for FC head by type."""
    if norm_type == 'batch':
        return nn.BatchNorm1d(size)
    if norm_type == 'layer':
        return nn.LayerNorm(size)
    if norm_type == 'none':
        return None
    raise ValueError(
        f"Unknown fc_norm: '{norm_type}'. Available: ['batch', 'layer', 'none']"
    )


class FCLayerBlock(nn.Module):
    """Linear → norm → activation → dropout, with optional residual add."""

    def __init__(self, in_size, out_size, fc_norm, activation, dropout):
        super().__init__()
        self.use_residual = in_size == out_size
        self.linear = nn.Linear(in_size, out_size)
        self.norm = build_norm_layer(fc_norm, out_size)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + x
        return out


class DualHeadOutput(nn.Module):
    """
    Direction/amplitude output head.

    The direction branch learns a signed component in [-1, 1], while the
    amplitude branch learns a non-negative magnitude. Their product remains a
    single log-return prediction, so the rest of the pipeline is unchanged.
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.direction = nn.Linear(input_size, 1)
        self.amplitude = nn.Linear(input_size, 1)

    def forward(self, x):
        direction = torch.tanh(self.direction(x))
        amplitude = F.softplus(self.amplitude(x))
        return direction * amplitude
