"""
Standalone LSTM architecture for the inference package.

This is a self-contained copy of ``lstm_logreturn_model/common/model.py`` so
the package can rebuild a trained model from ``model_config.json`` +
``lstm_model.pth`` WITHOUT importing anything from the training project. Keep
the forward pass byte-for-byte equivalent to the training model — any drift
here silently corrupts predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# HELPERS (copy of common/model_variants.py)
# ===========================================================================

def _get_activation(name: str) -> nn.Module:
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.01),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: '{name}'")
    return activations[name]


def _build_norm_layer(norm_type: str, size: int):
    if norm_type == 'batch':
        return nn.BatchNorm1d(size)
    if norm_type == 'layer':
        return nn.LayerNorm(size)
    if norm_type == 'none':
        return None
    raise ValueError(f"Unknown fc_norm: '{norm_type}'")


class FCLayerBlock(nn.Module):
    """Linear -> norm -> activation -> dropout, with optional residual add."""

    def __init__(self, in_size, out_size, fc_norm, activation, dropout):
        super().__init__()
        self.use_residual = in_size == out_size
        self.linear = nn.Linear(in_size, out_size)
        self.norm = _build_norm_layer(fc_norm, out_size)
        self.activation = _get_activation(activation)
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
    """Direction/amplitude output head (tanh * softplus)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.direction = nn.Linear(input_size, 1)
        self.amplitude = nn.Linear(input_size, 1)

    def forward(self, x):
        direction = torch.tanh(self.direction(x))
        amplitude = F.softplus(self.amplitude(x))
        return direction * amplitude


# ===========================================================================
# ATTENTION
# ===========================================================================

class FeatureAttention(nn.Module):
    """Squeeze-and-excitation gate for per-feature reweighting before RNN."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg = x.mean(dim=1)
        weights = self.gate(avg)
        return x * weights.unsqueeze(1)


class TemporalAttention(nn.Module):
    """Multi-head temporal attention (identical to the training model)."""

    def __init__(self, input_size: int, n_heads: int = 1):
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if input_size % n_heads != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by "
                f"n_heads ({n_heads})")
        self.n_heads = int(n_heads)
        self.input_size = int(input_size)
        self.head_dim = input_size // n_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, self.head_dim),
                nn.Tanh(),
                nn.Linear(self.head_dim, 1),
            )
            for _ in range(self.n_heads)
        ])
        self.content_heads = nn.ModuleList([
            nn.Linear(input_size, self.head_dim)
            for _ in range(self.n_heads)
        ])
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, input_size)

    def forward(self, rnn_output):
        contexts = []
        for i, head in enumerate(self.heads):
            scores = head(rnn_output)
            weights = torch.softmax(scores, dim=1)
            head_proj = self.content_heads[i](rnn_output)
            ctx = (head_proj * weights).sum(dim=1)
            contexts.append(ctx)
        concat = torch.cat(contexts, dim=-1)
        return self.out_proj(concat)


# ===========================================================================
# MODEL
# ===========================================================================

class LSTMLogReturnModel(nn.Module):
    """Configurable LSTM/GRU + attention + FC head (inference copy)."""

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, bidirectional=False, use_attention=False,
                 fc_layers=None, activation='relu', fc_norm='layer',
                 rnn_type='lstm', use_input_norm=False,
                 attention_heads=1, lstm_dropout=None, fc_dropout=None,
                 arch_type='legacy', use_feature_attention=False,
                 use_rnn_norm=False, pooling_mode='last',
                 use_conv1d_front=False, conv1d_kernel=5,
                 use_fc_residual=False, use_learnable_output_scale=False):
        super().__init__()
        if arch_type not in ('legacy', 'dual_head'):
            raise ValueError(
                f"Unknown arch_type: '{arch_type}'. "
                "Available: ['legacy', 'dual_head']")
        self.arch_type = arch_type
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_input_norm = bool(use_input_norm)
        self.attention_heads = int(attention_heads)

        self.lstm_dropout = (
            float(lstm_dropout) if lstm_dropout is not None else float(dropout))
        self.fc_dropout = (
            float(fc_dropout) if fc_dropout is not None else float(dropout))

        if self.use_input_norm:
            self.input_norm = nn.LayerNorm(input_size)
        else:
            self.input_norm = None

        self.use_feature_attention = bool(use_feature_attention)
        if self.use_feature_attention:
            self.feature_attention = FeatureAttention(input_size)
        else:
            self.feature_attention = None

        self.use_rnn_norm = bool(use_rnn_norm)
        self.pooling_mode = pooling_mode
        if pooling_mode not in ('last', 'mean', 'max'):
            raise ValueError(
                f"Unknown pooling_mode: '{pooling_mode}'. "
                "Available: ['last', 'mean', 'max']")
        self.use_fc_residual = bool(use_fc_residual)
        self.use_learnable_output_scale = bool(use_learnable_output_scale)

        if bool(use_conv1d_front):
            kernel = int(conv1d_kernel)
            if kernel < 1:
                raise ValueError(f"conv1d_kernel must be >= 1, got {kernel}")
            self.conv1d_front = nn.Conv1d(
                input_size, input_size, kernel_size=kernel,
                padding=kernel // 2,
            )
            self.conv1d_activation = _get_activation(activation)
        else:
            self.conv1d_front = None
            self.conv1d_activation = None

        rnn_dropout = self.lstm_dropout if num_layers > 1 else 0
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=rnn_dropout,
                               bidirectional=bidirectional)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=rnn_dropout,
                              bidirectional=bidirectional)
        else:
            raise ValueError(f"Unknown rnn_type: '{rnn_type}'")

        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        if self.use_rnn_norm:
            self.rnn_norm = nn.LayerNorm(rnn_output_size)
        else:
            self.rnn_norm = None

        if use_attention:
            self.attention = TemporalAttention(
                rnn_output_size, n_heads=self.attention_heads)
        else:
            self.attention = None

        self.fc_norm = fc_norm
        fc_sizes = fc_layers if fc_layers else [rnn_output_size // 2]
        prev_size = rnn_output_size
        if self.use_fc_residual:
            self.fc_blocks = nn.ModuleList([
                FCLayerBlock(
                    prev_size if i == 0 else fc_sizes[i - 1],
                    fc_size, fc_norm, activation, self.fc_dropout,
                )
                for i, fc_size in enumerate(fc_sizes)
            ])
            prev_size = fc_sizes[-1] if fc_sizes else rnn_output_size
            self.fc = None
            if self.arch_type == 'legacy':
                self.fc_out = nn.Linear(prev_size, 1)
                self.output_head = None
            else:
                self.fc_out = None
                self.output_head = DualHeadOutput(prev_size)
        else:
            self.fc_blocks = None
            self.fc_out = None
            layers = []
            for fc_size in fc_sizes:
                layers.append(nn.Linear(prev_size, fc_size))
                norm = _build_norm_layer(fc_norm, fc_size)
                if norm is not None:
                    layers.append(norm)
                layers.append(_get_activation(activation))
                layers.append(nn.Dropout(self.fc_dropout))
                prev_size = fc_size
            if self.arch_type == 'legacy':
                layers.append(nn.Linear(prev_size, 1))
                self.fc = nn.Sequential(*layers)
                self.output_head = None
            else:
                self.fc = nn.Sequential(*layers)
                self.output_head = DualHeadOutput(prev_size)

        if self.use_learnable_output_scale:
            self.output_scale = nn.Parameter(torch.ones(1))
            self.output_bias = nn.Parameter(torch.zeros(1))
        else:
            self.output_scale = None
            self.output_bias = None

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.feature_attention is not None:
            x = self.feature_attention(x)
        if self.conv1d_front is not None:
            x = self.conv1d_front(x.transpose(1, 2)).transpose(1, 2)
            x = self.conv1d_activation(x)
        rnn_out, _ = self.rnn(x)

        if self.rnn_norm is not None:
            rnn_out = self.rnn_norm(rnn_out)

        if self.attention is not None:
            context = self.attention(rnn_out)
        elif self.pooling_mode == 'last':
            context = rnn_out[:, -1, :]
        elif self.pooling_mode == 'mean':
            context = rnn_out.mean(dim=1)
        else:
            context = rnn_out.max(dim=1).values

        if self.use_fc_residual:
            h = context
            for block in self.fc_blocks:
                h = block(h)
            if self.output_head is not None:
                out = self.output_head(h)
            else:
                out = self.fc_out(h)
        else:
            features = self.fc(context)
            if self.output_head is not None:
                out = self.output_head(features)
            else:
                out = features

        if self.output_scale is not None:
            out = out * self.output_scale + self.output_bias
        return out


def build_model_from_config(config: dict, input_size: int) -> LSTMLogReturnModel:
    """Reconstruct the architecture described by a model_config.json dict."""
    return LSTMLogReturnModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False),
        use_attention=config.get('use_attention', False),
        # Same default as lstm_test.py: a config without fc_layers was built with [64].
        fc_layers=config.get('fc_layers', [64]),
        activation=config.get('activation', 'relu'),
        fc_norm=config.get('fc_norm', 'layer'),
        rnn_type=config.get('rnn_type', 'lstm'),
        use_input_norm=config.get('use_input_norm', False),
        attention_heads=config.get('attention_heads', 1),
        lstm_dropout=config.get('lstm_dropout'),
        fc_dropout=config.get('fc_dropout'),
        arch_type=config.get('arch_type', 'legacy'),
        use_feature_attention=config.get('use_feature_attention', False),
        use_rnn_norm=config.get('use_rnn_norm', False),
        pooling_mode=config.get('pooling_mode', 'last'),
        use_conv1d_front=config.get('use_conv1d_front', False),
        conv1d_kernel=config.get('conv1d_kernel', 5),
        use_fc_residual=config.get('use_fc_residual', False),
        use_learnable_output_scale=config.get(
            'use_learnable_output_scale', False),
    )
