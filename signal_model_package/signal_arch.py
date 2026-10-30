"""
SignalModel architecture and neural building blocks for signal_model_package.
Pure inference implementation supporting all canonical variants:
- Standalone DualHead (CP-162): direction (tanh) * strength (softplus) + bias
- Standalone Legacy: Linear head with learnable scale/bias
- Submodel Fusion: context + submodel predictions via BatchNorm1d/ZScore projection
Self-contained, without training-specific dependencies.
"""
from __future__ import annotations

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# ACTIVATIONS & NORMALISATION
# =========================================================================

def get_activation(name: str) -> nn.Module:
    name_clean = str(name).lower()
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.01),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),
        'mish': nn.Mish(),
    }
    if name_clean not in activations:
        raise ValueError(f"Unknown activation: '{name}'. Available: {list(activations.keys())}")
    return activations[name_clean]


def build_norm_layer(norm_type: str, size: int) -> nn.Module | None:
    norm_clean = str(norm_type).lower()
    if norm_clean == 'batch':
        return nn.BatchNorm1d(size)
    if norm_clean == 'layer':
        return nn.LayerNorm(size)
    if norm_clean in ('none', '0', 'false'):
        return None
    raise ValueError(f"Unknown fc_norm: '{norm_type}'. Available: ['batch', 'layer', 'none']")


# =========================================================================
# FC RESIDUAL BLOCK
# =========================================================================

class FCLayerBlock(nn.Module):
    """Linear -> norm -> activation -> dropout, with residual add when in_size == out_size."""

    def __init__(self, in_size: int, out_size: int, fc_norm: str, activation: str, dropout: float = 0.0):
        super().__init__()
        self.use_residual = (in_size == out_size)
        self.linear = nn.Linear(in_size, out_size)
        self.norm = build_norm_layer(fc_norm, out_size)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + x
        return out


# =========================================================================
# ATTENTION MECHANISMS
# =========================================================================

class FeatureAttention(nn.Module):
    """Squeeze-and-excitation gate for dynamic feature weighting before RNN."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1)
        weights = self.gate(avg)
        return x * weights.unsqueeze(1)


class TemporalAttention(nn.Module):
    """Multi-head temporal attention over the RNN sequence output."""

    def __init__(self, input_size: int, n_heads: int = 1, use_content_heads: bool = True):
        super().__init__()
        # Same checks as lstm_signal_model/common/model.py: floor division would
        # silently shrink head_dim and produce a different (unloadable) layout.
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if input_size % n_heads != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.n_heads = int(n_heads)
        self.input_size = int(input_size)
        self.head_dim = input_size // self.n_heads
        self.use_content_heads = bool(use_content_heads)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, self.head_dim),
                nn.Tanh(),
                nn.Linear(self.head_dim, 1),
            )
            for _ in range(self.n_heads)
        ])
        if self.use_content_heads:
            self.content_heads = nn.ModuleList([
                nn.Linear(input_size, self.head_dim)
                for _ in range(self.n_heads)
            ])
            self.out_proj = nn.Linear(self.n_heads * self.head_dim, input_size)
        else:
            self.content_heads = None
            self.out_proj = nn.Linear(input_size, input_size)

    def forward(self, rnn_output: torch.Tensor) -> torch.Tensor:
        contexts = []
        for i, head in enumerate(self.heads):
            scores = head(rnn_output)
            weights = torch.softmax(scores, dim=1)
            if self.content_heads is not None:
                head_proj = self.content_heads[i](rnn_output)
                ctx = (head_proj * weights).sum(dim=1)
            else:
                ctx = (rnn_output * weights).sum(dim=1)
            contexts.append(ctx)
        concat = torch.cat(contexts, dim=-1) if self.content_heads is not None else contexts[0]
        return self.out_proj(concat)


# =========================================================================
# OUTPUT HEADS (DUALHEAD & FUSION)
# =========================================================================

class DualHeadOutput(nn.Module):
    """
    Direction and magnitude output head for Signal Model (CP-162).
    Direction: tanh(Linear(h)) in [-1, 1].
    Strength: softplus(Linear(h)) in [0, inf).
    """

    def __init__(self, in_size: int, init_bias: float = -2.944):
        super().__init__()
        self.direction = nn.Linear(in_size, 1, bias=False)
        self.strength = nn.Linear(in_size, 1, bias=False)
        self.bias = nn.Parameter(torch.full((2,), float(init_bias)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.tanh(self.direction(x))
        s = F.softplus(self.strength(x))
        buy_logit = d * s + self.bias[0]
        sell_logit = -d * s + self.bias[1]
        return torch.cat([buy_logit, sell_logit], dim=-1)


class SubmodelFixedZScore(nn.Module):
    """Fixed Z-score scaling for submodel return predictions (1 / target_std)."""

    def __init__(self, target_std: float = 0.001):
        super().__init__()
        self.scale = 1.0 / max(float(target_std), 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SubmodelFusionHead(nn.Module):
    """Fusion head combining primary RNN context with submodel predictions."""

    def __init__(self, primary_size: int, submodel_size: int, out_features: int = 2,
                 use_submodel_norm: bool = True, submodel_norm_type: str = 'auto'):
        super().__init__()
        self.submodel_size = int(submodel_size)
        self.use_submodel_norm = bool(use_submodel_norm)
        norm_type = str(submodel_norm_type or 'auto').lower()

        if not self.use_submodel_norm or norm_type in ('none', '0', 'false'):
            self.submodel_norm = None
        elif norm_type == 'batch' or norm_type == 'auto':
            self.submodel_norm = nn.BatchNorm1d(self.submodel_size)
        elif norm_type == 'zscore':
            self.submodel_norm = SubmodelFixedZScore(target_std=0.001)
        elif norm_type == 'layer':
            self.submodel_norm = nn.LayerNorm(self.submodel_size)
        else:
            self.submodel_norm = nn.BatchNorm1d(self.submodel_size)

        self.fusion_dim = primary_size + self.submodel_size
        self.projection = nn.Linear(self.fusion_dim, primary_size)
        self.activation = nn.GELU()
        self.output_linear = nn.Linear(primary_size, out_features)

    def forward(self, primary_ctx: torch.Tensor, submodel_preds: torch.Tensor) -> torch.Tensor:
        if self.submodel_norm is not None:
            submodel_preds = self.submodel_norm(submodel_preds)
        fused = torch.cat([primary_ctx, submodel_preds], dim=-1)
        projected = self.activation(self.projection(fused))
        return self.output_linear(projected)


# =========================================================================
# MAIN SIGNAL MODEL
# =========================================================================

class SignalModel(nn.Module):
    """
    LSTM/GRU + attention/pooling + FC -> dual sigmoid probabilities [p_buy, p_sell].
    Self-contained pure inference PyTorch module.
    """

    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.35, bidirectional: bool = False, use_attention: bool = True,
                 fc_layers: list[int] | None = None, activation: str = 'gelu', fc_norm: str = 'layer',
                 rnn_type: str = 'lstm', use_input_norm: bool = True,
                 attention_heads: int = 1, lstm_dropout: float | None = None, fc_dropout: float | None = None,
                 use_feature_attention: bool = False, output_mode: str = 'dual', arch_type: str = 'legacy',
                 submodel_size: int = 0, use_submodel_norm: bool = True, submodel_norm_type: str = 'auto',
                 use_conv1d_front: bool = False, conv1d_kernel: int = 5, pooling_mode: str = 'last',
                  use_fc_residual: bool = False, use_rnn_norm: bool = False,
                  use_learnable_output_scale: bool = False,
                  use_content_heads: bool = True, **kwargs):
        super().__init__()
        self.arch_type = str(arch_type or 'legacy').lower()
        self.output_mode = str(output_mode or 'dual').lower()
        self.rnn_type = str(rnn_type or 'lstm').lower()
        self.bidirectional = bool(bidirectional)
        self.use_attention = bool(use_attention)
        self.use_input_norm = bool(use_input_norm)
        self.pooling_mode = str(pooling_mode or 'last').lower()
        self.use_fc_residual = bool(use_fc_residual)
        self.use_rnn_norm = bool(use_rnn_norm)
        self.use_learnable_output_scale = bool(use_learnable_output_scale)
        self.submodel_size = int(submodel_size or 0)

        # 1. Input Norm & Feature Attention
        self.input_norm = nn.LayerNorm(input_size) if self.use_input_norm else None
        self.feature_attention = FeatureAttention(input_size) if use_feature_attention else None

        # 2. Conv1D Front-end
        if bool(use_conv1d_front):
            kernel = int(conv1d_kernel)
            self.conv1d_front = nn.Conv1d(input_size, input_size, kernel_size=kernel, padding=kernel // 2)
            self.conv1d_act = get_activation(activation)
        else:
            self.conv1d_front = None
            self.conv1d_act = None

        # 3. RNN
        rnn_dropout = (float(lstm_dropout) if lstm_dropout is not None else float(dropout)) if num_layers > 1 else 0.0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                               dropout=rnn_dropout, bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                              dropout=rnn_dropout, bidirectional=self.bidirectional)
        else:
            raise ValueError(f"Unknown rnn_type: '{self.rnn_type}'")

        rnn_output_size = hidden_size * (2 if self.bidirectional else 1)
        self.rnn_norm = nn.LayerNorm(rnn_output_size) if self.use_rnn_norm else None

        # 4. Temporal Attention / Pooling
        if self.use_attention:
            self.attention = TemporalAttention(
                rnn_output_size, n_heads=int(attention_heads or 1), use_content_heads=bool(use_content_heads)
            )
        else:
            self.attention = None

        # 5. FC Head
        fc_sizes = fc_layers if fc_layers else [hidden_size // 2]
        fc_drop = float(fc_dropout) if fc_dropout is not None else float(dropout)
        n_out = 2 if self.output_mode == 'dual' else 1

        if self.use_fc_residual:
            self.fc_blocks = nn.ModuleList()
            prev_size = rnn_output_size
            for fc_size in fc_sizes:
                self.fc_blocks.append(FCLayerBlock(prev_size, fc_size, fc_norm, activation, fc_drop))
                prev_size = fc_size
            self.fc = None
        else:
            self.fc_blocks = None
            layers = []
            prev_size = rnn_output_size
            for fc_size in fc_sizes:
                layers.append(nn.Linear(prev_size, fc_size))
                norm = build_norm_layer(fc_norm, fc_size)
                if norm is not None:
                    layers.append(norm)
                layers.append(get_activation(activation))
                layers.append(nn.Dropout(fc_drop))
                prev_size = fc_size
            self.fc = nn.Sequential(*layers)

        self.output_linear = nn.Linear(prev_size, n_out)

        # 6. Output Architecture
        if self.arch_type == 'submodel_fusion' and self.submodel_size > 0:
            self.fusion_head = SubmodelFusionHead(
                primary_size=prev_size,
                submodel_size=self.submodel_size,
                out_features=n_out,
                use_submodel_norm=bool(use_submodel_norm),
                submodel_norm_type=str(submodel_norm_type or 'auto'),
            )
            self.dual_head = None
        elif self.arch_type == 'dual_head':
            self.fusion_head = None
            self.dual_head = DualHeadOutput(prev_size)
        else:  # 'legacy'
            self.fusion_head = None
            self.dual_head = None

        # 7. Learnable output scale + bias
        if self.use_learnable_output_scale:
            self.output_scale = nn.Parameter(torch.ones(n_out))
            self.output_bias = nn.Parameter(torch.zeros(n_out))
        else:
            self.output_scale = None
            self.output_bias = None

    def forward(self, x: torch.Tensor, submodel_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.feature_attention is not None:
            x = self.feature_attention(x)

        if self.conv1d_front is not None:
            x = self.conv1d_front(x.transpose(1, 2)).transpose(1, 2)
            x = self.conv1d_act(x)

        rnn_out, _ = self.rnn(x)

        if self.rnn_norm is not None:
            rnn_out = self.rnn_norm(rnn_out)

        if self.attention is not None:
            context = self.attention(rnn_out)
        elif self.pooling_mode == 'mean':
            context = rnn_out.mean(dim=1)
        elif self.pooling_mode == 'max':
            context = rnn_out.max(dim=1).values
        else:
            context = rnn_out[:, -1, :]

        if self.use_fc_residual:
            h = context
            for block in self.fc_blocks:
                h = block(h)
        else:
            h = self.fc(context)

        if self.arch_type == 'submodel_fusion':
            if submodel_inputs is not None:
                logits = self.fusion_head(h, submodel_inputs)
            else:
                warnings.warn("arch_type='submodel_fusion' but submodel_inputs is None — fallback to output_linear.", stacklevel=2)
                logits = self.output_linear(h)
        elif self.arch_type == 'dual_head':
            logits = self.dual_head(h)
        else:
            logits = self.output_linear(h)

        if self.output_scale is not None:
            logits = logits * self.output_scale + self.output_bias

        return torch.sigmoid(logits)


# =========================================================================
# FACTORY
# =========================================================================

def build_signal_model_from_config(config: dict, input_size: int, use_content_heads: bool = True) -> SignalModel:
    """Instantiate SignalModel from resolved model_config.json dict."""
    return SignalModel(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 32),
        num_layers=config.get('num_layers', 1),
        dropout=config.get('dropout', 0.35),
        bidirectional=config.get('bidirectional', False),
        use_attention=config.get('use_attention', True),
        fc_layers=config.get('fc_layers', [32]),
        activation=config.get('activation', 'gelu'),
        fc_norm=config.get('fc_norm', 'layer'),
        rnn_type=config.get('rnn_type', 'lstm'),
        use_input_norm=config.get('use_input_norm', True),
        attention_heads=config.get('attention_heads', 1),
        lstm_dropout=config.get('lstm_dropout'),
        fc_dropout=config.get('fc_dropout'),
        use_feature_attention=config.get('use_feature_attention', False),
        output_mode=config.get('output_mode', 'dual'),
        arch_type=config.get('arch_type', 'legacy'),
        submodel_size=config.get('submodel_size', 0),
        use_submodel_norm=config.get('use_submodel_norm', True),
        submodel_norm_type=config.get('submodel_norm_type', 'auto'),
        use_conv1d_front=config.get('use_conv1d_front', False),
        conv1d_kernel=config.get('conv1d_kernel', 5),
        pooling_mode=config.get('pooling_mode', 'last'),
        use_fc_residual=config.get('use_fc_residual', False),
        use_rnn_norm=config.get('use_rnn_norm', False),
        use_learnable_output_scale=config.get('use_learnable_output_scale', False),
        use_content_heads=use_content_heads,
    )
