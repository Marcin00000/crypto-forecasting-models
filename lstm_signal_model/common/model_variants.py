"""
Optional output heads and FC building blocks for Signal model variants.

Independent implementation for lstm_signal_model.
Supports: 'legacy' | 'dual_head' | 'submodel_fusion'
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from .model import get_activation, build_norm_layer


class SubmodelFixedZScore(nn.Module):
    """
    Fixed Z-score scaler for submodel predictions.
    Scales log-return predictions (std ~= 0.001) by 1/target_std to match context scale (~1.0).
    Works independently of batch size and feature dimension.
    """
    def __init__(self, target_std: float = 0.001):
        super().__init__()
        self.scale = 1.0 / max(float(target_std), 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SubmodelFusionHead(nn.Module):
    """
    Fusion head for dynamic submodel integration (Online mode).

    Combines context vector from primary Signal RNN with submodel prediction/embedding
    vectors before final output linear layer.

    Args:
        primary_size: int — dimension of primary RNN context vector
        submodel_size: int — number of submodel prediction outputs
        out_features: int — number of final output predictions (1 or 2)
        use_submodel_norm: bool — legacy boolean flag for enabling normalization.
        submodel_norm_type: str — normalization method:
            - 'auto': always nn.BatchNorm1d(submodel_size), for both D=1 and D>1
              (since DP3.1; LayerNorm is never selected by 'auto').
            - 'batch': nn.BatchNorm1d(submodel_size) (normalizes across batch samples).
            - 'zscore': SubmodelFixedZScore(0.001) (fixed target std scaling).
            - 'layer': nn.LayerNorm(submodel_size) (normalizes across feature dims).
            - 'none': no normalization.
    """

    def __init__(self, primary_size: int, submodel_size: int, out_features: int = 2,
                 use_submodel_norm: bool = True,
                 submodel_norm_type: str = 'auto'):
        super().__init__()
        self.submodel_size = submodel_size
        self.use_submodel_norm = bool(use_submodel_norm)
        self.submodel_norm_type = str(submodel_norm_type or 'auto').lower()

        if not self.use_submodel_norm or self.submodel_norm_type in ('none', '0', 'false'):
            self.submodel_norm = None
        elif self.submodel_norm_type == 'batch':
            self.submodel_norm = nn.BatchNorm1d(submodel_size)
        elif self.submodel_norm_type == 'zscore':
            self.submodel_norm = SubmodelFixedZScore(target_std=0.001)
        elif self.submodel_norm_type == 'layer':
            self.submodel_norm = nn.LayerNorm(submodel_size)
        elif self.submodel_norm_type == 'auto':
            # Use BatchNorm1d across batch samples for both D=1 and D>1.
            # LayerNorm(D) for D>=2 subtracts the sample mean across submodels,
            # which destroys the common directional market signal (bull/bear).
            self.submodel_norm = nn.BatchNorm1d(submodel_size)
        else:
            # Fallback
            self.submodel_norm = nn.BatchNorm1d(submodel_size)

        self.fusion_dim = primary_size + submodel_size
        self.projection = nn.Linear(self.fusion_dim, primary_size)
        self.activation = nn.GELU()
        self.output_linear = nn.Linear(primary_size, out_features)

    def forward(self, primary_ctx, submodel_preds):
        """
        Args:
            primary_ctx: (batch, primary_size)
            submodel_preds: (batch, submodel_size)
        """
        if self.submodel_norm is not None:
            submodel_preds = self.submodel_norm(submodel_preds)
        fused = torch.cat([primary_ctx, submodel_preds], dim=-1)
        projected = self.activation(self.projection(fused))
        return self.output_linear(projected)


class DualHeadOutput(nn.Module):
    """
    Direction and magnitude output head for Signal Model (CP-162).

    Uses bias=False on direction and strength layers to prevent collision
    with the master learnable neutral intercept self.bias.
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
