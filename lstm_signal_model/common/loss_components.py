"""
Loss components for Signal Model.

Isolated, modular penalty terms for probabilistic dual-sigmoid predictions (p_buy, p_sell).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossComponent(nn.Module):
    """
    Focal Loss component for handling class imbalance / neutral dominance.

    Down-weights easy neutral predictions (pred ~ 0.5) and focuses gradients
    on hard signal samples.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_factor = (1 - p_t) ** self.gamma
        return (focal_factor * bce).mean()


class AsymmetricLossComponent(nn.Module):
    """
    Asymmetric weighting component for Buy vs Sell errors.

    Allows different penalty weights for false buy signals vs false sell signals.
    """

    def __init__(self, w_buy: float = 1.0, w_sell: float = 1.0):
        super().__init__()
        self.w_buy = float(w_buy)
        self.w_sell = float(w_sell)

    def forward(self, pred, target):
        bce_buy = F.binary_cross_entropy(pred[:, 0], target[:, 0]) * self.w_buy
        bce_sell = F.binary_cross_entropy(pred[:, 1], target[:, 1]) * self.w_sell
        return bce_buy + bce_sell


class StdMatchLossComponent(nn.Module):
    """
    Standard deviation matching penalty.

    Penalizes |std(pred) - target_std|, preventing signal collapse to constant 0.5.
    """

    def __init__(self, target_std: float = 0.20, weight: float = 0.1):
        super().__init__()
        self.target_std = float(target_std)
        self.weight = float(weight)

    def forward(self, pred):
        pred_std = pred.std(dim=0).mean()
        penalty = (pred_std - self.target_std).abs()
        return self.weight * penalty


class BuySellCorrelationPenalty(nn.Module):
    """
    Penalty for simultaneous high buy and sell predictions.

    Forces p_buy and p_sell to be mutually exclusive on the same candle.
    penalty = mean(relu(p_buy + p_sell - 1.0)^2)
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = float(weight)

    def forward(self, pred):
        buy = pred[:, 0]
        sell = pred[:, 1]
        overlap = F.relu(buy + sell - 1.0)
        return self.weight * (overlap ** 2).mean()


# =========================================================================
# ECONOMIC UTILITY & METRIC PARITY COMPONENTS (CP-173)
# =========================================================================

def signal_utility_penalty(p_buy: torch.Tensor, p_sell: torch.Tensor,
                           y_raw: torch.Tensor, weight: float = 0.05) -> torch.Tensor:
    """
    Differentiable utility penalty for Signal Model.
    Rewards p_buy when y_raw > 0 and p_sell when y_raw < 0.
    utility = (p_buy - p_sell) * y_raw
    penalty = -weight * mean(utility)
    """
    if weight <= 0.0:
        return torch.tensor(0.0, device=p_buy.device)
    utility = (p_buy - p_sell) * y_raw
    return weight * (-utility.mean())


def batch_r2(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable batch R² coefficient of determination.
    R² = 1 - SS_res / (SS_tot + eps)
    """
    if y_pred.numel() < 2:
        return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)


def batch_ccc(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable batch Lin's Concordance Correlation Coefficient.
    CCC = 2 · cov(p, t) / (var(p) + var(t) + (mean(p) - mean(t))²)
    """
    if y_pred.numel() < 2:
        return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    mean_p, mean_t = y_pred.mean(), y_true.mean()
    var_p = y_pred.var(unbiased=False)
    var_t = y_true.var(unbiased=False)
    cov_pt = ((y_pred - mean_p) * (y_true - mean_t)).mean()
    denom = var_p + var_t + (mean_p - mean_t) ** 2 + eps
    return torch.clamp(2.0 * cov_pt / denom, min=-1.0, max=1.0)


def cr_ceiling_penalty(y_pred: torch.Tensor, y_true: torch.Tensor,
                       ceiling: float = 1.5, eps: float = 1e-8) -> torch.Tensor:
    """
    Penalise std(pred)/std(true) above ceiling.
    In Signal, complements StdMatchLossComponent with asymmetric ceiling cap.
    """
    std_p = torch.sqrt(y_pred.var(unbiased=False) + eps)
    std_t = torch.sqrt(y_true.var(unbiased=False) + eps)
    ratio = std_p / (std_t + eps)
    return torch.relu(ratio - ceiling)


# =========================================================================
# REGIME-ROBUST OPTIMIZATION (CP-183)
# =========================================================================

class GroupDROLoss(nn.Module):
    """
    Distributionally Robust Optimization across volatility/market regimes.
    Maintains adaptive group weights updated via Exponentiated Gradient Ascent.
    """

    def __init__(self, n_groups: int = 3, eta: float = 0.01):
        super().__init__()
        self.n_groups = int(n_groups)
        self.eta = float(eta)
        self.register_buffer('group_weights', torch.ones(self.n_groups) / float(self.n_groups))

    def forward(self, per_sample_losses: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        group_losses = torch.zeros(self.n_groups, device=per_sample_losses.device)
        valid_groups = []
        for g in range(self.n_groups):
            mask = (group_ids == g)
            if mask.any():
                group_losses[g] = per_sample_losses[mask].mean()
                valid_groups.append(g)
            else:
                group_losses[g] = 0.0

        if not valid_groups:
            return per_sample_losses.mean()

        with torch.no_grad():
            self.group_weights = self.group_weights * torch.exp(self.eta * group_losses)
            sum_w = self.group_weights.sum()
            if sum_w > 0:
                self.group_weights = self.group_weights / sum_w

        return (self.group_weights * group_losses).sum()
