"""
Pure mathematical helpers and shared data utilities for loss modules.

This module deliberately contains only stateless functions (no nn.Module).
It is consumed by both `losses.py` (LSTMCustomLoss / native losses factory)
and `signal_loss.py` (SignalLoss v2) to avoid duplicated math and to keep
each loss class focused on composition rather than primitives.

Functions:
    Pre-training data helpers:
        compute_sample_weights(...)     — per-sample weights from |y_raw|
        resolve_signal_threshold(...)   — fixed/empirical threshold from y_raw

    Differentiable batch statistics (autograd-aware):
        batch_pearson(pred, true)       — Pearson correlation
        batch_r2(pred, true)            — coefficient of determination
        batch_ccc(pred, true)           — concordance correlation
        log_variance_penalty(pred, true)— relu(log_var_true - log_var_pred)
"""
from __future__ import annotations

import numpy as np
import torch


# =========================================================================
# DATA-DRIVEN HELPERS (numpy, run before/between epochs)
# =========================================================================

def resolve_signal_threshold(y_raw, mode='fixed', fixed_value=0.001,
                             percentile=50.0):
    """
    Resolve signal threshold from training data.

    Args:
        y_raw:       np.ndarray — raw log-return targets
        mode:        'fixed' or 'empirical'
        fixed_value: float — threshold for mode='fixed'
        percentile:  float — percentile of |y| for mode='empirical'

    Returns:
        float — resolved threshold value
    """
    if mode == 'fixed':
        return fixed_value
    if mode == 'empirical':
        return float(np.percentile(np.abs(y_raw), percentile))
    raise ValueError(f"Unknown signal_threshold_mode: {mode}")


def compute_sample_weights(y_raw, mode='percentile', percentile=70.0,
                           low_w=0.1, clip_max=5.0):
    """
    Compute per-sample weights from |y_raw| to emphasize large moves.

    Modes:
        'abs'        — weight = |y| (raw absolute value)
        'abs_clip'   — weight = clip(|y|, 0, clip_max)
        'rank'       — weight = rank(|y|) / n (uniform 0..1)
        'percentile' — weight = 1.0 if |y| >= P(percentile), else low_w
    """
    abs_y = np.abs(y_raw)

    if mode == 'abs':
        weights = abs_y / (abs_y.mean() + 1e-10)
    elif mode == 'abs_clip':
        weights = np.clip(abs_y / (abs_y.mean() + 1e-10), 0, clip_max)
    elif mode == 'rank':
        ranks = abs_y.argsort().argsort().astype(np.float32)
        weights = ranks / max(len(ranks) - 1, 1)
    elif mode == 'percentile':
        threshold = np.percentile(abs_y, percentile)
        weights = np.where(abs_y >= threshold, 1.0, low_w)
    else:
        raise ValueError(f"Unknown sample_weight_mode: {mode}")

    return weights.astype(np.float32)


# =========================================================================
# BATCH STATISTICS (autograd-aware tensor ops)
# =========================================================================

def batch_pearson(y_pred, y_true, eps=1e-8):
    """
    Pearson correlation over a batch. Safe for constant predictions:
    returns 0.0 (→ max penalty when caller does `1 - ρ`) when std(pred) or
    std(true) is ~0. Single-sample batches return 0.0.
    """
    if y_pred.numel() < 2:
        return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    p = y_pred - y_pred.mean()
    t = y_true - y_true.mean()
    denom = torch.sqrt((p * p).sum() * (t * t).sum() + eps)
    return (p * t).sum() / (denom + eps)


def batch_r2(y_pred, y_true, eps=1e-12):
    """
    Differentiable coefficient of determination over a batch.

    R² = 1 - SS_res / SS_tot, where SS_tot uses mean(y_true). When predictions
    collapse to a constant: R² ≈ -var(true)/var(true) → close to 0 or below.
    When y_pred == y_true: R² = 1. The gradient is non-zero at collapse,
    which is the key property exploited by F2.V1-3 (R² penalty).
    """
    if y_pred.numel() < 2:
        return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    ss_res = ((y_pred - y_true) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return torch.clamp(r2, min=-1.0)


def batch_ccc(y_pred, y_true, eps=1e-12):
    """
    Concordance Correlation Coefficient — penalises simultaneously:
        — low correlation between pred and true
        — different variances (scale mismatch)
        — different means (bias)

    CCC = 2 · cov(p, t) / (var(p) + var(t) + (mean(p) - mean(t))²)

    Returns 1.0 for ideal predictions, 0.0 for uncorrelated/biased outputs.
    """
    if y_pred.numel() < 2:
        return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    mean_p, mean_t = y_pred.mean(), y_true.mean()
    var_p = y_pred.var(unbiased=False)
    var_t = y_true.var(unbiased=False)
    cov_pt = ((y_pred - mean_p) * (y_true - mean_t)).mean()
    denom = var_p + var_t + (mean_p - mean_t) ** 2 + eps
    return torch.clamp(2.0 * cov_pt / denom, min=-1.0, max=1.0)


def std_match_penalty(y_pred, y_true, eps=1e-8):
    """
    Symmetric standard-deviation matching penalty — targets CR ≈ 1.

    penalty = | 1 - std(pred) / std(true) |

    Unlike `log_variance_penalty` / the one-sided variance penalty (which only
    push variance *up* and can drive the model into amplitude explosion), this
    term penalises BOTH collapse (std_pred → 0 ⇒ penalty → 1) and explosion
    (std_pred ≫ std_true ⇒ penalty grows), so the optimiser is steered toward
    a compression ratio near 1 — exactly the regime where the probabilistic
    BUY/SELL signal is well-scaled. Scale-invariant, differentiable
    (abs has a sub-gradient at 0), bounded below at 0.
    """
    std_p = torch.sqrt(y_pred.var(unbiased=False) + eps)
    std_t = torch.sqrt(y_true.var(unbiased=False) + eps)
    return torch.abs(1.0 - std_p / (std_t + eps))


def threshold_directional_penalty(y_pred, y_true, threshold, eps=1e-8,
                                    soft_threshold=False):
    """
    Directional penalty restricted to economically meaningful moves.

    Only samples with |y_true| >= threshold contribute (small moves are noise
    for signal generation). For those samples the term is
    ``relu(-sign(y_true) * y_pred)`` — zero when the predicted sign is correct,
    |pred| when wrong — normalised by the mean |y_true| of the masked subset so
    it stays dimensionless. Returns 0 when no sample crosses the threshold.

    When soft_threshold=True, weight samples by
    sigmoid((|y|-T)/(0.25*T)) instead of a hard mask.
    """
    if soft_threshold:
        thr = max(float(threshold), 1e-8)
        weight = torch.sigmoid((torch.abs(y_true) - thr) / (0.25 * thr))
        if weight.sum() < 1e-8:
            return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
        raw_pen = torch.relu(-torch.sign(y_true).detach() * y_pred)
        denom = (torch.abs(y_true) * weight).sum().detach() / (
            weight.sum().detach() + eps) + eps
        return (raw_pen * weight).sum() / (weight.sum() + eps) / denom
    mask = (torch.abs(y_true) >= threshold)
    if not bool(mask.any()):
        return torch.zeros((), device=y_pred.device, dtype=y_pred.dtype)
    yp = y_pred[mask]
    yt = y_true[mask]
    raw_pen = torch.relu(-torch.sign(yt).detach() * yp)
    denom = torch.abs(yt).mean().detach() + eps
    return raw_pen.mean() / denom


def cr_ceiling_penalty(y_pred, y_true, ceiling=1.5, eps=1e-8):
    """Penalise std(pred)/std(true) above ceiling."""
    std_p = torch.sqrt(y_pred.var(unbiased=False) + eps)
    std_t = torch.sqrt(y_true.var(unbiased=False) + eps)
    ratio = std_p / (std_t + eps)
    return torch.relu(ratio - ceiling)


def log_variance_penalty(y_pred, y_true, eps=1e-6, clamp_min=-15.0):
    """
    Log-variance penalty term (raw, unweighted) — relu(log_var_true - log_var_pred).

    Unlike the linear `relu(1 - var(pred)/var(true))` penalty, the log-space
    form has a non-zero gradient at the collapse point (var(pred) → 0). The
    gradient is amplified inversely with var(pred) because
    d(log(var))/d(pred_i) = 2·(pred_i - mean(pred)) / (batch · var(pred)).

    Returns the raw (unweighted) penalty value. The caller multiplies by its
    weight and adds to the total loss.
    """
    log_var_p = torch.log(y_pred.var(unbiased=False) + eps).clamp(min=clamp_min)
    log_var_t = torch.log(y_true.var(unbiased=False) + eps)
    return torch.clamp(log_var_t - log_var_p, min=0.0)


# =========================================================================
# REGIME-ROBUST OPTIMIZATION (CP-183)
# =========================================================================

class GroupDROLoss(torch.nn.Module):
    """
    Distributionally Robust Optimization across volatility/market regimes (CP-183).
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
