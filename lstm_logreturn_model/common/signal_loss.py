"""
SignalLoss (v2) — minimalist loss for direction-quality regression.

A small, anti-collapse loss built around three observations:

* The signal quality the strategy cares about is *direction* of log returns,
  not exact magnitude. Pinning the loss to MAE (instead of Huber/MSE) keeps
  the gradient bounded and avoids the over-attention to outliers that pushes
  predictions toward the mean.
* The custom v1 loss (`LSTMCustomLoss`) accumulated many specialised terms
  (dead-zone, amplified-zone, sample weights, mis-direction). They make
  tuning fragile. V2 drops all of those.
* The remaining failure mode — prediction collapse to a constant — is
  addressed by three optional batch-level penalties from
  `loss_components`: R², CCC, and log-variance. Each has a non-zero
  gradient at the collapse point, so the optimiser is pushed away from
  flat predictions even when MAE alone is near its minimum.

Component breakdown is stored on `self._last_components` so the training
loop can log it identically to `LSTMCustomLoss`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_components import (
    batch_ccc, batch_r2, cr_ceiling_penalty, log_variance_penalty,
    std_match_penalty, threshold_directional_penalty,
)


__all__ = ['SignalLoss']


class SignalLoss(nn.Module):
    """
    Minimalist MAE + collapse-aware penalties.

    Optional F7 extensions:
        ema_penalties / ema_alpha — smooth batch-level penalty noise
        label_smoothing / label_smoothing_alpha — soften dead-zone targets

    Optional Faza 1b extensions:
        use_utility / utility_weight — differentiable proxy aligned with
            probabilistic BUY/SELL (requires per-sample sigma in forward)
        soft_threshold — sigmoid weights instead of hard |y|>=T masks
        use_cr_ceiling / cr_ceiling / cr_ceiling_weight — penalise CR explosion
    """

    def __init__(self,
                 use_r2_penalty: bool = True,
                 r2_penalty_weight: float = 0.01,
                 use_ccc_penalty: bool = True,
                 ccc_penalty_weight: float = 0.01,
                 use_log_var_penalty: bool = False,
                 log_var_penalty_weight: float = 0.005,
                 use_std_match_penalty: bool = False,
                 std_match_penalty_weight: float = 0.05,
                 use_dir_threshold_penalty: bool = False,
                 dir_threshold_penalty_weight: float = 0.1,
                 use_direction_bce: bool = False,
                 direction_bce_weight: float = 0.01,
                 signal_threshold: float = 0.001,
                 ema_penalties: bool = False,
                 ema_alpha: float = 0.1,
                 label_smoothing: bool = False,
                 label_smoothing_alpha: float = 0.5,
                 use_utility: bool = False,
                 utility_weight: float = 0.05,
                 soft_threshold: bool = False,
                 use_cr_ceiling: bool = False,
                 cr_ceiling: float = 1.5,
                 cr_ceiling_weight: float = 0.1,
                 target_scaler=None):
        super().__init__()
        self.use_r2_penalty = bool(use_r2_penalty)
        self.r2_penalty_weight = float(r2_penalty_weight)
        self.use_ccc_penalty = bool(use_ccc_penalty)
        self.ccc_penalty_weight = float(ccc_penalty_weight)
        self.use_log_var_penalty = bool(use_log_var_penalty)
        self.log_var_penalty_weight = float(log_var_penalty_weight)
        self.use_std_match_penalty = bool(use_std_match_penalty)
        self.std_match_penalty_weight = float(std_match_penalty_weight)
        self.use_dir_threshold_penalty = bool(use_dir_threshold_penalty)
        self.dir_threshold_penalty_weight = float(dir_threshold_penalty_weight)
        self.use_direction_bce = bool(use_direction_bce)
        self.direction_bce_weight = float(direction_bce_weight)
        self.signal_threshold = float(signal_threshold)
        self.ema_penalties = bool(ema_penalties)
        self.ema_alpha = float(ema_alpha)
        self.label_smoothing = bool(label_smoothing)
        self.label_smoothing_alpha = float(label_smoothing_alpha)
        self.use_utility = bool(use_utility)
        self.utility_weight = float(utility_weight)
        self.soft_threshold = bool(soft_threshold)
        self.use_cr_ceiling = bool(use_cr_ceiling)
        self.cr_ceiling = float(cr_ceiling)
        self.cr_ceiling_weight = float(cr_ceiling_weight)
        self.target_scaler = target_scaler

        self.register_buffer('_ema_r2_pen', torch.zeros(1))
        self.register_buffer('_ema_ccc_pen', torch.zeros(1))
        self.register_buffer('_ema_log_var_pen', torch.zeros(1))
        self.register_buffer('_ema_std_match_pen', torch.zeros(1))

        self._has_target_transform = False
        if target_scaler is not None:
            scaler = target_scaler.scaler
            scale = float(getattr(scaler, 'scale_', [1.0])[0])
            if target_scaler.method == 'standard':
                center = float(scaler.mean_[0])
            elif target_scaler.method == 'robust':
                center = float(scaler.center_[0])
            else:
                center = 0.0
            self.register_buffer(
                '_target_scale', torch.tensor(scale, dtype=torch.float32))
            self.register_buffer(
                '_target_center', torch.tensor(center, dtype=torch.float32))
            self._has_target_transform = True

        self._last_components: dict = {}

    def _to_raw(self, y_scaled: torch.Tensor) -> torch.Tensor:
        if self._has_target_transform:
            return y_scaled * self._target_scale + self._target_center
        return y_scaled

    def _smooth_targets(self, y_true: torch.Tensor) -> torch.Tensor:
        """Shrink targets in the dead zone toward zero (F7.D)."""
        if not self.label_smoothing:
            return y_true
        alpha = self.label_smoothing_alpha
        if self.target_scaler is not None:
            scaler = self.target_scaler.scaler
            if self.target_scaler.method == 'robust':
                center = float(scaler.center_[0])
                scale = float(scaler.scale_[0])
            else:
                center = float(scaler.mean_[0])
                scale = float(np.sqrt(scaler.var_[0]))
            y_raw = y_true * scale + center
            dead = torch.abs(y_raw) < self.signal_threshold
        else:
            thr_scaled = self.signal_threshold
            dead = torch.abs(y_true) < thr_scaled
        if not bool(dead.any()):
            return y_true
        smoothed = (1.0 - alpha) * y_true
        return torch.where(dead, smoothed, y_true)

    def _ema_blend(self, raw_pen: torch.Tensor, buffer_name: str) -> torch.Tensor:
        """Blend raw penalty with EMA buffer (F7.A)."""
        if not self.ema_penalties:
            return raw_pen
        buf = getattr(self, buffer_name)
        alpha = self.ema_alpha
        if self.training:
            with torch.no_grad():
                buf.mul_(1.0 - alpha).add_(alpha * raw_pen.detach())
        return alpha * raw_pen + (1.0 - alpha) * buf.detach()

    def forward(self, y_pred, y_true, indices=None, sigma=None):
        del indices
        y_pred = y_pred.squeeze(-1)
        y_true = y_true.squeeze(-1)
        y_true_loss = self._smooth_targets(y_true)

        base = F.l1_loss(y_pred, y_true_loss)
        total = base

        comps = {'base': float(base.detach().item())}

        if self.use_r2_penalty:
            r2 = batch_r2(y_pred, y_true_loss)
            r2_pen = self.r2_penalty_weight * torch.clamp(1.0 - r2, min=0.0)
            r2_pen = self._ema_blend(r2_pen, '_ema_r2_pen')
            total = total + r2_pen
            comps['r2'] = float(r2.detach().item())
            comps['r2_penalty'] = float(r2_pen.detach().item())

        if self.use_ccc_penalty:
            ccc = batch_ccc(y_pred, y_true_loss)
            ccc_pen = self.ccc_penalty_weight * torch.clamp(1.0 - ccc, min=0.0)
            ccc_pen = self._ema_blend(ccc_pen, '_ema_ccc_pen')
            total = total + ccc_pen
            comps['ccc'] = float(ccc.detach().item())
            comps['ccc_penalty'] = float(ccc_pen.detach().item())

        if self.use_log_var_penalty:
            raw_lv = log_variance_penalty(y_pred, y_true_loss)
            lv_pen = self.log_var_penalty_weight * raw_lv
            lv_pen = self._ema_blend(lv_pen, '_ema_log_var_pen')
            log_var_pred = float(
                torch.log(y_pred.var(unbiased=False) + 1e-6).detach().item())
            total = total + lv_pen
            comps['log_var_pred'] = log_var_pred
            comps['log_var_penalty'] = float(lv_pen.detach().item())

        if self.use_std_match_penalty:
            raw_sm = std_match_penalty(y_pred, y_true_loss)
            sm_pen = self.std_match_penalty_weight * raw_sm
            sm_pen = self._ema_blend(sm_pen, '_ema_std_match_pen')
            total = total + sm_pen
            comps['std_match'] = float(raw_sm.detach().item())
            comps['std_match_penalty'] = float(sm_pen.detach().item())

        if self.use_dir_threshold_penalty:
            raw_dt = threshold_directional_penalty(
                y_pred, y_true_loss, self.signal_threshold,
                soft_threshold=self.soft_threshold)
            dt_pen = self.dir_threshold_penalty_weight * raw_dt
            total = total + dt_pen
            comps['dir_threshold_penalty'] = float(dt_pen.detach().item())

        if self.use_direction_bce:
            thr = self.signal_threshold
            scale = max(abs(thr), 1e-8)
            logits = y_pred / scale
            target = (y_true_loss > 0).to(dtype=y_pred.dtype)
            if self.soft_threshold:
                weight = torch.sigmoid(
                    (torch.abs(y_true_loss) - thr) / (0.25 * thr))
                if weight.sum() >= 1e-8:
                    raw_bce = F.binary_cross_entropy_with_logits(
                        logits, target, reduction='none')
                    bce_pen = (self.direction_bce_weight
                               * (raw_bce * weight).sum()
                               / (weight.sum() + 1e-8))
                    total = total + bce_pen
                    comps['direction_bce'] = float(
                        raw_bce.mean().detach().item())
                    comps['direction_bce_penalty'] = float(
                        bce_pen.detach().item())
            else:
                mask = torch.abs(y_true_loss) >= thr
                if bool(mask.any()):
                    target_m = target[mask]
                    logits_m = logits[mask]
                    raw_bce = F.binary_cross_entropy_with_logits(
                        logits_m, target_m)
                    bce_pen = self.direction_bce_weight * raw_bce
                    total = total + bce_pen
                    comps['direction_bce'] = float(raw_bce.detach().item())
                    comps['direction_bce_penalty'] = float(
                        bce_pen.detach().item())

        if self.use_utility and sigma is not None:
            sigma_safe = torch.maximum(
                sigma.to(y_pred.device, dtype=y_pred.dtype),
                torch.tensor(1e-8, device=y_pred.device, dtype=y_pred.dtype))
            mu_raw = self._to_raw(y_pred)
            y_raw = self._to_raw(y_true)
            thr = self.signal_threshold
            soft_buy = torch.tanh((mu_raw - thr) / sigma_safe)
            soft_sell = torch.tanh((-thr - mu_raw) / sigma_safe)
            util_raw = soft_buy * y_raw + soft_sell * (-y_raw)
            util_pen = self.utility_weight * (-util_raw.mean())
            total = total + util_pen
            comps['utility_penalty'] = float(util_pen.detach().item())

        if self.use_cr_ceiling:
            raw_cr = cr_ceiling_penalty(
                y_pred, y_true_loss, ceiling=self.cr_ceiling)
            cr_pen = self.cr_ceiling_weight * raw_cr
            total = total + cr_pen
            comps['cr_ceiling_penalty'] = float(cr_pen.detach().item())

        comps['total'] = float(total.detach().item())
        self._last_components = comps
        return total
