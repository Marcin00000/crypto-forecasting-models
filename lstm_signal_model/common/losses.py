"""
Loss functions for Signal Model.

Provides:
  - SignalBCELoss:                BCE on dual (pred_buy, pred_sell) vs oracle
  - SignalBCEWithDiversityLoss:   BCE + diversity penalty (B4)
  - SignalCustomLoss:             Modular custom loss combining focal, asymmetric, std_match, buy/sell correlation penalties
  - ScheduledLossWrapper:         Ramp-up wrapper that activates auxiliary terms
  - get_signal_loss:              Factory from loss_type string
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_components import (
    FocalLossComponent,
    AsymmetricLossComponent,
    StdMatchLossComponent,
    BuySellCorrelationPenalty,
    signal_utility_penalty,
    batch_r2,
    batch_ccc,
    cr_ceiling_penalty,
)


class SignalBCELoss(nn.Module):
    """Binary Cross-Entropy on paired signal predictions vs oracle targets."""

    def __init__(self, label_smoothing_alpha: float = 0.0):
        super().__init__()
        self.label_smoothing_alpha = float(label_smoothing_alpha)

    def forward(self, pred, target, y_raw=None):
        if self.label_smoothing_alpha > 0:
            target_smooth = target * (1.0 - self.label_smoothing_alpha) + 0.5 * self.label_smoothing_alpha
        else:
            target_smooth = target
        return F.binary_cross_entropy(pred, target_smooth)


class SignalBCEWithDiversityLoss(nn.Module):
    """BCE loss augmented with an anti-collapse diversity penalty (B4)."""

    def __init__(self, diversity_weight: float = 1.0,
                 diversity_target: float = 0.30,
                 label_smoothing_alpha: float = 0.0):
        super().__init__()
        self.diversity_weight = float(diversity_weight)
        self.diversity_target = float(diversity_target)
        self.label_smoothing_alpha = float(label_smoothing_alpha)
        self._ramp_weight = 1.0
        self._last_components: dict = {}

    def set_ramp_weight(self, w: float):
        """Called by ScheduledLossWrapper to update the ramp multiplier."""
        self._ramp_weight = float(w)

    def forward(self, pred, target, y_raw=None):
        if self.label_smoothing_alpha > 0:
            target_smooth = target * (1.0 - self.label_smoothing_alpha) + 0.5 * self.label_smoothing_alpha
        else:
            target_smooth = target

        bce = F.binary_cross_entropy(pred, target_smooth)
        diversity = (pred - 0.5).abs().mean()
        penalty = torch.clamp(
            torch.tensor(self.diversity_target, dtype=pred.dtype,
                         device=pred.device) - diversity,
            min=0.0)
        aux = self.diversity_weight * self._ramp_weight * penalty
        total = bce + aux
        self._last_components = {
            'bce': float(bce.item()),
            'diversity': float(diversity.item()),
            'diversity_penalty': float(penalty.item()),
            'aux_weighted': float(aux.item()),
        }
        return total


class SignalCustomLoss(nn.Module):
    """
    Modular custom loss for Signal Model.

    Combines base loss (BCE or Focal) with optional auxiliary penalties:
    - Asymmetric weights for buy vs sell
    - Diversity penalty
    - Std match penalty
    - Buy/Sell correlation penalty
    - Label smoothing
    - Signal utility penalty (CP-173)
    - Batch R² / CCC correlation loss (CP-173)
    - CR ceiling penalty (CP-173)
    """

    def __init__(self, use_focal: bool = False, focal_gamma: float = 2.0,
                 w_buy: float = 1.0, w_sell: float = 1.0,
                 diversity_weight: float = 0.0, diversity_target: float = 0.30,
                 use_std_match: bool = False, std_match_weight: float = 0.1,
                 use_corr_penalty: bool = False, corr_penalty_weight: float = 0.5,
                 label_smoothing_alpha: float = 0.0,
                 use_signal_utility: bool = False, signal_utility_weight: float = 0.05,
                 use_r2_loss: bool = False, r2_loss_weight: float = 0.1,
                 use_ccc_loss: bool = False, ccc_loss_weight: float = 0.1,
                 use_cr_ceiling: bool = False, cr_ceiling_weight: float = 0.1,
                 cr_ceiling: float = 1.5):
        super().__init__()
        self.use_focal = use_focal
        self.focal = FocalLossComponent(gamma=focal_gamma) if use_focal else None
        self.asymmetric = AsymmetricLossComponent(w_buy=w_buy, w_sell=w_sell)
        self.diversity_weight = float(diversity_weight)
        self.diversity_target = float(diversity_target)
        self.std_match = StdMatchLossComponent(weight=std_match_weight) if use_std_match else None
        self.corr_penalty = BuySellCorrelationPenalty(weight=corr_penalty_weight) if use_corr_penalty else None
        self.label_smoothing_alpha = float(label_smoothing_alpha)

        # CP-173 components
        self.use_signal_utility = use_signal_utility
        self.signal_utility_weight = float(signal_utility_weight)
        self.use_r2_loss = use_r2_loss
        self.r2_loss_weight = float(r2_loss_weight)
        self.use_ccc_loss = use_ccc_loss
        self.ccc_loss_weight = float(ccc_loss_weight)
        self.use_cr_ceiling = use_cr_ceiling
        self.cr_ceiling_weight = float(cr_ceiling_weight)
        self.cr_ceiling = float(cr_ceiling)

        self._ramp_weight = 1.0
        self._last_components: dict = {}

    def set_ramp_weight(self, w: float):
        self._ramp_weight = float(w)

    def forward(self, pred, target, y_raw=None):
        if self.label_smoothing_alpha > 0:
            target_smooth = target * (1.0 - self.label_smoothing_alpha) + 0.5 * self.label_smoothing_alpha
        else:
            target_smooth = target

        if self.use_focal:
            base_loss = self.focal(pred, target_smooth)
        else:
            base_loss = self.asymmetric(pred, target_smooth)

        total = base_loss
        components = {'base_loss': float(base_loss.item())}

        if self.diversity_weight > 0:
            diversity = (pred - 0.5).abs().mean()
            penalty = torch.clamp(
                torch.tensor(self.diversity_target, dtype=pred.dtype, device=pred.device) - diversity,
                min=0.0)
            aux_div = self.diversity_weight * self._ramp_weight * penalty
            total = total + aux_div
            components['diversity_penalty'] = float(aux_div.item())

        if self.std_match is not None:
            std_loss = self.std_match(pred) * self._ramp_weight
            total = total + std_loss
            components['std_match_loss'] = float(std_loss.item())

        if self.corr_penalty is not None:
            corr_loss = self.corr_penalty(pred) * self._ramp_weight
            total = total + corr_loss
            components['corr_penalty'] = float(corr_loss.item())

        # CP-173 additions
        if self.use_signal_utility and self.signal_utility_weight > 0:
            # If explicit y_raw not passed, derive direction from target oracle difference
            actual_y = y_raw if y_raw is not None else (target[:, 0] - target[:, 1])
            util_pen = signal_utility_penalty(pred[:, 0], pred[:, 1], actual_y, weight=self.signal_utility_weight)
            util_loss = util_pen * self._ramp_weight
            total = total + util_loss
            components['signal_utility_loss'] = float(util_loss.item())

        if self.use_r2_loss and self.r2_loss_weight > 0:
            r2_b = batch_r2(pred[:, 0], target[:, 0])
            r2_s = batch_r2(pred[:, 1], target[:, 1])
            r2_loss = self.r2_loss_weight * self._ramp_weight * (1.0 - 0.5 * (r2_b + r2_s))
            total = total + r2_loss
            components['r2_loss'] = float(r2_loss.item())

        if self.use_ccc_loss and self.ccc_loss_weight > 0:
            ccc_b = batch_ccc(pred[:, 0], target[:, 0])
            ccc_s = batch_ccc(pred[:, 1], target[:, 1])
            ccc_loss = self.ccc_loss_weight * self._ramp_weight * (1.0 - 0.5 * (ccc_b + ccc_s))
            total = total + ccc_loss
            components['ccc_loss'] = float(ccc_loss.item())

        if self.use_cr_ceiling and self.cr_ceiling_weight > 0:
            cr_b = cr_ceiling_penalty(pred[:, 0], target[:, 0], ceiling=self.cr_ceiling)
            cr_s = cr_ceiling_penalty(pred[:, 1], target[:, 1], ceiling=self.cr_ceiling)
            cr_pen = self.cr_ceiling_weight * self._ramp_weight * 0.5 * (cr_b + cr_s)
            total = total + cr_pen
            components['cr_ceiling_penalty'] = float(cr_pen.item())

        self._last_components = components
        return total


class ScheduledLossWrapper(nn.Module):
    """Wrapper that ramps up auxiliary loss components over the first N epochs."""

    def __init__(self, base_loss, ramp_epochs=0):
        super().__init__()
        self._base = base_loss
        self._ramp = max(int(ramp_epochs), 0)
        self._current_weight = 1.0 if self._ramp == 0 else 0.0

    def set_epoch(self, epoch: int):
        """Update ramp weight for current epoch (1-based)."""
        if self._ramp > 0:
            self._current_weight = min(1.0, epoch / float(self._ramp))
            if hasattr(self._base, 'set_ramp_weight'):
                self._base.set_ramp_weight(self._current_weight)

    @property
    def ramp_weight(self) -> float:
        return self._current_weight

    @property
    def _last_components(self) -> dict:
        return getattr(self._base, '_last_components', {})

    def forward(self, pred, target, y_raw=None):
        return self._base(pred, target, y_raw=y_raw)


def get_signal_loss(loss_type: str, ramp_epochs: int = 0,
                    diversity_weight: float = 1.0,
                    diversity_target: float = 0.30,
                    focal_gamma: float = 2.0,
                    w_buy: float = 1.0, w_sell: float = 1.0,
                    std_match_weight: float = 0.1,
                    corr_penalty_weight: float = 0.5,
                    label_smoothing_alpha: float = 0.0,
                    use_corr_penalty: bool | None = None,
                    use_signal_utility: bool = False,
                    signal_utility_weight: float = 0.05,
                    use_r2_loss: bool = False,
                    r2_loss_weight: float = 0.1,
                    use_ccc_loss: bool = False,
                    ccc_loss_weight: float = 0.1,
                    use_cr_ceiling: bool = False,
                    cr_ceiling_weight: float = 0.1,
                    cr_ceiling: float = 1.5):
    """Factory: build loss function from type string."""
    corr_on = bool(use_corr_penalty) if use_corr_penalty is not None else None

    if loss_type in ('bce', 'bce_diversity') and corr_on:
        logging.warning(
            "use_corr_penalty=True ignored for loss_type='%s' — the buy/sell "
            "correlation penalty is only available in SignalCustomLoss "
            "('focal', 'asymmetric', 'custom', 'custom_signal').", loss_type)

    if loss_type == 'bce':
        base = SignalBCELoss(label_smoothing_alpha=label_smoothing_alpha)
    elif loss_type == 'bce_diversity':
        base = SignalBCEWithDiversityLoss(
            diversity_weight=diversity_weight,
            diversity_target=diversity_target,
            label_smoothing_alpha=label_smoothing_alpha)
    elif loss_type == 'focal':
        base = SignalCustomLoss(
            use_focal=True, focal_gamma=focal_gamma,
            w_buy=w_buy, w_sell=w_sell,
            use_corr_penalty=bool(corr_on),
            corr_penalty_weight=corr_penalty_weight,
            label_smoothing_alpha=label_smoothing_alpha,
            use_signal_utility=use_signal_utility,
            signal_utility_weight=signal_utility_weight,
            use_r2_loss=use_r2_loss, r2_loss_weight=r2_loss_weight,
            use_ccc_loss=use_ccc_loss, ccc_loss_weight=ccc_loss_weight,
            use_cr_ceiling=use_cr_ceiling, cr_ceiling_weight=cr_ceiling_weight,
            cr_ceiling=cr_ceiling)
    elif loss_type == 'asymmetric':
        base = SignalCustomLoss(
            use_focal=False, w_buy=w_buy, w_sell=w_sell,
            use_corr_penalty=bool(corr_on),
            corr_penalty_weight=corr_penalty_weight,
            label_smoothing_alpha=label_smoothing_alpha,
            use_signal_utility=use_signal_utility,
            signal_utility_weight=signal_utility_weight,
            use_r2_loss=use_r2_loss, r2_loss_weight=r2_loss_weight,
            use_ccc_loss=use_ccc_loss, ccc_loss_weight=ccc_loss_weight,
            use_cr_ceiling=use_cr_ceiling, cr_ceiling_weight=cr_ceiling_weight,
            cr_ceiling=cr_ceiling)
    elif loss_type in ('custom', 'custom_signal'):
        base = SignalCustomLoss(
            use_focal=True, focal_gamma=focal_gamma,
            w_buy=w_buy, w_sell=w_sell,
            diversity_weight=diversity_weight, diversity_target=diversity_target,
            use_std_match=True, std_match_weight=std_match_weight,
            use_corr_penalty=(True if corr_on is None else corr_on),
            corr_penalty_weight=corr_penalty_weight,
            label_smoothing_alpha=label_smoothing_alpha,
            use_signal_utility=use_signal_utility,
            signal_utility_weight=signal_utility_weight,
            use_r2_loss=use_r2_loss, r2_loss_weight=r2_loss_weight,
            use_ccc_loss=use_ccc_loss, ccc_loss_weight=ccc_loss_weight,
            use_cr_ceiling=use_cr_ceiling, cr_ceiling_weight=cr_ceiling_weight,
            cr_ceiling=cr_ceiling)
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            "Available: ['bce', 'bce_diversity', 'focal', 'asymmetric', 'custom', 'custom_signal']")
    return ScheduledLossWrapper(base, ramp_epochs=ramp_epochs)
