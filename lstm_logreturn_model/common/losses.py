"""
Custom loss functions for LSTM Log-Return pipeline.

Provides:
  - PseudoHuberLoss:   smooth approximation of Huber loss
  - QuantileLoss:      asymmetric (pinball) loss
  - LSTMCustomLoss:    unified loss with sample weights, misdir penalty,
                       dead-zone, amplified-zone, correlation/variance/
                       directional penalties
  - get_loss_function: factory from LSTMConfig

Pure math helpers (`batch_pearson`, `batch_r2`, `batch_ccc`,
`log_variance_penalty`, `compute_sample_weights`, `resolve_signal_threshold`)
live in `loss_components.py` and are re-exported here for backward
compatibility — existing `from common.losses import compute_sample_weights`
imports continue to work.
"""
import numpy as np
import torch
import torch.nn as nn

from .loss_components import (
    batch_ccc,
    batch_pearson,
    batch_r2,
    compute_sample_weights,
    cr_ceiling_penalty,
    log_variance_penalty,
    resolve_signal_threshold,
)
from .signal_loss import SignalLoss


# Public API re-exports (kept stable for callers that historically imported
# from `common.losses`). New code should import from `common.loss_components`.
__all__ = [
    'PseudoHuberLoss', 'QuantileLoss', 'LSTMCustomLoss', 'SignalLoss',
    'get_loss_function',
    'resolve_signal_threshold', 'compute_sample_weights',
]


# =========================================================================
# PSEUDO-HUBER LOSS
# =========================================================================

class PseudoHuberLoss(nn.Module):
    """
    Pseudo-Huber loss: delta^2 * (sqrt(1 + (r/delta)^2) - 1)

    Smooth approximation of Huber loss. Differentiable everywhere.
    """

    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred, y_true, weights=None):
        """
        Args:
            y_pred:  (batch,) or (batch, 1)
            y_true:  (batch,) or (batch, 1)
            weights: (batch,) optional per-sample weights

        Returns:
            scalar loss
        """
        y_pred = y_pred.squeeze(-1)
        y_true = y_true.squeeze(-1)
        r = y_pred - y_true
        loss = self.delta ** 2 * (torch.sqrt(1 + (r / self.delta) ** 2) - 1)

        if weights is not None:
            loss = loss * weights

        return loss.mean()


# =========================================================================
# QUANTILE LOSS (F2.4)
# =========================================================================

class QuantileLoss(nn.Module):
    """
    Quantile (pinball) loss — asymmetric absolute error.

    loss = max(q * (y - ŷ), (q - 1) * (y - ŷ))

    For q=0.5 it reduces to MAE / 2. For q=0.7 over-predictions are penalised
    more heavily than under-predictions (biases model toward higher values),
    for q=0.3 the opposite. Useful as a research alternative to MAE/Huber
    when the cost of over/under prediction is asymmetric.
    """

    def __init__(self, q=0.5):
        super().__init__()
        if not 0.0 < q < 1.0:
            raise ValueError(f"Quantile q must be in (0, 1), got {q}")
        self.q = float(q)

    def forward(self, y_pred, y_true, weights=None):
        y_pred = y_pred.squeeze(-1)
        y_true = y_true.squeeze(-1)
        err = y_true - y_pred
        loss = torch.maximum(self.q * err, (self.q - 1.0) * err)
        if weights is not None:
            loss = loss * weights
        return loss.mean()


# =========================================================================
# UNIFIED CUSTOM LOSS
# =========================================================================

class LSTMCustomLoss(nn.Module):
    """
    Unified custom loss with configurable components:

    1. Base loss:           Pseudo-Huber (configurable delta)
    2. Sample weights:      per-sample weight from |y_raw| (optional)
    3. Misdir penalty:      extra weight when sign(pred) != sign(true) (optional)
    4. Dead-zone:           reduced weight for |y_raw| < threshold (optional)
    5. Amplified zone:      boosted weight for |y_raw| >= threshold (optional)
                            — counterpart to dead-zone (F2.3 anti-collapse).
    6. Correlation penalty: additive batch-level term = w * (1 - Pearson(pred,true))
                            — penalizes flat predictions (F2.1a anti-collapse).
    7. Variance penalty:    additive batch-level term = w * relu(1 - var(pred)/var(true))
                            — directly penalizes prediction variance below target
                            variance (F2.1b anti-collapse).
    8. Directional loss:    additive batch-level term (F2.2) proportional to
                            |pred| when sign(pred) != sign(true), normalized
                            by mean(|true|). Bounded roughly in [0, weight].

    Components 1–5 are combined multiplicatively as per-sample multipliers.
    Components 6, 7, 8 are added at batch level after the mean. All are
    scale-invariant or normalised so they contribute at most `weight` when
    the model fully collapses / mispredicts.
    """

    def __init__(self, delta=1.0,
                 use_weights=False, weight_mode='percentile',
                 weight_percentile=70.0, weight_low_w=0.1,
                 use_misdir=False, misdir_penalty=2.0, misdir_quantile=0.7,
                 use_dead_zone=False, dead_zone_threshold=0.001,
                 dead_zone_weight=0.1,
                 use_amplified_zone=False, amplified_zone_threshold=0.002,
                 amplified_zone_weight=2.0,
                 use_correlation_penalty=False,
                 correlation_penalty_weight=0.1,
                 use_variance_penalty=False,
                 variance_penalty_weight=0.1,
                 use_directional_loss=False,
                 directional_loss_weight=0.5,
                 use_log_variance_penalty=False,
                 log_variance_penalty_weight=0.01,
                 use_r2_penalty=False,
                 r2_penalty_weight=0.01,
                 use_ccc_penalty=False,
                 ccc_penalty_weight=0.01,
                 use_cr_ceiling=False,
                 cr_ceiling=1.5,
                 cr_ceiling_weight=0.1,
                 target_scaler=None):
        super().__init__()
        self.delta = delta
        self.use_weights = use_weights
        self.weight_mode = weight_mode
        self.weight_percentile = weight_percentile
        self.weight_low_w = weight_low_w
        self.use_misdir = use_misdir
        self.misdir_penalty = misdir_penalty
        self.misdir_quantile = misdir_quantile
        self.use_dead_zone = use_dead_zone
        self.dead_zone_weight = dead_zone_weight
        self.use_amplified_zone = use_amplified_zone
        self.amplified_zone_threshold = float(amplified_zone_threshold)
        self.amplified_zone_weight = float(amplified_zone_weight)
        self.use_correlation_penalty = use_correlation_penalty
        self.correlation_penalty_weight = float(correlation_penalty_weight)
        self.use_variance_penalty = use_variance_penalty
        self.variance_penalty_weight = float(variance_penalty_weight)
        self.use_directional_loss = use_directional_loss
        self.directional_loss_weight = float(directional_loss_weight)
        # F2.V1-1: log-variance penalty (non-zero gradient at collapse).
        self.use_log_variance_penalty = use_log_variance_penalty
        self.log_variance_penalty_weight = float(log_variance_penalty_weight)
        # F2.V1-3: R² penalty — penalises low fit / negative R².
        self.use_r2_penalty = use_r2_penalty
        self.r2_penalty_weight = float(r2_penalty_weight)
        # F2.V1-4: CCC penalty — penalises low concordance correlation.
        self.use_ccc_penalty = use_ccc_penalty
        self.ccc_penalty_weight = float(ccc_penalty_weight)
        self.use_cr_ceiling = use_cr_ceiling
        self.cr_ceiling = float(cr_ceiling)
        self.cr_ceiling_weight = float(cr_ceiling_weight)
        self.target_scaler = target_scaler

        # F2.V1-2: misdir threshold set once per epoch on RAW y. None = fall
        # back to per-batch quantile (legacy / unset). Stored as scalar tensor
        # without device assignment; we lazy-move to y_pred.device in forward.
        self._misdir_thr = None

        # Dead-zone / amplified-zone checks are performed in RAW (original)
        # space. We keep the thresholds as given and, when targets are scaled,
        # we un-scale y_true inside forward() using torch buffers derived from
        # the target scaler. This avoids the subtle bug where center != 0
        # (e.g. RobustScaler using median) shifts the comparison.
        self.dead_zone_threshold = float(dead_zone_threshold)

        # Register torch buffers for on-device un-scaling (no CPU-GPU sync
        # per batch; moves with model.to(device); saved in state_dict).
        needs_raw_transform = use_dead_zone or use_amplified_zone
        if needs_raw_transform and target_scaler is not None:
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
        else:
            self._has_target_transform = False

        # Pre-computed batch weights (set externally per-epoch if needed)
        self._batch_weights = None

        # Per-batch loss component breakdown (detached floats). Populated each
        # forward() call so train_epoch/validate_epoch can accumulate stats
        # for diagnostics. Keys depend on which components are active.
        self._last_components: dict = {}

    def set_misdir_threshold(self, y_raw_all):
        """
        F2.V1-2: precompute the misdirection threshold once per epoch.

        The old behaviour computed `torch.quantile(abs_true, q)` for each
        batch, which is (a) unstable on small batches and (b) sensitive to
        whether the batch happened to contain large moves. Computing the
        threshold from the full epoch's raw |y| stabilises it.

        Pass either the RAW (un-scaled) targets or the same scaled targets
        used during training — the only requirement is consistency across
        the entire epoch.
        """
        if not self.use_misdir:
            self._misdir_thr = None
            return
        thr = float(np.percentile(np.abs(np.asarray(y_raw_all)),
                                  self.misdir_quantile * 100.0))
        self._misdir_thr = torch.tensor(thr, dtype=torch.float32)

    def set_epoch_weights(self, y_raw_all):
        """
        Pre-compute sample weights from raw targets for the entire dataset.
        Called once per epoch (or once at start if targets don't change).

        Args:
            y_raw_all: np.ndarray — raw log-return targets for the dataset
        """
        if self.use_weights:
            w = compute_sample_weights(
                y_raw_all, mode=self.weight_mode,
                percentile=self.weight_percentile,
                low_w=self.weight_low_w,
            )
            self._batch_weights = torch.FloatTensor(w)
        else:
            self._batch_weights = None

    def forward(self, y_pred, y_true, indices=None):
        """
        Args:
            y_pred:  (batch,) or (batch, 1) — model predictions (scaled)
            y_true:  (batch,) or (batch, 1) — targets (scaled)
            indices: (batch,) — sample indices for weight lookup (optional)

        Returns:
            scalar loss
        """
        y_pred = y_pred.squeeze(-1)
        y_true = y_true.squeeze(-1)

        # 1. Base pseudo-huber loss per sample
        r = y_pred - y_true
        loss = self.delta ** 2 * (torch.sqrt(1 + (r / self.delta) ** 2) - 1)

        # 2. Per-sample multiplier
        multiplier = torch.ones_like(loss)

        # 2a. Sample weights (from pre-computed)
        if self._batch_weights is not None and indices is not None:
            w = self._batch_weights[indices].to(loss.device)
            multiplier = multiplier * w

        # 2b. Misdir penalty: sign(pred) != sign(true) on large moves.
        # F2.V1-2: prefer epoch-level threshold (stable, see set_misdir_threshold)
        # and fall back to per-batch quantile when the threshold was not set.
        if self.use_misdir:
            abs_true = torch.abs(y_true)
            if self._misdir_thr is not None:
                quantile_val = self._misdir_thr.to(abs_true.device,
                                                   dtype=abs_true.dtype)
            else:
                quantile_val = torch.quantile(abs_true, self.misdir_quantile)
            is_large = abs_true >= quantile_val
            is_misdirected = torch.sign(y_pred) != torch.sign(y_true)
            penalty_mask = (is_large & is_misdirected).float()
            multiplier = multiplier * (1.0 + penalty_mask * (self.misdir_penalty - 1.0))

        # Compute y_raw once if either dead-zone or amplified-zone is on.
        y_true_raw = None
        if self.use_dead_zone or self.use_amplified_zone:
            if self._has_target_transform:
                y_true_raw = y_true * self._target_scale + self._target_center
            else:
                y_true_raw = y_true

        # 2c. Dead-zone: reduce weight for small moves (check in RAW space)
        if self.use_dead_zone:
            abs_true_raw = torch.abs(y_true_raw)
            is_small = (abs_true_raw < self.dead_zone_threshold).float()
            dz_multiplier = is_small * self.dead_zone_weight + (1.0 - is_small)
            multiplier = multiplier * dz_multiplier

        # 2d. Amplified zone (F2.3): boost weight for large moves (RAW space)
        #     |y_raw| >= threshold → multiplier × amplified_zone_weight (> 1).
        #     Complementary to dead_zone; together they form a two-sided
        #     selectivity (suppress small / amplify large).
        if self.use_amplified_zone:
            abs_true_raw = torch.abs(y_true_raw)
            is_large = (abs_true_raw >= self.amplified_zone_threshold).float()
            az_multiplier = is_large * self.amplified_zone_weight + (1.0 - is_large)
            multiplier = multiplier * az_multiplier

        loss = loss * multiplier
        base = loss.mean()
        total = base

        # Track per-batch component values (detached floats for diagnostics).
        comps = {'base': float(base.detach().item())}

        # 3. Correlation penalty (F2.1a) — batch-level additive term.
        #    Encourages pred to co-move with true. If pred collapses to a
        #    constant, std(pred)=0 → corr=0 → penalty = weight. Computed
        #    in scaled space (correlation is scale-invariant).
        if self.use_correlation_penalty:
            corr = batch_pearson(y_pred, y_true)
            corr_penalty = self.correlation_penalty_weight * (1.0 - corr)
            total = total + corr_penalty
            comps['corr'] = float(corr.detach().item())
            comps['corr_penalty'] = float(corr_penalty.detach().item())

        # 4. Variance penalty (F2.1b) — batch-level additive term.
        #    Directly attacks flatness: penalizes when var(pred) < var(true).
        #    Uses ratio in [0, ∞), clamped by relu so no bonus for excess
        #    variance. Scale-invariant (both variances in scaled space).
        #    var_ratio = var(pred) / var(true); penalty = w * relu(1 - ratio).
        if self.use_variance_penalty:
            var_p = y_pred.var(unbiased=False)
            var_t = y_true.var(unbiased=False)
            var_ratio = var_p / (var_t + 1e-12)
            var_penalty = self.variance_penalty_weight * torch.clamp(
                1.0 - var_ratio, min=0.0)
            total = total + var_penalty
            comps['var_ratio'] = float(var_ratio.detach().item())
            comps['var_penalty'] = float(var_penalty.detach().item())

        # 5. Directional loss (F2.2) — batch-level additive term.
        #    relu(-sign(true) * pred) is 0 when correctly directed and |pred|
        #    when mis-directed. Normalised by mean(|y_true|) to stay
        #    dimensionless; bounded roughly in [0, weight] when pred scale
        #    matches true scale. Returns 0 when pred collapses to 0 — that
        #    case is handled by correlation/variance penalties, not here.
        if self.use_directional_loss:
            true_sign = torch.sign(y_true).detach()
            raw_pen = torch.relu(-true_sign * y_pred)
            denom = torch.abs(y_true).mean().detach() + 1e-8
            dir_loss = self.directional_loss_weight * raw_pen.mean() / denom
            total = total + dir_loss
            comps['dir_loss'] = float(dir_loss.detach().item())

        # 6. Log-variance penalty (F2.V1-1) — relu(log_var_true - log_var_pred)
        #    Non-zero gradient at the collapse point (var(pred) → 0), where
        #    the linear variance penalty already saturated. Complementary to
        #    `use_variance_penalty` rather than a replacement.
        if self.use_log_variance_penalty:
            raw_lv = log_variance_penalty(y_pred, y_true)
            lv_pen = self.log_variance_penalty_weight * raw_lv
            total = total + lv_pen
            comps['log_var_pen_raw'] = float(raw_lv.detach().item())
            comps['log_var_penalty'] = float(lv_pen.detach().item())

        # 7. R² penalty (F2.V1-3) — relu(1 - R²)
        #    R²=1 → penalty 0; R²<<1 (constant prediction) → penalty ≈ weight.
        #    Clamping at 0 ensures we never reward over-fitting beyond R²>1.
        if self.use_r2_penalty:
            r2 = batch_r2(y_pred, y_true)
            r2_pen = self.r2_penalty_weight * torch.clamp(1.0 - r2, min=0.0)
            total = total + r2_pen
            comps['r2'] = float(r2.detach().item())
            comps['r2_penalty'] = float(r2_pen.detach().item())

        # 8. CCC penalty (F2.V1-4) — weight * (1 - CCC)
        #    Sensitive to bias / scale mismatch in addition to correlation,
        #    so it complements the pure Pearson correlation penalty (#3).
        if self.use_ccc_penalty:
            ccc = batch_ccc(y_pred, y_true)
            ccc_pen = self.ccc_penalty_weight * torch.clamp(1.0 - ccc, min=0.0)
            total = total + ccc_pen
            comps['ccc'] = float(ccc.detach().item())
            comps['ccc_penalty'] = float(ccc_pen.detach().item())

        if self.use_cr_ceiling:
            raw_cr = cr_ceiling_penalty(
                y_pred, y_true, ceiling=self.cr_ceiling)
            cr_pen = self.cr_ceiling_weight * raw_cr
            total = total + cr_pen
            comps['cr_ceiling_penalty'] = float(cr_pen.detach().item())

        comps['total'] = float(total.detach().item())
        self._last_components = comps
        return total


# =========================================================================
# FACTORY
# =========================================================================

def get_loss_function(config, target_scaler=None, signal_threshold=None):
    """
    Create loss function from LSTMConfig.

    Args:
        config:           LSTMConfig class or compatible object
        target_scaler:    TargetScaler instance or None
        signal_threshold: float or None (for dead-zone threshold)

    Returns:
        nn.Module — loss function
    """
    loss_type = config.LOSS_TYPE

    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'huber':
        return nn.HuberLoss(delta=config.HUBER_DELTA)
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif loss_type == 'pseudo_huber':
        return PseudoHuberLoss(delta=config.HUBER_DELTA)
    elif loss_type == 'quantile':
        return QuantileLoss(q=getattr(config, 'QUANTILE_LOSS_Q', 0.5))
    elif loss_type == 'signal':
        return SignalLoss(
            use_r2_penalty=getattr(config, 'SIGNAL_USE_R2', True),
            r2_penalty_weight=getattr(config, 'SIGNAL_R2_WEIGHT', 0.01),
            use_ccc_penalty=getattr(config, 'SIGNAL_USE_CCC', True),
            ccc_penalty_weight=getattr(config, 'SIGNAL_CCC_WEIGHT', 0.01),
            use_log_var_penalty=getattr(config, 'SIGNAL_USE_LOG_VAR', False),
            log_var_penalty_weight=getattr(
                config, 'SIGNAL_LOG_VAR_WEIGHT', 0.005),
            use_std_match_penalty=getattr(
                config, 'SIGNAL_USE_STD_MATCH', False),
            std_match_penalty_weight=getattr(
                config, 'SIGNAL_STD_MATCH_WEIGHT', 0.05),
            use_dir_threshold_penalty=getattr(
                config, 'SIGNAL_USE_DIR_THRESHOLD', False),
            dir_threshold_penalty_weight=getattr(
                config, 'SIGNAL_DIR_THRESHOLD_WEIGHT', 0.1),
            use_direction_bce=getattr(
                config, 'SIGNAL_USE_DIRECTION_BCE', False),
            direction_bce_weight=getattr(
                config, 'SIGNAL_DIRECTION_BCE_WEIGHT', 0.01),
            signal_threshold=(signal_threshold
                              or getattr(config, 'SIGNAL_THRESHOLD_VALUE',
                                         0.001)),
            ema_penalties=getattr(config, 'SIGNAL_EMA_PENALTIES', False),
            ema_alpha=getattr(config, 'SIGNAL_EMA_ALPHA', 0.1),
            label_smoothing=getattr(config, 'SIGNAL_LABEL_SMOOTHING', False),
            label_smoothing_alpha=getattr(
                config, 'SIGNAL_LABEL_SMOOTHING_ALPHA', 0.5),
            use_utility=getattr(config, 'SIGNAL_USE_UTILITY', False),
            utility_weight=getattr(config, 'SIGNAL_UTILITY_WEIGHT', 0.05),
            soft_threshold=getattr(config, 'SIGNAL_SOFT_THRESHOLD', False),
            use_cr_ceiling=getattr(config, 'USE_CR_CEILING', False),
            cr_ceiling=getattr(config, 'CR_CEILING', 1.5),
            cr_ceiling_weight=getattr(config, 'CR_CEILING_WEIGHT', 0.1),
            target_scaler=target_scaler,
        )
    elif loss_type == 'custom':
        dz_threshold = signal_threshold or config.SIGNAL_THRESHOLD_VALUE
        return LSTMCustomLoss(
            delta=config.HUBER_DELTA,
            use_weights=config.USE_SAMPLE_WEIGHTS,
            weight_mode=config.SAMPLE_WEIGHT_MODE,
            weight_percentile=config.SAMPLE_WEIGHT_PERCENTILE,
            weight_low_w=config.SAMPLE_WEIGHT_LOW_W,
            use_misdir=config.USE_MISDIR_PENALTY,
            misdir_penalty=config.MISDIR_PENALTY,
            misdir_quantile=config.MISDIR_MIN_QUANTILE,
            use_dead_zone=config.USE_DEAD_ZONE,
            dead_zone_threshold=dz_threshold,
            dead_zone_weight=config.DEAD_ZONE_WEIGHT,
            use_amplified_zone=getattr(
                config, 'USE_AMPLIFIED_ZONE', False),
            amplified_zone_threshold=getattr(
                config, 'AMPLIFIED_ZONE_THRESHOLD', 0.002),
            amplified_zone_weight=getattr(
                config, 'AMPLIFIED_ZONE_WEIGHT', 2.0),
            use_correlation_penalty=getattr(
                config, 'USE_CORRELATION_PENALTY', False),
            correlation_penalty_weight=getattr(
                config, 'CORRELATION_PENALTY_WEIGHT', 0.1),
            use_variance_penalty=getattr(
                config, 'USE_VARIANCE_PENALTY', False),
            variance_penalty_weight=getattr(
                config, 'VARIANCE_PENALTY_WEIGHT', 0.1),
            use_directional_loss=getattr(
                config, 'USE_DIRECTIONAL_LOSS', False),
            directional_loss_weight=getattr(
                config, 'DIRECTIONAL_LOSS_WEIGHT', 0.5),
            use_log_variance_penalty=getattr(
                config, 'USE_LOG_VARIANCE_PENALTY', False),
            log_variance_penalty_weight=getattr(
                config, 'LOG_VARIANCE_PENALTY_WEIGHT', 0.01),
            use_r2_penalty=getattr(
                config, 'USE_R2_PENALTY', False),
            r2_penalty_weight=getattr(
                config, 'R2_PENALTY_WEIGHT', 0.01),
            use_ccc_penalty=getattr(
                config, 'USE_CCC_PENALTY', False),
            ccc_penalty_weight=getattr(
                config, 'CCC_PENALTY_WEIGHT', 0.01),
            use_cr_ceiling=getattr(config, 'USE_CR_CEILING', False),
            cr_ceiling=getattr(config, 'CR_CEILING', 1.5),
            cr_ceiling_weight=getattr(config, 'CR_CEILING_WEIGHT', 0.1),
            target_scaler=target_scaler,
        )
    else:
        raise ValueError(f"Unknown LOSS_TYPE: '{loss_type}'")
