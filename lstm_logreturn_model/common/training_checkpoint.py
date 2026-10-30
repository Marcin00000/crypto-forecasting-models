"""Checkpoint metric helpers: val_signal score, SWA weight averaging."""
from __future__ import annotations

import numpy as np
import torch

from lstm_config import LSTMConfig
from .simulation import evaluate_signal_and_simulation

VAL_SIGNAL_W_CAPTURE = 0.5
VAL_SIGNAL_W_DIRACC = 0.3
VAL_SIGNAL_W_ACTIVITY = 0.2


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def compute_val_signal_score(sig_out):
    """Return (score in [0,1], monitor_loss for minimisation)."""
    capture_log = sig_out.get('sim_capture_log')
    dir_acc = sig_out.get('dir_acc_strong')
    pct_strong = sig_out.get('pct_strong')

    capture_term = (_clamp01(capture_log)
                    if capture_log is not None else 0.0)
    dir_term = (_clamp01((dir_acc - 50.0) / 10.0)
                if dir_acc is not None else 0.0)
    activity_term = (_clamp01(pct_strong / 20.0)
                     if pct_strong is not None else 0.0)

    score = (VAL_SIGNAL_W_CAPTURE * capture_term
             + VAL_SIGNAL_W_DIRACC * dir_term
             + VAL_SIGNAL_W_ACTIVITY * activity_term)
    return score, -score


def update_swa(swa_state, model, n_swa):
    """Running average of model weights on CPU (in-place update).

    Only floating-point tensors are averaged. Non-floating buffers (e.g.
    `num_batches_tracked` in `nn.BatchNorm1d`, dtype int64) are copied
    verbatim from the current model — averaging them would produce a float
    result that cannot be cast back into an int64 buffer in-place.
    """
    current = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if swa_state is None:
        return current, n_swa
    for k in swa_state:
        if torch.is_floating_point(swa_state[k]):
            swa_state[k] += (current[k] - swa_state[k]) / n_swa
        else:
            swa_state[k] = current[k]
    return swa_state, n_swa


def resolve_monitor_loss(*, metric_name, train_loss, val_loss, diag_rec,
                         val_signal_ctx, epoch, val_signal_warned,
                         y_val_pred_scaled, y_val_true_scaled_cached,
                         target_scaler, logger):
    """
    Pick monitor_loss and metric_label for checkpoint / patience.

    Returns (monitor_loss, metric_label, diag_updates, warned, domain_switch).
  """
    val_signal_min_epoch = getattr(LSTMConfig, 'VAL_SIGNAL_MIN_EPOCH', 5)
    diag_updates = {}

    if metric_name == 'train':
        return train_loss, 'train_loss', diag_updates, val_signal_warned, False

    if metric_name == 'val_composite':
        cr = diag_rec.get('compression_ratio')
        cr_valid = cr is not None and np.isfinite(cr)
        cr_penalty = abs(1.0 - cr) if cr_valid else 0.0
        if not cr_valid:
            logger.debug(
                "val_composite: CR unavailable — falling back to pure val_loss"
            )
        cr_weight = getattr(LSTMConfig, 'BEST_MODEL_CR_WEIGHT', 0.01)
        monitor_loss = val_loss + cr_weight * cr_penalty
        cr_label = f"{cr:.4f}" if cr_valid else "n/a"
        return (monitor_loss,
                f"val_composite(cr={cr_label}, w={cr_weight:.4g})",
                diag_updates, val_signal_warned, False)

    if metric_name == 'val_signal':
        if val_signal_ctx is not None and epoch + 1 >= val_signal_min_epoch:
            if target_scaler is not None:
                y_val_pred_raw = target_scaler.inverse_transform(
                    y_val_pred_scaled)
                y_val_true_raw = target_scaler.inverse_transform(
                    y_val_true_scaled_cached)
            else:
                y_val_pred_raw = y_val_pred_scaled
                y_val_true_raw = y_val_true_scaled_cached

            sig_out = evaluate_signal_and_simulation(
                y_val_true_raw, y_val_pred_raw,
                val_signal_ctx['close'],
                lookback=val_signal_ctx['lookback'],
                horizon=val_signal_ctx['horizon'],
                threshold=val_signal_ctx['threshold'],
                window_sigma=val_signal_ctx['window_sigma'],
                min_signal=val_signal_ctx['min_signal'],
                sigma=val_signal_ctx.get('sigma'),
            )
            val_signal_score, monitor_loss = compute_val_signal_score(sig_out)
            domain_switch = (epoch + 1 == val_signal_min_epoch)
            diag_updates = {
                'val_signal_score': val_signal_score,
                'val_pct_strong': sig_out.get('pct_strong'),
                'val_dir_acc_strong': sig_out.get('dir_acc_strong'),
                'val_sim_capture_log': sig_out.get('sim_capture_log'),
            }
            return (monitor_loss,
                    f"val_signal(score={val_signal_score:.4f})",
                    diag_updates, val_signal_warned, domain_switch)

        if val_signal_ctx is None and not val_signal_warned:
            logger.warning(
                "BEST_MODEL_METRIC='val_signal' but val_signal_ctx "
                "is missing — falling back to val_loss for checkpoint"
            )
            val_signal_warned = True
    return val_loss, 'val_loss', diag_updates, val_signal_warned, False


def build_val_signal_ctx(val_close, lookback, horizon, signal_threshold, cfg,
                         sigma=None):
    """Context dict for per-epoch val_signal checkpointing."""
    if val_close is None:
        return None
    return {
        'close': val_close,
        'lookback': lookback,
        'horizon': horizon,
        'threshold': (signal_threshold
                      if signal_threshold is not None
                      else cfg.SIGNAL_THRESHOLD_VALUE),
        'window_sigma': cfg.SIGNAL_WINDOW_SIGMA,
        'min_signal': cfg.SIGNAL_MIN_STRONG,
        'sigma': sigma,
    }
