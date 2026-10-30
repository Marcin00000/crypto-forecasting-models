"""
Checkpoint metric helpers for Signal Model: val_signal score and SWA.

Analogous to lstm_logreturn_model/common/training_checkpoint.py but adapted
to signal semantics:
  - signal model outputs p_buy/p_sell directly (no log-return → compute_signals)
  - val_signal score uses corr_avg + pct_strong + sim_capture_log
    (replaces dir_acc_strong from the LR variant)
"""
from __future__ import annotations

import numpy as np
import torch

VAL_SIGNAL_W_CAPTURE = 0.50
VAL_SIGNAL_W_CORR = 0.30
VAL_SIGNAL_W_ACTIVITY = 0.20


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# =========================================================================
# SWA
# =========================================================================

def update_swa(swa_state: dict | None, model: torch.nn.Module,
               n_swa: int) -> tuple[dict, int]:
    """
    Running average of model weights on CPU (in-place update).

    Only floating-point tensors are averaged. Non-floating buffers
    (e.g. ``num_batches_tracked`` in ``nn.BatchNorm1d``, dtype int64) are
    copied verbatim from the current model — averaging them would yield a
    float that cannot be stored back in an int64 buffer.

    Args:
        swa_state: previous running-average state dict, or None on first call.
        model:     current model (on any device).
        n_swa:     current SWA step counter (1-based, passed in from caller).

    Returns:
        (updated_swa_state, n_swa) — n_swa unchanged (caller increments it).
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


# =========================================================================
# VAL_SIGNAL CHECKPOINT
# =========================================================================

def compute_val_signal_score(sig_metrics: dict) -> tuple[float, float]:
    """
    Compute a composite val_signal score from signal evaluation metrics.

    Score components (all clipped to [0, 1] before weighting):
      - capture  (0.50): sim_capture_log clamped to [0, 1]
      - corr     (0.30): (corr_buy + corr_sell) / 2, clamped to [0, 1]
      - activity (0.20): pct_strong / 20 (full points at 20%)

    Returns:
        (score in [0,1], monitor_loss = -score for minimisation)
    """
    capture_log = sig_metrics.get('sim_capture_log')
    corr_buy = sig_metrics.get('corr_buy', 0.0) or 0.0
    corr_sell = sig_metrics.get('corr_sell', 0.0) or 0.0
    pct_strong = sig_metrics.get('pct_strong', 0.0) or 0.0

    capture_term = _clamp01(capture_log) if capture_log is not None else 0.0
    corr_avg = (corr_buy + corr_sell) / 2.0
    corr_term = _clamp01(max(corr_avg, 0.0))
    activity_term = _clamp01(pct_strong / 20.0)

    score = (VAL_SIGNAL_W_CAPTURE * capture_term
             + VAL_SIGNAL_W_CORR * corr_term
             + VAL_SIGNAL_W_ACTIVITY * activity_term)
    return score, -score


def resolve_monitor_loss(*, metric_name: str, val_loss: float,
                         diag_entry: dict | None,
                         val_signal_ctx: dict | None,
                         epoch: int,
                         val_signal_warned: bool,
                         val_preds: np.ndarray | None,
                         val_targets: np.ndarray | None,
                         val_signal_min_epoch: int,
                         logger) -> tuple[float, str, dict, bool, bool]:
    """
    Pick monitor_loss and metric_label for checkpoint / patience tracking.

    Supported metric_name values:
        'val'           → raw val_loss (BCE)
        'val_composite' → val_loss + diversity_weight * (1 - diversity)
                          (diversity comes from diag_entry)
        'val_signal'    → -score from compute_val_signal_score
                          (active only from val_signal_min_epoch onward)

    Args:
        metric_name:          BEST_MODEL_METRIC setting (str).
        val_loss:             raw BCE validation loss (float).
        diag_entry:           per-epoch diagnostic dict from SignalEpochDiagnostics
                              (may be None if diagnostics disabled).
        val_signal_ctx:       context dict for val_signal evaluation (see
                              build_val_signal_ctx), or None.
        epoch:                current epoch index (0-based).
        val_signal_warned:    whether the "ctx missing" warning was emitted.
        val_preds:            (n, 2) model predictions on val set in [0,1].
        val_targets:          (n, 2) oracle targets on val set in [0,1].
        val_signal_min_epoch: earliest epoch at which val_signal is activated.
        logger:               training logger.

    Returns:
        (monitor_loss, metric_label, diag_updates, val_signal_warned, domain_switch)
        domain_switch=True on the epoch where val_signal first activates.
    """
    diag_updates: dict = {}

    if metric_name == 'val_composite':
        diversity = 0.0
        if diag_entry:
            diversity = diag_entry.get('signal_diversity', 0.0) or 0.0
        weight = val_signal_ctx.get('diversity_weight', 0.01) if val_signal_ctx else 0.01
        monitor_loss = val_loss + weight * (1.0 - diversity)
        return (monitor_loss,
                f"val_composite(div={diversity:.4f}, w={weight:.4g})",
                diag_updates, val_signal_warned, False)

    if metric_name == 'val_signal':
        if (val_signal_ctx is not None
                and val_preds is not None
                and val_targets is not None
                and epoch + 1 >= val_signal_min_epoch):

            from .simulation import evaluate_signal_simulation, derive_sim_capture_log
            from .metrics import signal_correlation, thresholded_precision

            min_signal = val_signal_ctx.get('min_signal', 60.0)
            horizon = val_signal_ctx.get('horizon', 1)
            y_true_lr = val_signal_ctx.get('y_true_lr')
            start_cash = val_signal_ctx.get('start_cash', 100.0)
            target_oracle = val_signal_ctx.get('target_oracle')

            sig_metrics: dict = {}
            sig_metrics.update(signal_correlation(val_preds, val_targets))

            if y_true_lr is not None:
                prec = thresholded_precision(val_preds, y_true_lr,
                                            min_signal_pct=min_signal)
                sig_metrics['pct_strong'] = prec.get('pct_strong', 0.0)

            if target_oracle is not None and y_true_lr is not None:
                sim_out = evaluate_signal_simulation(
                    val_preds, target_oracle, y_true_lr,
                    horizon=horizon, min_signal=min_signal,
                    start_cash=start_cash)
                sig_metrics['sim_capture_log'] = sim_out.get('sim_capture_log')
            else:
                sig_metrics['sim_capture_log'] = None

            val_signal_score, monitor_loss = compute_val_signal_score(sig_metrics)
            domain_switch = (epoch + 1 == val_signal_min_epoch)

            diag_updates = {
                'val_signal_score': val_signal_score,
                'val_pct_strong': sig_metrics.get('pct_strong'),
                'val_corr_buy': sig_metrics.get('corr_buy'),
                'val_corr_sell': sig_metrics.get('corr_sell'),
                'val_sim_capture_log': sig_metrics.get('sim_capture_log'),
            }
            return (monitor_loss,
                    f"val_signal(score={val_signal_score:.4f})",
                    diag_updates, val_signal_warned, domain_switch)

        if val_signal_ctx is None and not val_signal_warned:
            logger.warning(
                "BEST_MODEL_METRIC='val_signal' but val_signal_ctx is missing"
                " — falling back to val_loss for checkpoint"
            )
            val_signal_warned = True

    return val_loss, 'val_loss', diag_updates, val_signal_warned, False


def build_val_signal_ctx(val_targets: np.ndarray | None,
                         y_true_val_lr: np.ndarray | None,
                         horizon: int,
                         min_signal: float,
                         diversity_weight: float = 0.01,
                         start_cash: float = 100.0) -> dict | None:
    """
    Build context dict for per-epoch val_signal checkpointing.

    Args:
        val_targets:       (n, 2) oracle targets (p_buy, p_sell) on val set.
        y_true_val_lr:     (n,) realised log-returns on val set.
        horizon:           forward horizon (used in simulation).
        min_signal:        strong-signal cutoff in [0, 100].
        diversity_weight:  weight for (1-diversity) in val_composite fallback.
        start_cash:        starting capital for simulation (default 100).

    Returns:
        dict ready to pass as val_signal_ctx, or None if inputs missing.
    """
    if val_targets is None or y_true_val_lr is None:
        return None
    return {
        'target_oracle': val_targets,
        'y_true_lr': y_true_val_lr,
        'horizon': horizon,
        'min_signal': min_signal,
        'start_cash': start_cash,
        'diversity_weight': diversity_weight,
    }
