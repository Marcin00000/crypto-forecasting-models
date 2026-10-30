"""
Metrics for Signal Model pipeline.

Signal-specific metrics: MAE, correlation, thresholded precision,
signal diversity, and calibration.
"""
import numpy as np

from .simulation import compute_signal_thresholds


def signal_mae(y_pred, y_oracle):
    """
    Mean absolute error per channel (buy, sell) and combined.

    Args:
        y_pred:   (n, 2) predictions in [0, 1]
        y_oracle: (n, 2) oracle targets in [0, 1]

    Returns:
        dict with mae_buy, mae_sell, mae_combined
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_oracle = np.asarray(y_oracle, dtype=np.float64)
    errors = np.abs(y_pred - y_oracle)
    return {
        'mae_buy': float(errors[:, 0].mean()),
        'mae_sell': float(errors[:, 1].mean()),
        'mae_combined': float(errors.mean()),
    }


def signal_correlation(y_pred, y_oracle):
    """
    Pearson correlation per channel.

    Returns:
        dict with corr_buy, corr_sell
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_oracle = np.asarray(y_oracle, dtype=np.float64)

    def _pearson(a, b):
        if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        'corr_buy': _pearson(y_pred[:, 0], y_oracle[:, 0]),
        'corr_sell': _pearson(y_pred[:, 1], y_oracle[:, 1]),
    }


def signal_diversity(y_pred):
    """
    Anti-collapse metric: mean(|pred - 0.5|).

    0.0 = all predictions are neutral (0.5) = total collapse.
    Higher = more decisive signals.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - 0.5)))


def threshold_activity_metrics(signal_buy, signal_sell, y_true_lr=None, *,
                               min_signal=60.0, threshold_mode='absolute',
                               percentile_cutoff=5.0, zscore_cutoff=2.0,
                               threshold_history='full', warmup=0,
                               signal_threshold=0.001, eligible=None):
    """Precision, calibration and activity based on shared threshold series."""
    buy = np.asarray(signal_buy, dtype=np.float64)
    sell = np.asarray(signal_sell, dtype=np.float64)
    n = min(len(buy), len(sell))
    if y_true_lr is not None:
        n = min(n, len(y_true_lr))
    buy, sell = buy[:n], sell[:n]
    thresholds = compute_signal_thresholds(
        buy, sell, mode=threshold_mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=threshold_history, warmup=warmup)
    valid = thresholds['eligible'].copy()
    if eligible is not None:
        valid &= np.asarray(eligible, dtype=bool)[:n]
    y_true = None
    if y_true_lr is not None:
        y_true = np.asarray(y_true_lr, dtype=np.float64)[:n]
        valid &= np.isfinite(y_true)
    strong_buy = valid & (buy >= thresholds['buy'])
    strong_sell = valid & (sell >= thresholds['sell'])
    n_valid = int(valid.sum())

    def _rate(mask, predicate):
        return float(predicate(y_true[mask]).mean()) if mask.any() else None

    def _last(values):
        finite = np.asarray(values)[np.isfinite(values)]
        return float(finite[-1]) if finite.size else None

    return {
        'precision_buy': _rate(strong_buy, lambda x: x > 0) if y_true is not None else None,
        'precision_sell': _rate(strong_sell, lambda x: x < 0) if y_true is not None else None,
        'calib_buy': _rate(strong_buy, lambda x: x > signal_threshold) if y_true is not None else None,
        'calib_sell': _rate(strong_sell, lambda x: x < -signal_threshold) if y_true is not None else None,
        'n_strong_buy': int(strong_buy.sum()),
        'n_strong_sell': int(strong_sell.sum()),
        'pct_strong': float((strong_buy | strong_sell).sum() / n_valid * 100.0)
        if n_valid else 0.0,
        'threshold_mode_eval': threshold_mode,
        'threshold_history_eval': threshold_history,
        'threshold_warmup_eval': int(warmup),
        'thresh_buy_eval': _last(thresholds['buy']),
        'thresh_sell_eval': _last(thresholds['sell']),
    }


def thresholded_precision(y_pred, y_true_log_returns, min_signal_pct=60.0,
                          threshold_mode='absolute', percentile_cutoff=5.0,
                          zscore_cutoff=2.0, signal_threshold=0.001,
                          threshold_history='full', warmup=0):
    """
    Precision of strong buy/sell signals against realised direction and economic threshold.

    Args:
        threshold_mode: 'absolute' | 'percentile' | 'zscore'
        signal_threshold: economic threshold T for calib_buy/calib_sell (default 0.001)
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return threshold_activity_metrics(
        y_pred[:, 0], y_pred[:, 1], y_true_log_returns,
        min_signal=float(min_signal_pct) / 100.0,
        threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        signal_threshold=signal_threshold,
        threshold_history=threshold_history,
        warmup=warmup)


def calculate_signal_metrics(y_pred, y_oracle, y_true_log_returns=None,
                             min_signal_pct=60.0,
                             threshold_mode='absolute',
                             percentile_cutoff=5.0,
                             zscore_cutoff=2.0,
                             threshold_history='full',
                             warmup=0):
    """
    Compute all signal metrics in one call.

    Args:
        y_pred:             (n, 2) model output in [0, 1]
        y_oracle:           (n, 2) oracle targets in [0, 1]
        y_true_log_returns: (n,) optional — for thresholded precision
        min_signal_pct:     threshold for strong signal
        threshold_mode:     'absolute' | 'percentile' | 'zscore'

    Returns:
        flat dict with all metrics
    """
    metrics = {}
    metrics.update(signal_mae(y_pred, y_oracle))
    metrics.update(signal_correlation(y_pred, y_oracle))
    metrics['diversity'] = signal_diversity(y_pred)

    if y_true_log_returns is not None:
        metrics.update(thresholded_precision(
            y_pred, y_true_log_returns, min_signal_pct,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            threshold_history=threshold_history,
            warmup=warmup))

    return metrics


# =========================================================================
# SIGNAL RANK SCORE (analogous to LR rank_v2)
# =========================================================================

RANK_ACTIVITY_FULL = 10.0
RANK_CORR_FULL = 0.5
RANK_DIVERSITY_FULL = 0.5
RANK_CAPTURE_LOG_FULL = 0.10
RANK_MAE_WORST = 0.25

RANK_W_ACTIVITY = 25
RANK_W_CORRELATION = 25
RANK_W_DIVERSITY = 20
RANK_W_CAPTURE = 25
RANK_W_MAE_HEALTH = 5


def compute_signal_rank_score(metrics: dict, threshold_mode: str = 'absolute') -> dict:
    """
    Compute composite rank score for a signal model (0–100).

    Components (sum = 100):
      - activity (25):    pct_strong / 10% → full 25 pts
      - correlation (25): corr_avg / 0.5 → full 25 pts
      - diversity (20):   diversity / 0.5 → full 20 pts
      - capture (25):     sim_capture_log / 0.10 → full 25 pts
      - mae_health (5):   (1 - mae_avg/0.25) → full 5 pts

    Gate DEAD: pct_strong < 1% → score=0, signal_class='DEAD' (when threshold_mode='absolute').

    RankScore uses only ``pct_strong`` and ``sim_capture_log``. For schema v2
    those aliases are primary absolute-20 gross plus canonical oracle; net
    fields such as ``sim_final_model_net`` are ignored.

    Returns:
        dict with rank_score, signal_class, rank_components,
        signal_rank_version.
    """
    pct_strong = metrics.get('pct_strong', 0.0) or 0.0
    corr_buy = metrics.get('corr_buy', 0.0) or 0.0
    corr_sell = metrics.get('corr_sell', 0.0) or 0.0
    diversity = metrics.get('diversity', 0.0) or 0.0
    sim_capture_log = metrics.get('sim_capture_log')
    mae_buy = metrics.get('mae_buy', 0.25)
    mae_sell = metrics.get('mae_sell', 0.25)

    corr_avg = (corr_buy + corr_sell) / 2.0
    mae_avg = (mae_buy + mae_sell) / 2.0

    if metrics.get('signal_rank_version') is not None:
        rank_ver = int(metrics['signal_rank_version'])
    elif sim_capture_log is not None:
        rank_ver = 2
    else:
        rank_ver = 1

    eval_mode = metrics.get('threshold_mode_eval', threshold_mode)
    if eval_mode == 'absolute' and pct_strong < 1.0:
        return {
            'rank_score': 0.0,
            'signal_class': 'DEAD',
            'rank_components': {
                'activity': 0.0,
                'correlation': 0.0,
                'diversity': 0.0,
                'capture': 0.0,
                'mae_health': 0.0,
            },
            'signal_rank_version': rank_ver,
        }

    activity = _clamp01(pct_strong / RANK_ACTIVITY_FULL) * RANK_W_ACTIVITY
    correlation = _clamp01(max(corr_avg, 0.0) / RANK_CORR_FULL) * RANK_W_CORRELATION
    div_score = _clamp01(diversity / RANK_DIVERSITY_FULL) * RANK_W_DIVERSITY

    if sim_capture_log is not None:
        capture = _clamp01(sim_capture_log / RANK_CAPTURE_LOG_FULL) * RANK_W_CAPTURE
    else:
        capture = 0.0

    mae_health = _clamp01(1.0 - mae_avg / RANK_MAE_WORST) * RANK_W_MAE_HEALTH

    rank_score = activity + correlation + div_score + capture + mae_health

    if rank_score > 70:
        signal_class = 'STRONG'
    elif rank_score >= 40:
        signal_class = 'OK'
    else:
        signal_class = 'WEAK'

    return {
        'rank_score': round(rank_score, 2),
        'signal_class': signal_class,
        'rank_components': {
            'activity': round(activity, 2),
            'correlation': round(correlation, 2),
            'diversity': round(div_score, 2),
            'capture': round(capture, 2),
            'mae_health': round(mae_health, 2),
        },
        'signal_rank_version': rank_ver,
    }


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, x))
