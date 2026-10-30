"""
Metrics calculation utilities for Random Forest model - log return only.

Includes standard regression metrics, threshold directional accuracy,
and the multidimensional Rank Score (0-100) from Simulation v2.
"""
from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    'pseudo_huber_loss',
    'calculate_metrics',
    'calculate_threshold_metrics',
    'compute_rank_score',
    'print_metrics_summary',
    'interpret_metrics',
    'RANK_CR_DEAD',
    'RANK_PCT_STRONG_DEAD',
    'RANK_METRIC_VERSION',
]

# Rank Score weightings and parameters (Sum = 100 pts)
RANK_CR_DEAD = 0.10
RANK_PCT_STRONG_DEAD = 1.0

RANK_PCT_STRONG_FULL = 20.0
RANK_CALIB_FLOOR = 0.50
RANK_CALIB_FULL = 0.70
RANK_DIRACC_FULL = 60.0

RANK_W_ACTIVITY = 25.0
RANK_W_CALIBRATION = 25.0
RANK_W_DIRECTION = 20.0
RANK_W_CAPTURE = 25.0
RANK_W_CR_HEALTH = 5.0

RANK_CAPTURE_LOG_FULL = 0.10
RANK_METRIC_VERSION = 2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def pseudo_huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    """
    Pseudo-Huber loss (mean over samples).
    L = delta^2 * (sqrt(1 + ((y_pred - y_true) / delta)^2) - 1)
    """
    residuals = y_pred - y_true
    return float(np.mean(delta ** 2 * (np.sqrt(1.0 + (residuals / delta) ** 2) - 1.0)))


def calculate_metrics(y_true_log, y_pred_log) -> dict:
    """
    Calculate regression metrics on log returns, including compression ratio.
    """
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)

    metrics = {
        'mae':  float(mean_absolute_error(y_true_log, y_pred_log)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        'mse':  float(mean_squared_error(y_true_log, y_pred_log)),
        'r2':   float(r2_score(y_true_log, y_pred_log)),
    }

    metrics['pseudo_huber']       = pseudo_huber_loss(y_true_log, y_pred_log, delta=1.0)
    metrics['pseudo_huber_large'] = pseudo_huber_loss(y_true_log, y_pred_log, delta=10.0)

    errors = y_pred_log - y_true_log
    metrics['error_mean']   = float(np.mean(errors))
    metrics['error_std']    = float(np.std(errors))
    metrics['error_median'] = float(np.median(errors))
    metrics['true_mean']    = float(np.mean(y_true_log))
    metrics['pred_mean']    = float(np.mean(y_pred_log))

    p_std = float(np.std(y_pred_log))
    t_std = float(np.std(y_true_log))
    cr = float(p_std / t_std) if t_std > 1e-12 else 0.0
    corr = float(np.corrcoef(y_pred_log, y_true_log)[0, 1]) if (p_std > 1e-12 and t_std > 1e-12) else 0.0

    metrics['pred_std'] = p_std
    metrics['true_std'] = t_std
    metrics['compression_ratio'] = cr
    metrics['corr'] = corr

    actual_dir = np.sign(y_true_log)
    pred_dir   = np.sign(y_pred_log)
    metrics['dir_acc'] = float(np.mean(actual_dir == pred_dir) * 100)

    return metrics


def calculate_threshold_metrics(y_true_log, y_pred_log, threshold: float) -> dict:
    """
    Calculate metrics for samples where |y_pred_log| >= threshold.
    """
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)
    mask = np.abs(y_pred_log) >= float(threshold)
    n_above = int(np.sum(mask))
    n_total = len(y_pred_log)

    if n_above == 0:
        return {
            'threshold': float(threshold),
            'n_above': 0,
            'pct_above': 0.0,
            't_mae': None,
            't_rmse': None,
            't_dir_acc': None,
            't_bias_buy': None,
            'mae_thresh': None,
            'rmse_thresh': None,
            'r2_thresh': None,
            'dir_acc_thresh': None,
            'bias_buy': None,
        }

    yt = y_true_log[mask]
    yp = y_pred_log[mask]

    t_mae = float(mean_absolute_error(yt, yp))
    t_rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    t_r2 = float(r2_score(yt, yp))
    t_dir_acc = float(np.mean(np.sign(yt) == np.sign(yp)) * 100)
    t_bias_buy = float(np.mean(yp > 0) * 100)

    return {
        'threshold': float(threshold),
        'n_above': n_above,
        'pct_above': float(n_above / n_total * 100),
        't_mae': t_mae,
        't_rmse': t_rmse,
        't_r2': t_r2,
        't_dir_acc': t_dir_acc,
        't_bias_buy': t_bias_buy,
        'mae_thresh': t_mae,
        'rmse_thresh': t_rmse,
        'r2_thresh': t_r2,
        'pseudo_huber_thresh': pseudo_huber_loss(yt, yp, delta=1.0),
        'pseudo_huber_large_thresh': pseudo_huber_loss(yt, yp, delta=10.0),
        'dir_acc_thresh': t_dir_acc,
        'bias_buy': t_bias_buy,
    }


def compute_rank_score(
    metrics: dict | None = None,
    sim_metrics: dict | None = None,
    sig_metrics: dict | None = None,
    pred_log_returns=None,
    true_log_returns=None,
    n_samples=None,
) -> dict:
    """
    Collapse simulation and signal quality metrics into a 0-100 Rank Score.
    """
    m = dict(metrics) if isinstance(metrics, dict) else {}
    if isinstance(sig_metrics, dict):
        m.update(sig_metrics)
    if isinstance(sim_metrics, dict):
        m.update(sim_metrics)

    p_std = m.get('pred_std')
    t_std = m.get('true_std')
    cr = m.get('compression_ratio')
    corr_val = m.get('corr')

    if pred_log_returns is not None and true_log_returns is not None:
        p_arr = np.asarray(pred_log_returns, dtype=np.float64)
        t_arr = np.asarray(true_log_returns, dtype=np.float64)
        if len(p_arr) > 1 and len(t_arr) > 1:
            p_std = float(np.std(p_arr))
            t_std = float(np.std(t_arr))
            cr = float(p_std / t_std) if t_std > 1e-12 else 0.0
            corr_val = float(np.corrcoef(p_arr, t_arr)[0, 1]) if (p_std > 1e-12 and t_std > 1e-12) else 0.0
            m['pred_std'] = p_std
            m['true_std'] = t_std
            m['compression_ratio'] = cr
            m['corr'] = corr_val

    pct_strong = m.get('pct_strong')

    if pct_strong is None or cr is None:
        return {'rank_score': None, 'signal_class': 'UNKNOWN_SIGNAL'}

    # Collapse gate — flat models cannot generate tradable signals
    if cr < RANK_CR_DEAD or pct_strong < RANK_PCT_STRONG_DEAD:
        return {
            'rank_score': 0.0,
            'signal_class': 'DEAD',
            'ranking_method': 'absolute_60_gross',
            'rank_metric_version': RANK_METRIC_VERSION,
            'compression_ratio': cr,
            'pred_std': p_std,
            'true_std': t_std,
            'corr': corr_val,
            'rank_components': {
                'activity': 0.0,
                'calibration': 0.0,
                'direction': 0.0,
                'capture': 0.0,
                'cr_health': 0.0,
            },
        }

    # 1. Signal activity (25 pts)
    activity = _clamp01(pct_strong / RANK_PCT_STRONG_FULL) * RANK_W_ACTIVITY

    # 2. Calibration (25 pts)
    calibs = [c for c in (m.get('calib_buy'), m.get('calib_sell')) if c is not None]
    if calibs:
        avg_calib = sum(calibs) / len(calibs)
        calib_norm = _clamp01((avg_calib - RANK_CALIB_FLOOR) / (RANK_CALIB_FULL - RANK_CALIB_FLOOR))
    else:
        avg_calib = None
        calib_norm = 0.0
    calibration = calib_norm * RANK_W_CALIBRATION

    # 3. Directional accuracy on strong samples (20 pts)
    dir_acc_strong = m.get('dir_acc_strong')
    if dir_acc_strong is not None:
        direction = _clamp01((dir_acc_strong - 50.0) / (RANK_DIRACC_FULL - 50.0)) * RANK_W_DIRECTION
    else:
        direction = 0.0

    # 4. Simulation capture (25 pts)
    # v2 uses log-scale capture; the legacy ratio is the fallback (as in core LR).
    capture_log = m.get('sim_capture_log')
    if capture_log is not None:
        capture = _clamp01(capture_log / RANK_CAPTURE_LOG_FULL) * RANK_W_CAPTURE
    else:
        capture_ratio = m.get('sim_capture_ratio')
        capture = _clamp01(capture_ratio) * RANK_W_CAPTURE if capture_ratio is not None else 0.0

    # 5. Compression ratio health (5 pts)
    cr_health = math.exp(-0.5 * ((cr - 1.0) / 0.5) ** 2) * RANK_W_CR_HEALTH

    # No rounding: the score must be bit-comparable with the core LR rank score.
    total_score = float(max(0.0, min(100.0, activity + calibration + direction + capture + cr_health)))

    # Classification identical to core LR (all available calibrations >= 0.55).
    if avg_calib is None:
        signal_class = 'UNKNOWN_SIGNAL'
    elif all(c >= 0.55 for c in calibs):
        signal_class = 'TRADEABLE_GOOD'
    else:
        signal_class = 'TRADEABLE_UNCAL'

    return {
        'rank_score': total_score,
        'signal_class': signal_class,
        'ranking_method': 'absolute_60_gross',
        'rank_metric_version': RANK_METRIC_VERSION if capture_log is not None else 1,
        'compression_ratio': cr,
        'pred_std': p_std,
        'true_std': t_std,
        'corr': corr_val,
        'rank_components': {
            'activity': float(activity),
            'calibration': float(calibration),
            'direction': float(direction),
            'capture': float(capture),
            'cr_health': float(cr_health),
        },
    }


def print_metrics_summary(metrics: dict, thresh_metrics: dict | None = None) -> None:
    """Print formatted metrics summary"""
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    print("\nLOG RETURN METRICS (ALL SAMPLES):")
    print(f"  MAE:                   {metrics['mae']:.6f}")
    print(f"  RMSE:                  {metrics['rmse']:.6f}")
    print(f"  R2:                    {metrics['r2']:.4f}")
    print(f"  Pseudo-Huber (d=1):    {metrics.get('pseudo_huber', 0.0):.6f}")
    print(f"  Pseudo-Huber (d=10):   {metrics.get('pseudo_huber_large', 0.0):.6f}")
    if 'compression_ratio' in metrics:
        print(f"  Compression Ratio:     {metrics['compression_ratio']:.4f}")
    if 'corr' in metrics:
        print(f"  Correlation (Pearson): {metrics['corr']:.4f}")

    print("\nERROR STATS:")
    print(f"  Mean:   {metrics['error_mean']:.6f}")
    print(f"  Median: {metrics['error_median']:.6f}")
    print(f"  Std:    {metrics['error_std']:.6f}")

    if 'dir_acc' in metrics:
        print("\nDIRECTIONAL (ALL SAMPLES):")
        print(f"  Accuracy: {metrics['dir_acc']:.2f}%")

    if thresh_metrics is not None and thresh_metrics.get('n_above', 0) > 0:
        t_val = thresh_metrics['threshold']
        n_val = thresh_metrics['n_above']
        pct_val = thresh_metrics['pct_above']
        print(f"\nTHRESHOLD METRICS (|y_pred| >= {t_val:.4f}):")
        print(f"  Samples above threshold: {n_val}  ({pct_val:.1f}% of all)")
        if thresh_metrics.get('t_mae') is not None:
            print(f"  MAE:                   {thresh_metrics['t_mae']:.6f}")
        if thresh_metrics.get('t_rmse') is not None:
            print(f"  RMSE:                  {thresh_metrics['t_rmse']:.6f}")
        if thresh_metrics.get('t_dir_acc') is not None:
            print(f"  Dir Acc:               {thresh_metrics['t_dir_acc']:.2f}%")
        if thresh_metrics.get('t_bias_buy') is not None:
            print(f"  Bias BUY:              {thresh_metrics['t_bias_buy']:.1f}%")


def interpret_metrics(metrics: dict, thresh_metrics: dict | None = None) -> None:
    """Provide interpretation of metrics quality"""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    r2 = metrics.get('r2', 0.0)
    if r2 > 0.1:
        print(f"  [+] Good fit: R2 = {r2:.4f}")
    elif r2 > 0.05:
        print(f"  [o] Moderate fit: R2 = {r2:.4f}")
    else:
        print(f"  [-] Weak fit: R2 = {r2:.4f}")

    dir_acc = metrics.get('dir_acc', 0.0)
    if dir_acc > 0:
        if dir_acc > 55:
            print(f"  [+] Good direction (all): {dir_acc:.1f}%")
        elif dir_acc > 50:
            print(f"  [o] Slight edge (all): {dir_acc:.1f}%")
        else:
            print(f"  [-] No edge (all): {dir_acc:.1f}%")
