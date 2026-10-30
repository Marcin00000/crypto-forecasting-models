"""
Metrics calculation utilities for LSTM price prediction pipeline.

Includes regression metrics in dollar price space, implied log-return metrics,
directional accuracies, and the unified Rank Score v2.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANK_CR_DEAD = 0.1
RANK_PCT_STRONG_DEAD = 1.0
RANK_PCT_STRONG_FULL = 20.0
RANK_CALIB_FLOOR = 0.5
RANK_CALIB_FULL = 0.7
RANK_DIRACC_FULL = 60.0
RANK_CR_HEALTH_SIGMA = 0.5

RANK_W_ACTIVITY = 25.0
RANK_W_CALIBRATION = 25.0
RANK_W_DIRECTION = 20.0
RANK_W_CAPTURE = 25.0
RANK_W_CR_HEALTH = 5.0

RANK_CAPTURE_LOG_FULL = 0.10
RANK_METRIC_VERSION = 2

__all__ = [
    'calculate_metrics',
    'calculate_price_metrics',
    'calculate_log_return_metrics',
    'calculate_directional_accuracy',
    'pseudo_huber_loss',
    'compute_rank_score',
    'print_metrics_summary',
    'interpret_metrics',
    'RANK_CR_DEAD',
    'RANK_PCT_STRONG_DEAD',
    'RANK_METRIC_VERSION',
]


def pseudo_huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    """Pseudo-Huber loss: delta^2 * (sqrt(1 + (r/delta)^2) - 1)."""
    r = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(delta ** 2 * (np.sqrt(1.0 + (r / delta) ** 2) - 1.0)))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def calculate_metrics(y_true, y_pred, price_scaler=None, use_log_prices: bool = False):
    """
    Calculate price metrics in dollar space ($).
    Backward-compatible with legacy lstm_test_price.py calls.
    """
    if price_scaler is not None:
        y_true_2d = y_true.reshape(-1, 1)
        y_pred_2d = y_pred.reshape(-1, 1)
        if use_log_prices:
            y_true_log = price_scaler.inverse_transform(y_true_2d).flatten()
            y_pred_log = price_scaler.inverse_transform(y_pred_2d).flatten()
            y_true_price = np.exp(y_true_log)
            y_pred_price = np.exp(y_pred_log)
        else:
            y_true_price = price_scaler.inverse_transform(y_true_2d).flatten()
            y_pred_price = price_scaler.inverse_transform(y_pred_2d).flatten()
    else:
        y_true_price = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred_price = np.asarray(y_pred, dtype=np.float64).flatten()

    metrics = calculate_price_metrics(y_true_price, y_pred_price)
    return metrics, y_true_price, y_pred_price


def calculate_price_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray) -> dict:
    """Calculate error metrics in raw currency ($) space."""
    y_true = np.asarray(y_true_price, dtype=np.float64)
    y_pred = np.asarray(y_pred_price, dtype=np.float64)
    errors = y_pred - y_true

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs(errors) / denom) * 100.0)
    if not np.isfinite(mape):
        mape = 999.99

    return {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'r2': r2,
        'mape': mape,
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
        'error_median': float(np.median(errors)),
        'true_mean': float(np.mean(y_true)),
        'pred_mean': float(np.mean(y_pred)),
    }


def calculate_log_return_metrics(y_true_log, y_pred_log, threshold: float | None = None) -> dict:
    """
    Calculate comprehensive regression metrics on log-return predictions.
    Matches the LR pipeline metrics schema.
    """
    y_true = np.asarray(y_true_log, dtype=np.float64)
    y_pred = np.asarray(y_pred_log, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors))) if n else 0.0
    mse = float(np.mean(errors ** 2)) if n else 0.0
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) if n else 0.0
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12)) if n else 0.0

    ph1 = pseudo_huber_loss(y_true, y_pred, delta=1.0) if n else 0.0
    ph10 = pseudo_huber_loss(y_true, y_pred, delta=10.0) if n else 0.0

    nonzero_mask = (y_true != 0) & (y_pred != 0)
    if nonzero_mask.sum() > 0:
        correct = np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])
        dir_acc = float(correct.mean() * 100.0)
    else:
        dir_acc = 0.0

    true_std = float(np.std(y_true)) if n else 0.0
    pred_std = float(np.std(y_pred)) if n else 0.0
    compression_ratio = (pred_std / true_std) if true_std > 0 else 0.0

    if true_std < 1e-12 or pred_std < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mse': mse,
        'r2': r2,
        'pseudo_huber': ph1,
        'pseudo_huber_large': ph10,
        'error_mean': float(np.mean(errors)) if n else 0.0,
        'error_std': float(np.std(errors)) if n else 0.0,
        'error_median': float(np.median(errors)) if n else 0.0,
        'true_mean': float(np.mean(y_true)) if n else 0.0,
        'true_std': true_std,
        'pred_mean': float(np.mean(y_pred)) if n else 0.0,
        'pred_std': pred_std,
        'compression_ratio': compression_ratio,
        'dir_acc': dir_acc,
        'corr': corr,
    }

    if threshold is not None and threshold > 0:
        t_mask = np.abs(y_true) >= threshold
        t_n = int(t_mask.sum())
        if t_n >= 10:
            yt_t = y_true[t_mask]
            yp_t = y_pred[t_mask]
            err_t = yp_t - yt_t
            t_ss_res = np.sum(err_t ** 2)
            t_ss_tot = np.sum((yt_t - np.mean(yt_t)) ** 2)
            t_r2 = float(1.0 - t_ss_res / max(t_ss_tot, 1e-12))
            t_nz = (yt_t != 0) & (yp_t != 0)
            t_dir = float((np.sign(yt_t[t_nz]) == np.sign(yp_t[t_nz])).mean() * 100) if t_nz.sum() > 0 else 0.0

            metrics.update({
                't_mae': float(np.mean(np.abs(err_t))),
                't_rmse': float(np.sqrt(np.mean(err_t ** 2))),
                't_r2': t_r2,
                't_pseudo_huber': pseudo_huber_loss(yt_t, yp_t, delta=1.0),
                't_pseudo_huber_large': pseudo_huber_loss(yt_t, yp_t, delta=10.0),
                't_dir_acc': t_dir,
                't_n_samples': t_n,
                't_pct_samples': float(t_n / len(y_true) * 100),
                't_threshold': float(threshold),
            })

    return metrics


def calculate_directional_accuracy(y_true_price, y_pred_price, current_prices) -> float:
    """Calculate directional prediction accuracy on prices."""
    cur = np.asarray(current_prices, dtype=np.float64)
    if len(cur) == 0 or np.all(cur == 0):
        return 0.0

    actual_dir = np.sign(np.asarray(y_true_price, dtype=np.float64) - cur)
    pred_dir = np.sign(np.asarray(y_pred_price, dtype=np.float64) - cur)
    return float(np.mean(actual_dir == pred_dir) * 100.0)


def compute_rank_score(metrics: dict) -> dict:
    """
    Compute composite 0-100 Rank Score v2 based on signal and simulation metrics.
    Includes Prediction Collapse gating (CR < 0.10 or pct_strong < 1.0% -> DEAD).
    """
    cr = metrics.get('compression_ratio')
    pct_strong = metrics.get('pct_strong')

    if pct_strong is None or cr is None:
        return {'rank_score': None, 'signal_class': 'UNKNOWN_SIGNAL'}

    if cr < RANK_CR_DEAD or pct_strong < RANK_PCT_STRONG_DEAD:
        return {
            'rank_score': 0.0,
            'signal_class': 'DEAD',
            'ranking_method': 'absolute_60_gross',
            'rank_metric_version': RANK_METRIC_VERSION,
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
    calibs = [c for c in (metrics.get('calib_buy'), metrics.get('calib_sell')) if c is not None]
    if calibs:
        avg_calib = sum(calibs) / len(calibs)
        calib_norm = _clamp01((avg_calib - RANK_CALIB_FLOOR) / (RANK_CALIB_FULL - RANK_CALIB_FLOOR))
    else:
        avg_calib = None
        calib_norm = 0.0
    calibration = calib_norm * RANK_W_CALIBRATION

    # 3. Directional strength (20 pts)
    dir_acc_strong = metrics.get('dir_acc_strong')
    if dir_acc_strong is not None:
        direction = _clamp01((dir_acc_strong - 50.0) / (RANK_DIRACC_FULL - 50.0)) * RANK_W_DIRECTION
    else:
        direction = 0.0

    # 4. Simulation capture (25 pts)
    capture_log = metrics.get('sim_capture_log')
    if capture_log is not None:
        capture = _clamp01(capture_log / RANK_CAPTURE_LOG_FULL) * RANK_W_CAPTURE
    else:
        capture_ratio = metrics.get('sim_capture_ratio')
        capture = _clamp01(capture_ratio) * RANK_W_CAPTURE if capture_ratio is not None else 0.0

    # 5. CR health (5 pts)
    cr_health = np.exp(-((cr - 1.0) ** 2) / (2.0 * RANK_CR_HEALTH_SIGMA ** 2)) * RANK_W_CR_HEALTH

    score = float(activity + calibration + direction + capture + cr_health)

    if avg_calib is None:
        signal_class = 'UNKNOWN_SIGNAL'
    elif all(c >= 0.55 for c in calibs):
        signal_class = 'TRADEABLE_GOOD'
    else:
        signal_class = 'TRADEABLE_UNCAL'

    return {
        'rank_score': score,
        'signal_class': signal_class,
        'ranking_method': 'absolute_60_gross',
        'rank_metric_version': RANK_METRIC_VERSION if capture_log is not None else 1,
        'rank_components': {
            'activity': float(activity),
            'calibration': float(calibration),
            'direction': float(direction),
            'capture': float(capture),
            'cr_health': float(cr_health),
        },
    }


def print_metrics_summary(metrics, dir_acc=None):
    """Print formatted summary of regression and price metrics."""
    print("\nPRICE METRICS:")
    if 'mae' in metrics:
        print(f"  MAE:   ${metrics['mae']:,.2f}")
    if 'rmse' in metrics:
        print(f"  RMSE:  ${metrics['rmse']:,.2f}")
    if 'mape' in metrics:
        print(f"  MAPE:  {metrics['mape']:.2f}%")
    if 'r2' in metrics:
        print(f"  R²:    {metrics['r2']:.4f}")

    if dir_acc is not None and dir_acc > 0:
        print(f"  Directional Acc: {dir_acc:.2f}%")


def interpret_metrics(metrics, dir_acc=None):
    """Print brief evaluation interpretation."""
    print("\n" + "=" * 60)
    print("METRICS INTERPRETATION")
    print("=" * 60)
    if 'mape' in metrics:
        if metrics['mape'] < 1.0:
            print("  ✓ EXCELLENT: MAPE < 1%")
        elif metrics['mape'] < 2.0:
            print("  ✓ GOOD: MAPE 1-2%")
        elif metrics['mape'] < 5.0:
            print("  ○ ACCEPTABLE: MAPE 2-5%")
        else:
            print("  ✗ POOR: MAPE > 5%")

    if 'r2' in metrics:
        r2 = metrics['r2']
        if r2 > 0.5:
            print(f"  ✓ Strong: R² = {r2:.3f}")
        elif r2 > 0.2:
            print(f"  ○ Moderate: R² = {r2:.3f}")
        else:
            print(f"  ✗ Weak: R² = {r2:.3f}")
