"""
Metrics for LSTM Log-Return pipeline.

Independent implementation (not shared with XGB) but identical metric set
for direct comparison between models.
"""
import numpy as np


def pseudo_huber_loss(y_true, y_pred, delta=1.0):
    """Pseudo-Huber loss: delta^2 * (sqrt(1 + (r/delta)^2) - 1)"""
    r = y_pred - y_true
    return float(np.mean(delta ** 2 * (np.sqrt(1 + (r / delta) ** 2) - 1)))


def calculate_metrics(y_true_log, y_pred_log, threshold=None):
    """
    Calculate comprehensive metrics for log-return predictions.

    All metrics computed in original log-return space
    (after inverse_transform if target was scaled).

    Args:
        y_true_log: np.ndarray — true log-returns
        y_pred_log: np.ndarray — predicted log-returns
        threshold:  float or None — signal threshold for thresholded metrics

    Returns:
        dict with keys:
          Base: mae, rmse, mse, r2, pseudo_huber, pseudo_huber_large,
                error_mean, error_std, error_median,
                true_mean, true_std, pred_mean, pred_std,
                dir_acc, corr
          Thresholded (if threshold set and enough samples):
                t_mae, t_rmse, t_r2, t_pseudo_huber, t_pseudo_huber_large,
                t_dir_acc, t_n_samples, t_pct_samples, t_threshold
    """
    y_true = np.asarray(y_true_log, dtype=np.float64)
    y_pred = np.asarray(y_pred_log, dtype=np.float64)
    errors = y_pred - y_true

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    ph1 = pseudo_huber_loss(y_true, y_pred, delta=1.0)
    ph10 = pseudo_huber_loss(y_true, y_pred, delta=10.0)

    # Directional accuracy: both nonzero and same sign
    nonzero_mask = (y_true != 0) & (y_pred != 0)
    if nonzero_mask.sum() > 0:
        correct = np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])
        dir_acc = float(correct.mean() * 100)
    else:
        dir_acc = 0.0

    # Compression ratio (CR) = std(pred) / std(true). Formalised here so it is
    # always present in metrics.json instead of being recomputed ad hoc by
    # batch scripts. CR < 0.1 == prediction collapse (model useless for
    # signals); CR > 1.5 == amplitude explosion.
    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
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
        'error_mean': float(np.mean(errors)),
        'error_std': float(np.std(errors)),
        'error_median': float(np.median(errors)),
        'true_mean': float(np.mean(y_true)),
        'true_std': true_std,
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': pred_std,
        'compression_ratio': compression_ratio,
        'dir_acc': dir_acc,
        'corr': corr,
    }

    # Thresholded metrics
    if threshold is not None and threshold > 0:
        t_mask = np.abs(y_true) >= threshold
        t_n = int(t_mask.sum())

        if t_n >= 10:
            yt = y_true[t_mask]
            yp = y_pred[t_mask]
            t_errors = yp - yt

            t_mae = float(np.mean(np.abs(t_errors)))
            t_mse = float(np.mean(t_errors ** 2))
            t_rmse = float(np.sqrt(t_mse))

            t_ss_res = np.sum(t_errors ** 2)
            t_ss_tot = np.sum((yt - np.mean(yt)) ** 2)
            t_r2 = float(1.0 - t_ss_res / max(t_ss_tot, 1e-12))

            t_nonzero = (yt != 0) & (yp != 0)
            if t_nonzero.sum() > 0:
                t_correct = np.sign(yt[t_nonzero]) == np.sign(yp[t_nonzero])
                t_dir_acc = float(t_correct.mean() * 100)
            else:
                t_dir_acc = 0.0

            metrics.update({
                't_mae': t_mae,
                't_rmse': t_rmse,
                't_r2': t_r2,
                't_pseudo_huber': pseudo_huber_loss(yt, yp, delta=1.0),
                't_pseudo_huber_large': pseudo_huber_loss(yt, yp, delta=10.0),
                't_dir_acc': t_dir_acc,
                't_n_samples': t_n,
                't_pct_samples': float(t_n / len(y_true) * 100),
                't_threshold': float(threshold),
            })

    return metrics


# =========================================================================
# RANK SCORE — single number for quick model triage (higher = better)
# =========================================================================

# Tuning knobs (calibrate against batches). Kept module-level so the ranking
# refresh script / analysis notebooks can import and reproduce the score.
RANK_CR_DEAD = 0.1            # CR below this == prediction collapse
RANK_PCT_STRONG_DEAD = 1.0    # < 1% strong signals == effectively dead
RANK_PCT_STRONG_FULL = 20.0   # >= 20% strong earns full activity points
RANK_CALIB_FLOOR = 0.5        # random calibration baseline
RANK_CALIB_FULL = 0.7         # calibration that earns full points
RANK_DIRACC_FULL = 60.0       # dir_acc_strong that earns full points
RANK_CR_HEALTH_SIGMA = 0.5    # width of the CR≈1 reward bell

RANK_W_ACTIVITY = 25.0
RANK_W_CALIBRATION = 25.0
RANK_W_DIRECTION = 20.0
RANK_W_CAPTURE = 25.0
RANK_W_CR_HEALTH = 5.0

# v2 rank metric: log-scale capture (10% of oracle log-return = full capture pts).
RANK_CAPTURE_LOG_FULL = 0.10
RANK_METRIC_VERSION = 2


def _clamp01(x):
    return max(0.0, min(1.0, float(x)))


def compute_rank_score(metrics):
    """
    Collapse the signal/simulation metrics into a single 0-100 rank score.

    Higher == more useful for generating trading signals. Models that collapse
    (CR < RANK_CR_DEAD) or never produce a strong signal score ~0 regardless of
    their MAE / t_dir_acc — that is the whole point: standard regression error
    rewards flat predictors, this score punishes them.

    Requires the signal/simulation keys produced by
    ``common.simulation.evaluate_signal_and_simulation`` to be present in
    ``metrics``. When they are absent the score is None (cannot be judged).

    Returns dict: {rank_score, signal_class, rank_components} or
    {rank_score: None, signal_class: 'UNKNOWN_SIGNAL'} when inputs are missing.
    """
    cr = metrics.get('compression_ratio')
    pct_strong = metrics.get('pct_strong')

    if pct_strong is None or cr is None:
        return {'rank_score': None, 'signal_class': 'UNKNOWN_SIGNAL'}

    # Collapse gate — a flat model is useless for signals.
    if cr < RANK_CR_DEAD or pct_strong < RANK_PCT_STRONG_DEAD:
        return {
            'rank_score': 0.0,
            'signal_class': 'DEAD',
            'rank_components': {
                'activity': 0.0, 'calibration': 0.0, 'direction': 0.0,
                'capture': 0.0, 'cr_health': 0.0,
            },
        }

    # 1. Signal activity
    activity = _clamp01(pct_strong / RANK_PCT_STRONG_FULL) * RANK_W_ACTIVITY

    # 2. Calibration (avg of available calib_buy / calib_sell)
    calibs = [c for c in (metrics.get('calib_buy'), metrics.get('calib_sell'))
              if c is not None]
    if calibs:
        avg_calib = sum(calibs) / len(calibs)
        calib_norm = _clamp01(
            (avg_calib - RANK_CALIB_FLOOR)
            / (RANK_CALIB_FULL - RANK_CALIB_FLOOR))
    else:
        avg_calib = None
        calib_norm = 0.0
    calibration = calib_norm * RANK_W_CALIBRATION

    # 3. Directional strength on strong samples
    dir_acc_strong = metrics.get('dir_acc_strong')
    if dir_acc_strong is not None:
        direction = _clamp01(
            (dir_acc_strong - 50.0) / (RANK_DIRACC_FULL - 50.0)
        ) * RANK_W_DIRECTION
    else:
        direction = 0.0

    # 4. Simulation capture — v2 uses log-scale when available; else legacy ratio.
    capture_log = metrics.get('sim_capture_log')
    if capture_log is not None:
        capture = (_clamp01(capture_log / RANK_CAPTURE_LOG_FULL)
                   * RANK_W_CAPTURE)
    else:
        capture_ratio = metrics.get('sim_capture_ratio')
        capture = (_clamp01(capture_ratio) * RANK_W_CAPTURE
                   if capture_ratio is not None else 0.0)

    # 5. CR health — reward CR≈1, punish collapse and explosion
    cr_health = (np.exp(-((cr - 1.0) ** 2)
                        / (2.0 * RANK_CR_HEALTH_SIGMA ** 2))
                 * RANK_W_CR_HEALTH)

    score = activity + calibration + direction + capture + cr_health

    # Classification (mirrors refresh_model_ranking signal classes)
    if avg_calib is None:
        signal_class = 'UNKNOWN_SIGNAL'
    elif all(c >= 0.55 for c in calibs):
        signal_class = 'TRADEABLE_GOOD'
    else:
        signal_class = 'TRADEABLE_UNCAL'

    return {
        'rank_score': float(score),
        'signal_class': signal_class,
        'ranking_method': 'absolute_60_gross',
        'rank_metric_version': (
            RANK_METRIC_VERSION
            if capture_log is not None else 1
        ),
        'rank_components': {
            'activity': float(activity),
            'calibration': float(calibration),
            'direction': float(direction),
            'capture': float(capture),
            'cr_health': float(cr_health),
        },
    }


def print_metrics_summary(metrics):
    """Print a formatted summary of metrics."""
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  RMSE:      {metrics['rmse']:.6f}")
    print(f"  R²:        {metrics['r2']:.4f}")
    print(f"  PH(δ=1):   {metrics['pseudo_huber']:.6f}")
    print(f"  PH(δ=10):  {metrics['pseudo_huber_large']:.6f}")
    print(f"  Dir Acc:   {metrics['dir_acc']:.2f}%")
    print(f"  Pred std:  {metrics['pred_std']:.6f}")
    print(f"  True std:  {metrics['true_std']:.6f}")

    if 't_mae' in metrics:
        t = metrics['t_threshold']
        n = metrics['t_n_samples']
        pct = metrics['t_pct_samples']
        print(f"\n  THRESHOLDED (|y| ≥ {t:.6f}, n={n}, {pct:.1f}%):")
        print(f"  t_MAE:     {metrics['t_mae']:.6f}")
        print(f"  t_RMSE:    {metrics['t_rmse']:.6f}")
        print(f"  t_R²:      {metrics['t_r2']:.4f}")
        print(f"  t_Dir Acc: {metrics['t_dir_acc']:.2f}%")


def interpret_metrics(metrics):
    """Print brief interpretation of metrics quality."""
    dir_acc = metrics['dir_acc']
    r2 = metrics['r2']

    if dir_acc > 55:
        print("  → Directional accuracy above random (>55%)")
    elif dir_acc > 50:
        print("  → Directional accuracy near random (~50%)")
    else:
        print("  → Directional accuracy below random (<50%)")

    if r2 > 0.1:
        print(f"  → R² = {r2:.4f}: model explains some variance")
    elif r2 > 0.0:
        print(f"  → R² = {r2:.4f}: weak explanatory power")
    else:
        print(f"  → R² = {r2:.4f}: model worse than mean prediction")
