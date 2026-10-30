"""
Metrics calculation utilities for XGBoost log-return pipeline.

Computes standard regression metrics (MAE, RMSE, R2, Pseudo-Huber),
thresholded metrics, compression ratio (CR), Pearson correlation,
and the multi-dimensional Rank Score (0-100) with collapse gating.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = [
    'pseudo_huber_loss',
    'calculate_metrics',
    'compute_rank_score',
    'print_metrics_summary',
    'interpret_metrics',
    'RANK_CR_DEAD',
    'RANK_PCT_STRONG_DEAD',
    'RANK_PCT_STRONG_FULL',
    'RANK_CALIB_FLOOR',
    'RANK_CALIB_FULL',
    'RANK_DIRACC_FULL',
    'RANK_CR_HEALTH_SIGMA',
    'RANK_W_ACTIVITY',
    'RANK_W_CALIBRATION',
    'RANK_W_DIRECTION',
    'RANK_W_CAPTURE',
    'RANK_W_CR_HEALTH',
    'RANK_CAPTURE_LOG_FULL',
    'RANK_METRIC_VERSION',
    '_jsonable',
]


def pseudo_huber_loss(y_true, y_pred, delta: float = 1.0) -> float:
    """
    Pseudo-Huber loss (mean over samples).
    L = delta^2 * (sqrt(1 + ((y_pred - y_true) / delta)^2) - 1)
    """
    residuals = y_pred - y_true
    return float(np.mean(delta ** 2 * (np.sqrt(1.0 + (residuals / delta) ** 2) - 1.0)))


def calculate_metrics(y_true_log, y_pred_log, threshold=None) -> dict:
    """
    Calculate comprehensive metrics on log-returns.

    Args:
        y_true_log: True log-returns (unscaled raw space)
        y_pred_log: Predicted log-returns (unscaled raw space)
        threshold:  Optional signal threshold. When > 0, computes additional
                    thresholded metrics (t_ prefix) on samples where
                    |y_true| >= threshold.

    Returns:
        dict: Base metrics + thresholded metrics
    """
    y_true = np.asarray(y_true_log, dtype=np.float64)
    y_pred = np.asarray(y_pred_log, dtype=np.float64)
    errors = y_pred - y_true

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    ph1 = pseudo_huber_loss(y_true, y_pred, delta=1.0)
    ph10 = pseudo_huber_loss(y_true, y_pred, delta=10.0)

    # Directional accuracy: sign match on nonzero entries
    nonzero_mask = (y_true != 0) & (y_pred != 0)
    if nonzero_mask.sum() > 0:
        dir_acc = float(np.mean(np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])) * 100.0)
    else:
        dir_acc = 0.0

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

    # Thresholded metrics (computed on samples where |y_true| >= threshold)
    if threshold is not None and threshold > 0:
        mask = np.abs(y_true) >= threshold
        n_above = int(np.sum(mask))
        metrics['t_n_samples'] = n_above
        metrics['t_pct_samples'] = float(n_above / max(len(y_true), 1) * 100.0)
        metrics['t_threshold'] = float(threshold)

        if n_above >= 10:
            yt = y_true[mask]
            yp = y_pred[mask]
            metrics['t_mae'] = float(mean_absolute_error(yt, yp))
            metrics['t_rmse'] = float(np.sqrt(mean_squared_error(yt, yp)))
            metrics['t_r2'] = float(r2_score(yt, yp))
            metrics['t_pseudo_huber'] = pseudo_huber_loss(yt, yp, delta=1.0)
            metrics['t_pseudo_huber_large'] = pseudo_huber_loss(yt, yp, delta=10.0)

            t_nonzero = (yt != 0) & (yp != 0)
            if t_nonzero.sum() > 0:
                metrics['t_dir_acc'] = float(np.mean(np.sign(yt[t_nonzero]) == np.sign(yp[t_nonzero])) * 100.0)
            else:
                metrics['t_dir_acc'] = 0.0

    return metrics


# =========================================================================
# RANK SCORE (0-100) — Multi-dimensional model utility metric
# =========================================================================

RANK_CR_DEAD = 0.1            # CR below this == prediction collapse
RANK_PCT_STRONG_DEAD = 1.0    # < 1% strong signals == effectively dead
RANK_PCT_STRONG_FULL = 20.0   # >= 20% strong earns full activity points
RANK_CALIB_FLOOR = 0.5        # random calibration baseline
RANK_CALIB_FULL = 0.7         # calibration that earns full points
RANK_DIRACC_FULL = 60.0       # dir_acc_strong that earns full points
RANK_CR_HEALTH_SIGMA = 0.5    # width of the CR≈1 reward Gaussian bell

RANK_W_ACTIVITY = 25.0
RANK_W_CALIBRATION = 25.0
RANK_W_DIRECTION = 20.0
RANK_W_CAPTURE = 25.0
RANK_W_CR_HEALTH = 5.0

RANK_CAPTURE_LOG_FULL = 0.10
RANK_METRIC_VERSION = 2


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_rank_score(
    metrics: dict = None,
    sim_metrics: dict = None,
    sig_metrics: dict = None,
    pred_log_returns=None,
    true_log_returns=None,
    n_samples=None,
) -> dict:
    """
    Collapse simulation and signal quality metrics into a 0-100 Rank Score.

    Accepts either a single pre-merged metrics dict, or separate components
    (sim_metrics, sig_metrics, pred/true arrays).
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
            if p_std > 1e-12 and t_std > 1e-12:
                corr_val = float(np.corrcoef(p_arr, t_arr)[0, 1])
            else:
                corr_val = 0.0
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

    # 1. Signal activity
    activity = _clamp01(pct_strong / RANK_PCT_STRONG_FULL) * RANK_W_ACTIVITY

    # 2. Calibration (average of available calib_buy and calib_sell)
    calibs = [c for c in (m.get('calib_buy'), m.get('calib_sell')) if c is not None]
    if calibs:
        avg_calib = sum(calibs) / len(calibs)
        calib_norm = _clamp01((avg_calib - RANK_CALIB_FLOOR) / (RANK_CALIB_FULL - RANK_CALIB_FLOOR))
    else:
        avg_calib = None
        calib_norm = 0.0
    calibration = calib_norm * RANK_W_CALIBRATION

    # 3. Directional accuracy on strong samples
    dir_acc_strong = m.get('dir_acc_strong')
    if dir_acc_strong is not None:
        direction = _clamp01((dir_acc_strong - 50.0) / (RANK_DIRACC_FULL - 50.0)) * RANK_W_DIRECTION
    else:
        direction = 0.0

    # 4. Simulation capture — v2 uses log-scale capture
    capture_log = m.get('sim_capture_log')
    if capture_log is not None:
        capture = _clamp01(capture_log / RANK_CAPTURE_LOG_FULL) * RANK_W_CAPTURE
    else:
        capture_ratio = m.get('sim_capture_ratio')
        capture = _clamp01(capture_ratio) * RANK_W_CAPTURE if capture_ratio is not None else 0.0

    # 5. CR health — Gaussian reward centered at CR=1.0
    cr_health = float(np.exp(-((cr - 1.0) ** 2) / (2.0 * RANK_CR_HEALTH_SIGMA ** 2))) * RANK_W_CR_HEALTH

    score = activity + calibration + direction + capture + cr_health

    # Signal classification
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


def format_rank_summary(rank_metrics: dict, primary_sim: dict = None) -> str:
    """Format Rank Score and simulation metrics into a clean text block."""
    lines = [
        "=" * 70,
        "MARKET SIMULATION & RANK SCORE (Simulation v2 Profile)",
        "=" * 70,
        f"  Rank Score:            {rank_metrics.get('rank_score', 0.0):.2f} / 100  ({rank_metrics.get('signal_class', 'UNKNOWN')})",
        f"  Compression Ratio:     {rank_metrics.get('compression_ratio', 0.0):.4f}",
        f"  Pred Std:              {rank_metrics.get('pred_std', 0.0):.6f}",
        f"  True Std:              {rank_metrics.get('true_std', 0.0):.6f}",
        f"  Signal Corr:           {rank_metrics.get('corr', 0.0):.4f}",
    ]
    if primary_sim:
        lines.extend([
            "-" * 70,
            f"PRIMARY SIMULATION ({primary_sim.get('fee_bps', 2.0):.1f} bps fee, sticky mode):",
            f"  Total Return:          {primary_sim.get('total_return_pct', 0.0):+.2f}%",
            f"  Sharpe Ratio:          {primary_sim.get('sharpe_ratio', 0.0):.2f}",
            f"  Max Drawdown:          {primary_sim.get('max_drawdown_pct', 0.0):.2f}%",
            f"  Win Rate:              {primary_sim.get('win_rate_pct', 0.0):.2f}% ({primary_sim.get('trades', 0)} trades)",
        ])
    return "\n".join(lines)


def print_metrics_summary(metrics: dict) -> None:
    """Print formatted metrics summary."""
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)

    print("\nLOG RETURN METRICS:")
    print(f"  MAE:                   {metrics['mae']:.6f}")
    print(f"  RMSE:                  {metrics['rmse']:.6f}")
    print(f"  R²:                    {metrics['r2']:.4f}")
    print(f"  Pseudo-Huber (δ=1):    {metrics['pseudo_huber']:.6f}")
    print(f"  Pseudo-Huber (δ=10):   {metrics['pseudo_huber_large']:.6f}")
    print(f"  CR (Compression):      {metrics.get('compression_ratio', 0.0):.4f}")
    print(f"  Corr (Pearson):        {metrics.get('corr', 0.0):.4f}")

    print("\nERROR STATS:")
    print(f"  Mean:   {metrics['error_mean']:.6f}")
    print(f"  Median: {metrics['error_median']:.6f}")
    print(f"  Std:    {metrics['error_std']:.6f}")

    if 'dir_acc' in metrics:
        print("\nDIRECTIONAL:")
        print(f"  Accuracy: {metrics['dir_acc']:.2f}%")

    if 't_mae' in metrics:
        print(f"\nTHRESHOLDED METRICS (|y| ≥ {metrics['t_threshold']:.6f}, "
              f"n={metrics['t_n_samples']} / {metrics['t_pct_samples']:.1f}%):")
        print(f"  MAE:                   {metrics['t_mae']:.6f}")
        print(f"  RMSE:                  {metrics['t_rmse']:.6f}")
        print(f"  R²:                    {metrics['t_r2']:.4f}")
        print(f"  Pseudo-Huber (δ=1):    {metrics['t_pseudo_huber']:.6f}")
        print(f"  Pseudo-Huber (δ=10):   {metrics['t_pseudo_huber_large']:.6f}")
        print(f"  Dir Acc:               {metrics['t_dir_acc']:.2f}%")

    if metrics.get('rank_score') is not None:
        print("\nSIMULATION & RANK SCORE (Simulation v2):")
        print(f"  Rank Score:            {metrics['rank_score']:.2f} / 100")
        print(f"  Signal Class:          {metrics.get('signal_class')}")
        print(f"  Sim Final Model:       ${metrics.get('sim_final_model', 0.0):.2f}")
        print(f"  Sim Final Oracle:      ${metrics.get('sim_final_oracle', 0.0):.2f}")
        if 'sim_capture_log' in metrics and metrics['sim_capture_log'] is not None:
            print(f"  Capture Log:           {metrics['sim_capture_log']:.4f}")
        print(f"  Pct Strong Signals:    {metrics.get('pct_strong', 0.0):.2f}%")


def interpret_metrics(metrics: dict) -> None:
    """Provide interpretation of metrics quality."""
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    r2 = metrics['r2']
    if r2 > 0.1:
        print(f"  ✓ Good fit: R² = {r2:.4f}")
    elif r2 > 0.05:
        print(f"  ○ Moderate fit: R² = {r2:.4f}")
    else:
        print(f"  ✗ Weak fit: R² = {r2:.4f}")

    dir_acc = metrics.get('dir_acc', 0)
    if dir_acc > 55:
        print(f"  ✓ Good direction: {dir_acc:.1f}%")
    elif dir_acc > 50:
        print(f"  ○ Slight edge: {dir_acc:.1f}%")
    else:
        print(f"  ✗ No edge: {dir_acc:.1f}%")

    cr = metrics.get('compression_ratio', 0.0)
    if cr < RANK_CR_DEAD:
        print(f"  ✗ Prediction collapse detected: CR = {cr:.4f} < {RANK_CR_DEAD}")
    elif 0.8 <= cr <= 1.2:
        print(f"  ✓ Healthy prediction variance: CR = {cr:.4f}")
    else:
        print(f"  ○ Variance non-standard: CR = {cr:.4f}")


def _jsonable(val):
    """Recursively convert numpy types and non-finite floats for safe JSON serialization."""
    if isinstance(val, (np.floating, float)):
        return float(val) if np.isfinite(val) else None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, np.ndarray):
        return [_jsonable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_jsonable(x) for x in val]
    return val
