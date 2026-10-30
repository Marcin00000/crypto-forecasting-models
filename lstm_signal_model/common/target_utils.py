"""
Oracle target generation for Signal Model.

Generates (p_buy, p_sell) targets from future log-returns using the same
probabilistic formula as lstm_model_package / simulation.py, ensuring
parity between training targets and inference signals.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def compute_signals(mu, sigma, threshold: float):
    """
    Vectorised probabilistic BUY / SELL signals in [0, 100].

    Identical formula to lstm_logreturn_model/common/simulation.py and
    lstm_model_package._compute_signals. Kept as a local copy to avoid
    importing from modules that may change in model-specific ways.

    signal_buy  = P(r >  threshold | mu, sigma) * 100
    signal_sell = P(r < -threshold | mu, sigma) * 100
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    signal_buy = (1.0 - ndtr((threshold - mu) / sigma_safe)) * 100.0
    signal_sell = ndtr((-threshold - mu) / sigma_safe) * 100.0
    signal_buy = np.clip(signal_buy, 0.0, 100.0)
    signal_sell = np.clip(signal_sell, 0.0, 100.0)
    return signal_buy, signal_sell


def compute_sigma_series(close, window: int) -> np.ndarray:
    """
    Rolling std of 1-step log-returns from close prices.

    Local copy of simulation.compute_sigma_series for isolation.
    First ``window`` entries are NaN.
    """
    import pandas as pd
    close = pd.Series(np.asarray(close, dtype=np.float64))
    log_ret = np.log(close / close.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy()


def align_sigma(sigma_series: np.ndarray, lookback: int,
                n_samples: int) -> np.ndarray:
    """
    Map per-sample index → sigma at the sample's last visible candle.

    For sample i the model sees close[i : i+lookback]; the last visible
    candle is i + lookback - 1.
    """
    idx = np.arange(n_samples) + (lookback - 1)
    max_idx = len(sigma_series) - 1
    idx = np.clip(idx, 0, max_idx)
    return sigma_series[idx]


def generate_oracle_targets(y_true_log_returns, sigma_aligned, threshold):
    """
    Generate oracle (p_buy, p_sell) from realised future log-returns.

    Uses the same probabilistic formula as compute_signals but with
    mu = y_true (perfect foresight). Output normalised to [0, 1].

    Args:
        y_true_log_returns: np.ndarray — realised future log-returns
        sigma_aligned:      np.ndarray — market volatility per sample
        threshold:          float — economic threshold T

    Returns:
        (oracle_buy, oracle_sell) — np.ndarrays in [0, 1]
    """
    oracle_buy, oracle_sell = compute_signals(
        y_true_log_returns, sigma_aligned, threshold)
    return oracle_buy / 100.0, oracle_sell / 100.0


def winsorize_targets(y_train, y_val=None, percentile=99.0):
    """
    Clip extreme log-return targets (0.5% lower, 0.5% upper for percentile=99.0).
    Bounds computed ONLY from y_train to avoid data leakage.

    Args:
        y_train:    np.ndarray — training log-returns
        y_val:      optional np.ndarray — validation log-returns
        percentile: float in (50, 100) — e.g. 99.0 clips the outer 1%

    Returns:
        y_train_w, y_val_w, clip_bounds
    """
    valid_train = y_train[np.isfinite(y_train)]
    if len(valid_train) == 0:
        return y_train, y_val, {}

    lower_pct = (100.0 - float(percentile)) / 2.0
    upper_pct = 100.0 - lower_pct

    lower = float(np.percentile(valid_train, lower_pct))
    upper = float(np.percentile(valid_train, upper_pct))

    y_train_w = np.clip(y_train, lower, upper)
    y_val_w = np.clip(y_val, lower, upper) if y_val is not None else None

    clip_bounds = {'lower': lower, 'upper': upper, 'percentile': percentile}
    return y_train_w, y_val_w, clip_bounds
