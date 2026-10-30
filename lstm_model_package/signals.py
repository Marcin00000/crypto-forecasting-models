"""
Fast, standalone signal and market volatility calculation for lstm_model_package.
Uses C-level scipy.special.ndtr for high-throughput batch evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

VOL_NORM_SIGMA_WINDOW = 100


def compute_sigma_series(close, window: int) -> np.ndarray:
    """
    Rolling standard deviation of 1-step log-returns from close prices.
    Returns 1-D numpy array aligned with close series. The first log-return is
    undefined (no previous close), so the first finite sigma sits at index `window`
    (the first `window` entries are NaN).
    """
    close_series = pd.Series(np.asarray(close, dtype=np.float64))
    log_ret = np.log(close_series / close_series.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy(dtype=np.float64)


def align_sigma(sigma_series: np.ndarray, lookback: int, n_samples: int) -> np.ndarray:
    """
    Align sigma to the last visible candle of each sample window.
    For sample i seeing close[i : i + lookback], last candle is i + lookback - 1.
    """
    idx = np.arange(n_samples) + (lookback - 1)
    max_idx = len(sigma_series) - 1
    idx = np.clip(idx, 0, max_idx)
    return sigma_series[idx]


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """
    Replace NaN entries in aligned sigma with the median of finite values.
    Provides fallback for initial rolling window history without data leakage.
    """
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    finite_mask = np.isfinite(sigma_arr)
    if np.any(finite_mask):
        median_val = float(np.median(sigma_arr[finite_mask]))
        return np.where(finite_mask, sigma_arr, median_val)
    return sigma_arr


def denormalize_mu_vol_norm(mu_norm: np.ndarray, close: np.ndarray, lookback: int) -> np.ndarray:
    """
    Denormalize target predictions for models trained with target_mode='vol_norm'.
    Uses fixed 100-candle rolling sigma (SIGNAL_WINDOW_SIGMA) per training protocol.
    mu_raw = mu_norm * sigma_100
    """
    sigma_series = compute_sigma_series(close, VOL_NORM_SIGMA_WINDOW)
    sigma_aligned = align_sigma(sigma_series, lookback, len(mu_norm))
    sigma_safe = fill_sigma_nan(sigma_aligned)
    return mu_norm * sigma_safe


def valid_sigma_mask(close, sigma_aligned: np.ndarray, lookback: int,
                     vol_norm: bool = False) -> np.ndarray:
    """
    Rows whose volatility is defined (live use drops the rolling-sigma warm-up).

    For target_mode='vol_norm' the prediction itself is rescaled by the fixed 100-candle
    sigma, so rows where that sigma is undefined are dropped even when the signal
    window is shorter (otherwise mu would rest on a median-filled rescaling).
    """
    mask = np.isfinite(sigma_aligned)
    if vol_norm:
        sigma_100 = align_sigma(
            compute_sigma_series(close, VOL_NORM_SIGMA_WINDOW), lookback, len(mask))
        mask &= np.isfinite(sigma_100)
    return mask


def compute_signals(mu: np.ndarray, sigma: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    High-performance vectorised probabilistic BUY / SELL signals in [0, 100].
    Computed using C-level Gaussian cumulative distribution function (ndtr).
    
    signal_buy  = P(r > threshold)  = Phi((mu - threshold) / sigma) * 100
    signal_sell = P(r < -threshold) = Phi((-threshold - mu) / sigma) * 100
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    
    # 1 - Phi((threshold - mu)/sigma) = Phi((mu - threshold)/sigma)
    buy_prob = ndtr((mu_arr - threshold) / sigma_safe) * 100.0
    sell_prob = ndtr((-threshold - mu_arr) / sigma_safe) * 100.0
    
    signal_buy = np.clip(buy_prob, 0.0, 100.0)
    signal_sell = np.clip(sell_prob, 0.0, 100.0)
    return signal_buy, signal_sell


def get_required_warmup(features: list[str]) -> int:
    """Determine minimum candle warmup required by the feature set."""
    if 'vol_regime' in features:
        return 530
    if 'vol_percentile_240' in features:
        return 254
    return 100
