"""
Standalone signal and market volatility calculation for lstm_price_model_package.

The formulas mirror lstm_model/common/signals.py so that package output equals the
signals stored by lstm_test_price.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


def compute_sigma_series(close, window: int = 100) -> np.ndarray:
    """
    Rolling standard deviation of 1-step log-returns from close prices.
    Returns an array aligned 1:1 with the close series (first `window` entries are NaN).
    """
    close_series = pd.Series(np.asarray(close, dtype=np.float64))
    valid_close = close_series.where(close_series > 0, np.nan)
    log_ret = np.log(valid_close / valid_close.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy(dtype=np.float64)


def align_sigma(sigma_series: np.ndarray, lookback: int, n_samples: int,
                offset: int | None = None) -> np.ndarray:
    """
    Map sample index to sigma at the sample's decision candle.

    For sample i with window [i, i + lookback) the last visible candle is
    t = i + lookback - 1, hence the default offset lookback - 1.
    """
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if len(sigma_series) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)
    actual_offset = int(lookback) - 1 if offset is None else int(offset)
    idx = np.clip(np.arange(n_samples) + actual_offset, 0, len(sigma_series) - 1)
    return sigma_series[idx]


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """
    Replace NaN entries with the median of the finite ones (warm-up fallback, no look-ahead
    beyond the batch itself). Only used when warm-up rows are kept (drop_warmup=False).
    """
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    finite_mask = np.isfinite(sigma_arr)
    if np.any(finite_mask):
        median_val = float(np.median(sigma_arr[finite_mask]))
        return np.where(finite_mask, sigma_arr, median_val)
    return sigma_arr


def compute_signals(mu: np.ndarray, sigma: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Probabilistic BUY / SELL signals in [0, 100] (Gaussian approach).

        signal_buy  = P(r >  threshold) = (1 - Phi((threshold - mu) / sigma)) * 100
        signal_sell = P(r < -threshold) = Phi((-threshold - mu) / sigma) * 100

    ndtr is the C implementation behind scipy.stats.norm.cdf, so the result is
    bit-identical to the training-side formula while avoiding per-call overhead.
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    signal_buy = (1.0 - ndtr((threshold - mu_arr) / sigma_safe)) * 100.0
    signal_sell = ndtr((-threshold - mu_arr) / sigma_safe) * 100.0
    return np.clip(signal_buy, 0.0, 100.0), np.clip(signal_sell, 0.0, 100.0)


def get_required_warmup(features: list[str]) -> int:
    """Minimum candle warm-up required by the feature set (raw mode)."""
    if 'vol_regime' in features:
        return 530
    if 'vol_percentile_240' in features:
        return 254
    return 100
