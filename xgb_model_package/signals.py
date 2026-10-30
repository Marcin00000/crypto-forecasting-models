"""
Market volatility and probabilistic BUY / SELL signals for xgb_model_package.

Standalone copy of the formulas in xgb_model/common/signals.py (the package
must not import from the training project). Signals follow Gaussian
"Approach A": two independent tail probabilities around a +/- threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


def compute_sigma_series(close, window: int) -> np.ndarray:
    """
    Rolling standard deviation of 1-candle log-returns from close prices.

    The first `window` entries are NaN (insufficient history). The array is
    aligned 1:1 with the close series.
    """
    close = pd.Series(np.asarray(close, dtype=np.float64))
    valid_close = close.where(close > 0, np.nan)
    log_ret = np.log(valid_close / valid_close.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy()


def align_sigma(sigma_series: np.ndarray, offset: int, n_samples: int) -> np.ndarray:
    """
    Map sample index i to sigma at row (offset + i) of the working frame.

    In the tabular models the decision candle of sample i is row lookback + i
    (features are shifted by one candle), so callers pass offset = lookback.
    """
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if len(sigma_series) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    idx = np.arange(n_samples) + int(offset)
    idx = np.clip(idx, 0, len(sigma_series) - 1)
    return sigma_series[idx]


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """
    Replace NaN entries with the median of finite entries.

    Lets signals be computed for warm-up rows without look-ahead on a single
    candle; the median is only a placeholder for rows that carry no sigma.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    finite = sigma[np.isfinite(sigma)]
    if len(finite) > 0:
        fill = float(np.median(finite))
        return np.where(np.isfinite(sigma), sigma, fill)
    return sigma


def compute_signals(mu, sigma, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised probabilistic BUY / SELL signals in [0, 100].

    signal_buy  = P(r >  threshold | mu, sigma) * 100
    signal_sell = P(r < -threshold | mu, sigma) * 100
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    signal_buy = (1.0 - ndtr((threshold - mu) / sigma_safe)) * 100.0
    signal_sell = ndtr((-threshold - mu) / sigma_safe) * 100.0
    return np.clip(signal_buy, 0.0, 100.0), np.clip(signal_sell, 0.0, 100.0)
