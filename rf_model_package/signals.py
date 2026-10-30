"""
Market volatility and probabilistic BUY / SELL signals for rf_model_package.

Formulas are identical to rf_model/common/signals.py (Gaussian "Approach A"), so signals
produced by the package match the ones stored by rf_test.py for the same mu, sigma, T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr


def compute_sigma_series(close, window: int) -> np.ndarray:
    """
    Rolling standard deviation of 1-candle log-returns, aligned 1:1 with `close`.

    The first `window` entries are NaN (insufficient history). Non-positive prices are
    masked so a corrupt tick cannot poison the whole rolling window with inf.
    """
    close_series = pd.Series(np.asarray(close, dtype=np.float64))
    valid_close = close_series.where(close_series > 0, np.nan)
    log_ret = np.log(valid_close / valid_close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std().to_numpy()


def align_sigma(sigma_series: np.ndarray, offset: int, n_samples: int) -> np.ndarray:
    """
    Sigma at the decision candle of each sample: sigma_series[i + offset].

    Tabular RF samples decide on candle `lookback + i` (features are shifted by one
    candle), so training code passes offset = lookback. The package gets the same
    alignment from the preserved row index instead of an explicit offset.
    """
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if len(sigma_series) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)
    idx = np.clip(np.arange(n_samples) + int(offset), 0, len(sigma_series) - 1)
    return sigma_series[idx]


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """Replace NaN sigma (rolling warm-up) with the median of finite values."""
    sigma = np.asarray(sigma, dtype=np.float64)
    finite = sigma[np.isfinite(sigma)]
    if len(finite) == 0:
        return sigma
    return np.where(np.isfinite(sigma), sigma, float(np.median(finite)))


def compute_signals(mu, sigma, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    signal_buy  = P(r >  T | mu, sigma) * 100
    signal_sell = P(r < -T | mu, sigma) * 100

    BUY + SELL < 100 always; the gap is the probability of the "hold" zone |r| <= T.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    signal_buy = (1.0 - ndtr((threshold - mu) / sigma_safe)) * 100.0
    signal_sell = ndtr((-threshold - mu) / sigma_safe) * 100.0
    return np.clip(signal_buy, 0.0, 100.0), np.clip(signal_sell, 0.0, 100.0)
