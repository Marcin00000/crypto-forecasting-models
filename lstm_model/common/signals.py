"""
Signal and volatility computation for LSTM price prediction pipeline.

Transforms continuous implied log-return predictions and market volatility into
probabilistic BUY/SELL signals via Gaussian Approach A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    'compute_sigma_series',
    'align_sigma',
    'fill_sigma_nan',
    'compute_signals',
    'compute_signal_thresholds',
    'compute_signal_metrics',
    '_last_finite',
]


def compute_sigma_series(close, window: int = 100) -> np.ndarray:
    """
    Rolling standard deviation of 1-step log-returns from close prices.

    The first `window` entries are NaN (insufficient history). Returns an
    array aligned 1:1 with close series.
    """
    close_series = pd.Series(np.asarray(close, dtype=np.float64))
    valid_close = close_series.where(close_series > 0, np.nan)
    log_ret = np.log(valid_close / valid_close.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy()


def align_sigma(sigma_series: np.ndarray, lookback: int, n_samples: int,
                offset: int | None = None) -> np.ndarray:
    """
    Map per-sample index to market volatility sigma at the sample's decision candle.

    In LSTM sequence generation, for sample i with window [i, i + lookback),
    the last visible decision candle is t = i + lookback - 1.
    Therefore, the default offset is lookback - 1.
    """
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if len(sigma_series) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    actual_offset = int(lookback) - 1 if offset is None else int(offset)
    idx = np.arange(n_samples) + actual_offset
    max_idx = len(sigma_series) - 1
    idx = np.clip(idx, 0, max_idx)
    return sigma_series[idx]


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """
    Replace NaN entries in aligned sigma with the median of finite entries.
    Provides in-sample fallback for warm-up history at the series beginning.
    Note: Simulation Profile v2 preserves NaNs in sigma without replacing them with median.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    finite = sigma[np.isfinite(sigma)]
    if len(finite) > 0:
        fill = float(np.median(finite))
        return np.where(np.isfinite(sigma), sigma, fill)
    return sigma


def compute_signals(mu, sigma, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised probabilistic BUY / SELL trading signals in [0, 100].

    signal_buy  = P(r >  threshold | mu, sigma) * 100
    signal_sell = P(r < -threshold | mu, sigma) * 100
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    signal_buy = (1.0 - norm.cdf((threshold - mu) / sigma_safe)) * 100.0
    signal_sell = norm.cdf((-threshold - mu) / sigma_safe) * 100.0
    signal_buy = np.clip(signal_buy, 0.0, 100.0)
    signal_sell = np.clip(signal_sell, 0.0, 100.0)
    return signal_buy, signal_sell


def _expanding_zscore_threshold(values, cutoff: float, warmup: int) -> np.ndarray:
    """Expanding population mean and std strictly using observations before t."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(clean)))
    sums2 = np.concatenate(([0.0], np.cumsum(clean * clean)))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    idx = np.arange(len(values))
    count = counts[idx]
    valid = count >= int(warmup)
    mean = np.divide(sums[idx], count, out=np.full(len(values), np.nan), where=valid)
    second = np.divide(sums2[idx], count, out=np.full(len(values), np.nan), where=valid)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    threshold = mean + float(cutoff) * std
    return threshold


def _expanding_quantile_threshold(values, percentile_cutoff: float, warmup: int) -> np.ndarray:
    """Expanding quantile shifted by 1 so candle t is strictly excluded."""
    quantile = 1.0 - float(percentile_cutoff) / 100.0
    if not (0.0 <= quantile <= 1.0):
        raise ValueError("percentile_cutoff must be between 0 and 100")
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .expanding(min_periods=int(warmup))
        .quantile(quantile, interpolation='linear')
        .shift(1)
        .to_numpy(dtype=np.float64)
    )


def compute_signal_thresholds(
    signal_buy,
    signal_sell,
    *,
    mode: str = 'absolute',
    min_signal: float = 60.0,
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    history: str = 'full',
    warmup: int = 0,
) -> dict:
    """
    Generate threshold arrays for BUY and SELL signals according to mode and history.
    """
    buy = np.asarray(signal_buy, dtype=np.float64)
    sell = np.asarray(signal_sell, dtype=np.float64)
    n = min(len(buy), len(sell))
    buy, sell = buy[:n], sell[:n]

    if mode not in {'absolute', 'percentile', 'zscore'}:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    if history not in {'full', 'expanding'}:
        raise ValueError(f"Unsupported threshold history: {history}")
    if int(warmup) < 0:
        raise ValueError("warmup must be non-negative")

    if mode == 'absolute':
        buy_t = np.full(n, float(min_signal), dtype=np.float64)
        sell_t = np.full(n, float(min_signal), dtype=np.float64)
    elif history == 'expanding':
        if mode == 'zscore':
            buy_t = _expanding_zscore_threshold(buy, zscore_cutoff, warmup)
            sell_t = _expanding_zscore_threshold(sell, zscore_cutoff, warmup)
        else:
            buy_t = _expanding_quantile_threshold(buy, percentile_cutoff, warmup)
            sell_t = _expanding_quantile_threshold(sell, percentile_cutoff, warmup)
    else:
        finite_buy = buy[np.isfinite(buy)]
        finite_sell = sell[np.isfinite(sell)]
        if mode == 'percentile':
            q = 1.0 - float(percentile_cutoff) / 100.0
            b_val = float(np.quantile(finite_buy, q)) if len(finite_buy) else np.nan
            s_val = float(np.quantile(finite_sell, q)) if len(finite_sell) else np.nan
        else:
            b_val = (
                float(np.mean(finite_buy) + float(zscore_cutoff) * np.std(finite_buy))
                if len(finite_buy) else np.nan
            )
            s_val = (
                float(np.mean(finite_sell) + float(zscore_cutoff) * np.std(finite_sell))
                if len(finite_sell) else np.nan
            )
        buy_t = np.full(n, b_val, dtype=np.float64)
        sell_t = np.full(n, s_val, dtype=np.float64)

    eligible = np.arange(n) >= int(warmup)
    eligible &= np.isfinite(buy) & np.isfinite(sell)
    eligible &= np.isfinite(buy_t) & np.isfinite(sell_t)

    return {
        'buy': buy_t,
        'sell': sell_t,
        'eligible': eligible,
        'mode': mode,
        'history': history,
        'warmup': int(warmup),
    }


def _last_finite(values) -> float | None:
    """Return the last finite value from array or None."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite[-1]) if finite.size else None


def compute_signal_metrics(
    mu,
    y_true,
    sigma,
    threshold: float,
    min_signal: float = 60.0,
    threshold_mode: str = 'absolute',
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    threshold_history: str = 'full',
    warmup: int = 0,
    eligible=None,
) -> dict:
    """
    Compute comprehensive signal metrics including strong activity, calibration, and direction.
    """
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = min(len(mu), len(y_true), len(sigma))
    mu, y_true, sigma = mu[:n], y_true[:n], sigma[:n]

    signal_buy, signal_sell = compute_signals(mu, sigma, threshold)
    thresholds = compute_signal_thresholds(
        signal_buy, signal_sell,
        mode=threshold_mode,
        min_signal=min_signal,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        history=threshold_history,
        warmup=warmup,
    )

    mask = np.asarray(thresholds['eligible'] if eligible is None else eligible, dtype=bool)[:n]
    n_el = int(mask.sum())
    buy_el = signal_buy[mask]
    sell_el = signal_sell[mask]
    yt_el = y_true[mask]
    buy_t = thresholds['buy'][mask]
    sell_t = thresholds['sell'][mask]

    strong_buy = buy_el >= buy_t
    strong_sell = sell_el >= sell_t
    strong_any = strong_buy | strong_sell
    pct_strong = float(strong_any.mean() * 100) if n_el else 0.0

    calib_buy = float((yt_el[strong_buy] > threshold).mean()) if strong_buy.any() else None
    calib_sell = float((yt_el[strong_sell] < -threshold).mean()) if strong_sell.any() else None

    if strong_any.any():
        pred_dir = np.where(buy_el >= sell_el, 1.0, -1.0)[strong_any]
        true_dir = np.sign(yt_el[strong_any])
        nz = true_dir != 0
        dir_acc_strong = float((pred_dir[nz] == true_dir[nz]).mean() * 100) if nz.any() else None
    else:
        dir_acc_strong = None

    return {
        'mu_std': float(np.std(mu[np.isfinite(sigma)])) if n else 0.0,
        'signal_buy_std': float(np.std(buy_el)) if n_el else 0.0,
        'signal_sell_std': float(np.std(sell_el)) if n_el else 0.0,
        'signal_min_strong': float(min_signal),
        'threshold_mode_eval': threshold_mode,
        'threshold_history_eval': threshold_history,
        'threshold_warmup_eval': int(warmup),
        'thresh_buy_eval': _last_finite(thresholds['buy']),
        'thresh_sell_eval': _last_finite(thresholds['sell']),
        'signal_window_sigma_valid_pct': float(np.isfinite(sigma).mean() * 100) if n else 0.0,
        'pct_strong': pct_strong,
        'pct_strong_buy': float(strong_buy.mean() * 100) if n_el else 0.0,
        'pct_strong_sell': float(strong_sell.mean() * 100) if n_el else 0.0,
        'n_strong': int(strong_any.sum()),
        'n_strong_buy': int(strong_buy.sum()),
        'n_strong_sell': int(strong_sell.sum()),
        'eligible_n': n_el,
        'calib_buy': calib_buy,
        'calib_sell': calib_sell,
        'dir_acc_strong': dir_acc_strong,
    }
