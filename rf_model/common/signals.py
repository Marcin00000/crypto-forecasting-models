"""
Signal and volatility computation for Random Forest log-return pipeline.

Transforms continuous log-return predictions and market volatility into
probabilistic BUY/SELL signals via Gaussian Approach A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

__all__ = [
    'compute_sigma_series',
    'align_sigma',
    'fill_sigma_nan',
    'compute_signals',
    'compute_signal_thresholds',
    'compute_signal_metrics',
    '_last_finite',
]


def compute_sigma_series(close, window: int) -> np.ndarray:
    """
    Rolling standard deviation of 1-minute log-returns from close prices.

    The first `window` entries are NaN (insufficient history). Returns an
    array aligned 1:1 with close series.
    """
    close_series = pd.Series(np.asarray(close, dtype=np.float64))
    valid_close = close_series.where(close_series > 0, np.nan)
    log_ret = np.log(valid_close / valid_close.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy()


def align_sigma(sigma_series: np.ndarray, offset: int, n_samples: int) -> np.ndarray:
    """
    Map per-sample index to market volatility sigma at the sample's decision candle.

    INVARIANT: In rf_model the base entry candle for sample i is close[lookback + i]
    (since features up to lookback-1 are used, and target starts at lookback + i).
    Therefore callers in rf_model pass offset = int(lookback).
    """
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if len(sigma_series) == 0:
        return np.full(n_samples, np.nan, dtype=np.float64)

    idx = np.arange(n_samples) + int(offset)
    max_idx = len(sigma_series) - 1
    idx = np.clip(idx, 0, max_idx)
    return sigma_series[idx]


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """
    Replace NaN entries in aligned sigma with the median of finite entries.
    Provides in-sample fallback for warm-up history at the series beginning.
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
    signal_buy = (1.0 - ndtr((threshold - mu) / sigma_safe)) * 100.0
    signal_sell = ndtr((-threshold - mu) / sigma_safe) * 100.0
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
    return np.where(valid, threshold, np.nan)


def _expanding_percentile_threshold(values, percentile_cutoff: float, warmup: int) -> np.ndarray:
    """Causal expanding quantile strictly using observations before t (vectorized, identical to LR)."""
    quantile = 1.0 - float(percentile_cutoff) / 100.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile_cutoff must be between 0 and 100")
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .expanding(min_periods=int(warmup))
        .quantile(quantile, interpolation='linear')
        .shift(1)
        .to_numpy(dtype=np.float64)
    )


def _last_finite(values: np.ndarray, default: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(default)
    return float(finite[-1])


def compute_signal_thresholds(
    buy_signals,
    sell_signals,
    *,
    mode: str = 'absolute',
    min_signal: float = 60.0,
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    history: str = 'full',
    warmup: int = 2000,
    eligible: np.ndarray | None = None,
) -> dict:
    """
    Compute per-step execution thresholds and mask of eligible steps.

    Modes:
      - absolute: fixed min_signal for all steps
      - percentile: top percentile_cutoff of historical signals
      - zscore: expanding mean + zscore_cutoff * std
    """
    buy = np.asarray(buy_signals, dtype=np.float64)
    sell = np.asarray(sell_signals, dtype=np.float64)
    n = len(buy)
    base_eligible = np.isfinite(buy) & np.isfinite(sell)
    if eligible is not None:
        base_eligible = base_eligible & np.asarray(eligible, dtype=bool)

    if mode == 'absolute':
        t_buy = np.full(n, float(min_signal), dtype=np.float64)
        t_sell = np.full(n, float(min_signal), dtype=np.float64)
        is_eligible = base_eligible
    elif mode == 'zscore':
        if history == 'expanding':
            t_buy = _expanding_zscore_threshold(buy, zscore_cutoff, warmup)
            t_sell = _expanding_zscore_threshold(sell, zscore_cutoff, warmup)
        else:
            b_fin = buy[base_eligible]
            s_fin = sell[base_eligible]
            tb = float(np.mean(b_fin) + zscore_cutoff * np.std(b_fin)) if len(b_fin) else 60.0
            ts = float(np.mean(s_fin) + zscore_cutoff * np.std(s_fin)) if len(s_fin) else 60.0
            t_buy = np.full(n, tb, dtype=np.float64)
            t_sell = np.full(n, ts, dtype=np.float64)
        is_eligible = base_eligible & np.isfinite(t_buy) & np.isfinite(t_sell)
    elif mode == 'percentile':
        if history == 'expanding':
            t_buy = _expanding_percentile_threshold(buy, percentile_cutoff, warmup)
            t_sell = _expanding_percentile_threshold(sell, percentile_cutoff, warmup)
        else:
            q = 100.0 - float(percentile_cutoff)
            b_fin = buy[base_eligible]
            s_fin = sell[base_eligible]
            tb = float(np.percentile(b_fin, q)) if len(b_fin) else 60.0
            ts = float(np.percentile(s_fin, q)) if len(s_fin) else 60.0
            t_buy = np.full(n, tb, dtype=np.float64)
            t_sell = np.full(n, ts, dtype=np.float64)
        is_eligible = base_eligible & np.isfinite(t_buy) & np.isfinite(t_sell)
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")

    return {
        'buy': t_buy,
        'sell': t_sell,
        'eligible': is_eligible,
        'mode': mode,
        'history': history,
        'warmup': int(warmup),
    }


def compute_signal_metrics(
    y_pred,
    y_true,
    sigma,
    threshold: float,
    *,
    min_signal: float = 60.0,
    threshold_mode: str = 'absolute',
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    threshold_history: str = 'full',
    warmup: int = 0,
    eligible: np.ndarray | None = None,
) -> dict:
    """
    Compute calibration, coverage, and direction accuracy for strong trading signals.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    buy, sell = compute_signals(y_pred, sigma, threshold)
    th = compute_signal_thresholds(
        buy,
        sell,
        mode=threshold_mode,
        min_signal=min_signal,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        history=threshold_history,
        warmup=warmup,
        eligible=eligible,
    )

    mask = th['eligible']
    n_eligible = int(np.sum(mask))
    if n_eligible == 0:
        return {
            'n_eligible': 0,
            'pct_strong': 0.0,
            'calib_buy': None,
            'calib_sell': None,
            'dir_acc_strong': None,
            'bias_buy_strong': None,
            'mean_spread': None,
            'first_buy_threshold': None,
            'last_buy_threshold': None,
            'first_sell_threshold': None,
            'last_sell_threshold': None,
            '_threshold_eligible': mask,
            '_buy_thresholds': th['buy'],
            '_sell_thresholds': th['sell'],
        }

    # Definitions mirror the core LR pipeline so that activity, calibration and
    # direction feeding the rank score are identical for identical predictions.
    strong_buy = mask & (buy >= th['buy'])
    strong_sell = mask & (sell >= th['sell'])
    strong_any = strong_buy | strong_sell
    n_strong = int(np.sum(strong_any))
    n_buy = int(np.sum(strong_buy))
    n_sell = int(np.sum(strong_sell))

    pct_strong = float(n_strong / n_eligible * 100.0) if n_eligible > 0 else 0.0
    calib_buy = float(np.mean(y_true[strong_buy] > threshold)) if n_buy > 0 else None
    calib_sell = float(np.mean(y_true[strong_sell] < -threshold)) if n_sell > 0 else None

    dir_acc_strong = None
    bias_buy_strong = None
    if n_strong > 0:
        pred_dir = np.where(buy >= sell, 1.0, -1.0)[strong_any]
        true_dir = np.sign(y_true[strong_any])
        # Flat realised moves (y_true == 0) carry no direction and are excluded.
        nonzero = true_dir != 0
        if nonzero.any():
            dir_acc_strong = float(np.mean(pred_dir[nonzero] == true_dir[nonzero]) * 100.0)
        bias_buy_strong = float(np.mean(pred_dir > 0) * 100.0)

    mean_spread = float(np.mean(np.abs(buy[mask] - sell[mask])))

    return {
        'n_eligible': n_eligible,
        'pct_strong': pct_strong,
        'calib_buy': calib_buy,
        'calib_sell': calib_sell,
        'dir_acc_strong': dir_acc_strong,
        'bias_buy_strong': bias_buy_strong,
        'mean_spread': mean_spread,
        'first_buy_threshold': float(th['buy'][mask][0]),
        'last_buy_threshold': float(th['buy'][mask][-1]),
        'first_sell_threshold': float(th['sell'][mask][0]),
        'last_sell_threshold': float(th['sell'][mask][-1]),
        '_threshold_eligible': mask,
        '_buy_thresholds': th['buy'],
        '_sell_thresholds': th['sell'],
    }
