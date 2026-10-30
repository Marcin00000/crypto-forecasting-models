"""
Signal + market-simulation metrics for the LSTM Log-Return pipeline.

This module turns raw log-return predictions into the trading-signal view the
project actually cares about (see lstm_3.md section 10) and a small,
intentionally naive market simulation used to rank models by *usefulness*
rather than regression error.

Why this exists
---------------
Standard regression metrics (MAE / R² / t_dir_acc) reward models that collapse
their predictions toward 0: a flat predictor has a tiny MAE yet produces no
tradable signal. The signal metrics + simulation here expose that failure mode
directly.

Signal model (Approach A — two independent thresholds, normal distribution),
identical to ``xgb_model_package._compute_signals`` so LSTM and XGB are
comparable:

    mu    = model prediction (log-return, raw space)
    sigma = rolling std of 1-step log-returns from close (market volatility)
    T     = economic threshold (minimum meaningful move)

    signal_buy  = P(r >  T) = (1 - Phi((T  - mu) / sigma)) * 100   [0-100]
    signal_sell = P(r < -T) = Phi((-T - mu) / sigma) * 100          [0-100]

Market simulation (per the project owner's spec):
    Start with $100, long+short all-in, no fees. Two runs over the test set:
      - oracle: mu = y_true   (upper bound — perfect foresight)
      - model:  mu = y_pred
    Both use the SAME signal->action logic. Non-overlapping horizon steps avoid
    counting the same forward move ``horizon`` times.
"""
from __future__ import annotations

import math

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
    'simulate_long_short',
    'evaluate_signal_and_simulation',
    'evaluate_all_simulation_modes',
    'evaluate_lr_simulation_v2',
    'evaluate_lr_simulation_profile',
    '_compute_thresholds',
    'derive_sim_capture_log',
]


# =========================================================================
# VOLATILITY (market sigma)
# =========================================================================

def compute_sigma_series(close, window: int) -> np.ndarray:
    """
    Rolling std of 1-step log-returns from close prices.

    The first ``window`` entries are NaN (insufficient history). Returns an
    array aligned 1:1 with ``close`` so callers can index by candle position.
    """
    close = pd.Series(np.asarray(close, dtype=np.float64))
    log_ret = np.log(close / close.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy()


def align_sigma(sigma_series: np.ndarray, lookback: int,
                n_samples: int) -> np.ndarray:
    """
    Map per-sample index -> sigma at the sample's "current" candle.

    For sample ``i`` the model sees close[i : i+lookback]; the last visible
    candle (the one a live signal would fire on) is ``i + lookback - 1``.
    Therefore ``sigma_aligned[i] = sigma_series[i + lookback - 1]``.

    Indices are clipped to the valid range; out-of-range / NaN sigma values are
    preserved as NaN so downstream code can mask them.
    """
    idx = np.arange(n_samples) + (lookback - 1)
    max_idx = len(sigma_series) - 1
    idx = np.clip(idx, 0, max_idx)
    return sigma_series[idx]


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """
    Replace NaN entries in an aligned sigma array with the median of the
    finite entries (in-sample fallback for insufficient rolling-window
    history at the start of a series).

    Used consistently wherever a sigma array feeds a component that cannot
    tolerate NaN (e.g. ``compute_signals`` -> isotonic calibrator predict) —
    keep any such downstream use in sync with this helper instead of
    duplicating the fill logic (see ``calibrator_fit.py`` and
    ``lstm_test.py::_posthoc_calibrator_metrics``).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    finite = sigma[np.isfinite(sigma)]
    if len(finite):
        fill = float(np.median(finite))
        return np.where(np.isfinite(sigma), sigma, fill)
    return sigma


# =========================================================================
# SIGNAL CALCULATION & THRESHOLDING
# =========================================================================

def compute_signals(mu, sigma, threshold: float):
    """
    Vectorised probabilistic BUY / SELL signals in [0, 100].

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
    """Population mean/std using finite observations strictly before t."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(clean)))
    sums2 = np.concatenate(([0.0], np.cumsum(clean * clean)))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    idx = np.arange(len(values))
    count = counts[idx]
    valid = count >= int(warmup)
    mean = np.divide(
        sums[idx], count, out=np.full(len(values), np.nan), where=valid)
    second = np.divide(
        sums2[idx], count, out=np.full(len(values), np.nan), where=valid)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return mean + float(cutoff) * std


def _expanding_quantile_threshold(
        values, percentile_cutoff: float, warmup: int) -> np.ndarray:
    """Linear expanding quantile, shifted so t is never included."""
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


def compute_signal_thresholds(signal_buy, signal_sell, *,
                              mode: str = 'absolute',
                              min_signal: float = 60.0,
                              percentile_cutoff: float = 5.0,
                              zscore_cutoff: float = 1.6,
                              history: str = 'full',
                              warmup: int = 0) -> dict:
    """Shared threshold series for metrics and simulation."""
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
            buy_t = _expanding_quantile_threshold(
                buy, percentile_cutoff, warmup)
            sell_t = _expanding_quantile_threshold(
                sell, percentile_cutoff, warmup)
    else:
        finite_buy = buy[np.isfinite(buy)]
        finite_sell = sell[np.isfinite(sell)]
        if mode == 'percentile':
            q = 1.0 - float(percentile_cutoff) / 100.0
            b_value = float(np.quantile(finite_buy, q)) if len(finite_buy) else np.nan
            s_value = float(np.quantile(finite_sell, q)) if len(finite_sell) else np.nan
        else:
            b_value = float(
                np.mean(finite_buy) + float(zscore_cutoff) * np.std(finite_buy)
            ) if len(finite_buy) else np.nan
            s_value = float(
                np.mean(finite_sell) + float(zscore_cutoff) * np.std(finite_sell)
            ) if len(finite_sell) else np.nan
        buy_t = np.full(n, b_value, dtype=np.float64)
        sell_t = np.full(n, s_value, dtype=np.float64)

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
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite[-1]) if finite.size else None


def _compute_thresholds(signal_buy, signal_sell, *, mode: str = 'absolute',
                        min_signal: float = 60.0,
                        percentile_cutoff: float = 5.0,
                        zscore_cutoff: float = 1.6) -> tuple[float, float]:
    """Legacy scalar thresholds (full-window). Kept for single-mode tests."""
    if mode == 'percentile':
        p_buy = float(np.percentile(signal_buy, 100.0 - percentile_cutoff))
        p_sell = float(np.percentile(signal_sell, 100.0 - percentile_cutoff))
        return p_buy, p_sell
    if mode == 'zscore':
        b_mean, b_std = float(np.mean(signal_buy)), float(np.std(signal_buy))
        s_mean, s_std = float(np.mean(signal_sell)), float(np.std(signal_sell))
        return b_mean + zscore_cutoff * b_std, s_mean + zscore_cutoff * s_std
    return float(min_signal), float(min_signal)


# =========================================================================
# SIGNAL METRICS
# =========================================================================

def compute_signal_metrics(mu, y_true, sigma, threshold: float,
                           min_signal: float = 60.0,
                           threshold_mode: str = 'absolute',
                           percentile_cutoff: float = 5.0,
                           zscore_cutoff: float = 1.6,
                           threshold_history: str = 'full',
                           warmup: int = 0,
                           eligible=None) -> dict:
    """
    Signal-quality metrics that expose collapse and calibration.

    Activity uses the same threshold series and eligible mask as simulation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = min(len(mu), len(y_true), len(sigma))
    mu, y_true, sigma = mu[:n], y_true[:n], sigma[:n]
    signal_buy, signal_sell = compute_signals(mu, sigma, threshold)
    thresholds = compute_signal_thresholds(
        signal_buy, signal_sell, mode=threshold_mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=threshold_history, warmup=warmup)
    mask = np.asarray(
        thresholds['eligible'] if eligible is None else eligible, dtype=bool)[:n]
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
    calib_buy = (float((yt_el[strong_buy] > threshold).mean())
                 if strong_buy.any() else None)
    calib_sell = (float((yt_el[strong_sell] < -threshold).mean())
                  if strong_sell.any() else None)
    if strong_any.any():
        pred_dir = np.where(buy_el >= sell_el, 1.0, -1.0)[strong_any]
        true_dir = np.sign(yt_el[strong_any])
        nz = true_dir != 0
        dir_acc_strong = (float((pred_dir[nz] == true_dir[nz]).mean() * 100)
                          if nz.any() else None)
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
        'signal_window_sigma_valid_pct': (
            float(np.isfinite(sigma).mean() * 100) if n else 0.0),
        'pct_strong': pct_strong,
        'pct_strong_buy': float(strong_buy.mean() * 100) if n_el else 0.0,
        'pct_strong_sell': float(strong_sell.mean() * 100) if n_el else 0.0,
        'n_strong_buy': int(strong_buy.sum()),
        'n_strong_sell': int(strong_sell.sum()),
        'eligible_n': n_el,
        'calib_buy': calib_buy,
        'calib_sell': calib_sell,
        'n_strong': int(strong_any.sum()),
        'dir_acc_strong': dir_acc_strong,
        '_buy_thresholds': thresholds['buy'],
        '_sell_thresholds': thresholds['sell'],
        '_threshold_eligible': mask,
    }


# =========================================================================
# MARKET SIMULATION (long + short, all-in)
# =========================================================================

def simulate_long_short(mu, y_true, sigma, threshold: float,
                        min_signal: float = 60.0, horizon: int = 1,
                        start_cash: float = 100.0,
                        fee_bps: float = 0.0,
                        threshold_mode: str = 'absolute',
                        percentile_cutoff: float = 5.0,
                        zscore_cutoff: float = 1.6,
                        position_mode: str = 'sticky',
                        threshold_history: str = 'full',
                        warmup: int = 0,
                        thresholds: dict | None = None) -> dict:
    """Naive $100 long+short simulation over non-overlapping horizon steps."""
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = min(len(mu), len(y_true), len(sigma))
    signal_buy, signal_sell = compute_signals(mu[:n], sigma[:n], threshold)
    if thresholds is None:
        thresholds = compute_signal_thresholds(
            signal_buy, signal_sell, mode=threshold_mode, min_signal=min_signal,
            percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
            history=threshold_history, warmup=warmup)
    step = max(int(horizon), 1)
    fee_factor = float(fee_bps) * 1e-4
    cash = float(start_cash)
    position = prev_position = 0
    n_trades = n_fee_legs = n_steps = 0
    n_long = n_short = n_flat = 0
    equity_curve = [cash]
    positions = []
    step_indices = []
    for i in range(0, n, step):
        b, s = signal_buy[i], signal_sell[i]
        if not thresholds['eligible'][i]:
            position = 0
        elif b >= thresholds['buy'][i] and b >= s:
            position = 1
        elif s >= thresholds['sell'][i]:
            position = -1
        elif position_mode == 'reset':
            position = 0
        if position != prev_position:
            n_trades += 1
            legs = 2 if (prev_position != 0 and position != 0) else 1
            n_fee_legs += legs
            if fee_factor > 0.0:
                cash *= (1.0 - fee_factor) ** legs
            prev_position = position
        r = y_true[i]
        if np.isfinite(r):
            cash *= float(np.exp(position * r))
        n_steps += 1
        if position == 1:
            n_long += 1
        elif position == -1:
            n_short += 1
        else:
            n_flat += 1
        equity_curve.append(cash)
        positions.append(position)
        step_indices.append(i)
    return {
        'final_value': cash,
        'return_pct': (cash / start_cash - 1.0) * 100.0,
        'n_trades': n_trades,
        'n_steps': n_steps,
        'n_long': n_long,
        'n_short': n_short,
        'n_flat': n_flat,
        'buy_threshold_eval': _last_finite(thresholds['buy']),
        'sell_threshold_eval': _last_finite(thresholds['sell']),
        'threshold_mode': threshold_mode,
        'threshold_history': threshold_history,
        'threshold_warmup': int(warmup),
        'n_fee_legs': n_fee_legs,
        'equity_curve': np.asarray(equity_curve, dtype=np.float64),
        'positions': np.asarray(positions, dtype=np.int8),
        'step_indices': np.asarray(step_indices, dtype=np.int64),
        '_buy_thresholds': thresholds['buy'],
        '_sell_thresholds': thresholds['sell'],
        '_threshold_eligible': thresholds['eligible'],
    }


def derive_sim_capture_log(
    sim_final_model: float,
    sim_final_oracle: float,
    sim_start_cash: float = 100.0,
) -> float | None:
    """Log-scale simulation capture for rank v2 (retroactive from stored sim_*)."""
    if sim_final_oracle <= sim_start_cash * 1.001:
        return None
    if sim_final_model <= 0:
        return 0.0
    return (
        math.log(sim_final_model / sim_start_cash)
        / math.log(sim_final_oracle / sim_start_cash)
    )


# =========================================================================
# TOP-LEVEL ENTRY (called by lstm_test)
# =========================================================================

def _thresholds_of(result: dict) -> dict:
    """Threshold bundle of a metrics/simulation result (the `thresholds` argument of simulate_long_short)."""
    keys = {'buy': '_buy_thresholds', 'sell': '_sell_thresholds', 'eligible': '_threshold_eligible'}
    return {name: result[key] for name, key in keys.items()}


def evaluate_signal_and_simulation(y_true, y_pred, close, *, lookback: int,
                                   horizon: int, threshold: float,
                                   window_sigma: int, min_signal: float = 60.0,
                                   start_cash: float = 100.0,
                                   fee_bps: float = 0.0,
                                   sigma: np.ndarray | None = None,
                                   threshold_mode: str = 'absolute',
                                   percentile_cutoff: float = 5.0,
                                   zscore_cutoff: float = 1.6,
                                   position_mode: str = 'sticky',
                                   threshold_history: str = 'full',
                                   warmup: int = 0) -> dict:
    """
    Compute signal metrics + oracle/model simulation for a tested model.

    Args:
        y_true:  realised log-returns per sample (raw space)
        y_pred:  predicted log-returns per sample (raw space)
        close:   full close-price series for the test CSV (length >= samples)
        lookback / horizon: sequence + target geometry
        threshold:    economic threshold T (e.g. 0.001)
        window_sigma: rolling window for market volatility
        min_signal:   strong-signal cutoff (e.g. 60)
        start_cash:   starting capital (default $100)
        fee_bps:      transaction fee in bps per leg
        sigma:        optional pre-aligned per-sample volatility array
        threshold_mode: 'absolute' | 'percentile' | 'zscore'
        percentile_cutoff: top X% for percentile mode
        zscore_cutoff:     Z-score multiplier for zscore mode
        position_mode:     'sticky' | 'reset'

    Returns a flat dict ready to merge into metrics.json.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)[:n]
    else:
        sigma_series = compute_sigma_series(close, window_sigma)
        sigma = align_sigma(sigma_series, lookback, n)

    out: dict = {}

    true_std = float(np.std(y_true)) if n else 0.0
    pred_std = float(np.std(y_pred)) if n else 0.0
    out['compression_ratio'] = (pred_std / true_std) if true_std > 0 else 0.0

    sig_metrics = compute_signal_metrics(
        y_pred, y_true, sigma, threshold,
        min_signal=min_signal,
        threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        threshold_history=threshold_history,
        warmup=warmup,
    )
    out.update({k: v for k, v in sig_metrics.items() if not str(k).startswith('_')})

    # Thresholds depend only on signals, not on fees: reuse them instead of recomputing the
    # expanding quantile/z-score for every simulation leg (model/oracle, gross/net).
    model_thresholds = _thresholds_of(sig_metrics)
    model_sim = simulate_long_short(
        y_pred, y_true, sigma, threshold,
        min_signal=min_signal, horizon=horizon, start_cash=start_cash,
        fee_bps=0.0, threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        position_mode=position_mode,
        threshold_history=threshold_history, warmup=warmup,
        thresholds=model_thresholds,
    )
    oracle_sim = simulate_long_short(
        y_true, y_true, sigma, threshold,
        min_signal=min_signal, horizon=horizon, start_cash=start_cash,
        fee_bps=0.0, threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        position_mode=position_mode,
        threshold_history=threshold_history, warmup=warmup,
    )

    oracle_final = oracle_sim['final_value']
    model_final = model_sim['final_value']
    oracle_profit = oracle_final - start_cash
    model_profit = model_final - start_cash
    if oracle_profit > 1e-9:
        capture = model_profit / oracle_profit
    elif oracle_profit < -1e-9:
        # negative-oracle regime: capture undefined, rank component neutral
        capture = None
    else:
        capture = None

    sim_capture_log = derive_sim_capture_log(
        model_final, oracle_final, start_cash,
    )

    out.update({
        'sim_start_cash': float(start_cash),
        'sim_final_model': model_final,
        'sim_final_oracle': oracle_final,
        'sim_return_model_pct': model_sim['return_pct'],
        'sim_return_oracle_pct': oracle_sim['return_pct'],
        'sim_capture_ratio': capture,
        'sim_capture_log': sim_capture_log,
        'sim_n_trades_model': model_sim['n_trades'],
        'sim_n_trades_oracle': oracle_sim['n_trades'],
        'sim_n_steps': model_sim['n_steps'],
        'signal_threshold': float(threshold),
        'signal_window_sigma': int(window_sigma),
        'threshold_mode': threshold_mode,
        'position_mode': position_mode,
        'buy_threshold_eval': model_sim.get('buy_threshold_eval'),
        'sell_threshold_eval': model_sim.get('sell_threshold_eval'),
    })
    if fee_bps > 0:
        model_net = simulate_long_short(
            y_pred, y_true, sigma, threshold,
            min_signal=min_signal, horizon=horizon,
            start_cash=start_cash, fee_bps=fee_bps,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            position_mode=position_mode,
            threshold_history=threshold_history, warmup=warmup,
            thresholds=model_thresholds,
        )
        oracle_thresholds = _thresholds_of(oracle_sim)
        oracle_net = simulate_long_short(
            y_true, y_true, sigma, threshold,
            min_signal=min_signal, horizon=horizon,
            start_cash=start_cash, fee_bps=fee_bps,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            position_mode=position_mode,
            threshold_history=threshold_history, warmup=warmup,
            thresholds=oracle_thresholds,
        )
        out['sim_final_model_net'] = model_net['final_value']
        out['sim_final_oracle_net'] = oracle_net['final_value']
        out['sim_fee_bps'] = float(fee_bps)
    return out


def evaluate_all_simulation_modes(y_true, y_pred, close, *, lookback: int,
                                  horizon: int, threshold: float,
                                  window_sigma: int, min_signal: float = 60.0,
                                  percentile_cutoff: float = 5.0,
                                  zscore_cutoff: float = 1.6,
                                  position_mode: str = 'sticky',
                                  start_cash: float = 100.0,
                                  fee_bps: float = 0.0,
                                  sigma: np.ndarray | None = None) -> dict:
    """
    Run evaluation and simulation across all 3 threshold modes:
      1. absolute   (min_signal, e.g. 60%)
      2. zscore     (mean + Z*std, e.g. 1.6 sigma)
      3. percentile (top X%, e.g. 5.0%)

    Returns dict with per-mode metrics and ranking comparisons.
    """
    from .metrics import compute_rank_score

    modes_config = [
        ('absolute', {
            'threshold_mode': 'absolute',
            'min_signal': min_signal,
            'percentile_cutoff': percentile_cutoff,
            'zscore_cutoff': zscore_cutoff,
        }),
        ('zscore', {
            'threshold_mode': 'zscore',
            'min_signal': min_signal,
            'percentile_cutoff': percentile_cutoff,
            'zscore_cutoff': zscore_cutoff,
        }),
        ('percentile', {
            'threshold_mode': 'percentile',
            'min_signal': min_signal,
            'percentile_cutoff': percentile_cutoff,
            'zscore_cutoff': zscore_cutoff,
        }),
    ]

    modes_results = {}
    for mode_name, cfg in modes_config:
        res = evaluate_signal_and_simulation(
            y_true, y_pred, close,
            lookback=lookback, horizon=horizon, threshold=threshold,
            window_sigma=window_sigma, min_signal=cfg['min_signal'],
            start_cash=start_cash, fee_bps=fee_bps, sigma=sigma,
            threshold_mode=cfg['threshold_mode'],
            percentile_cutoff=cfg['percentile_cutoff'],
            zscore_cutoff=cfg['zscore_cutoff'],
            position_mode=position_mode,
        )
        rank_info = compute_rank_score(res)
        res.update(rank_info)
        modes_results[mode_name] = res

    best_rank_mode = max(
        modes_results.keys(),
        key=lambda m: (modes_results[m].get('rank_score') or 0.0)
    )
    best_sim_mode = max(
        modes_results.keys(),
        key=lambda m: (modes_results[m].get('sim_final_model') or 0.0)
    )

    return {
        'modes': modes_results,
        'best_mode_by_rank': best_rank_mode,
        'best_mode_by_sim': best_sim_mode,
    }


def evaluate_lr_simulation_v2(*args, **kwargs):
    from .simulation_profile import evaluate_lr_simulation_v2 as _evaluate
    return _evaluate(*args, **kwargs)


evaluate_lr_simulation_profile = evaluate_lr_simulation_v2
