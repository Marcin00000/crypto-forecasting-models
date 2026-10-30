"""
Market simulation engine for LSTM price prediction pipeline.

Performs long/short capital simulation over non-overlapping horizon steps
and computes market capture metrics relative to an omniscient oracle.
"""
from __future__ import annotations

import math
import numpy as np

from .signals import (
    align_sigma,
    compute_sigma_series,
    compute_signals,
    compute_signal_thresholds,
    compute_signal_metrics,
    _last_finite,
)

__all__ = [
    'simulate_long_short',
    'derive_sim_capture_log',
    'evaluate_signal_and_simulation',
]


def simulate_long_short(
    mu,
    y_true,
    sigma,
    threshold: float,
    *,
    min_signal: float = 60.0,
    horizon: int = 1,
    start_cash: float = 100.0,
    fee_bps: float = 0.0,
    threshold_mode: str = 'absolute',
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    position_mode: str = 'sticky',
    threshold_history: str = 'full',
    warmup: int = 0,
    thresholds: dict | None = None,
) -> dict:
    """
    Naive $100 long+short simulation over non-overlapping horizon steps.
    """
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = min(len(mu), len(y_true), len(sigma))
    signal_buy, signal_sell = compute_signals(mu[:n], sigma[:n], threshold)

    if thresholds is None:
        thresholds = compute_signal_thresholds(
            signal_buy,
            signal_sell,
            mode=threshold_mode,
            min_signal=min_signal,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            history=threshold_history,
            warmup=warmup,
        )

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


def evaluate_signal_and_simulation(
    y_true,
    y_pred,
    close=None,
    *,
    lookback: int,
    horizon: int,
    threshold: float,
    window_sigma: int = 100,
    min_signal: float = 60.0,
    start_cash: float = 100.0,
    fee_bps: float = 0.0,
    sigma: np.ndarray | None = None,
    threshold_mode: str = 'absolute',
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    position_mode: str = 'sticky',
    threshold_history: str = 'full',
    warmup: int = 0,
) -> dict:
    """
    Compute signal metrics + oracle/model simulation for a tested model.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)[:n]
    else:
        if close is None:
            raise ValueError("close or sigma is required")
        sigma_series = compute_sigma_series(close, window_sigma)
        sigma = align_sigma(sigma_series, lookback, n)

    out: dict = {}
    true_std = float(np.std(y_true)) if n else 0.0
    pred_std = float(np.std(y_pred)) if n else 0.0
    out['compression_ratio'] = (pred_std / true_std) if true_std > 0 else 0.0

    out.update({
        key: value for key, value in compute_signal_metrics(
            y_pred,
            y_true,
            sigma,
            threshold,
            min_signal=min_signal,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            threshold_history=threshold_history,
            warmup=warmup,
        ).items() if not str(key).startswith('_')
    })

    model_sim = simulate_long_short(
        y_pred,
        y_true,
        sigma,
        threshold,
        min_signal=min_signal,
        horizon=horizon,
        start_cash=start_cash,
        fee_bps=0.0,
        threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        position_mode=position_mode,
        threshold_history=threshold_history,
        warmup=warmup,
    )
    oracle_sim = simulate_long_short(
        y_true,
        y_true,
        sigma,
        threshold,
        min_signal=min_signal,
        horizon=horizon,
        start_cash=start_cash,
        fee_bps=0.0,
        threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        position_mode=position_mode,
        threshold_history=threshold_history,
        warmup=warmup,
    )

    oracle_final = oracle_sim['final_value']
    model_final = model_sim['final_value']
    oracle_profit = oracle_final - start_cash

    out['sim_start_cash'] = float(start_cash)
    out['sim_final_oracle'] = float(oracle_final)
    out['sim_final_model'] = float(model_final)
    out['sim_return_oracle_pct'] = float(oracle_sim['return_pct'])
    out['sim_return_model_pct'] = float(model_sim['return_pct'])
    out['sim_capture_ratio'] = (
        float((model_final - start_cash) / oracle_profit)
        if oracle_profit > 1e-9 else None
    )
    out['sim_capture_log'] = derive_sim_capture_log(model_final, oracle_final, start_cash)
    out['sim_n_trades_oracle'] = int(oracle_sim['n_trades'])
    out['sim_n_trades_model'] = int(model_sim['n_trades'])
    out['sim_n_steps'] = int(model_sim['n_steps'])
    out['sim_min_signal'] = float(min_signal)
    out['sim_fee_bps'] = float(fee_bps)

    if fee_bps > 0.0:
        model_net = simulate_long_short(
            y_pred,
            y_true,
            sigma,
            threshold,
            min_signal=min_signal,
            horizon=horizon,
            start_cash=start_cash,
            fee_bps=fee_bps,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            position_mode=position_mode,
            threshold_history=threshold_history,
            warmup=warmup,
        )
        out['sim_final_model_net'] = float(model_net['final_value'])
        out['sim_return_model_net_pct'] = float(model_net['return_pct'])

    return out
