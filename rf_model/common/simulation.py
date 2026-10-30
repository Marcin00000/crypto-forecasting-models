"""
Trading simulation engine for Random Forest log-return pipeline.

Performs naive long/short capital simulation over non-overlapping horizon steps
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
    Simulate $100 long/short trading over non-overlapping horizon steps.

    Position logic:
      - long (+1): buy signal >= threshold and buy >= sell
      - short (-1): sell signal >= threshold and sell > buy
      - sticky: maintain prior position when neither threshold fires
      - reset: exit to cash when neither threshold fires

    Returns:
      Dictionary with final_value, return_pct, trade counts, fee legs, and equity curve.
    """
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    step = max(int(horizon), 1)
    indices = np.arange(0, len(mu), step)
    n_steps = len(indices)

    if thresholds is None:
        buy_sig, sell_sig = compute_signals(mu, sigma, threshold)
        thresholds = compute_signal_thresholds(
            buy_sig,
            sell_sig,
            mode=threshold_mode,
            min_signal=min_signal,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            history=threshold_history,
            warmup=warmup,
        )
    else:
        buy_sig, sell_sig = compute_signals(mu, sigma, threshold)

    t_buy = thresholds['buy']
    t_sell = thresholds['sell']
    eligible = thresholds['eligible']

    cash = float(start_cash)
    current_pos = 0.0  # -1.0, 0.0, +1.0
    equity = [cash]
    positions: list[float] = []
    trades = 0
    fee_legs = 0
    fee_factor = float(fee_bps) * 1e-4

    n_long = 0
    n_short = 0
    n_flat = 0

    for i in indices:
        if not eligible[i]:
            if position_mode == 'reset' and current_pos != 0.0:
                fee_legs += 1
                cash *= (1.0 - fee_factor)
                current_pos = 0.0
            equity.append(cash)
            positions.append(float(current_pos))
            n_flat += 1
            continue

        b = buy_sig[i]
        s = sell_sig[i]
        tb = t_buy[i]
        ts = t_sell[i]

        new_pos = current_pos
        if b >= tb and b >= s:
            new_pos = 1.0
        elif s >= ts and s > b:
            new_pos = -1.0
        elif position_mode == 'reset':
            new_pos = 0.0

        if new_pos != current_pos:
            legs = 2 if (current_pos != 0.0 and new_pos != 0.0) else 1
            fee_legs += legs
            cash *= math.pow(1.0 - fee_factor, legs)
            trades += 1
            current_pos = new_pos

        if current_pos == 1.0:
            n_long += 1
        elif current_pos == -1.0:
            n_short += 1
        else:
            n_flat += 1

        actual_ret = y_true[i]
        if current_pos != 0.0 and np.isfinite(actual_ret):
            pnl_mult = math.exp(current_pos * float(actual_ret))
            cash *= pnl_mult

        equity.append(cash)
        positions.append(float(current_pos))

    ret_pct = ((cash / float(start_cash)) - 1.0) * 100.0 if start_cash > 0 else 0.0

    return {
        'final_value': float(cash),
        'return_pct': float(ret_pct),
        'n_trades': int(trades),
        'n_fee_legs': int(fee_legs),
        'n_steps': int(n_steps),
        'n_long': int(n_long),
        'n_short': int(n_short),
        'n_flat': int(n_flat),
        'equity': equity,
        'positions': positions,
        '_threshold_eligible': eligible,
        '_buy_thresholds': t_buy,
        '_sell_thresholds': t_sell,
    }


def derive_sim_capture_log(
    model_final: float,
    oracle_final: float,
    start_cash: float = 100.0,
    *,
    sim_start_cash: float | None = None,
) -> float | None:
    """
    Log-return ratio of model growth to omniscient oracle growth.

    Defined as: log(model_final / start_cash) / log(oracle_final / start_cash).
    Returns None if oracle did not achieve positive log growth (<= start_cash * 1.001).
    """
    if sim_start_cash is not None:
        start_cash = sim_start_cash
    if start_cash <= 0.0 or oracle_final <= start_cash * 1.001:
        return None
    if model_final <= 0.0:
        return 0.0
    r_oracle = math.log(oracle_final / start_cash)
    r_model = math.log(model_final / start_cash)
    return float(r_model / r_oracle)


def evaluate_signal_and_simulation(
    y_pred,
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
    threshold_history: str = 'full',
    warmup: int = 0,
) -> dict:
    """
    Compute signal calibration metrics, long/short simulation, and oracle capture.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    buy_sig, sell_sig = compute_signals(y_pred, sigma, threshold)
    model_th = compute_signal_thresholds(
        buy_sig,
        sell_sig,
        mode=threshold_mode,
        min_signal=min_signal,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        history=threshold_history,
        warmup=warmup,
    )

    sim = simulate_long_short(
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
        position_mode='sticky',
        threshold_history=threshold_history,
        warmup=warmup,
        thresholds=model_th,
    )

    # Oracle uses true log-returns as predictions (zero fee)
    buy_orc, sell_orc = compute_signals(y_true, sigma, threshold)
    orc_th = compute_signal_thresholds(
        buy_orc,
        sell_orc,
        mode=threshold_mode,
        min_signal=min_signal,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        history=threshold_history,
        warmup=warmup,
    )
    sim_oracle = simulate_long_short(
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
        position_mode='sticky',
        threshold_history=threshold_history,
        warmup=warmup,
        thresholds=orc_th,
    )

    sig_metrics = compute_signal_metrics(
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
        eligible=model_th['eligible'],
    )

    capture_log = derive_sim_capture_log(
        sim['final_value'], sim_oracle['final_value'], start_cash
    )
    oracle_profit = sim_oracle['final_value'] - start_cash
    capture_ratio = (
        float((sim['final_value'] - start_cash) / oracle_profit)
        if oracle_profit > 1e-9 else None
    )

    out = {
        'sim_final_value': sim['final_value'],
        'sim_return_pct': sim['return_pct'],
        'sim_n_trades': sim['n_trades'],
        'sim_n_fee_legs': sim['n_fee_legs'],
        'sim_n_steps': sim['n_steps'],
        'sim_n_long': sim['n_long'],
        'sim_n_short': sim['n_short'],
        'sim_n_flat': sim['n_flat'],
        'oracle_final_value': sim_oracle['final_value'],
        'oracle_return_pct': sim_oracle['return_pct'],
        'oracle_n_trades': sim_oracle['n_trades'],
        'sim_capture_ratio': capture_ratio,
        'sim_capture_log': capture_log,
    }
    for k, v in sig_metrics.items():
        if not k.startswith('_'):
            out[k] = v

    return out
