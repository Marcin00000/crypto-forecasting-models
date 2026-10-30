"""
Market simulation for the LSTM Signal Model pipeline.

Signal models output direct p_buy / p_sell in [0, 1]. Simulation uses those
probabilities scaled to [0, 100] — NOT log-return → compute_signals (LR path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    'probs_to_signal_strength',
    'compute_signal_thresholds',
    'threshold_activity_metrics',
    'simulate_from_signals',
    'evaluate_signal_simulation',
    'evaluate_signal_simulation_v2',
    'evaluate_signal_simulation_profile',
    'threshold_activity_metrics',
    'derive_sim_capture_log',
]


def probs_to_signal_strength(probs):
    """
    Scale model/oracle probabilities from [0, 1] to signal strength [0, 100].

    Args:
        probs: (n, 2) array or tuple of (buy, sell) 1d arrays in [0, 1]

    Returns:
        (signal_buy, signal_sell) as float64 arrays in [0, 100]
    """
    if isinstance(probs, (tuple, list)) and len(probs) == 2:
        buy = np.asarray(probs[0], dtype=np.float64)
        sell = np.asarray(probs[1], dtype=np.float64)
    else:
        arr = np.asarray(probs, dtype=np.float64)
        buy = arr[:, 0]
        sell = arr[:, 1]
    return buy * 100.0, sell * 100.0


def _expanding_zscore_threshold(values, cutoff: float, warmup: int) -> np.ndarray:
    """Population mean/std threshold using finite observations strictly before t."""
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
    """Linear expanding quantile, shifted so the value at t is never included."""
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
                              zscore_cutoff: float = 2.0,
                              history: str = 'full',
                              warmup: int = 0) -> dict:
    """
    Build threshold series shared by metrics and simulation.

    ``history='expanding'`` is causal: thresholds at t use only finite values
    from [0, t). Dynamic thresholds are unavailable until ``warmup`` prior
    observations exist. ``history='full'`` preserves legacy single-mode.
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
            buy_t = _expanding_quantile_threshold(
                buy, percentile_cutoff, warmup)
            sell_t = _expanding_quantile_threshold(
                sell, percentile_cutoff, warmup)
    else:
        finite_buy = buy[np.isfinite(buy)]
        finite_sell = sell[np.isfinite(sell)]
        if mode == 'percentile':
            q = 1.0 - float(percentile_cutoff) / 100.0
            b_value = float(np.quantile(finite_buy, q))
            s_value = float(np.quantile(finite_sell, q))
        else:
            b_value = float(
                np.mean(finite_buy) + float(zscore_cutoff) * np.std(finite_buy))
            s_value = float(
                np.mean(finite_sell) + float(zscore_cutoff) * np.std(finite_sell))
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


def simulate_from_signals(signal_buy, signal_sell, y_true_lr, *,
                          min_signal: float, horizon: int,
                          start_cash: float = 100.0,
                          fee_bps: float = 0.0,
                          threshold_mode: str = 'absolute',
                          percentile_cutoff: float = 5.0,
                          zscore_cutoff: float = 2.0,
                          position_mode: str = 'sticky',
                          threshold_history: str = 'full',
                          warmup: int = 0,
                          thresholds: dict | None = None) -> dict:
    """
    Naive $100 long+short simulation over non-overlapping horizon steps.

    Uses precomputed signal strengths in [0, 100] — no Gaussian transform.

    Args:
        fee_bps: Transaction cost in basis points applied on each position
                 change (one-way). E.g. fee_bps=5 → 0.05% per trade side.
        threshold_mode: 'absolute' | 'percentile' | 'zscore'
        position_mode: 'sticky' (keep position until opposite signal) | 'reset' (return to 0 on neutral)
    """
    signal_buy = np.asarray(signal_buy, dtype=np.float64)
    signal_sell = np.asarray(signal_sell, dtype=np.float64)
    y_true = np.asarray(y_true_lr, dtype=np.float64)
    n = min(len(signal_buy), len(signal_sell), len(y_true))
    fee_factor = float(fee_bps) * 1e-4  # bps → fraction

    if thresholds is None:
        thresholds = compute_signal_thresholds(
            signal_buy, signal_sell, mode=threshold_mode, min_signal=min_signal,
            percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
            history=threshold_history, warmup=warmup,
        )

    step = max(int(horizon), 1)
    cash = float(start_cash)
    position = 0
    prev_position = 0
    n_trades = 0
    n_fee_legs = 0
    n_steps = 0
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
        else:
            if position_mode == 'reset':
                position = 0

        if position != prev_position:
            n_trades += 1
            legs = 2 if (prev_position != 0 and position != 0) else 1
            n_fee_legs += legs
            # apply fee per leg on position change
            if fee_factor > 0.0:
                cash *= ((1.0 - fee_factor) ** legs)
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


def _thresholds_of(sim: dict) -> dict:
    """Threshold bundle of a finished simulation, in the shape simulate_from_signals(thresholds=) expects."""
    return {
        'buy': sim['_buy_thresholds'],
        'sell': sim['_sell_thresholds'],
        'eligible': sim['_threshold_eligible'],
    }


def evaluate_signal_simulation(y_pred, target_oracle, y_true_lr, *,
                               horizon: int,
                               min_signal: float = 60.0,
                               oracle_min_signal: float = 60.0,
                               start_cash: float = 100.0,
                               fee_bps: float = 0.0,
                               threshold_mode: str = 'absolute',
                               percentile_cutoff: float = 5.0,
                               zscore_cutoff: float = 2.0,
                               oracle_threshold_mode: str | None = None,
                               oracle_percentile_cutoff: float | None = None,
                               oracle_zscore_cutoff: float | None = None,
                               position_mode: str = 'sticky',
                               threshold_history: str = 'full',
                               warmup: int = 0,
                               oracle_threshold_history: str | None = None,
                               oracle_warmup: int | None = None) -> dict:
    """
    Run model + oracle simulation for a signal model on held-out data.

    Args:
        y_pred:            (n, 2) model predictions in [0, 1]
        target_oracle:     (n, 2) oracle targets in [0, 1]
        y_true_lr:         (n,) realised log-returns
        horizon:           forward horizon for non-overlapping steps
        min_signal:        strong-signal cutoff in [0, 100] for model (e.g. 60)
        oracle_min_signal: canonical strong-signal cutoff for oracle benchmark (default 60.0)
        start_cash:        starting capital (e.g. 100)
        fee_bps:           transaction cost in basis points (one-way, per trade).
        threshold_mode:    'absolute' | 'percentile' | 'zscore' for model
        percentile_cutoff: percentile cutoff for model (e.g. 5.0)
        zscore_cutoff:     zscore cutoff for model (e.g. 2.0)
        oracle_threshold_mode: threshold mode for oracle (defaults to threshold_mode if None)
        oracle_percentile_cutoff: percentile cutoff for oracle (defaults to percentile_cutoff if None)
        oracle_zscore_cutoff: zscore cutoff for oracle (defaults to zscore_cutoff if None)
        position_mode:     'sticky' | 'reset'

    Returns:
        Flat dict with sim_* metrics ready to merge into metrics.json.
    """
    buy_m, sell_m = probs_to_signal_strength(y_pred)
    buy_o, sell_o = probs_to_signal_strength(target_oracle)

    o_mode = oracle_threshold_mode if oracle_threshold_mode is not None else threshold_mode
    o_p_cutoff = oracle_percentile_cutoff if oracle_percentile_cutoff is not None else percentile_cutoff
    o_z_cutoff = oracle_zscore_cutoff if oracle_zscore_cutoff is not None else zscore_cutoff
    o_history = (oracle_threshold_history if oracle_threshold_history is not None
                 else threshold_history)
    o_warmup = oracle_warmup if oracle_warmup is not None else warmup

    model_sim = simulate_from_signals(
        buy_m, sell_m, y_true_lr,
        min_signal=min_signal, horizon=horizon,
        start_cash=start_cash, fee_bps=fee_bps,
        threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        position_mode=position_mode,
        threshold_history=threshold_history,
        warmup=warmup,
    )
    oracle_sim = simulate_from_signals(
        buy_o, sell_o, y_true_lr,
        min_signal=oracle_min_signal, horizon=horizon,
        start_cash=start_cash, fee_bps=fee_bps,
        threshold_mode=o_mode,
        percentile_cutoff=o_p_cutoff,
        zscore_cutoff=o_z_cutoff,
        position_mode=position_mode,
        threshold_history=o_history,
        warmup=o_warmup,
    )

    if fee_bps > 0.0:
        # Thresholds depend only on signals, not on fees: reuse them from the net run.
        model_sim_gross = simulate_from_signals(
            buy_m, sell_m, y_true_lr,
            min_signal=min_signal, horizon=horizon,
            start_cash=start_cash, fee_bps=0.0,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            position_mode=position_mode,
            threshold_history=threshold_history,
            warmup=warmup,
            thresholds=_thresholds_of(model_sim),
        )
        oracle_sim_gross = simulate_from_signals(
            buy_o, sell_o, y_true_lr,
            min_signal=oracle_min_signal, horizon=horizon,
            start_cash=start_cash, fee_bps=0.0,
            threshold_mode=o_mode,
            percentile_cutoff=o_p_cutoff,
            zscore_cutoff=o_z_cutoff,
            position_mode=position_mode,
            threshold_history=o_history,
            warmup=o_warmup,
            thresholds=_thresholds_of(oracle_sim),
        )
    else:
        model_sim_gross = model_sim
        oracle_sim_gross = oracle_sim

    oracle_profit = oracle_sim['final_value'] - start_cash
    model_profit = model_sim['final_value'] - start_cash
    capture = (model_profit / oracle_profit) if oracle_profit > 1e-9 else None

    out = {
        'sim_start_cash': float(start_cash),
        'sim_final_model': model_sim_gross['final_value'],
        'sim_final_oracle': oracle_sim_gross['final_value'],
        'sim_return_model_pct': model_sim_gross['return_pct'],
        'sim_return_oracle_pct': oracle_sim_gross['return_pct'],
        'sim_final_model_net': model_sim['final_value'],
        'sim_final_oracle_net': oracle_sim['final_value'],
        'sim_return_model_net_pct': model_sim['return_pct'],
        'sim_return_oracle_net_pct': oracle_sim['return_pct'],
        'sim_capture_ratio': capture,
        'sim_capture_log': derive_sim_capture_log(
            model_sim['final_value'], oracle_sim['final_value'], start_cash),
        'sim_n_trades_model': model_sim['n_trades'],
        'sim_n_trades_oracle': oracle_sim['n_trades'],
        'sim_n_steps': model_sim['n_steps'],
        'sim_min_signal': float(min_signal),
        'sim_horizon': int(horizon),
        'sim_fee_bps': float(fee_bps),
        # backward-compat alias
        'sim_model_final': model_sim['final_value'],
        'sim_model_return_pct': model_sim['return_pct'],
        'sim_oracle_final': oracle_sim['final_value'],
        'sim_oracle_return_pct': oracle_sim['return_pct'],
        '_model_sim': model_sim,
        '_oracle_sim': oracle_sim,
    }
    return out


def threshold_activity_metrics(*args, **kwargs):
    """Backward-compatible export; implementation lives in ``metrics``."""
    from .metrics import threshold_activity_metrics as _activity
    return _activity(*args, **kwargs)


def evaluate_signal_simulation_v2(*args, **kwargs):
    """Causal ranking profile; implemented in ``simulation_profile``."""
    from .simulation_profile import evaluate_signal_simulation_v2 as _eval_v2
    return _eval_v2(*args, **kwargs)


evaluate_signal_simulation_profile = evaluate_signal_simulation_v2


def derive_sim_capture_log(sim_final_model: float, sim_final_oracle: float,
                           start_cash: float = 100.0) -> float | None:
    """
    Log-scale capture metric analogous to rank_v2 in LR.

    capture_log = log(model/start) / log(oracle/start)
    Returns None when oracle didn't produce meaningful profit.
    """
    min_oracle_ratio = 1.001
    oracle_ratio = sim_final_oracle / start_cash
    if oracle_ratio <= min_oracle_ratio:
        return None
    model_ratio = sim_final_model / start_cash
    if model_ratio <= 0:
        return 0.0
    log_oracle = np.log(oracle_ratio)
    log_model = np.log(model_ratio)
    return float(log_model / log_oracle)
