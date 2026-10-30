"""Causal ``lr_simulation_v2`` profile assembled from shared Gaussian thresholds."""
from __future__ import annotations

import numpy as np

from .simulation import (
    align_sigma,
    compute_sigma_series,
    compute_signal_metrics,
    compute_signal_thresholds,
    compute_signals,
    derive_sim_capture_log,
    simulate_long_short,
)

RANKING_METHOD = 'absolute_60_gross'
SIMULATION_PROFILE = 'lr_simulation_v2'
VARIANT_KEYS = (
    'absolute_60',
    'absolute_70',
    'zscore_expanding',
    'percentile_expanding',
)
_COMPACT_KEYS = (
    'final_value', 'return_pct', 'n_trades', 'n_fee_legs', 'n_steps',
    'n_long', 'n_short', 'n_flat',
)


def zscore_cutoff_for_asset(asset: str) -> float:
    name = str(asset).upper()
    if 'ETH' in name:
        return 1.6
    if 'BTC' in name or (name[:1] == 'B' and (len(name) == 1 or name[1].isdigit())):
        return 1.5
    return 1.6


def _compact_sim(sim: dict) -> dict:
    return {key: sim[key] for key in _COMPACT_KEYS}


def _threshold_summary(sim: dict) -> dict:
    eligible = sim['_threshold_eligible']

    def _summary(values):
        selected = np.asarray(values, dtype=np.float64)[eligible]
        selected = selected[np.isfinite(selected)]
        if not selected.size:
            return {'first': None, 'last': None, 'mean': None}
        return {
            'first': float(selected[0]),
            'last': float(selected[-1]),
            'mean': float(np.mean(selected)),
        }

    return {
        'scale': '0_100',
        'buy': _summary(sim['_buy_thresholds']),
        'sell': _summary(sim['_sell_thresholds']),
    }


def _shared_thresholds(
        buy_m, sell_m, buy_o, sell_o, *, mode: str, min_signal: float,
        percentile_cutoff: float, zscore_cutoff: float, history: str,
        warmup: int) -> tuple[dict, dict]:
    model_th = compute_signal_thresholds(
        buy_m, sell_m, mode=mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup)
    oracle_th = compute_signal_thresholds(
        buy_o, sell_o, mode=mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup)
    shared = model_th['eligible'] & oracle_th['eligible']
    return dict(model_th, eligible=shared), dict(oracle_th, eligible=shared)


def _simulate_pair(mu, y_true, sigma, *, threshold: float, horizon: int,
                   start_cash: float, fee_bps: float, min_signal: float,
                   mode: str, percentile_cutoff: float, zscore_cutoff: float,
                   history: str, warmup: int, thresholds: dict) -> dict:
    return simulate_long_short(
        mu, y_true, sigma, threshold, min_signal=min_signal, horizon=horizon,
        start_cash=start_cash, fee_bps=fee_bps, threshold_mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        position_mode='sticky', threshold_history=history, warmup=warmup,
        thresholds=thresholds)


def build_lr_simulation_variant(
        y_pred, y_true, sigma, *, threshold: float, horizon: int,
        start_cash: float, mode: str, min_signal: float,
        percentile_cutoff: float, zscore_cutoff: float, history: str,
        warmup: int, fee_bps: float, include_net: bool) -> tuple:
    buy_m, sell_m = compute_signals(y_pred, sigma, threshold)
    buy_o, sell_o = compute_signals(y_true, sigma, threshold)
    model_th, oracle_th = _shared_thresholds(
        buy_m, sell_m, buy_o, sell_o, mode=mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup)
    model_gross = _simulate_pair(
        y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, fee_bps=0.0, min_signal=min_signal, mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup, thresholds=model_th)
    oracle_gross = _simulate_pair(
        y_true, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, fee_bps=0.0, min_signal=min_signal, mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup, thresholds=oracle_th)
    activity = compute_signal_metrics(
        y_pred, y_true, sigma, threshold, min_signal=min_signal,
        threshold_mode=mode, percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff, threshold_history=history, warmup=warmup,
        eligible=model_th['eligible'])
    activity = {key: value for key, value in activity.items()
                if not key.startswith('_')}
    capture_log = derive_sim_capture_log(
        model_gross['final_value'], oracle_gross['final_value'], start_cash)
    oracle_profit = oracle_gross['final_value'] - start_cash
    capture_ratio = (
        (model_gross['final_value'] - start_cash) / oracle_profit
        if oracle_profit > 1e-9 else None)
    variant = {
        'config': {
            'threshold_mode': mode,
            'threshold_history': history,
            'min_signal': float(min_signal) if mode == 'absolute' else None,
            'percentile_cutoff': (
                float(percentile_cutoff) if mode == 'percentile' else None),
            'zscore_cutoff': float(zscore_cutoff) if mode == 'zscore' else None,
            'warmup': int(warmup),
            'position_mode': 'sticky',
            'gross_fee_bps': 0.0,
        },
        'gross': {
            'model': _compact_sim(model_gross),
            'oracle': _compact_sim(oracle_gross),
            'capture_log': capture_log,
        },
        'oracle': _compact_sim(oracle_gross),
        'activity': activity,
        'position_changes': {
            'model': model_gross['n_trades'],
            'oracle': oracle_gross['n_trades'],
        },
        'thresholds': {
            'model': _threshold_summary(model_gross),
            'oracle': _threshold_summary(oracle_gross),
        },
    }
    if include_net:
        model_net = _simulate_pair(
            y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
            start_cash=start_cash, fee_bps=fee_bps, min_signal=min_signal,
            mode=mode, percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff, history=history, warmup=warmup,
            thresholds=model_th)
        oracle_net = _simulate_pair(
            y_true, y_true, sigma, threshold=threshold, horizon=horizon,
            start_cash=start_cash, fee_bps=fee_bps, min_signal=min_signal,
            mode=mode, percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff, history=history, warmup=warmup,
            thresholds=oracle_th)
        variant['config']['net_fee_bps'] = float(fee_bps)
        variant['net'] = {
            'model': _compact_sim(model_net),
            'oracle': _compact_sim(oracle_net),
            'capture_log': derive_sim_capture_log(
                model_net['final_value'], oracle_net['final_value'], start_cash),
        }
    return variant, model_gross, oracle_gross, capture_ratio


def _resolve_sigma(y_true, close, lookback, window_sigma, sigma):
    n = len(np.asarray(y_true))
    if sigma is not None:
        return np.asarray(sigma, dtype=np.float64)[:n]
    if close is None:
        raise ValueError("close or sigma is required")
    return align_sigma(compute_sigma_series(close, window_sigma), lookback, n)


def evaluate_lr_simulation_v2(
        y_true, y_pred, close=None, *, lookback: int, horizon: int,
        threshold: float, window_sigma: int = 100, asset: str = 'ETH',
        start_cash: float = 100.0, primary_fee_bps: float = 2.0,
        auxiliary_fee_bps: float = 2.0, warmup: int = 2000,
        position_mode: str = 'sticky', sigma=None) -> dict:
    """Primary A60 gross+net2; auxiliary variants also report gross 0 / net 2."""
    if position_mode != 'sticky':
        raise ValueError("lr_simulation_v2 requires position_mode='sticky'")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    sigma = _resolve_sigma(y_true, close, lookback, window_sigma, sigma)
    zscore_cutoff = zscore_cutoff_for_asset(asset)
    definitions = (
        ('absolute_60', 'absolute', 60.0, 5.0, zscore_cutoff, 'full', 0,
         primary_fee_bps),
        ('absolute_70', 'absolute', 70.0, 5.0, zscore_cutoff, 'full', 0,
         auxiliary_fee_bps),
        ('zscore_expanding', 'zscore', 60.0, 5.0, zscore_cutoff, 'expanding',
         warmup, auxiliary_fee_bps),
        ('percentile_expanding', 'percentile', 60.0, 5.0, zscore_cutoff,
         'expanding', warmup, auxiliary_fee_bps),
    )
    variants = {}
    private = {}
    for (name, mode, minimum, percentile, zscore, history, variant_warmup,
         fee) in definitions:
        variant, model_g, oracle_g, capture_ratio = build_lr_simulation_variant(
            y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
            start_cash=start_cash, mode=mode, min_signal=minimum,
            percentile_cutoff=percentile, zscore_cutoff=zscore,
            history=history, warmup=variant_warmup, fee_bps=fee,
            include_net=True)
        variants[name] = variant
        private[name] = (model_g, oracle_g, capture_ratio)

    primary = variants['absolute_60']
    model_gross, oracle_gross, capture_ratio = private['absolute_60']
    model_net = primary['net']['model']
    oracle_net = primary['net']['oracle']
    true_std = float(np.std(y_true)) if n else 0.0
    pred_std = float(np.std(y_pred)) if n else 0.0

    def _gross_final(name: str) -> float:
        return variants[name]['gross']['model']['final_value']

    def _net_final(name: str) -> float:
        return variants[name]['net']['model']['final_value']

    return {
        'simulation_schema_version': 2,
        'simulation_profile': SIMULATION_PROFILE,
        'ranking_method': RANKING_METHOD,
        'simulation_variants': variants,
        'compression_ratio': (pred_std / true_std) if true_std > 0 else 0.0,
        'sim_start_cash': float(start_cash),
        'sim_final_model': primary['gross']['model']['final_value'],
        'sim_final_oracle': primary['gross']['oracle']['final_value'],
        'sim_return_model_pct': primary['gross']['model']['return_pct'],
        'sim_return_oracle_pct': primary['gross']['oracle']['return_pct'],
        'sim_final_model_net': model_net['final_value'],
        'sim_final_oracle_net': oracle_net['final_value'],
        'sim_return_model_net_pct': model_net['return_pct'],
        'sim_return_oracle_net_pct': oracle_net['return_pct'],
        'sim_capture_log': primary['gross']['capture_log'],
        'sim_capture_ratio': capture_ratio,
        'sim_n_trades_model': primary['gross']['model']['n_trades'],
        'sim_n_trades_oracle': primary['gross']['oracle']['n_trades'],
        'sim_n_steps': primary['gross']['model']['n_steps'],
        'sim_min_signal': 60.0,
        'sim_horizon': int(horizon),
        'sim_fee_bps': float(primary_fee_bps),
        'sim_final_absolute_60': _gross_final('absolute_60'),
        'sim_final_absolute_60_net': _net_final('absolute_60'),
        'sim_final_absolute_70': _gross_final('absolute_70'),
        'sim_final_absolute_70_net': _net_final('absolute_70'),
        'sim_final_zscore_expanding': _gross_final('zscore_expanding'),
        'sim_final_zscore_expanding_net': _net_final('zscore_expanding'),
        'sim_final_percentile_expanding': _gross_final('percentile_expanding'),
        'sim_final_percentile_expanding_net': _net_final(
            'percentile_expanding'),
        'pct_strong': primary['activity']['pct_strong'],
        'pct_strong_buy': primary['activity']['pct_strong_buy'],
        'pct_strong_sell': primary['activity']['pct_strong_sell'],
        'calib_buy': primary['activity']['calib_buy'],
        'calib_sell': primary['activity']['calib_sell'],
        'n_strong': primary['activity']['n_strong'],
        'n_strong_buy': primary['activity']['n_strong_buy'],
        'n_strong_sell': primary['activity']['n_strong_sell'],
        'dir_acc_strong': primary['activity']['dir_acc_strong'],
        'mu_std': primary['activity']['mu_std'],
        'signal_buy_std': primary['activity']['signal_buy_std'],
        'signal_sell_std': primary['activity']['signal_sell_std'],
        'signal_min_strong': 60.0,
        'threshold_mode_eval': 'absolute',
        'threshold_history_eval': 'full',
        'threshold_warmup_eval': 0,
        'thresh_buy_eval': primary['activity']['thresh_buy_eval'],
        'thresh_sell_eval': primary['activity']['thresh_sell_eval'],
        '_model_sim': model_gross,
        '_oracle_sim': oracle_gross,
    }


evaluate_lr_simulation_profile = evaluate_lr_simulation_v2
