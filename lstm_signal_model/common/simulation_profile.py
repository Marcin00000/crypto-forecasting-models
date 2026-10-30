"""Causal ``signal_simulation_v2`` profile assembled from shared thresholds."""
from __future__ import annotations

import numpy as np

from .metrics import threshold_activity_metrics
from .simulation import (
    compute_signal_thresholds,
    derive_sim_capture_log,
    probs_to_signal_strength,
    simulate_from_signals,
)

RANKING_METHOD = 'absolute_20_gross'
VARIANT_KEYS = (
    'absolute_20',
    'absolute_30',
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
    model_th = dict(model_th, eligible=shared)
    oracle_th = dict(oracle_th, eligible=shared)
    return model_th, oracle_th


def _simulate_pair(buy, sell, y_true_lr, *, horizon: int, start_cash: float,
                   fee_bps: float, min_signal: float, mode: str,
                   percentile_cutoff: float, zscore_cutoff: float,
                   history: str, warmup: int, thresholds: dict) -> dict:
    return simulate_from_signals(
        buy, sell, y_true_lr, min_signal=min_signal, horizon=horizon,
        start_cash=start_cash, fee_bps=fee_bps, threshold_mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        position_mode='sticky', threshold_history=history, warmup=warmup,
        thresholds=thresholds)


def build_signal_simulation_variant(
        y_pred, target_oracle, y_true_lr, *, horizon: int, start_cash: float,
        mode: str, min_signal: float, percentile_cutoff: float,
        zscore_cutoff: float, history: str, warmup: int, fee_bps: float,
        include_net: bool) -> tuple[dict, dict, dict]:
    buy_m, sell_m = probs_to_signal_strength(y_pred)
    buy_o, sell_o = probs_to_signal_strength(target_oracle)
    model_th, oracle_th = _shared_thresholds(
        buy_m, sell_m, buy_o, sell_o, mode=mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup)
    model_gross = _simulate_pair(
        buy_m, sell_m, y_true_lr, horizon=horizon, start_cash=start_cash,
        fee_bps=0.0, min_signal=min_signal, mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup, thresholds=model_th)
    oracle_gross = _simulate_pair(
        buy_o, sell_o, y_true_lr, horizon=horizon, start_cash=start_cash,
        fee_bps=0.0, min_signal=min_signal, mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup, thresholds=oracle_th)
    activity = threshold_activity_metrics(
        buy_m, sell_m, y_true_lr, min_signal=min_signal, threshold_mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        threshold_history=history, warmup=warmup,
        eligible=model_th['eligible'])
    activity = {key: value for key, value in activity.items()
                if not key.startswith('_')}
    activity['eligible_n'] = int(model_th['eligible'].sum())
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
            buy_m, sell_m, y_true_lr, horizon=horizon, start_cash=start_cash,
            fee_bps=fee_bps, min_signal=min_signal, mode=mode,
            percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
            history=history, warmup=warmup, thresholds=model_th)
        oracle_net = _simulate_pair(
            buy_o, sell_o, y_true_lr, horizon=horizon, start_cash=start_cash,
            fee_bps=fee_bps, min_signal=min_signal, mode=mode,
            percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
            history=history, warmup=warmup, thresholds=oracle_th)
        variant['config']['net_fee_bps'] = float(fee_bps)
        variant['net'] = {
            'model': _compact_sim(model_net),
            'oracle': _compact_sim(oracle_net),
            'capture_log': derive_sim_capture_log(
                model_net['final_value'], oracle_net['final_value'], start_cash),
        }
    return variant, model_gross, oracle_gross, capture_ratio


def evaluate_signal_simulation_v2(
        y_pred, target_oracle, y_true_lr, *, horizon: int,
        asset: str = 'ETH', start_cash: float = 100.0,
        primary_fee_bps: float = 2.0,
        auxiliary_fee_bps: float = 2.0,
        warmup: int = 2000,
        position_mode: str = 'sticky') -> dict:
    """Primary A20 gross+net2; auxiliary variants also report gross 0 / net 2."""
    if position_mode != 'sticky':
        raise ValueError("signal_simulation_v2 requires position_mode='sticky'")
    zscore_cutoff = zscore_cutoff_for_asset(asset)
    definitions = (
        ('absolute_20', 'absolute', 20.0, 5.0, zscore_cutoff, 'full', 0,
         primary_fee_bps),
        ('absolute_30', 'absolute', 30.0, 5.0, zscore_cutoff, 'full', 0,
         auxiliary_fee_bps),
        ('zscore_expanding', 'zscore', 20.0, 5.0, zscore_cutoff, 'expanding',
         warmup, auxiliary_fee_bps),
        ('percentile_expanding', 'percentile', 20.0, 5.0, zscore_cutoff,
         'expanding', warmup, auxiliary_fee_bps),
    )
    variants = {}
    private = {}
    for (name, mode, minimum, percentile, zscore, history, variant_warmup,
         fee) in definitions:
        variant, model_g, oracle_g, capture_ratio = build_signal_simulation_variant(
            y_pred, target_oracle, y_true_lr, horizon=horizon,
            start_cash=start_cash, mode=mode, min_signal=minimum,
            percentile_cutoff=percentile, zscore_cutoff=zscore,
            history=history, warmup=variant_warmup, fee_bps=fee,
            include_net=True)
        variants[name] = variant
        private[name] = (model_g, oracle_g, capture_ratio)

    primary = variants['absolute_20']
    model_gross, oracle_gross, capture_ratio = private['absolute_20']
    model_net = primary['net']['model']
    oracle_net = primary['net']['oracle']

    def _gross_final(name: str) -> float:
        return variants[name]['gross']['model']['final_value']

    def _net_final(name: str) -> float:
        return variants[name]['net']['model']['final_value']

    return {
        'simulation_schema_version': 2,
        'simulation_profile': 'signal_simulation_v2',
        'ranking_method': RANKING_METHOD,
        'simulation_variants': variants,
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
        'sim_min_signal': 20.0,
        'sim_horizon': int(horizon),
        'sim_fee_bps': float(primary_fee_bps),
        'sim_final_absolute_20': _gross_final('absolute_20'),
        'sim_final_absolute_20_net': _net_final('absolute_20'),
        'sim_final_absolute_30': _gross_final('absolute_30'),
        'sim_final_absolute_30_net': _net_final('absolute_30'),
        'sim_final_zscore_expanding': _gross_final('zscore_expanding'),
        'sim_final_zscore_expanding_net': _net_final('zscore_expanding'),
        'sim_final_percentile_expanding': _gross_final('percentile_expanding'),
        'sim_final_percentile_expanding_net': _net_final('percentile_expanding'),
        'pct_strong': primary['activity']['pct_strong'],
        'precision_buy': primary['activity']['precision_buy'],
        'precision_sell': primary['activity']['precision_sell'],
        'calib_buy': primary['activity']['calib_buy'],
        'calib_sell': primary['activity']['calib_sell'],
        'n_strong_buy': primary['activity']['n_strong_buy'],
        'n_strong_sell': primary['activity']['n_strong_sell'],
        'threshold_mode_eval': 'absolute',
        'threshold_history_eval': 'full',
        'threshold_warmup_eval': 0,
        'thresh_buy_eval': primary['activity']['thresh_buy_eval'],
        'thresh_sell_eval': primary['activity']['thresh_sell_eval'],
        'sim_model_final': primary['gross']['model']['final_value'],
        'sim_model_return_pct': primary['gross']['model']['return_pct'],
        'sim_oracle_final': primary['gross']['oracle']['final_value'],
        'sim_oracle_return_pct': primary['gross']['oracle']['return_pct'],
        '_model_sim': model_gross,
        '_oracle_sim': oracle_gross,
    }


evaluate_signal_simulation_profile = evaluate_signal_simulation_v2
