"""
Simulation v2 profile for Random Forest log-return pipeline.

Standardises model evaluation across 4 variants:
  - absolute_60 (primary ranking benchmark, gross + net 2bps)
  - absolute_70 (conservative benchmark, gross + net 2bps)
  - zscore_expanding (adaptive Z-score, asset-calibrated, gross + net 2bps)
  - percentile_expanding (adaptive quantile top 5%, gross + net 2bps)
"""
from __future__ import annotations

import numpy as np

from .signals import (
    align_sigma,
    compute_sigma_series,
    compute_signal_metrics,
    compute_signal_thresholds,
    compute_signals,
)
from .simulation import (
    derive_sim_capture_log,
    simulate_long_short,
)

__all__ = [
    'RANKING_METHOD',
    'SIMULATION_PROFILE',
    'VARIANT_KEYS',
    'zscore_cutoff_for_asset',
    'evaluate_rf_simulation_v2',
    'evaluate_rf_simulation_profile',
]

RANKING_METHOD = 'absolute_60_gross'
SIMULATION_PROFILE = 'rf_simulation_v2'
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


def zscore_cutoff_for_asset(asset: str | None) -> float:
    """Return asset-specific Z-score cutoff (1.6 for ETH, 1.5 for BTC)."""
    if not asset:
        return 1.6
    name = str(asset).upper()
    if 'ETH' in name:
        return 1.6
    if 'BTC' in name or (name[:1] == 'B' and (len(name) == 1 or name[1].isdigit())):
        return 1.5
    return 1.6


def _compact_sim(sim: dict) -> dict:
    return {key: sim[key] for key in _COMPACT_KEYS if key in sim}


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
    buy_m, sell_m, buy_o, sell_o,
    *,
    mode: str,
    min_signal: float,
    percentile_cutoff: float,
    zscore_cutoff: float,
    history: str,
    warmup: int,
) -> tuple[dict, dict]:
    model_th = compute_signal_thresholds(
        buy_m, sell_m, mode=mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup
    )
    oracle_th = compute_signal_thresholds(
        buy_o, sell_o, mode=mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup
    )
    shared = model_th['eligible'] & oracle_th['eligible']
    return dict(model_th, eligible=shared), dict(oracle_th, eligible=shared)


def _simulate_pair(
    mu, y_true, sigma, *,
    threshold: float, horizon: int, start_cash: float, fee_bps: float,
    min_signal: float, mode: str, percentile_cutoff: float, zscore_cutoff: float,
    history: str, warmup: int, thresholds: dict
) -> dict:
    return simulate_long_short(
        mu, y_true, sigma, threshold,
        min_signal=min_signal, horizon=horizon, start_cash=start_cash,
        fee_bps=fee_bps, threshold_mode=mode, percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff, position_mode='sticky',
        threshold_history=history, warmup=warmup, thresholds=thresholds
    )


def build_rf_simulation_variant(
    y_pred, y_true, sigma, *,
    threshold: float, horizon: int, start_cash: float, mode: str,
    min_signal: float, percentile_cutoff: float, zscore_cutoff: float,
    history: str, warmup: int, fee_bps: float, include_net: bool = True
) -> dict:
    """Build a single simulation variant with gross and optional net trading legs."""
    buy_m, sell_m = compute_signals(y_pred, sigma, threshold)
    buy_o, sell_o = compute_signals(y_true, sigma, threshold)

    model_th, oracle_th = _shared_thresholds(
        buy_m, sell_m, buy_o, sell_o,
        mode=mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup
    )

    model_gross = _simulate_pair(
        y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, fee_bps=0.0, min_signal=min_signal, mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup, thresholds=model_th
    )
    oracle_gross = _simulate_pair(
        y_true, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, fee_bps=0.0, min_signal=min_signal, mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=history, warmup=warmup, thresholds=oracle_th
    )

    activity = compute_signal_metrics(
        y_pred, y_true, sigma, threshold,
        min_signal=min_signal, threshold_mode=mode,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        threshold_history=history, warmup=warmup,
        eligible=model_th['eligible']
    )
    activity = {k: v for k, v in activity.items() if not k.startswith('_')}

    capture_log = derive_sim_capture_log(
        model_gross['final_value'], oracle_gross['final_value'], start_cash
    )
    oracle_profit = oracle_gross['final_value'] - start_cash
    capture_ratio = (
        float((model_gross['final_value'] - start_cash) / oracle_profit)
        if oracle_profit > 1e-9 else None
    )

    variant = {
        'config': {
            'threshold_mode': mode,
            'threshold_history': history,
            'min_signal': float(min_signal) if mode == 'absolute' else None,
            'percentile_cutoff': float(percentile_cutoff) if mode == 'percentile' else None,
            'zscore_cutoff': float(zscore_cutoff) if mode == 'zscore' else None,
            'warmup': int(warmup),
            'position_mode': 'sticky',
            'gross_fee_bps': 0.0,
        },
        'gross': {
            'model': _compact_sim(model_gross),
            'oracle': _compact_sim(oracle_gross),
            'capture_log': capture_log,
            'capture_ratio': capture_ratio,
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
        'positions': model_gross.get('positions', []),
    }

    if include_net:
        model_net = _simulate_pair(
            y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
            start_cash=start_cash, fee_bps=fee_bps, min_signal=min_signal,
            mode=mode, percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff, history=history, warmup=warmup,
            thresholds=model_th
        )
        variant['config']['net_fee_bps'] = float(fee_bps)
        variant['net'] = {
            'model': _compact_sim(model_net),
            'capture_log': derive_sim_capture_log(
                model_net['final_value'], oracle_gross['final_value'], start_cash
            ),
        }

    return variant


def evaluate_rf_simulation_v2(
    y_true,
    y_pred,
    *,
    close: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    lookback: int = 48,
    horizon: int = 12,
    threshold: float = 0.001,
    start_cash: float = 100.0,
    primary_fee_bps: float = 2.0,
    auxiliary_fee_bps: float = 2.0,
    asset: str | None = None,
    sigma_window: int = 100,
    first_valid_idx: int = 0,
    warmup: int = 2000,
) -> dict:
    """
    Evaluate Random Forest model under the standardized Simulation v2 profile.

    Argument order (y_true, y_pred) matches the core LR/XGB/LSTM entry points.
    `threshold` must be the same signal threshold T as used for the compared models;
    `warmup` is the expanding-history warmup of the zscore/percentile variants.

    NOTE ON SIGMA HANDLING: In accordance with the LR simulation v2 contract,
    NaNs from rolling volatility warmup are PRESERVED and not filled with median.
    This guarantees that steps without historical volatility are inactive (eligible=False).
    Offset accounts for lookback and any leading feature warmup rows (first_valid_idx).
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    n_samples = len(y_pred)

    if sigma is None:
        if close is None:
            raise ValueError("Either 'sigma' or 'close' must be provided.")
        raw_sigma = compute_sigma_series(close, window=sigma_window)
        # RF entry candle is close[lookback + first_valid_idx + i]
        sigma_offset = int(lookback) + int(first_valid_idx)
        sigma = align_sigma(raw_sigma, offset=sigma_offset, n_samples=n_samples)
    else:
        sigma = np.asarray(sigma, dtype=np.float64)

    z_cutoff = zscore_cutoff_for_asset(asset)

    variants: dict[str, dict] = {}

    # Variant 1: absolute_60 (Primary Ranking Variant)
    variants['absolute_60'] = build_rf_simulation_variant(
        y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, mode='absolute', min_signal=60.0,
        percentile_cutoff=5.0, zscore_cutoff=z_cutoff, history='full',
        warmup=0, fee_bps=primary_fee_bps, include_net=True
    )

    # Variant 2: absolute_70 (Conservative)
    variants['absolute_70'] = build_rf_simulation_variant(
        y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, mode='absolute', min_signal=70.0,
        percentile_cutoff=5.0, zscore_cutoff=z_cutoff, history='full',
        warmup=0, fee_bps=auxiliary_fee_bps, include_net=True
    )

    # Variant 3: zscore_expanding (Asset Calibrated)
    variants['zscore_expanding'] = build_rf_simulation_variant(
        y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, mode='zscore', min_signal=60.0,
        percentile_cutoff=5.0, zscore_cutoff=z_cutoff, history='expanding',
        warmup=warmup, fee_bps=auxiliary_fee_bps, include_net=True
    )

    # Variant 4: percentile_expanding (Top 5%)
    variants['percentile_expanding'] = build_rf_simulation_variant(
        y_pred, y_true, sigma, threshold=threshold, horizon=horizon,
        start_cash=start_cash, mode='percentile', min_signal=60.0,
        percentile_cutoff=5.0, zscore_cutoff=z_cutoff, history='expanding',
        warmup=warmup, fee_bps=auxiliary_fee_bps, include_net=True
    )

    primary_v = variants['absolute_60']
    act = primary_v['activity']
    primary_gross = primary_v['gross']['model']

    primary_flat = {
        'sim_final_value': primary_gross['final_value'],
        'sim_return_pct': primary_gross['return_pct'],
        'sim_n_trades': primary_gross['n_trades'],
        'sim_n_fee_legs': primary_gross['n_fee_legs'],
        'sim_n_steps': primary_gross['n_steps'],
        'sim_n_long': primary_gross['n_long'],
        'sim_n_short': primary_gross['n_short'],
        'sim_n_flat': primary_gross['n_flat'],
        'oracle_final_value': primary_v['gross']['oracle']['final_value'],
        'oracle_return_pct': primary_v['gross']['oracle']['return_pct'],
        'oracle_n_trades': primary_v['gross']['oracle']['n_trades'],
        'sim_capture_ratio': primary_v['gross']['capture_ratio'],
        'sim_capture_log': primary_v['gross']['capture_log'],
        'pct_strong': act['pct_strong'],
        'calib_buy': act['calib_buy'],
        'calib_sell': act['calib_sell'],
        'dir_acc_strong': act['dir_acc_strong'],
        'bias_buy_strong': act['bias_buy_strong'],
        'mean_spread': act['mean_spread'],
        'sim_positions': primary_v.get('positions', []),
    }

    return {
        'profile': SIMULATION_PROFILE,
        'ranking_method': RANKING_METHOD,
        'zscore_cutoff': z_cutoff,
        'primary': primary_flat,
        'variants': variants,
    }


def evaluate_rf_simulation_profile(
    y_true,
    y_pred,
    *,
    close: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    lookback: int = 48,
    horizon: int = 12,
    threshold: float = 0.001,
    start_cash: float = 100.0,
    fee_bps: float = 2.0,
    auxiliary_fee_bps: float = 2.0,
    asset: str | None = None,
    profile: str = SIMULATION_PROFILE,
    first_valid_idx: int = 0,
    warmup: int = 2000,
) -> dict:
    """Wrapper supporting future profiles while defaulting to Simulation v2."""
    if profile != SIMULATION_PROFILE:
        raise ValueError(f"Unsupported simulation profile: {profile}")
    return evaluate_rf_simulation_v2(
        y_true,
        y_pred,
        close=close,
        sigma=sigma,
        lookback=lookback,
        horizon=horizon,
        threshold=threshold,
        start_cash=start_cash,
        primary_fee_bps=fee_bps,
        auxiliary_fee_bps=auxiliary_fee_bps,
        asset=asset,
        first_valid_idx=first_valid_idx,
        warmup=warmup,
    )
