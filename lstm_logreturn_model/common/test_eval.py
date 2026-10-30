"""Signal/simulation evaluation helpers used by lstm_test."""
from __future__ import annotations

import os

import joblib
import numpy as np

from lstm_config import LSTMConfig
from .id_utils import detect_symbol
from .metrics import compute_rank_score
from .simulation import (
    align_sigma,
    compute_signals,
    compute_sigma_series,
    evaluate_all_simulation_modes,
    evaluate_lr_simulation_v2,
    evaluate_signal_and_simulation,
    fill_sigma_nan,
)


def evaluate_signal_metrics(y_true, y_pred, close_full, config,
                            signal_threshold, fee_bps=0.0,
                            threshold_mode='absolute', percentile_cutoff=5.0,
                            zscore_cutoff=1.6, position_mode='sticky',
                            simulation_profile='v2', auxiliary_fee_bps=2.0):
    """Compute signal-quality + simulation metrics and the rank score."""
    threshold = signal_threshold or config.get('signal_threshold') or 0.001
    if close_full is None:
        print("  ⚠ close series unavailable — skipping signal/simulation metrics")
        return {'rank_score': None, 'signal_class': 'UNKNOWN_SIGNAL'}

    window_sigma = int(getattr(LSTMConfig, 'SIGNAL_WINDOW_SIGMA', 100))
    min_signal = float(getattr(LSTMConfig, 'SIGNAL_MIN_STRONG', 60.0))
    start_cash = float(getattr(LSTMConfig, 'SIM_START_CASH', 100.0))
    asset = (config.get('symbol') or config.get('data_id')
             or detect_symbol(str(config.get('train_path', ''))) or 'ETH')

    if simulation_profile == 'v2':
        sig = evaluate_lr_simulation_v2(
            y_true, y_pred, close_full,
            lookback=config['lookback'], horizon=config['horizon'],
            threshold=float(threshold), window_sigma=window_sigma,
            asset=asset, start_cash=start_cash,
            primary_fee_bps=float(fee_bps),
            auxiliary_fee_bps=float(auxiliary_fee_bps),
            position_mode='sticky',
        )
        sig.update(compute_rank_score(sig))
        return sig

    sig = evaluate_signal_and_simulation(
        y_true, y_pred, close_full,
        lookback=config['lookback'], horizon=config['horizon'],
        threshold=float(threshold), window_sigma=window_sigma,
        min_signal=min_signal, start_cash=start_cash, fee_bps=fee_bps,
        threshold_mode=threshold_mode,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        position_mode=position_mode,
    )
    sig.update(compute_rank_score(sig))
    multi_modes = evaluate_all_simulation_modes(
        y_true, y_pred, close_full,
        lookback=config['lookback'], horizon=config['horizon'],
        threshold=float(threshold), window_sigma=window_sigma,
        min_signal=min_signal,
        percentile_cutoff=percentile_cutoff,
        zscore_cutoff=zscore_cutoff,
        position_mode=position_mode,
        start_cash=start_cash, fee_bps=fee_bps,
    )
    sig['multi_mode_simulation'] = multi_modes
    return sig


def multi_window_eval(y_true, y_pred, close_full, config, signal_threshold,
                      n_windows, fee_bps=0.0):
    """Chronological sub-window signal metrics (opt-in)."""
    n = len(y_true)
    if n_windows < 2 or n < n_windows:
        return None
    chunk = n // n_windows
    per_window = []
    lookback = config['lookback']
    horizon = config['horizon']
    threshold = signal_threshold or config.get('signal_threshold') or 0.001
    window_sigma = int(getattr(LSTMConfig, 'SIGNAL_WINDOW_SIGMA', 100))
    min_signal = float(getattr(LSTMConfig, 'SIGNAL_MIN_STRONG', 60.0))
    start_cash = float(getattr(LSTMConfig, 'SIM_START_CASH', 100.0))

    for w in range(n_windows):
        start = w * chunk
        end = n if w == n_windows - 1 else (w + 1) * chunk
        yt = y_true[start:end]
        yp = y_pred[start:end]
        if len(yt) < 10:
            continue
        close_off = lookback - 1 + start
        close_slice = close_full[close_off:close_off + len(yt)]
        if len(close_slice) < len(yt):
            continue
        block = evaluate_signal_and_simulation(
            yt, yp, close_slice,
            lookback=lookback, horizon=horizon,
            threshold=float(threshold), window_sigma=window_sigma,
            min_signal=min_signal, start_cash=start_cash, fee_bps=fee_bps,
        )
        rank = compute_rank_score(block)
        per_window.append({
            'sim_final_model': block.get('sim_final_model'),
            'rank_score': rank.get('rank_score'),
            'pct_strong': block.get('pct_strong'),
            'dir_acc_strong': block.get('dir_acc_strong'),
        })

    if not per_window:
        return None
    sims = [p['sim_final_model'] for p in per_window if p['sim_final_model']]
    ranks = [p['rank_score'] for p in per_window if p['rank_score'] is not None]
    return {
        'n_windows': n_windows,
        'per_window': per_window,
        'sim_window_min': min(sims) if sims else None,
        'sim_window_median': float(np.median(sims)) if sims else None,
        'rank_window_min': min(ranks) if ranks else None,
        'rank_window_std': float(np.std(ranks)) if len(ranks) > 1 else 0.0,
    }


def posthoc_calibrator_metrics(y_true, y_pred, close_full, config, model_dir,
                               signal_threshold):
    """Isotonic post-hoc calibration metrics when calibrator.pkl exists."""
    cal_path = os.path.join(model_dir, 'calibrator.pkl')
    if not os.path.exists(cal_path) or close_full is None:
        return {}
    try:
        from sklearn.isotonic import IsotonicRegression  # noqa: F401
    except ImportError:
        return {}
    cal = joblib.load(cal_path)
    threshold = float(cal.get('threshold', signal_threshold or 0.001))
    window_sigma = int(cal.get('window_sigma',
                               getattr(LSTMConfig, 'SIGNAL_WINDOW_SIGMA', 100)))
    lookback = config['lookback']
    sigma = fill_sigma_nan(align_sigma(
        compute_sigma_series(close_full, window_sigma), lookback, len(y_pred)))
    buy_raw, sell_raw = compute_signals(y_pred, sigma, threshold)
    buy_model = cal.get('buy')
    sell_model = cal.get('sell')
    if buy_model is None or sell_model is None:
        return {}
    buy_cal = np.clip(buy_model.predict(buy_raw), 0.0, 100.0)
    sell_cal = np.clip(sell_model.predict(sell_raw), 0.0, 100.0)
    min_signal = float(getattr(LSTMConfig, 'SIGNAL_MIN_STRONG', 60.0))
    strong_buy = buy_cal >= min_signal
    strong_sell = sell_cal >= min_signal
    calib_buy = (float((y_true[strong_buy] > threshold).mean())
                 if strong_buy.any() else None)
    calib_sell = (float((y_true[strong_sell] < -threshold).mean())
                  if strong_sell.any() else None)
    base = evaluate_signal_and_simulation(
        y_true, y_pred, close_full,
        lookback=config['lookback'], horizon=config['horizon'],
        threshold=threshold, window_sigma=window_sigma,
        min_signal=min_signal,
    )
    patched = dict(base)
    patched['calib_buy'] = calib_buy
    patched['calib_sell'] = calib_sell
    rank = compute_rank_score(patched)
    return {
        'calib_buy_posthoc': calib_buy,
        'calib_sell_posthoc': calib_sell,
        'rank_score_posthoc': rank.get('rank_score'),
    }
