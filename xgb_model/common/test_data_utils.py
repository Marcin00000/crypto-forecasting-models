"""
Test data loading utilities for XGBoost model evaluation.

Handles loading held-out test splits (test.csv, agent.csv, sym.csv)
and raw test CSVs with raw close price preservation for market simulation.
"""
from __future__ import annotations

import os
import json
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from .flattening import flatten_features, align_features_and_target
from .metrics import _jsonable

__all__ = [
    'load_test_data_only', 'load_test_csv', '_extract_close',
    'find_results_path', 'load_results', 'save_results', 'build_xgb_test_summary',
]


def _extract_close(df: pd.DataFrame) -> np.ndarray | None:
    """Return raw close-price series as float64 array, or None if unavailable."""
    if 'close' not in df.columns:
        return None
    return df['close'].to_numpy(dtype=np.float64)


def _resolve_features_from_metadata(base_features, feature_set, custom_features, has_buy_pressure,
                                    features=None, columns=None):
    """
    Feature list for a test run.

    `features` (model_config['features']) is authoritative: it is exactly what the
    model was trained on, so re-deriving it from a feature_set name could silently
    yield another list (set definitions change, datasets differ). Every requested
    column must exist in the CSV (`columns`), otherwise ValueError. Without `features`
    the lookup falls back to feature_set / custom_features under the strict policy.
    """
    if features is not None:
        available = set(columns) if columns is not None else set(base_features)
        missing = [f for f in features if f not in available]
        if missing:
            raise ValueError(
                f"Test data lacks columns required by the model (config['features']): {missing}"
            )
        return list(features)

    from .data_utils import _resolve_features
    return _resolve_features(base_features, feature_set, custom_features, has_buy_pressure)


def load_test_data_only(
    data_path: str,
    flattening_method: str | None = None,
    feature_set: str | None = None,
    custom_features: list[str] | None = None,
    lookback: int | None = None,
    horizon: int | None = None,
    test_file_type: str = 'test',
    return_close: bool = False,
    features: list[str] | None = None,
) -> tuple[dict, dict] | tuple[dict, dict, np.ndarray | None]:
    """
    Load test data from a prepared dataset directory (INVARIANT: offset=lookback).

    `features` is the model's own feature list (config['features']); when given it
    takes precedence over feature_set / custom_features.
    """
    from .data_utils import create_log_return_target

    valid_splits = ('test', 'agent', 'sym', 'val', 'train')
    if test_file_type not in valid_splits:
        raise ValueError(
            f"Invalid test_file_type: {test_file_type!r}. Must be one of {valid_splits}"
        )

    metadata_path = os.path.join(data_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata.json in {data_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    csv_file = f'{test_file_type}.csv'
    test_csv = os.path.join(data_path, csv_file)
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Missing {csv_file} in {data_path}")

    test_df = pd.read_csv(test_csv)

    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Feature scaler not found in {data_path}")
    feature_scaler = joblib.load(scaler_path)

    base_features = metadata['features']
    lookback = lookback if lookback is not None else metadata['lookback']
    horizon = horizon if horizon is not None else metadata['horizon']
    # From the column list: E114/E115 flag buy_pressure as true without having the column
    has_buy_pressure = 'buy_pressure' in base_features

    features = _resolve_features_from_metadata(
        base_features, feature_set, custom_features, has_buy_pressure,
        features=features, columns=test_df.columns,
    )

    if flattening_method is None:
        flattening_method = metadata.get('flattening_method', 'hybrid')

    X_test_flat, flat_features = flatten_features(
        test_df, features, lookback, method=flattening_method
    )
    y_test = create_log_return_target(test_df, lookback, horizon)

    # Same alignment as training: drop lookback rows, trim NaN prefix from X and y together
    X_test_flat, y_test, _ = align_features_and_target(X_test_flat, y_test, lookback, horizon)
    X_test = X_test_flat.values

    close_full = _extract_close(test_df)

    data = {
        'X_test': X_test,
        'y_test': y_test,
        'close_full': close_full,
        'feature_scaler': feature_scaler,
    }

    metadata.update({
        'flat_features': flat_features,
        'n_flat_features': len(flat_features),
        'n_features': len(features),
        'features': features,
        'lookback': lookback,
        'horizon': horizon,
        'flattening_method': flattening_method,
        'has_buy_pressure': has_buy_pressure,
        'test_file_type': test_file_type,
    })

    if return_close:
        return data, metadata, close_full
    return data, metadata


def load_test_csv(
    csv_path: str,
    config: dict,
    feature_scaler,
    return_close: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """
    Load raw CSV file as held-out test data (runs universal FE).
    Preserves raw close prices before scaling.
    """
    from feature_engineering import prepare_features
    from .data_utils import create_log_return_target

    df = pd.read_csv(csv_path)
    df = prepare_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    features = config['features']
    lookback = config['lookback']
    horizon = config['horizon']
    flat_method = config['flattening_method']

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"CSV missing features required by model: {missing}")

    close_full = _extract_close(df)

    # Scale only feature columns
    df[features] = feature_scaler.transform(df[features])

    X_flat, flat_features = flatten_features(df, features, lookback, method=flat_method)
    y_all = create_log_return_target(df, lookback, horizon)
    X_flat, y_test, _ = align_features_and_target(X_flat, y_all, lookback, horizon)
    X_test = X_flat.values

    metadata = {
        'data_id': 'CSV',
        'features': features,
        'flat_features': flat_features,
        'n_features': len(features),
        'n_flat_features': len(flat_features),
        'lookback': lookback,
        'horizon': horizon,
        'flattening_method': flat_method,
        'source_file': os.path.basename(csv_path),
    }

    if return_close:
        return X_test, y_test, metadata, close_full
    return X_test, y_test, metadata


# =========================================================================
# FORMATTING HELPERS FOR TEST SUMMARY
# =========================================================================

def _fmt(value, spec, missing='n/a'):
    if value is None:
        return missing
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return missing


_ORACLE_DIAG_HEADER = (
    'ORACLE DIAGNOSTICS (future-known, non-deployable reference)'
)


def _is_v2_metrics(metrics: dict) -> bool:
    if int(metrics.get('simulation_schema_version') or 0) == 2:
        return True
    return bool(metrics.get('simulation_variants'))


def build_xgb_test_summary(
    *args,
    test_id: str | None = None,
    model_id: str | None = None,
    data_id: str | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
    sim_results: dict | None = None,
    rank_metrics: dict | None = None,
    n_samples: int | None = None,
    horizon: int | None = None,
    effective_fee_bps: float = 0.0,
    test_file_type: str = 'test',
    timestamp: str | None = None,
    **kwargs,
) -> str:
    """Build standardized, comprehensive test summary report for XGBoost."""
    if len(args) >= 1 and isinstance(args[0], dict) and metrics is None:
        metrics = args[0]
    if len(args) >= 2 and isinstance(args[1], dict) and config is None:
        config = args[1]

    metrics = metrics or kwargs.get('metrics', {})
    config = config or kwargs.get('config', {})
    sim_results = sim_results or metrics.get('simulation') or {}
    rank_metrics = rank_metrics or metrics.get('rank_metrics') or {}

    t_id = test_id or kwargs.get('test_id', 'UNKNOWN')
    m_id = model_id or kwargs.get('model_id', 'UNKNOWN')
    d_id = data_id or kwargs.get('data_id', 'UNKNOWN')
    h = horizon if horizon is not None else config.get('horizon', 12)
    n = n_samples if n_samples is not None else kwargs.get('n_samples', 0)
    stamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    is_v2 = _is_v2_metrics(metrics) or bool(sim_results.get('simulation_variants'))

    lines = [
        '=' * 70,
        'XGBOOST MODEL TEST SUMMARY',
        '=' * 70,
        '',
        f'Test ID:    {t_id}',
        f'Model ID:   {m_id}',
        f'Data ID:    {d_id}',
        f'Date:       {stamp}',
        f'Horizon:    t+{h}',
        f'Samples:    {n:,}',
        f'Data used:  {test_file_type}.csv',
    ]

    total_r = config.get('total_rounds', metrics.get('total_rounds'))
    if total_r is not None or 'best_iteration' in config:
        save_f = config.get('save_from', 'last')
        best_i = config.get('best_iteration', 'n/a')
        lines.append(f"Training ended:        round {total_r} (saved from: {save_f}, best: {best_i})")

    lines.extend([
        '',
        'METRICS (log-return space):',
        f"  MAE:                   {_fmt(metrics.get('mae'), '.6f')}",
        f"  RMSE:                  {_fmt(metrics.get('rmse'), '.6f')}",
        f"  R²:                    {_fmt(metrics.get('r2'), '.4f')}",
        f"  Pseudo-Huber (δ=1):    {_fmt(metrics.get('pseudo_huber'), '.6f')}",
        f"  Pseudo-Huber (δ=10):   {_fmt(metrics.get('pseudo_huber_large'), '.6f')}",
        f"  Dir Acc:               {_fmt(metrics.get('dir_acc'), '.2f')}%",
        f"  Compression Ratio:     {_fmt(metrics.get('compression_ratio') or rank_metrics.get('compression_ratio'), '.4f')}",
    ])

    if 't_mae' in metrics:
        lines.extend([
            '',
            f"THRESHOLDED METRICS (|y| ≥ {_fmt(metrics.get('t_threshold'), '.6f')}, n={metrics.get('t_n_samples')}):",
            f"  t_MAE:                 {_fmt(metrics.get('t_mae'), '.6f')}",
            f"  t_RMSE:                {_fmt(metrics.get('t_rmse'), '.6f')}",
            f"  t_R²:                  {_fmt(metrics.get('t_r2'), '.4f')}",
            f"  t_Pseudo-Huber (δ=1):  {_fmt(metrics.get('t_pseudo_huber'), '.6f')}",
            f"  t_Pseudo-Huber (δ=10): {_fmt(metrics.get('t_pseudo_huber_large'), '.6f')}",
            f"  t_Dir Acc:             {_fmt(metrics.get('t_dir_acc'), '.2f')}%",
        ])

    # Portfolio simulation section with exact dollar amounts
    start_cash = float(sim_results.get('sim_start_cash') or metrics.get('sim_start_cash') or 100.0)
    gross_pct = metrics.get('sim_total_return_pct')
    if gross_pct is None:
        gross_pct = sim_results.get('sim_return_model_pct', 0.0)
    net_pct = metrics.get('sim_total_return_net_pct')
    if net_pct is None:
        net_pct = sim_results.get('sim_return_model_net_pct', 0.0)

    gross_val = metrics.get('sim_final_model') or sim_results.get('sim_final_model')
    if gross_val is None:
        gross_val = start_cash * (1.0 + float(gross_pct) / 100.0)
    net_val = metrics.get('sim_final_model_net') or sim_results.get('sim_final_model_net')
    if net_val is None:
        net_val = start_cash * (1.0 + float(net_pct) / 100.0)

    trades = metrics.get('sim_trades')
    if trades is None:
        trades = sim_results.get('sim_n_trades_model', 0)

    lines.extend([
        '',
        '=' * 70,
        'MARKET SIMULATION & RANK SCORE (Simulation v2 Profile)',
        '=' * 70,
        f"  Rank Score:            {_fmt(metrics.get('rank_score') or rank_metrics.get('rank_score'), '.2f')} / 100  ({metrics.get('signal_class') or rank_metrics.get('signal_class', 'UNKNOWN')})",
        f"  Compression Ratio:     {_fmt(metrics.get('compression_ratio') or rank_metrics.get('compression_ratio'), '.4f')}",
        f"  Pred Std:              {_fmt(metrics.get('pred_std') or rank_metrics.get('pred_std'), '.6f')}",
        f"  True Std:              {_fmt(metrics.get('true_std') or rank_metrics.get('true_std'), '.6f')}",
        f"  Signal Corr:           {_fmt(metrics.get('corr') or rank_metrics.get('corr'), '.4f')}",
        '----------------------------------------------------------------------',
        f'PRIMARY SIMULATION ({effective_fee_bps:.1f} bps fee, sticky mode):',
        f'  Initial Capital:       ${start_cash:.2f}',
        f'  Final Value (Gross):   ${gross_val:,.2f} ({gross_pct:+.2f}%)',
        f'  Final Value (Net):     ${net_val:,.2f} ({net_pct:+.2f}%)',
        f'  Trades:                {trades}',
    ])

    variants = metrics.get('simulation_variants') or sim_results.get('simulation_variants') or {}
    if variants:
        lines.extend(['', 'SIMULATION VARIANTS (model):'])
        var_labels = [
            ('absolute_60', 'VARIANT 1: ABSOLUTE 60.0 (A60)'),
            ('absolute_70', 'VARIANT 2: ABSOLUTE 70.0 (A70)'),
            ('zscore_expanding', 'VARIANT 3: Z-SCORE EXPANDING (Z-expanding)'),
            ('percentile_expanding', 'VARIANT 4: PERCENTILE EXPANDING (Pct-expanding)'),
        ]
        for key, title in var_labels:
            item = variants.get(key) or {}
            gross_item = item.get('gross') if isinstance(item.get('gross'), dict) else {}
            net_item = item.get('net') if isinstance(item.get('net'), dict) else {}
            activity = item.get('activity') or {}

            g_val = gross_item.get('final_value') or item.get('final_value')
            g_ret = gross_item.get('return_pct') if 'return_pct' in gross_item else item.get('return_pct', 0.0)
            n_val = net_item.get('final_value') or item.get('final_value_net')
            n_ret = net_item.get('return_pct') if 'return_pct' in net_item else item.get('return_net_pct', 0.0)

            if g_val is None and g_ret is not None:
                g_val = start_cash * (1.0 + float(g_ret) / 100.0)
            if n_val is None and n_ret is not None:
                n_val = start_cash * (1.0 + float(n_ret) / 100.0)

            n_tr = gross_item.get('n_trades') or item.get('n_trades', 'n/a')
            n_legs = gross_item.get('n_fee_legs') or net_item.get('n_fee_legs') or net_item.get('fee_legs') or 'n/a'
            p_str = activity.get('pct_strong', activity.get('pct_active', metrics.get('pct_strong')))

            lines.extend([
                f'  {title}:',
                f"    model gross: ${_fmt(g_val, '.2f')} ({_fmt(g_ret, '+.2f')}%)",
                f"    model net:   ${_fmt(n_val, '.2f')} ({_fmt(n_ret, '+.2f')}%)",
                f"    trades: {n_tr}  fee legs: {n_legs}",
                f"    activity: pct_strong={_fmt(p_str, '.1f')}%",
            ])

    comps = metrics.get('rank_components') or (rank_metrics.get('rank_components') if rank_metrics else None)
    if comps:
        lines.extend(['', 'RANK COMPONENTS BREAKDOWN:'])
        for k in ['activity', 'calibration', 'direction', 'capture', 'cr_health']:
            if k in comps:
                lines.append(f"  {k}: {_fmt(comps[k], '.2f')}")
        for k, v in comps.items():
            if k not in ['activity', 'calibration', 'direction', 'capture', 'cr_health']:
                lines.append(f"  {k}: {_fmt(v, '.2f')}")

    lines.extend(['', '-' * 70, _ORACLE_DIAG_HEADER, '-' * 70])
    oracle_val = metrics.get('sim_final_oracle') or sim_results.get('sim_final_oracle')
    oracle_ret = metrics.get('sim_return_oracle_pct') or sim_results.get('sim_return_oracle_pct')
    oracle_tr = metrics.get('sim_n_trades_oracle') or sim_results.get('sim_n_trades_oracle')
    if oracle_val is not None:
        lines.append(f"  Oracle Value:          ${oracle_val:,.2f} ({oracle_ret:+.2f}%), Trades: {oracle_tr}")
    if variants:
        for key, title in [('absolute_60', 'A60'), ('absolute_70', 'A70'), ('zscore_expanding', 'Z-expanding'), ('percentile_expanding', 'Pct-expanding')]:
            item = variants.get(key) or {}
            orc = (item.get('gross') or {}).get('oracle') or item.get('oracle') or {}
            f_val = orc.get('final_value')
            r_pct = orc.get('return_pct')
            if f_val is not None:
                lines.append(f"  {title}: ${_fmt(f_val, '.2f')} ({_fmt(r_pct, '+.2f')}%) trades={orc.get('n_trades', 'n/a')}")
    cap_log = metrics.get('sim_capture_log') or sim_results.get('sim_capture_log')
    if cap_log is not None:
        lines.append(f"  capture_log (primary): {_fmt(cap_log, '.3f')}")
    lines.append('')
    return '\n'.join(lines)


# =========================================================================
# STORAGE & PARQUET UTILITIES (CP-161)
# =========================================================================

def find_results_path(run_dir: str | Path) -> Path | None:
    """Locate results.parquet or results.csv in run_dir (parquet has priority)."""
    run_path = Path(run_dir)
    pq = run_path / "results.parquet"
    if pq.is_file():
        return pq
    csv = run_path / "results.csv"
    if csv.is_file():
        return csv
    return None


def load_results(run_dir_or_file: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load results DataFrame with automatic format verification (parquet or csv)."""
    target = Path(run_dir_or_file)
    if target.is_dir():
        file_path = find_results_path(target)
    elif target.is_file():
        file_path = target
    else:
        file_path = find_results_path(target.parent) if target.parent.is_dir() else None

    if file_path is None:
        raise FileNotFoundError(f"Neither results.parquet nor results.csv found in: {target}")

    is_parquet = file_path.suffix.lower() == '.parquet'
    if not is_parquet:
        try:
            with open(file_path, 'rb') as f:
                is_parquet = (f.read(4) == b'PAR1')
        except Exception:
            pass

    loaders = (pd.read_parquet, pd.read_csv) if is_parquet else (pd.read_csv, pd.read_parquet)
    kwargs = {'columns': columns} if is_parquet else {'usecols': columns}
    fallback_kwargs = {'usecols': columns} if is_parquet else {'columns': columns}

    try:
        return loaders[0](file_path, **kwargs)
    except Exception:
        try:
            return loaders[1](file_path, **fallback_kwargs)
        except Exception as e:
            raise IOError(f"Failed to read results file {file_path}: {e}")


def save_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    sim_results: dict | None = None,
    format_choice: str = 'parquet',
    compression: str = 'snappy',
    results_format: str | None = None,
) -> list[str]:
    """Save model predictions table as Parquet (preferred) or CSV (CP-161)."""
    fmt = (results_format or format_choice or 'parquet').lower()
    comp = None if compression in ('none', None) else compression
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        'actual_log': y_true, 'predicted_log': y_pred,
        'actual': y_true, 'predicted': y_pred,
        'error': y_pred - y_true, 'abs_error': np.abs(y_pred - y_true),
        'actual_dir': np.sign(y_true), 'pred_dir': np.sign(y_pred),
        'correct_dir': (np.sign(y_true) == np.sign(y_pred)).astype(int),
    })

    if sim_results:
        for k, col in [('signal_buy', 'signal_buy'), ('signal_sell', 'signal_sell'), ('primary_positions', 'signal_pos')]:
            if k in sim_results:
                df[col] = sim_results[k][:len(df)]

    saved = []
    if fmt in ('parquet', 'both'):
        pq = os.path.join(output_dir, 'results.parquet')
        try:
            df.to_parquet(pq, compression=comp, index=False)
            saved.append(pq)
        except (ImportError, Exception) as e:
            print(f"  [WARN] Failed to write parquet ({e}). Falling back to CSV.")
            csv = os.path.join(output_dir, 'results.csv')
            df.to_csv(csv, index=False)
            saved.append(csv)

    if fmt in ('csv', 'both'):
        csv = os.path.join(output_dir, 'results.csv')
        if csv not in saved:
            df.to_csv(csv, index=False)
            saved.append(csv)

    return saved
