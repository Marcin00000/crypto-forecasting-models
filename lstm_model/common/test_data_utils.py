"""
Test data loading, result persistence, and formatting utilities for LSTM price model.

Enforces CSV priority over NPY, exact sequence length geometry (n = N - lookback - horizon),
Parquet persistence with sticky signal_pos, and robust JSON serialization.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_utils import create_sequences
from .id_utils import resolve_dataset_path

__all__ = [
    'load_test_data',
    'generate_sticky_signal_pos',
    'save_results',
    'load_results',
    'build_lstm_test_summary',
]


def _resolve_prepared_dir(path_or_id: str) -> str:
    """Resolve a dataset reference (path, repo-relative path or ID/name prefix) to its directory."""
    if os.path.isfile(path_or_id):
        parent = os.path.dirname(path_or_id) or "."
        if os.path.isfile(os.path.join(parent, "metadata.json")):
            return parent
    if os.path.isdir(path_or_id):
        return path_or_id
    for ref in (path_or_id, os.path.basename(path_or_id)):
        try:
            return str(resolve_dataset_path(ref))
        except FileNotFoundError:
            continue
    return path_or_id


def load_test_data(
    test_path: str,
    lookback: int,
    horizon: int,
    features: list[str],
    price_scaler=None,
    feature_scaler=None,
    test_file_type: str = 'test',
    use_log_prices: bool = True,
) -> dict[str, Any]:
    """
    Load test dataset enforcing strict priority of CSV files over legacy NPY files.

    Returns dictionary containing:
      - X_test: 3D sequence array of shape (n_samples, lookback, n_features)
      - y_test: 1D target array
      - current_prices: 1D entry prices at candle t = i + lookback - 1
      - future_prices: 1D target prices at candle t + horizon
      - close_full: Raw unscaled close price series from the full test split
      - n_samples: Number of evaluated samples (len(df) - lookback - horizon)
    """
    resolved_dir = _resolve_prepared_dir(test_path)
    metadata_path = os.path.join(resolved_dir, 'metadata.json')
    metadata: dict = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    # 1. Check for CSV split according to test_file_type
    target_csv_name = f"{test_file_type}.csv" if not test_file_type.endswith('.csv') else test_file_type
    csv_candidate = os.path.join(resolved_dir, target_csv_name)
    if not os.path.exists(csv_candidate) and target_csv_name == 'test.csv':
        # Fallback to test_raw.csv or direct path
        alt = os.path.join(resolved_dir, 'test_raw.csv')
        if os.path.exists(alt):
            csv_candidate = alt

    if os.path.exists(csv_candidate):
        df = pd.read_csv(csv_candidate)
        close_full = df['close'].values.astype(np.float64).copy()

        # When scale_csv is true, features in CSV are already scaled
        scale_csv = metadata.get('scalers', {}).get('scale_csv', True)
        if not scale_csv and feature_scaler is not None:
            df[features] = feature_scaler.transform(df[features])

        # Sequence creation with exact length n = len(df) - lookback - horizon
        X_test, y_test, current_prices = create_sequences(df, features, lookback, horizon)

        # Future prices correspond to prices[i + lookback + horizon - 1]
        n_samples = len(current_prices)
        future_prices = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            future_prices[i] = close_full[i + lookback + horizon - 1]

        return {
            'X_test': X_test,
            'y_test': y_test,
            'current_prices': current_prices.astype(np.float64),
            'future_prices': future_prices,
            'close_full': close_full,
            'n_samples': n_samples,
            'metadata': metadata,
            'source_file': csv_candidate,
        }

    # Direct CSV file argument fallback
    if test_path.endswith('.csv') and os.path.exists(test_path):
        df = pd.read_csv(test_path)
        close_full = df['close'].values.astype(np.float64).copy()
        if feature_scaler is not None:
            df[features] = feature_scaler.transform(df[features])
        X_test, y_test, current_prices = create_sequences(df, features, lookback, horizon)
        n_samples = len(current_prices)
        future_prices = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            future_prices[i] = close_full[i + lookback + horizon - 1]
        return {
            'X_test': X_test,
            'y_test': y_test,
            'current_prices': current_prices.astype(np.float64),
            'future_prices': future_prices,
            'close_full': close_full,
            'n_samples': n_samples,
            'metadata': metadata,
            'source_file': test_path,
        }

    raise FileNotFoundError(
        f"Unable to find required test CSV split in {resolved_dir} (expected {target_csv_name})."
    )


def generate_sticky_signal_pos(
    step_indices: np.ndarray,
    positions: np.ndarray,
    n_samples: int,
    horizon: int,
) -> np.ndarray:
    """
    Project simulation positions into continuous sample-level signal_pos array.
    Maintains sticky position values between decision steps.
    """
    signal_pos = np.zeros(n_samples, dtype=np.int8)
    if len(step_indices) == 0 or len(positions) == 0:
        return signal_pos

    for idx, (start_idx, pos) in enumerate(zip(step_indices, positions)):
        end_idx = step_indices[idx + 1] if (idx + 1 < len(step_indices)) else n_samples
        signal_pos[start_idx:end_idx] = int(pos)

    return signal_pos


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert NaN / Inf to None and cast numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items() if not str(k).startswith('_')}
    elif isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.ndarray,)):
        return _clean_for_json(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results(
    output_dir: str | Path,
    df_results: pd.DataFrame,
    metrics: dict[str, Any],
    test_id: str,
    model_id: str | None = None,
) -> None:
    """
    Persist evaluation results as Parquet (with CSV companion), metrics.json, and summary.txt.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parquet with fallback to CSV
    parquet_path = os.path.join(output_dir, 'results.parquet')
    csv_path = os.path.join(output_dir, 'results.csv')

    try:
        df_results.to_parquet(parquet_path, index=False)
    except Exception as err:
        print(f"[WARN] Parquet engine unavailable or error ({err}), companion CSV will serve as primary.")
    df_results.to_csv(csv_path, index=False)

    # 2. Sanitized metrics.json
    metrics_path = os.path.join(output_dir, 'metrics.json')
    cleaned_metrics = _clean_for_json(metrics)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_metrics, f, indent=2, allow_nan=False)

    # 3. Comprehensive summary.txt
    summary_path = os.path.join(output_dir, 'summary.txt')
    m_id = model_id or metrics.get('model_id', 'UNKNOWN')
    summary_txt = build_lstm_test_summary(cleaned_metrics, test_id, m_id)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_txt)


def load_results(results_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load results DataFrame and metrics dictionary from a completed evaluation directory."""
    parquet_path = os.path.join(results_dir, 'results.parquet')
    csv_path = os.path.join(results_dir, 'results.csv')
    metrics_path = os.path.join(results_dir, 'metrics.json')

    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            df = pd.read_csv(csv_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing results file in {results_dir}")

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    return df, metrics


def _fmt(val: Any, spec: str, missing: str = 'n/a') -> str:
    if val is None:
        return missing
    try:
        return format(float(val), spec)
    except Exception:
        return str(val)


def _money_pair(final_value: Any, return_pct: Any) -> str:
    return f"${_fmt(final_value, '.2f')} ({_fmt(return_pct, '+.2f')}%)"


_VARIANT_ORDER = (
    ('absolute_60', 'A60'),
    ('absolute_70', 'A70'),
    ('zscore_expanding', 'Z-expanding'),
    ('percentile_expanding', 'Pct-expanding'),
)


def build_lstm_test_summary(
    metrics: dict[str, Any],
    test_id: str,
    model_id: str,
) -> str:
    """Format comprehensive evaluation report for console output and summary.txt."""
    data_id = metrics.get('data_id', 'UNKNOWN')
    stamp = metrics.get('evaluated_at', '')[:19].replace('T', ' ')
    horizon = metrics.get('horizon', 12)
    lookback = metrics.get('lookback', 48)
    n_samples = metrics.get('n_samples') or metrics.get('sim_n_steps') or 'n/a'
    test_file_type = metrics.get('test_file_type', 'test')

    lines = [
        "=" * 70,
        "LSTM PRICE PREDICTION TEST SUMMARY",
        "=" * 70,
        "",
        f"Test ID:    {test_id}",
        f"Model ID:   {model_id}",
        f"Data ID:    {data_id}",
        f"Date:       {stamp}",
        f"Lookback:   {lookback}",
        f"Horizon:    t+{horizon}",
        f"Samples:    {n_samples}",
        f"Data used:  {test_file_type}.csv",
    ]

    if metrics.get('epochs_trained') is not None:
        lines.append(
            f"Training ended: epoch {metrics['epochs_trained']} "
            f"(best checkpoint: epoch {metrics.get('best_epoch', 'n/a')})"
        )

    lines.extend([
        "",
        "PRICE ACCURACY ($):",
        f"  MAE Price:        ${_fmt(metrics.get('mae_price'), ',.2f')}",
        f"  RMSE Price:       ${_fmt(metrics.get('rmse_price'), ',.2f')}",
        f"  MAPE Price:       {_fmt(metrics.get('mape_price'), '.2f')}%",
        f"  R² Price:         {_fmt(metrics.get('r2_price'), '.4f')}",
        f"  Price Dir Acc:    {_fmt(metrics.get('price_dir_acc'), '.2f')}%",
        "",
        "METRICS (implied log-return space μ vs r_true):",
        f"  MAE:              {_fmt(metrics.get('mae'), '.6f')}",
        f"  RMSE:             {_fmt(metrics.get('rmse'), '.6f')}",
        f"  R²:               {_fmt(metrics.get('r2'), '.4f')}",
        f"  Dir Acc:          {_fmt(metrics.get('dir_acc'), '.2f')}%",
        f"  CR:               {_fmt(metrics.get('compression_ratio'), '.4f')}",
        "",
        "EVALUATION (Simulation v2):",
        f"  Profile:          {metrics.get('simulation_profile', 'lstm_simulation_v2')}",
        f"  Ranking method:   {metrics.get('ranking_method', 'absolute_60_gross')}",
        "  Variants:         A60, A70, Z-expanding, Pct-expanding",
        "",
        "SIGNAL METRICS:",
        f"  pct_strong:       {_fmt(metrics.get('pct_strong'), '.2f')}%",
        f"  calib_buy:        {_fmt(metrics.get('calib_buy'), '.3f')}",
        f"  calib_sell:       {_fmt(metrics.get('calib_sell'), '.3f')}",
        f"  dir_acc_strong:   {_fmt(metrics.get('dir_acc_strong'), '.2f')}%",
        f"  corr:             {_fmt(metrics.get('corr'), '.4f')}",
    ])

    variants = metrics.get('simulation_variants') or {}
    if variants:
        lines.extend(["", "SIMULATION VARIANTS (model):"])
        for key, label in _VARIANT_ORDER:
            item = variants.get(key) or {}
            cfg = item.get('config') or {}
            gross = (item.get('gross') or {}).get('model') or {}
            net = (item.get('net') or {}).get('model') or {}
            activity = item.get('activity') or {}
            lines.extend([
                f"  {label} ({key}):",
                f"    config: mode={cfg.get('threshold_mode', 'n/a')} history={cfg.get('threshold_history', 'n/a')} "
                f"cutoff={cfg.get('min_signal', cfg.get('zscore_cutoff', cfg.get('percentile_cutoff', 'n/a')))} "
                f"warmup={cfg.get('warmup', 0)} fees net={_fmt(cfg.get('net_fee_bps', 2.0), '.1f')} bps",
                f"    model gross: {_money_pair(gross.get('final_value'), gross.get('return_pct'))}",
                f"    model net:   {_money_pair(net.get('final_value'), net.get('return_pct'))}",
                f"    trades: {gross.get('n_trades', 'n/a')}  fee legs: {net.get('n_fee_legs', 'n/a')}",
                f"    activity: pct_strong={_fmt(activity.get('pct_strong'), '.1f')}%",
            ])

    if 'rank_score' in metrics:
        lines.extend([
            "",
            "RANK SCORE:",
            f"  rank_score:       {_fmt(metrics.get('rank_score'), '.2f')} / 100 [{metrics.get('signal_class', 'UNKNOWN')}]",
            f"  components:       {metrics.get('rank_components')}",
            f"  ranking_method:   {metrics.get('ranking_method', 'absolute_60_gross')}",
        ])

    lines.extend(["", "-" * 70, "ORACLE DIAGNOSTICS (reference)", "-" * 70])
    if variants:
        for key, label in _VARIANT_ORDER:
            item = variants.get(key) or {}
            oracle = ((item.get('gross') or {}).get('oracle') or item.get('oracle') or {})
            lines.append(
                f"  {label}: {_money_pair(oracle.get('final_value'), oracle.get('return_pct'))} "
                f"trades={oracle.get('n_trades', 'n/a')}"
            )
        lines.append(f"  capture_log (primary): {_fmt(metrics.get('sim_capture_log'), '.4f')}")

    lines.append("=" * 70)
    return "\n".join(lines)
