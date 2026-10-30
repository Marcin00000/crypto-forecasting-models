"""
Test data loading and results saving for LSTM Log-Return pipeline.

Split from lstm_test.py to keep main files under 500 lines.
"""
import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data_utils import create_sequences
from .target_utils import create_log_return_target


# =========================================================================
# TEST DATA LOADING
# =========================================================================

def load_test_data(test_path, config, feature_scaler, test_file_type='test'):
    """
    Load test data and create sequences + log-return targets.

    Args:
        test_path:      Path to prepared directory or raw CSV
        config:         Model config dict (features, lookback, horizon)
        feature_scaler: Fitted feature scaler
        test_file_type: 'test' | 'sym' | 'agent' — which CSV to load from directory

    Returns:
        X_test, y_test (raw log-returns, not scaled), close_full

        ``close_full`` is the full (un-truncated) raw close-price series from
        the same DataFrame the sequences were built from. It is needed to
        compute market volatility (sigma) and run the trading simulation in
        ``lstm_test``; rolling sigma requires the lookback warm-up history, so
        the whole series is returned rather than a per-sample slice.
    """
    features = config['features']
    lookback = config['lookback']
    horizon = config['horizon']

    # Prepared directory
    if os.path.isdir(test_path):
        csv_file = f'{test_file_type}.csv'
        csv_path = os.path.join(test_path, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"{csv_file} not found in {test_path}\n"
                f"Available: {[f for f in os.listdir(test_path) if f.endswith('.csv')]}"
            )

        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_file}: {len(df):,} rows")

        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in {csv_file}: {missing[:5]}...")

        X_test = create_sequences(df, features, lookback)
        y_test = create_log_return_target(df, lookback, horizon)
        close_full = _extract_close(df)

        n = min(len(X_test), len(y_test))
        if abs(len(X_test) - len(y_test)) != horizon:
            warnings.warn(
                f"X/y length mismatch: len(X)={len(X_test)}, len(y)={len(y_test)}, "
                f"diff={abs(len(X_test) - len(y_test))} (expected {horizon}).",
                stacklevel=2,
            )
        X_test, y_test = X_test[:n], y_test[:n]

        print(f"Test sequences: {len(X_test):,}")
        return X_test, y_test, close_full

    # Raw CSV
    elif test_path.endswith('.csv'):
        from feature_engineering import prepare_features

        df = pd.read_csv(test_path)
        print(f"Loaded raw CSV: {len(df):,} rows")

        df = prepare_features(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        print(f"After feature engineering: {len(df):,} rows")

        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in CSV: {missing[:5]}...")

        available = [f for f in features if f in df.columns]
        # Capture close BEFORE scaling features (close is not a model feature,
        # so feature_scaler does not touch it, but we read it explicitly to be
        # robust to future changes).
        close_full = _extract_close(df)
        df[available] = feature_scaler.transform(df[available])

        X_test = create_sequences(df, available, lookback)
        y_test = create_log_return_target(df, lookback, horizon)

        n = min(len(X_test), len(y_test))
        if abs(len(X_test) - len(y_test)) != horizon:
            warnings.warn(
                f"X/y length mismatch: len(X)={len(X_test)}, len(y)={len(y_test)}, "
                f"diff={abs(len(X_test) - len(y_test))} (expected {horizon}).",
                stacklevel=2,
            )
        X_test, y_test = X_test[:n], y_test[:n]

        print(f"Test sequences: {len(X_test):,}")
        return X_test, y_test, close_full

    else:
        raise ValueError(
            f"Unrecognised test_path: {test_path}\n"
            "Expected: prepared-data directory or .csv file"
        )


def _extract_close(df):
    """Return the raw close-price series as float64, or None if unavailable."""
    if 'close' not in df.columns:
        return None
    return df['close'].to_numpy(dtype=np.float64)


# =========================================================================
# SAVE RESULTS
# =========================================================================

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


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
    """Load results DataFrame with automatic format verification (parquet or csv).

    Args:
        run_dir_or_file: Path to run directory containing results or direct path to file.
        columns: Optional list of columns to load.

    Returns:
        pd.DataFrame containing test results.
    """
    target = Path(run_dir_or_file)
    if target.is_dir():
        file_path = find_results_path(target)
        if file_path is None:
            raise FileNotFoundError(f"Neither results.parquet nor results.csv found in: {target}")
    elif target.is_file():
        file_path = target
    else:
        file_path = find_results_path(target.parent) if target.parent.is_dir() else None
        if file_path is None:
            raise FileNotFoundError(f"Results file or directory not found: {target}")

    # Automatic format verification: check extension first, verify magic bytes if needed
    is_parquet = file_path.suffix.lower() == '.parquet'
    if not is_parquet:
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header == b'PAR1':
                    is_parquet = True
        except Exception:
            pass

    if is_parquet:
        try:
            return pd.read_parquet(file_path, columns=columns)
        except Exception as e:
            try:
                return pd.read_csv(file_path, usecols=columns)
            except Exception:
                raise IOError(f"Failed to read parquet file {file_path}: {e}")
    else:
        try:
            return pd.read_csv(file_path, usecols=columns)
        except Exception as e:
            try:
                return pd.read_parquet(file_path, columns=columns)
            except Exception:
                raise IOError(f"Failed to read csv file {file_path}: {e}")


def save_results(y_true, y_pred, metrics, config, test_id, model_id,
                 data_id, output_dir, horizon, test_file_type,
                 results_format='parquet', compression='snappy'):
    """Save test results: Parquet/CSV, metrics JSON, summary."""
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true),
        'actual_dir': np.sign(y_true),
        'pred_dir': np.sign(y_pred),
        'correct_dir': (np.sign(y_true) == np.sign(y_pred)).astype(int),
    })

    comp = None if compression in ('none', None) else compression
    if results_format == 'parquet':
        pq_path = os.path.join(output_dir, 'results.parquet')
        df.to_parquet(pq_path, compression=comp, index=False)
        print("  ✓ results.parquet")
    elif results_format == 'csv':
        csv_path = os.path.join(output_dir, 'results.csv')
        df.to_csv(csv_path, index=False)
        print("  ✓ results.csv")
    else:
        raise ValueError(f"Unsupported results_format: {results_format}")

    payload = _jsonable(metrics)
    payload.setdefault('test_file_type', test_file_type)
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print("  ✓ metrics.json")

    summary = build_lr_test_summary(
        metrics, config, test_id=test_id, model_id=model_id, data_id=data_id,
        n_samples=len(y_true), horizon=horizon, test_file_type=test_file_type)
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary)
    print("  ✓ summary.txt")


def _fmt(value, spec, missing='n/a'):
    if value is None:
        return missing
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return missing


_VARIANT_ORDER = (
    ('absolute_60', 'A60'),
    ('absolute_70', 'A70'),
    ('zscore_expanding', 'Z-expanding'),
    ('percentile_expanding', 'Pct-expanding'),
)
_ORACLE_DIAG_HEADER = (
    'ORACLE DIAGNOSTICS (future-known, non-deployable reference)'
)


def _is_v2_metrics(metrics):
    if int(metrics.get('simulation_schema_version') or 0) == 2:
        return True
    return bool(metrics.get('simulation_variants'))


def _money_pair(final_value, return_pct):
    return f"${_fmt(final_value, '.2f')} ({_fmt(return_pct, '+.2f')}%)"


def _variant_cutoff(cfg):
    mode = cfg.get('threshold_mode')
    if mode == 'absolute':
        return cfg.get('min_signal')
    if mode == 'zscore':
        return cfg.get('zscore_cutoff')
    if mode == 'percentile':
        return cfg.get('percentile_cutoff')
    return None


def build_lr_test_summary(metrics, config, *, test_id, model_id, data_id,
                          n_samples, horizon, test_file_type='test',
                          timestamp=None):
    """Single source of truth for summary.txt (v2 and single profiles)."""
    stamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    is_v2 = _is_v2_metrics(metrics)
    lines = [
        '=' * 70,
        'LSTM LOG-RETURN TEST SUMMARY',
        '=' * 70,
        '',
        f'Test ID:    {test_id}',
        f'Model ID:   {model_id}',
        f'Data ID:    {data_id}',
        f'Date:       {stamp}',
        f'Horizon:    t+{horizon}',
        f'Samples:    {n_samples:,}',
        f'Data used:  {test_file_type}.csv',
    ]
    if metrics.get('epochs_trained') is not None:
        lines.append(
            f"Training ended: epoch {metrics['epochs_trained']}"
            f"  (best checkpoint: epoch {metrics.get('best_epoch', 'n/a')})")
    lines.extend([
        '',
        'METRICS (log-return space):',
        f"  MAE:      {_fmt(metrics.get('mae'), '.6f')}",
        f"  RMSE:     {_fmt(metrics.get('rmse'), '.6f')}",
        f"  R²:       {_fmt(metrics.get('r2'), '.4f')}",
        f"  PH(δ=1):  {_fmt(metrics.get('pseudo_huber'), '.6f')}",
        f"  Dir Acc:  {_fmt(metrics.get('dir_acc'), '.2f')}%",
        f"  CR:       {_fmt(metrics.get('compression_ratio'), '.4f')}",
    ])
    if 't_mae' in metrics:
        lines.extend([
            f"  THRESHOLDED (|y| ≥ {_fmt(metrics.get('t_threshold'), '.6f')},"
            f" n={metrics.get('t_n_samples')}):",
            f"  t_MAE:     {_fmt(metrics.get('t_mae'), '.6f')}",
            f"  t_RMSE:    {_fmt(metrics.get('t_rmse'), '.6f')}",
            f"  t_R²:      {_fmt(metrics.get('t_r2'), '.4f')}",
            f"  t_Dir Acc: {_fmt(metrics.get('t_dir_acc'), '.2f')}%",
        ])
    lines.extend(['', 'EVALUATION:'])
    if is_v2:
        lines.append(
            f"  Profile:         {metrics.get('simulation_profile', 'v2')}")
        lines.append(
            f"  Ranking method:  {metrics.get('ranking_method', 'n/a')}")
        lines.append('  Variants:        A60, A70, Z-expanding, Pct-expanding')
    else:
        lines.append('  Profile:         single')
        lines.append(
            f"  Threshold mode:  {metrics.get('threshold_mode_eval', metrics.get('threshold_mode', 'n/a'))}")
    lines.extend([
        '',
        'SIGNAL METRICS:',
        f"  pct_strong:         {_fmt(metrics.get('pct_strong'), '.2f')}%",
        f"  calib_buy:          {_fmt(metrics.get('calib_buy'), '.3f')}",
        f"  calib_sell:         {_fmt(metrics.get('calib_sell'), '.3f')}",
        f"  dir_acc_strong:     {_fmt(metrics.get('dir_acc_strong'), '.2f')}%",
    ])
    if is_v2:
        variants = metrics.get('simulation_variants') or {}
        lines.extend(['', 'SIMULATION VARIANTS (model):'])
        for key, label in _VARIANT_ORDER:
            item = variants.get(key) or {}
            cfg = item.get('config') or {}
            gross = (item.get('gross') or {}).get('model') or {}
            net = (item.get('net') or {}).get('model') or {}
            activity = item.get('activity') or {}
            lines.extend([
                f'  {label} ({key}):',
                f"    config: mode={cfg.get('threshold_mode', 'n/a')} "
                f"history={cfg.get('threshold_history', 'n/a')} "
                f"cutoff={_fmt(_variant_cutoff(cfg), 'g')} "
                f"warmup={cfg.get('warmup', 'n/a')} "
                f"fees gross={_fmt(cfg.get('gross_fee_bps'), '.1f')} "
                f"net={_fmt(cfg.get('net_fee_bps'), '.1f')} bps",
                f"    model gross: {_money_pair(gross.get('final_value'), gross.get('return_pct'))}",
                f"    model net:   {_money_pair(net.get('final_value'), net.get('return_pct'))}",
                f"    trades: {gross.get('n_trades', net.get('n_trades', 'n/a'))}  "
                f"fee legs: {gross.get('n_fee_legs', net.get('n_fee_legs', 'n/a'))}",
                f"    activity: pct_strong={_fmt(activity.get('pct_strong'), '.1f')}% "
                f"n_strong={activity.get('n_strong', 'n/a')}",
            ])
    elif 'sim_final_model' in metrics:
        lines.extend([
            '',
            'SIMULATION (model):',
            f"  Model gross:  {_money_pair(metrics.get('sim_final_model'), metrics.get('sim_return_model_pct'))}",
            f"  Model net:    {_money_pair(metrics.get('sim_final_model_net'), metrics.get('sim_return_model_net_pct'))}",
            f"  Trades:       {metrics.get('sim_n_trades_model', 'n/a')}",
        ])
    if 'rank_score' in metrics:
        lines.extend([
            '',
            'RANK SCORE:',
            f"  rank_score:   {_fmt(metrics.get('rank_score'), '.2f')} / 100 "
            f"[{metrics.get('signal_class', 'n/a')}]",
            f"  components:   {metrics.get('rank_components')}",
            f"  ranking_method: {metrics.get('ranking_method', 'n/a')}",
        ])
    lines.extend(['', '-' * 70, _ORACLE_DIAG_HEADER, '-' * 70])
    if is_v2:
        variants = metrics.get('simulation_variants') or {}
        for key, label in _VARIANT_ORDER:
            item = variants.get(key) or {}
            oracle = ((item.get('gross') or {}).get('oracle')
                      or item.get('oracle') or {})
            lines.append(
                f"  {label}: {_money_pair(oracle.get('final_value'), oracle.get('return_pct'))} "
                f"trades={oracle.get('n_trades', 'n/a')}")
        lines.append(
            f"  capture_log (primary): {_fmt(metrics.get('sim_capture_log'), '.3f')}")
    elif 'sim_final_oracle' in metrics:
        lines.extend([
            f"  Oracle gross: {_money_pair(metrics.get('sim_final_oracle'), metrics.get('sim_return_oracle_pct'))}",
            f"  Capture log:  {_fmt(metrics.get('sim_capture_log'), '.3f')}",
        ])
    lines.append('')
    return '\n'.join(lines)
