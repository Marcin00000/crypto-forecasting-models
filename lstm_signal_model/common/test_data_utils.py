"""
Test data utilities for LSTM Signal Model pipeline.

Auto-detection, inference, and results saving for signal_test.py.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .path_utils import resolve_repo_path


# =========================================================================
# MODEL / DATA AUTO-DETECTION
# =========================================================================

def extract_signal_model_id(model_dir: str) -> str:
    """Extract SIG### from model directory path."""
    basename = os.path.basename(model_dir.rstrip(os.sep))
    if '_' in basename and basename.startswith('SIG'):
        candidate = basename.split('_')[0]
        if len(candidate) >= 6 and candidate[3:6].isdigit():
            return candidate
    return 'UNKNOWN'


def auto_select_model_dir(models_dir: str) -> str:
    """Auto-select latest trained signal model from signal_models/."""
    if not os.path.exists(models_dir):
        raise FileNotFoundError(
            f"No models directory found: {models_dir}\n"
            "Train a model first: python signal_train.py --data_path <path>"
        )

    model_dirs = sorted([
        d for d in os.listdir(models_dir)
        if d.startswith('SIG') and os.path.isdir(os.path.join(models_dir, d))
        and os.path.exists(os.path.join(models_dir, d, 'model_config.json'))
        and os.path.exists(os.path.join(models_dir, d, 'signal_model.pth'))
    ])

    if not model_dirs:
        raise FileNotFoundError(
            f"No trained signal models found in {models_dir}\n"
            "Train a model first: python signal_train.py --data_path <path>"
        )

    latest = os.path.join(models_dir, model_dirs[-1])
    print(f"Auto-selected latest model: {latest}")
    return latest


def auto_detect_data_path(config, base_dir: str) -> str | None:
    """
    Auto-detect prepared data directory with test.csv.

    Strategies:
      1. config train_path (relative values resolve against the repo root)
         if it contains test.csv
      2. search data_preparation/prepared_data by data_id (canonical location)
    """
    train_path = resolve_repo_path(config.get('train_path', ''))
    data_id = config.get('data_id', '')

    if train_path and os.path.isdir(train_path):
        if os.path.exists(os.path.join(train_path, 'test.csv')):
            print(f"Auto-detected: test.csv in training directory")
            return train_path

    if data_id and data_id != 'UNKNOWN':
        search_dirs = [
            os.path.join(base_dir, '..', 'data_preparation', 'prepared_data'),
        ]
        for search_dir in search_dirs:
            search_dir = os.path.normpath(search_dir)
            if not os.path.exists(search_dir):
                continue
            for name in sorted(os.listdir(search_dir), reverse=True):
                if not name.startswith(data_id):
                    continue
                cpath = os.path.join(search_dir, name)
                if os.path.isdir(cpath) and os.path.exists(
                        os.path.join(cpath, 'test.csv')):
                    print(f"Auto-detected by data_id ({data_id}): {cpath}")
                    return cpath

    return None


def resolve_model_dir(model_dir, models_dir: str) -> str:
    """Resolve short SIG### id or full path to model directory."""
    if model_dir is None:
        return auto_select_model_dir(models_dir)
    if os.path.sep not in model_dir and not os.path.exists(model_dir):
        candidate = os.path.join(models_dir, model_dir)
        if os.path.isdir(candidate):
            return candidate
        for d in os.listdir(models_dir):
            if d.startswith(model_dir + '_'):
                return os.path.join(models_dir, d)
    return model_dir


# =========================================================================
# INFERENCE
# =========================================================================

def run_inference(model, X_test, device, batch_size=4096, submodel_preds=None):
    """Generate dual-channel predictions in batches with progress bar.

    Args:
        submodel_preds: optional (N, submodel_dim) numpy array of
            pre-computed submodel predictions. When provided, sliced
            per batch and passed to model.forward(submodel_inputs=...).
    """
    model.eval()
    preds_list = []
    n_batches = (len(X_test) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in tqdm(range(0, len(X_test), batch_size),
                          desc="Predicting", total=n_batches):
            end = min(start + batch_size, len(X_test))
            x_batch = torch.from_numpy(X_test[start:end]).to(device)
            sub_batch = None
            if submodel_preds is not None:
                sub_batch = torch.from_numpy(
                    submodel_preds[start:end].astype(np.float32)).to(device)
            preds_list.append(
                model(x_batch, submodel_inputs=sub_batch).cpu().numpy())

    return np.concatenate(preds_list, axis=0)


# =========================================================================
# SUMMARY (single source of truth for summary.txt and console)
# =========================================================================

_VARIANT_ORDER = (
    ('absolute_20', 'A20'),
    ('absolute_30', 'A30'),
    ('zscore_expanding', 'Z-expanding'),
    ('percentile_expanding', 'Pct-expanding'),
)
_ORACLE_DIAG_HEADER = (
    'ORACLE DIAGNOSTICS (future-known, non-deployable reference)'
)


def _fmt(value, spec, missing='n/a'):
    if value is None:
        return missing
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return missing


def _is_v2_metrics(metrics):
    if int(metrics.get('simulation_schema_version') or 0) == 2:
        return True
    profile = metrics.get('simulation_profile')
    return profile in ('v2', 'signal_simulation_v2') or bool(
        metrics.get('simulation_variants'))


def _corr_avg(metrics):
    if metrics.get('corr_avg') is not None:
        return metrics['corr_avg']
    buy, sell = metrics.get('corr_buy'), metrics.get('corr_sell')
    if isinstance(buy, (int, float)) and isinstance(sell, (int, float)):
        return (buy + sell) / 2.0
    return None


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


def _append_identity(lines, *, test_id, model_id, data_id, n_samples,
                     horizon, config, model_dir, data_path, timestamp):
    lines.extend([
        '=' * 70,
        'LSTM SIGNAL MODEL TEST SUMMARY',
        '=' * 70,
        '',
        f'Test ID:    {test_id}',
        f'Model ID:   {model_id}',
    ])
    if model_dir:
        lines.append(f'Model dir:  {model_dir}')
    lines.append(f'Data ID:    {data_id}')
    if data_path:
        lines.append(f'Data path:  {data_path}')
    lines.extend([
        f'Date:       {timestamp}',
        f'Horizon:    t+{horizon}',
        f'Samples:    {n_samples:,}',
    ])
    if config.get('best_epoch') is not None:
        lines.append(f'Best epoch: {config["best_epoch"]}')


def _append_evaluation(lines, metrics, *, is_v2):
    lines.extend(['', 'EVALUATION:'])
    if is_v2:
        lines.append(
            f"  Profile:         {metrics.get('simulation_profile', 'v2')}")
        lines.append(
            f"  Ranking method:  {metrics.get('ranking_method', 'n/a')}")
        lines.append('  Variants:        A20, A30, Z-expanding, Pct-expanding')
    else:
        lines.append('  Profile:         single')
        lines.append(
            f"  Threshold mode:  {metrics.get('threshold_mode_eval', 'n/a')}")
        lines.append(
            f"  History:         {metrics.get('threshold_history_eval', 'n/a')}")
        lines.append(
            f"  Warmup:          {metrics.get('threshold_warmup_eval', 'n/a')}")
        lines.append(
            f"  Min signal:      {metrics.get('min_signal_eval', 'n/a')}")
    lines.append(
        f"  Thresh buy:      {_fmt(metrics.get('thresh_buy_eval'), '.5f')}")
    lines.append(
        f"  Thresh sell:     {_fmt(metrics.get('thresh_sell_eval'), '.5f')}")


def _append_signal_metrics(lines, metrics):
    lines.extend([
        '',
        'SIGNAL METRICS:',
        f"  MAE buy:           {_fmt(metrics.get('mae_buy'), '.6f')}",
        f"  MAE sell:          {_fmt(metrics.get('mae_sell'), '.6f')}",
        f"  MAE combined:      {_fmt(metrics.get('mae_combined'), '.6f')}",
        f"  Corr buy:          {_fmt(metrics.get('corr_buy'), '.4f')}",
        f"  Corr sell:         {_fmt(metrics.get('corr_sell'), '.4f')}",
        f"  Corr avg:          {_fmt(_corr_avg(metrics), '.4f')}",
        f"  Diversity:         {_fmt(metrics.get('diversity'), '.4f')}",
        f"  Precision buy:     {_fmt(metrics.get('precision_buy'), '.4f')}",
        f"  Precision sell:    {_fmt(metrics.get('precision_sell'), '.4f')}",
        f"  Calibration buy:   {_fmt(metrics.get('calib_buy'), '.4f')}",
        f"  Calibration sell:  {_fmt(metrics.get('calib_sell'), '.4f')}",
        f"  Strong buy:        {metrics.get('n_strong_buy', 'n/a')}",
        f"  Strong sell:       {metrics.get('n_strong_sell', 'n/a')}",
        f"  % strong:          {_fmt(metrics.get('pct_strong'), '.1f')}%",
    ])


def _append_v2_variants(lines, metrics):
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
            f"n_strong_buy={activity.get('n_strong_buy', 'n/a')} "
            f"n_strong_sell={activity.get('n_strong_sell', 'n/a')}",
        ])


def _append_single_simulation(lines, metrics):
    if 'sim_final_model' not in metrics:
        return
    lines.extend([
        '',
        'SIMULATION (model):',
        f"  Fee:          {_fmt(metrics.get('sim_fee_bps'), '.1f')} bps net",
        f"  Model gross:  {_money_pair(metrics.get('sim_final_model'), metrics.get('sim_return_model_pct'))}",
        f"  Model net:    {_money_pair(metrics.get('sim_final_model_net'), metrics.get('sim_return_model_net_pct'))}",
        f"  Trades:       {metrics.get('sim_n_trades_model', 'n/a')}",
    ])


def _append_rank(lines, metrics):
    if 'rank_score' not in metrics:
        return
    lines.extend([
        '',
        'RANK SCORE:',
        f"  rank_score:   {_fmt(metrics.get('rank_score'), '.2f')} / 100 "
        f"[{metrics.get('signal_class', 'n/a')}]",
        f"  components:   {metrics.get('rank_components')}",
        f"  rank_version: {metrics.get('signal_rank_version', 'n/a')}",
    ])


def _append_oracle_diagnostics(lines, metrics, *, is_v2):
    lines.extend(['', '-' * 70, _ORACLE_DIAG_HEADER, '-' * 70])
    lines.append(
        '  Primary oracle contributes only to the diagnostic capture '
        'component of RankScore.'
    )
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
        return
    if 'sim_final_oracle' not in metrics:
        lines.append('  n/a')
        return
    lines.extend([
        f"  Oracle gross: {_money_pair(metrics.get('sim_final_oracle'), metrics.get('sim_return_oracle_pct'))}",
        f"  Oracle net:   {_money_pair(metrics.get('sim_final_oracle_net'), metrics.get('sim_return_oracle_net_pct'))}",
        f"  Trades:       {metrics.get('sim_n_trades_oracle', 'n/a')}",
        f"  Capture:      {_fmt(metrics.get('sim_capture_ratio'), '.3f')}",
        f"  Capture log:  {_fmt(metrics.get('sim_capture_log'), '.3f')}",
    ])


def build_signal_test_summary(
        metrics, config, *, test_id, model_id, data_id, n_samples, horizon,
        model_dir=None, data_path=None, timestamp=None):
    """Build complete summary.txt text for v2 or single evaluation profiles."""
    stamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    is_v2 = _is_v2_metrics(metrics)
    lines = []
    _append_identity(
        lines, test_id=test_id, model_id=model_id, data_id=data_id,
        n_samples=n_samples, horizon=horizon, config=config or {},
        model_dir=model_dir, data_path=data_path, timestamp=stamp)
    _append_evaluation(lines, metrics, is_v2=is_v2)
    _append_signal_metrics(lines, metrics)
    if is_v2:
        _append_v2_variants(lines, metrics)
    else:
        _append_single_simulation(lines, metrics)
    _append_rank(lines, metrics)
    _append_oracle_diagnostics(lines, metrics, is_v2=is_v2)
    lines.append('')
    return '\n'.join(lines)


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
        if file_path is None:
            raise FileNotFoundError(f"Neither results.parquet nor results.csv found in: {target}")
    elif target.is_file():
        file_path = target
    else:
        file_path = find_results_path(target.parent) if target.parent.is_dir() else None
        if file_path is None:
            raise FileNotFoundError(f"Results file or directory not found: {target}")

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


def find_simulation_steps_path(run_dir: str | Path) -> Path | None:
    """Locate simulation_steps.parquet or simulation_steps.csv in run_dir (parquet priority)."""
    run_path = Path(run_dir)
    pq = run_path / "simulation_steps.parquet"
    if pq.is_file():
        return pq
    csv = run_path / "simulation_steps.csv"
    if csv.is_file():
        return csv
    return None


def load_simulation_steps(run_dir_or_file: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load simulation_steps DataFrame with automatic format verification (parquet or csv)."""
    target = Path(run_dir_or_file)
    if target.is_dir():
        file_path = find_simulation_steps_path(target)
        if file_path is None:
            raise FileNotFoundError(f"Neither simulation_steps.parquet nor simulation_steps.csv found in: {target}")
    elif target.is_file():
        file_path = target
    else:
        file_path = find_simulation_steps_path(target.parent) if target.parent.is_dir() else None
        if file_path is None:
            raise FileNotFoundError(f"simulation_steps file or directory not found: {target}")

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


def save_signal_results(y_pred, target_oracle, y_true_lr, metrics, config,
                        test_id, model_id, data_id, output_dir, horizon,
                        min_signal_pct=60.0, sim_result=None,
                        thresh_buy=None, thresh_sell=None,
                        model_dir=None, data_path=None,
                        results_format='parquet', compression='snappy'):
    """Save test results: results.parquet/csv, simulation_steps.parquet/csv, metrics, summary."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    target_oracle = np.asarray(target_oracle, dtype=np.float64)
    y_true_lr = np.asarray(y_true_lr, dtype=np.float64)
    threshold = min_signal_pct / 100.0
    t_buy = threshold if thresh_buy is None else float(thresh_buy)
    t_sell = threshold if thresh_sell is None else float(thresh_sell)

    strong_buy = (y_pred[:, 0] >= t_buy).astype(int)
    strong_sell = (y_pred[:, 1] >= t_sell).astype(int)

    df_results = pd.DataFrame({
        'oracle_buy': target_oracle[:, 0],
        'oracle_sell': target_oracle[:, 1],
        'pred_buy': y_pred[:, 0],
        'pred_sell': y_pred[:, 1],
        'error_buy': y_pred[:, 0] - target_oracle[:, 0],
        'error_sell': y_pred[:, 1] - target_oracle[:, 1],
        'abs_error_buy': np.abs(y_pred[:, 0] - target_oracle[:, 0]),
        'abs_error_sell': np.abs(y_pred[:, 1] - target_oracle[:, 1]),
        'y_true_lr': y_true_lr,
        'signal_buy_pct': y_pred[:, 0] * 100.0,
        'signal_sell_pct': y_pred[:, 1] * 100.0,
        'oracle_buy_pct': target_oracle[:, 0] * 100.0,
        'oracle_sell_pct': target_oracle[:, 1] * 100.0,
        'strong_buy': strong_buy,
        'strong_sell': strong_sell,
        'correct_dir_buy': ((strong_buy == 1) & (y_true_lr > 0)).astype(int),
        'correct_dir_sell': ((strong_sell == 1) & (y_true_lr < 0)).astype(int),
    })

    comp = None if compression in ('none', None) else compression
    if results_format == 'parquet':
        pq_path = os.path.join(output_dir, 'results.parquet')
        df_results.to_parquet(pq_path, compression=comp, index=False)
        print("  results.parquet")
    elif results_format == 'csv':
        csv_path = os.path.join(output_dir, 'results.csv')
        df_results.to_csv(csv_path, index=False)
        print("  results.csv")
    else:
        raise ValueError(f"Unsupported results_format: {results_format}")

    if sim_result is not None:
        model_sim = sim_result.get('_model_sim', {})
        oracle_sim = sim_result.get('_oracle_sim', {})
        m_idx = model_sim.get('step_indices', [])
        o_idx = oracle_sim.get('step_indices', [])
        n_steps = max(len(m_idx), len(o_idx))
        if n_steps > 0:
            rows = []
            for k in range(n_steps):
                row = {'step': k}
                if k < len(m_idx):
                    row['sample_idx'] = int(m_idx[k])
                    row['position_model'] = int(model_sim['positions'][k])
                    row['equity_model'] = float(model_sim['equity_curve'][k + 1])
                    row['y_true_at_step'] = float(y_true_lr[m_idx[k]])
                if k < len(o_idx):
                    row['position_oracle'] = int(oracle_sim['positions'][k])
                    row['equity_oracle'] = float(oracle_sim['equity_curve'][k + 1])
                rows.append(row)
            df_steps = pd.DataFrame(rows)
            if results_format == 'parquet':
                df_steps.to_parquet(
                    os.path.join(output_dir, 'simulation_steps.parquet'),
                    compression=comp, index=False)
                print("  simulation_steps.parquet")
            elif results_format == 'csv':
                df_steps.to_csv(
                    os.path.join(output_dir, 'simulation_steps.csv'), index=False)
                print("  simulation_steps.csv")

    metrics_save = {k: v for k, v in metrics.items()
                    if not k.startswith('_')}
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_save, f, indent=2, default=str)
    print("  metrics.json")

    summary = build_signal_test_summary(
        metrics, config or {},
        test_id=test_id, model_id=model_id, data_id=data_id,
        n_samples=len(y_pred), horizon=horizon,
        model_dir=model_dir, data_path=data_path)
    with open(os.path.join(output_dir, 'summary.txt'), 'w',
              encoding='utf-8') as f:
        f.write(summary)
    print("  summary.txt")
    return summary
