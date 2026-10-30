"""
XGBoost Model Testing and Market Simulation Evaluation
"""
import sys
import os
import json
import warnings
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from xgb_config import XGBConfig
from common.test_data_utils import (
    load_test_data_only, load_test_csv,
    save_results as save_results_to_disk, build_xgb_test_summary,
)
from common.id_utils import get_next_test_id, extract_model_id, extract_data_id, resolve_repo_path
from common.metrics import (
    calculate_metrics, print_metrics_summary, interpret_metrics,
    compute_rank_score, format_rank_summary, _jsonable,
)
from common.simulation_profile import evaluate_xgb_simulation_v2
from common.sample_weighting import resolve_signal_threshold
from common.visualization import plot_test_results, plot_feature_importance
from common.log_utils import (
    log_run_header, log_model_info, log_test_run_info, log_test_complete,
)

warnings.filterwarnings('ignore')


def load_model(model_dir):
    """Load trained XGBoost model and configuration."""
    config_path = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing model_config.json in {model_dir}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, 'xgb_model.json'))

    feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))

    target_scaler = None
    target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)

    return model, config, feature_scaler, target_scaler


def auto_detect_test_path(config):
    """
    Auto-detect test data path from training configuration.

    Search order:
      1. train_path from the model config (repo-relative or absolute), if it
         contains metadata.json + test.csv
      2. data_preparation/prepared_data/ by data_id (canonical prepared dirs)
      3. Family copies (older, possibly stale): xgb_model/prepared_data/, then
         rf_model/rf_prepared_data/
    """
    train_path = resolve_repo_path(config.get('train_path', ''))
    data_id    = config.get('data_id', 'UNKNOWN')

    if os.path.isdir(train_path) and os.path.exists(
            os.path.join(train_path, 'metadata.json')):
        if os.path.exists(os.path.join(train_path, 'test.csv')):
            return train_path

    if data_id in ('RAW', 'UNKNOWN'):
        return None

    def _search_prep_dir(base):
        if not os.path.exists(base):
            return None
        # '<id>_' prefix keeps E10 from matching E109
        matching = [
            d for d in os.listdir(base)
            if d.startswith(f'{data_id}_') and os.path.isdir(os.path.join(base, d))
        ]
        if not matching:
            return None
        candidate = os.path.join(base, sorted(matching)[-1])
        if os.path.exists(os.path.join(candidate, 'test.csv')):
            return candidate
        return None

    project_root = os.path.dirname(XGBConfig.BASE_DIR)

    for base in (
        os.path.join(project_root, 'data_preparation', 'prepared_data'),
        XGBConfig.PREPARED_DATA_DIR,
        os.path.join(project_root, 'rf_model', 'rf_prepared_data'),
    ):
        result = _search_prep_dir(base)
        if result:
            return result

    return None


def test_model(
    model_dir=None,
    test_path=None,
    test_file_type='test',
    simulation_profile='v2',
    min_signal=60.0,
    fee_bps=None,
    auxiliary_fee_bps=None,
    save_results=True,
    results_format='parquet',
    results_compression='snappy',
):
    """Test XGBoost model on test data with market simulation and ranking."""
    if model_dir is None:
        if not os.path.exists(XGBConfig.MODELS_DIR):
            raise FileNotFoundError(f"Models directory not found: {XGBConfig.MODELS_DIR}")

        model_dirs = [d for d in os.listdir(XGBConfig.MODELS_DIR)
                      if d.startswith('X') and os.path.isdir(os.path.join(XGBConfig.MODELS_DIR, d))]

        if not model_dirs:
            raise FileNotFoundError(f"No trained models found in {XGBConfig.MODELS_DIR}")

        model_dir = os.path.join(XGBConfig.MODELS_DIR, sorted(model_dirs)[-1])
        print(f"Auto-selected latest model: {os.path.basename(model_dir)}")

    if not os.path.isabs(model_dir) and not os.path.exists(model_dir):
        model_dir = os.path.join(XGBConfig.MODELS_DIR, model_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_id = extract_model_id(model_dir)
    log_run_header("XGBOOST MODEL TESTING & SIMULATION", newline=True)

    print(f"\nLoading model: {model_dir}")
    model, config, feature_scaler, target_scaler = load_model(model_dir)
    log_model_info(model_id, config, target_scaler)

    if test_path is None:
        test_path = auto_detect_test_path(config)
        if test_path is None:
            raise FileNotFoundError(
                f"Cannot auto-detect test data for model {model_id}.\n"
                f"Please specify: --test_path <path>"
            )
        print(f"\n[OK] Auto-detected test data: {test_path}")
    else:
        print(f"\nTest data: {test_path}")

    print("\nLoading test data...")
    close_full = None
    if os.path.isdir(test_path):
        data, metadata, close_full = load_test_data_only(
            test_path,
            test_file_type=test_file_type,
            flattening_method=config['flattening_method'],
            feature_set=config.get('feature_set'),
            custom_features=config.get('custom_features'),
            features=config['features'],
            lookback=config['lookback'],
            horizon=config['horizon'],
            return_close=True
        )
        X_test = data['X_test']
        y_test = data['y_test']
    elif test_path.endswith('.csv'):
        X_test, y_test, metadata, close_full = load_test_csv(
            test_path, config, feature_scaler
        )
    else:
        raise ValueError(f"Invalid test_path: {test_path}. Expected directory or .csv file")

    test_id = get_next_test_id(XGBConfig.LOGS_DIR)
    data_id = extract_data_id(metadata) if metadata else config['data_id']
    date_short = datetime.now().strftime('%d%m')
    run_name = f"{test_id}_test_results_{model_id}_{data_id}_{date_short}"
    output_dir = os.path.join(XGBConfig.LOGS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    log_test_run_info(test_id, len(X_test), output_dir)
    log_run_header("GENERATING PREDICTIONS", newline=True)

    saved_iter = model.attr('saved_iteration')
    if saved_iter is not None:
        n_saved = int(saved_iter)
        iter_range = (0, n_saved)
    else:
        iter_range = (0, model.num_boosted_rounds())

    expected_cols = config.get('n_flat_features')
    if expected_cols is not None and X_test.shape[1] != expected_cols:
        print(f"[SHAPE WARNING] X_test cols={X_test.shape[1]}, "
              f"model expects {expected_cols} flat features — mismatch may cause errors")

    dtest = xgb.DMatrix(X_test)
    y_pred_scaled = model.predict(dtest, iteration_range=iter_range)

    if target_scaler is not None:
        y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
    else:
        y_pred_log = y_pred_scaled

    print(f"Generated {len(y_pred_log)} predictions (iter_range={iter_range})")
    log_run_header("EVALUATION METRICS", newline=True)

    signal_threshold_enabled = config.get('signal_threshold_enabled', XGBConfig.SIGNAL_THRESHOLD_ENABLED)
    resolved_threshold = None
    if signal_threshold_enabled:
        saved_threshold = config.get('signal_threshold_resolved')
        if saved_threshold is not None:
            resolved_threshold = float(saved_threshold)
            print(f"\nSignal threshold: {resolved_threshold:.6f} (from model config)")
        else:
            _saved_mode = config.get('signal_threshold_mode', XGBConfig.SIGNAL_THRESHOLD_MODE)
            _fixed_value = config.get('signal_threshold_value', XGBConfig.SIGNAL_THRESHOLD_VALUE)
            if _saved_mode == 'empirical':
                print(
                    f"\n[WARN] Model used signal_threshold_mode='empirical' but y_train is "
                    f"unavailable during testing — falling back to fixed threshold "
                    f"({_fixed_value}) to avoid data leakage."
                )
            resolved_threshold = resolve_signal_threshold(
                y_test,
                mode='fixed',
                fixed_value=_fixed_value,
            )
            print(f"\nSignal threshold: {resolved_threshold:.6f} (fallback fixed — model config missing)")

    metrics = calculate_metrics(y_test, y_pred_log, threshold=resolved_threshold)
    metrics['test_id'] = test_id
    metrics['model_id'] = model_id
    print_metrics_summary(metrics)
    interpret_metrics(metrics)

    # Market Simulation & Rank Score
    sim_results = None
    rank_metrics = None
    # None = use the profile default; an explicit 0.0 is a valid gross (no-fee) run.
    effective_primary_fee = fee_bps if fee_bps is not None else XGBConfig.SIM_PRIMARY_FEE_BPS
    effective_aux_fee = (
        auxiliary_fee_bps if auxiliary_fee_bps is not None else XGBConfig.SIM_AUXILIARY_FEE_BPS
    )

    if simulation_profile == 'v2' and close_full is not None:
        log_run_header("MARKET SIMULATION & RANK SCORE (Simulation v2)", newline=True)
        symbol = config.get('symbol', 'ETHUSDT')
        lookback = int(config.get('lookback', XGBConfig.DEFAULT_LOOKBACK))
        horizon = int(config.get('horizon', XGBConfig.DEFAULT_HORIZON))
        sim_thresh = resolved_threshold if resolved_threshold is not None else XGBConfig.SIGNAL_THRESHOLD_VALUE

        sim_results = evaluate_xgb_simulation_v2(
            y_true=y_test,
            y_pred=y_pred_log,
            close=close_full,
            lookback=lookback,
            horizon=horizon,
            threshold=sim_thresh,
            min_signal=min_signal,
            asset=symbol,
            window_sigma=XGBConfig.SIGNAL_WINDOW_SIGMA,
            primary_fee_bps=effective_primary_fee,
            auxiliary_fee_bps=effective_aux_fee,
        )

        rank_metrics = compute_rank_score(
            metrics=sim_results,
            pred_log_returns=y_pred_log,
            true_log_returns=y_test,
            n_samples=len(y_test)
        )
        sim_results['rank_metrics'] = rank_metrics

        model_sim = sim_results.get('_model_sim', {})
        sim_summary = {
            'fee_bps': effective_primary_fee,
            'total_return_pct': sim_results.get('sim_return_model_pct', 0.0),
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate_pct': 0.0,
            'trades': sim_results.get('sim_n_trades_model', 0),
        }
        if 'equity_curve' in model_sim:
            eq = np.asarray(model_sim['equity_curve'], dtype=np.float64)
            if len(eq) > 1:
                cummax = np.maximum.accumulate(eq)
                dd = (eq - cummax) / np.maximum(cummax, 1e-12) * 100.0
                sim_summary['max_drawdown_pct'] = float(np.min(dd))

        # Flatten top-level simulation and rank metrics into metrics contract
        metrics['simulation_schema_version'] = sim_results.get('simulation_schema_version', 2)
        metrics['simulation_profile'] = simulation_profile
        metrics['ranking_method'] = sim_results.get('ranking_method', 'absolute_60_gross')
        metrics['simulation_variants'] = sim_results.get('simulation_variants', {})
        metrics['rank_components'] = rank_metrics.get('rank_components', {})
        metrics['sim_start_cash'] = sim_results.get('sim_start_cash', 100.0)
        metrics['sim_final_model'] = sim_results.get('sim_final_model')
        metrics['sim_final_model_net'] = sim_results.get('sim_final_model_net')
        metrics['sim_final_oracle'] = sim_results.get('sim_final_oracle')
        metrics['sim_return_oracle_pct'] = sim_results.get('sim_return_oracle_pct')
        metrics['sim_n_trades_oracle'] = sim_results.get('sim_n_trades_oracle')
        metrics['sim_capture_log'] = sim_results.get('sim_capture_log')
        metrics['pct_strong'] = sim_results.get('pct_strong')
        metrics['calib_buy'] = sim_results.get('calib_buy')
        metrics['calib_sell'] = sim_results.get('calib_sell')
        metrics['dir_acc_strong'] = sim_results.get('dir_acc_strong')
        metrics['rank_score'] = rank_metrics.get('rank_score')
        metrics['signal_class'] = rank_metrics.get('signal_class')
        metrics['compression_ratio'] = rank_metrics.get('compression_ratio')
        metrics['pred_std'] = rank_metrics.get('pred_std')
        metrics['true_std'] = rank_metrics.get('true_std')
        metrics['corr'] = rank_metrics.get('corr')
        metrics['sim_total_return_pct'] = sim_results.get('sim_return_model_pct')
        metrics['sim_total_return_net_pct'] = sim_results.get('sim_return_model_net_pct')
        metrics['sim_trades'] = sim_results.get('sim_n_trades_model')
        # Preserve clean nested simulation dict without raw full arrays or private debug objects
        metrics['simulation'] = {
            k: v for k, v in sim_results.items()
            if not k.startswith('_') and k not in ('signal_buy', 'signal_sell', 'primary_positions')
        }

        print("\n" + format_rank_summary(rank_metrics, sim_summary))
    else:
        metrics['rank_score'] = None
        metrics['signal_class'] = 'UNKNOWN_SIGNAL'
        metrics['simulation_schema_version'] = None
        metrics['simulation_profile'] = simulation_profile
        metrics['rank_components'] = {}
        metrics['sim_start_cash'] = None
        if close_full is None and simulation_profile == 'v2':
            print("\n[WARN] Simulation v2 skipped: close price series unavailable in test data.")

    # Visualizations & Saving
    if save_results:
        log_run_header("GENERATING VISUALIZATIONS", newline=True)
        plot_test_results(y_test, y_pred_log, output_dir, config['horizon'])

        if config.get('flat_features'):
            plot_feature_importance(model, config['flat_features'], output_dir, top_n=30)

        # Save results
        log_run_header("SAVING RESULTS", newline=True)

        saved_paths = save_results_to_disk(
            y_true=y_test,
            y_pred=y_pred_log,
            output_dir=output_dir,
            sim_results=sim_results,
            format_choice=results_format,
            compression=results_compression,
        )
        for p in saved_paths:
            print(f"  [OK] {os.path.basename(p)}")

        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(_jsonable(metrics), f, indent=2)
        print("  [OK] metrics.json")

        summary_txt = build_xgb_test_summary(
            test_id=test_id,
            model_id=model_id,
            data_id=data_id,
            config=config,
            metrics=metrics,
            sim_results=sim_results,
            rank_metrics=rank_metrics,
            n_samples=len(y_test),
            effective_fee_bps=effective_primary_fee,
        )
        with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write(summary_txt)
        print("  [OK] summary.txt")

    log_test_complete(test_id, model_id, len(y_test), metrics, output_dir)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test XGBoost model')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--test_file_type', type=str, default='test',
                        choices=['test', 'agent', 'sym', 'val', 'train'])
    parser.add_argument('--simulation_profile', type=str, default='v2',
                        choices=['none', 'legacy', 'v2'])
    parser.add_argument('--min_signal', type=float, default=60.0)
    parser.add_argument('--fee_bps', type=float, default=None,
                        help='Primary fee in bps (default: XGBConfig.SIM_PRIMARY_FEE_BPS; 0 = no fee)')
    parser.add_argument('--auxiliary_fee_bps', type=float, default=None,
                        help='Auxiliary fee in bps (default: XGBConfig.SIM_AUXILIARY_FEE_BPS)')
    parser.add_argument('--save_results', action='store_true', default=True,
                        help='Save test results, metrics.json, and visualizations to disk (default: True)')
    parser.add_argument('--no_save_results', dest='save_results', action='store_false',
                        help='Skip saving test results, metrics.json, and visualizations to disk')
    parser.add_argument('--results_format', type=str, default='parquet',
                        choices=['parquet', 'csv', 'both'],
                        help='Predictions output format (parquet, csv, both; default: parquet)')
    parser.add_argument('--results_compression', type=str, default='snappy',
                        choices=['snappy', 'gzip', 'brotli', 'none'],
                        help='Compression for parquet format (default: snappy)')
    args = parser.parse_args()

    try:
        test_model(
            model_dir=args.model_dir,
            test_path=args.test_path,
            test_file_type=args.test_file_type,
            simulation_profile=args.simulation_profile,
            min_signal=args.min_signal,
            fee_bps=args.fee_bps,
            auxiliary_fee_bps=args.auxiliary_fee_bps,
            save_results=args.save_results,
            results_format=args.results_format,
            results_compression=args.results_compression,
        )
    except (FileNotFoundError, Exception) as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        if not isinstance(e, FileNotFoundError):
            import traceback
            traceback.print_exc()
        raise
