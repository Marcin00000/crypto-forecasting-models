"""
Random Forest Model Testing and Simulation v2 Evaluation.
"""
from __future__ import annotations

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from rf_config import RFConfig
from common.id_utils import (
    get_next_test_id, extract_model_id, extract_data_id, resolve_repo_path,
)
from common.metrics import (
    calculate_metrics,
    calculate_threshold_metrics,
    compute_rank_score,
    print_metrics_summary,
    interpret_metrics,
)
from common.test_data_utils import (
    load_test_data_only,
    load_test_csv,
    save_results,
    build_rf_test_summary,
)
from common.simulation_profile import evaluate_rf_simulation_v2
from common.signals import compute_signals
from common.visualization import plot_test_results, plot_feature_importance

warnings.filterwarnings('ignore')

__all__ = [
    'load_model',
    'auto_detect_test_path',
    'test_model',
]


def load_model(model_dir: str):
    """Load trained Random Forest model and configuration."""
    config_path = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing model_config.json in {model_dir}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model_path = os.path.join(model_dir, 'rf_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing rf_model.pkl in {model_dir}")
    model = joblib.load(model_path)

    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(model_dir, 'scaler_features.pkl')
    feature_scaler = joblib.load(scaler_path)

    target_scaler = None
    target_scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    if os.path.exists(target_scaler_path):
        target_scaler = joblib.load(target_scaler_path)

    return model, config, feature_scaler, target_scaler


def auto_detect_test_path(config: dict) -> str | None:
    """
    Auto-detect test data from the training configuration. Search order:
    (1) train_path stored in the config (repo-relative, or absolute in older models),
    (2) data_preparation/prepared_data/{data_id}_*, (3) family directories holding older
    dataset copies (last, because they may be stale).
    """
    train_path = resolve_repo_path(config.get('train_path', ''))
    data_id = config.get('data_id', 'UNKNOWN')

    if os.path.isdir(train_path) and os.path.exists(os.path.join(train_path, 'metadata.json')):
        if os.path.exists(os.path.join(train_path, 'test.csv')):
            return train_path

    if data_id in ('RAW', 'UNKNOWN'):
        return None

    def _search_prep_dir(base: str) -> str | None:
        if not os.path.exists(base):
            return None
        matching = [
            d for d in os.listdir(base)
            if d.startswith(data_id) and os.path.isdir(os.path.join(base, d))
        ]
        if not matching:
            return None
        candidate = os.path.join(base, sorted(matching)[-1])
        if os.path.exists(os.path.join(candidate, 'test.csv')):
            return candidate
        return None

    project_root = os.path.dirname(RFConfig.BASE_DIR)

    for base in (
        os.path.join(project_root, 'data_preparation', 'prepared_data'),
        RFConfig.PREPARED_DATA_DIR,
        os.path.join(project_root, 'xgb_model', 'prepared_data'),
    ):
        res = _search_prep_dir(base)
        if res:
            return res

    return None


def test_model(
    model_dir: str | None = None,
    test_path: str | None = None,
    *,
    test_file_type: str = 'test',
    simulation_profile: str = 'rf_simulation_v2',
    fee_bps: float = 2.0,
    auxiliary_fee_bps: float = 2.0,
    results_format: str = 'parquet',
    compute_permutation_importance: bool = False,
) -> dict:
    """Test Random Forest model on test data with Simulation v2 evaluation."""
    if model_dir is None:
        if not os.path.exists(RFConfig.MODELS_DIR):
            raise FileNotFoundError(f"Models directory not found: {RFConfig.MODELS_DIR}")
        model_dirs = [
            d for d in os.listdir(RFConfig.MODELS_DIR)
            if d.startswith('R') and os.path.isdir(os.path.join(RFConfig.MODELS_DIR, d))
        ]
        if not model_dirs:
            raise FileNotFoundError(f"No trained models found in {RFConfig.MODELS_DIR}")
        model_dir = os.path.join(RFConfig.MODELS_DIR, sorted(model_dirs)[-1])
        print(f"Auto-selected latest model: {os.path.basename(model_dir)}")

    if not os.path.isabs(model_dir) and not os.path.exists(model_dir):
        model_dir = os.path.join(RFConfig.MODELS_DIR, model_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_id = extract_model_id(model_dir)

    print("\n" + "=" * 80)
    print("RANDOM FOREST MODEL TESTING & SIMULATION v2")
    print("=" * 80)

    model, config, feature_scaler, target_scaler = load_model(model_dir)
    data_id = config.get('data_id', 'UNKNOWN')
    symbol = config.get('symbol', 'BTC')

    print(f"Model ID:   {model_id}")
    print(f"Data ID:    {data_id}")
    print(f"Symbol:     {symbol.upper()}")
    print(f"Horizon:    t+{config['horizon']}")
    print(f"Features:   {config['n_features']} -> {config['n_flat_features']} flat")
    print(f"Flattening: {config['flattening_method']}")

    if test_path is None:
        test_path = auto_detect_test_path(config)
        if test_path is None:
            raise FileNotFoundError(f"Cannot auto-detect test data for model {model_id}.")
        print(f"\n[OK] Auto-detected test data: {test_path}")
    else:
        print(f"\nTest data: {test_path}")

    test_id = get_next_test_id(RFConfig.LOGS_DIR)
    date_short = datetime.now().strftime('%d%m')
    run_name = f"{test_id}_test_results_{model_id}_{data_id}_{date_short}"
    output_dir = os.path.join(RFConfig.LOGS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nTest ID:    {test_id}")
    print(f"Output:     {output_dir}")

    # Load data with close price
    data = None
    if os.path.isdir(test_path):
        data, metadata, close_full = load_test_data_only(
            test_path,
            flattening_method=config['flattening_method'],
            feature_set=config.get('feature_set'),
            custom_features=config.get('custom_features'),
            lookback=config['lookback'],
            horizon=config['horizon'],
            test_file_type=test_file_type,
            return_close=True,
            features=config.get('features'),
        )
        X_test = data['X_test']
        y_test = data['y_test']
    elif test_path.endswith('.csv'):
        X_test, y_test, metadata, close_full = load_test_csv(
            test_path, config, feature_scaler, return_close=True
        )
    else:
        raise ValueError(f"Invalid test_path: {test_path}")

    print(f"Samples:    {len(X_test)}")

    # Predictions
    y_pred_scaled = model.predict(X_test)
    if target_scaler is not None:
        y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
    else:
        y_pred_log = y_pred_scaled

    # Regression metrics
    metrics = calculate_metrics(y_test, y_pred_log)
    # One threshold T for threshold metrics, simulation and saved signals. Models trained
    # before the keys were persisted fall back to the current RFConfig default.
    dir_acc_threshold = config.get(
        'signal_threshold',
        config.get('dir_acc_threshold', RFConfig.DIR_ACC_THRESHOLD),
    )
    thresh_metrics = None
    if dir_acc_threshold is not None:
        thresh_metrics = calculate_threshold_metrics(
            y_test, y_pred_log, threshold=dir_acc_threshold
        )

    print_metrics_summary(metrics, thresh_metrics)
    interpret_metrics(metrics, thresh_metrics)

    # Simulation v2
    sim_profile = None
    rank_dict = None
    buy_signals, sell_signals = None, None

    first_valid_idx = 0
    if isinstance(data, dict):
        first_valid_idx = int(data.get('first_valid_idx', 0))
    elif isinstance(metadata, dict):
        first_valid_idx = int(metadata.get('first_valid_idx', 0))

    if close_full is not None and len(close_full) > 0:
        print("\n" + "=" * 80)
        print("SIMULATION v2 EVALUATION")
        print("=" * 80)

        # Single signal threshold T for the simulation and the saved signals.
        signal_threshold = float(dir_acc_threshold) if dir_acc_threshold else 0.001
        sim_profile = evaluate_rf_simulation_v2(
            y_test,
            y_pred_log,
            close=close_full,
            lookback=int(config['lookback']),
            horizon=int(config['horizon']),
            threshold=signal_threshold,
            primary_fee_bps=float(fee_bps),
            auxiliary_fee_bps=float(auxiliary_fee_bps),
            asset=symbol,
            first_valid_idx=first_valid_idx,
        )

        rank_dict = compute_rank_score(
            metrics=metrics,
            sim_metrics=sim_profile['primary'],
            sig_metrics=sim_profile['primary'],
            pred_log_returns=y_pred_log,
            true_log_returns=y_test,
        )

        # Compute signals for saving results
        threshold_val = signal_threshold
        from common.signals import compute_sigma_series, align_sigma
        raw_sigma = compute_sigma_series(close_full, window=100)
        aligned_sigma = align_sigma(
            raw_sigma,
            offset=int(config['lookback']) + first_valid_idx,
            n_samples=len(y_pred_log),
        )
        buy_signals, sell_signals = compute_signals(y_pred_log, aligned_sigma, threshold_val)

        print(f"  Rank Score:   {rank_dict.get('rank_score', 0.0):.2f}")
        print(f"  Signal Class: {rank_dict.get('signal_class', 'UNKNOWN')}")

    # Permutation importance if requested
    perm_results = None
    if compute_permutation_importance:
        print("\n" + "=" * 80)
        print("COMPUTING GROUPED PERMUTATION IMPORTANCE")
        print("=" * 80)
        from common.feature_importance import compute_permutation_importance_grouped
        flat_feats = metadata.get('flat_features', config.get('flat_features', []))
        base_feats = config.get('features', [])
        perm_results = compute_permutation_importance_grouped(
            model, X_test, y_test, flat_feats, base_feats,
            target_scaler=target_scaler, random_state=42
        )
        with open(os.path.join(output_dir, 'permutation_importance.json'), 'w') as f:
            json.dump(perm_results, f, indent=2)
        print("  [OK] permutation_importance.json")

    # Visualizations
    plot_test_results(y_test, y_pred_log, output_dir, config['horizon'])
    flat_features = config.get('flat_features')
    if flat_features:
        plot_feature_importance(model, flat_features, output_dir, top_n=30)

    # Save results DataFrame
    res_dict = {
        'actual_log': y_test,
        'predicted_log': y_pred_log,
        'error': y_pred_log - y_test,
    }
    if buy_signals is not None and sell_signals is not None:
        res_dict['signal_buy'] = buy_signals
        res_dict['signal_sell'] = sell_signals
        pos = np.where(buy_signals >= 60.0, 1.0, np.where(sell_signals >= 60.0, -1.0, 0.0))
        res_dict['signal_pos'] = pos
        if sim_profile and 'sim_positions' in sim_profile.get('primary', {}):
            sim_p = sim_profile['primary']['sim_positions']
            horizon_step = max(int(config.get('horizon', 12)), 1)
            step_indices = np.arange(0, len(pos), horizon_step)
            sim_pos_dense = np.zeros(len(pos), dtype=np.float64)
            for s_idx, p_val in zip(step_indices, sim_p):
                sim_pos_dense[s_idx:s_idx + horizon_step] = p_val
            res_dict['sim_pos'] = sim_pos_dense

    res_df = pd.DataFrame(res_dict)
    res_path = save_results(output_dir, res_df, results_format=results_format)
    print(f"  [OK] {res_path.name}")

    # Save metrics JSON
    metrics_out = dict(metrics)
    if thresh_metrics:
        metrics_out['threshold_metrics'] = thresh_metrics
    if sim_profile:
        metrics_out['simulation_v2'] = sim_profile
        # Surface primary simulation metrics at top level for ranking and reporting tools
        prim = sim_profile.get('primary', {})
        for k in [
            'sim_return_pct', 'sim_final_value', 'sim_n_trades', 'sim_capture_log',
            'sim_capture_ratio', 'pct_strong', 'calib_buy', 'calib_sell',
            'dir_acc_strong', 'mean_spread'
        ]:
            if k in prim and k not in metrics_out:
                metrics_out[k] = prim[k]
    if rank_dict:
        metrics_out.update(rank_dict)
    metrics_out['test_file_type'] = test_file_type
    metrics_out['results_format'] = results_format

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print("  [OK] metrics.json")

    # Save summary.txt
    summary_txt = build_rf_test_summary(
        model_id=model_id,
        test_id=test_id,
        metrics=metrics,
        thresh_metrics=thresh_metrics,
        sim_profile=sim_profile,
        rank_score_dict=rank_dict,
        test_file_type=test_file_type,
        results_format=results_format,
        results_path=res_path,
    )
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_txt)
    print("  [OK] summary.txt")

    print("\n" + "=" * 80)
    print(f"TEST COMPLETE -- {test_id} ({model_id})")
    print("=" * 80)

    return metrics_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Random Forest model with Simulation v2')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--test_file_type', type=str, default='test',
                        choices=['test', 'agent', 'sym', 'val', 'train'])
    parser.add_argument('--simulation_profile', type=str, default='rf_simulation_v2')
    parser.add_argument('--fee_bps', type=float, default=2.0)
    parser.add_argument('--auxiliary_fee_bps', type=float, default=2.0)
    parser.add_argument('--results_format', type=str, default='parquet', choices=['parquet', 'csv'])
    parser.add_argument('--compute_permutation_importance', action='store_true', default=False)

    args = parser.parse_args()

    test_model(
        model_dir=args.model_dir,
        test_path=args.test_path,
        test_file_type=args.test_file_type,
        simulation_profile=args.simulation_profile,
        fee_bps=args.fee_bps,
        auxiliary_fee_bps=args.auxiliary_fee_bps,
        results_format=args.results_format,
        compute_permutation_importance=args.compute_permutation_importance,
    )
