"""
LSTM Log-Return Prediction — Testing

Test a trained model on held-out data. Supports:
  - Auto-detection of latest model and test data
  - VAL_SOURCE awareness (test.csv vs sym.csv)
  - Prepared CSV directories and raw CSV files
  - Full metrics + visualizations
"""
import os
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import joblib
from tqdm import tqdm

from lstm_config import LSTMConfig
from common.model import LSTMLogReturnModel
from common.target_utils import TargetScaler
from common.metrics import (
    calculate_metrics, print_metrics_summary, interpret_metrics,
)
from common.training_utils import _apply_precision
from common.visualization import plot_test_results, plot_feature_importance
from common.log_utils import log_run_header, log_test_complete
from common.test_data_utils import load_test_data, save_results
from common.test_eval import (
    evaluate_signal_metrics as _evaluate_signal_metrics,
    multi_window_eval as _multi_window_eval,
    posthoc_calibrator_metrics as _posthoc_calibrator_metrics,
)
from common.id_utils import (
    get_next_test_id, extract_model_id, extract_data_id,
)
from common.path_utils import resolve_repo_path
from common.wandb_utils import ExperimentTracker
from common.feature_importance import compute_permutation_importance


# =========================================================================
# MODEL LOADING
# =========================================================================

def load_model(model_dir, checkpoint='best'):
    """
    Load trained LSTM model and all artifacts.

    Args:
        checkpoint: 'best' loads lstm_model.pth; 'swa' loads lstm_model_swa.pth.

    Returns:
        model, config, feature_scaler, target_scaler, device
    """
    config_path = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"model_config.json not found in {model_dir}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMLogReturnModel(
        input_size=config['n_features'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False),
        use_attention=config.get('use_attention', False),
        fc_layers=config.get('fc_layers', [64]),
        activation=config.get('activation', 'relu'),
        fc_norm=config.get('fc_norm', 'layer'),
        arch_type=config.get('arch_type', 'legacy'),
        use_input_norm=config.get('use_input_norm', False),
        rnn_type=config.get('rnn_type', 'lstm'),
        attention_heads=config.get('attention_heads', 1),
        lstm_dropout=config.get('lstm_dropout', None),
        fc_dropout=config.get('fc_dropout', None),
        use_feature_attention=config.get('use_feature_attention', False),
        use_rnn_norm=config.get('use_rnn_norm', False),
        pooling_mode=config.get('pooling_mode', 'last'),
        use_conv1d_front=config.get('use_conv1d_front', False),
        conv1d_kernel=config.get('conv1d_kernel', 5),
        use_fc_residual=config.get('use_fc_residual', False),
        use_learnable_output_scale=config.get(
            'use_learnable_output_scale', False),
    )

    pth_name = 'lstm_model_swa.pth' if checkpoint == 'swa' else 'lstm_model.pth'
    pth_path = os.path.join(model_dir, pth_name)
    if not os.path.exists(pth_path):
        raise FileNotFoundError(
            f"{pth_name} not found in {model_dir} (checkpoint={checkpoint!r})"
        )
    state_dict = torch.load(pth_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    feature_scaler = joblib.load(
        os.path.join(model_dir, 'feature_scaler.pkl')
    )

    target_scaler = None
    ts_path = os.path.join(model_dir, 'target_scaler.pkl')
    if os.path.exists(ts_path):
        target_scaler = joblib.load(ts_path)

    return model, config, feature_scaler, target_scaler, device


# =========================================================================
# AUTO-DETECTION: MODEL
# =========================================================================

def auto_select_model_dir():
    """Auto-select latest trained model from lstm_lr_models/."""
    base_dir = LSTMConfig.MODELS_DIR
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"No models directory found: {base_dir}\n"
            "Train a model first: python lstm_train.py --data_path <path>"
        )

    model_dirs = sorted([
        d for d in os.listdir(base_dir)
        if d.startswith('LR') and not d.startswith('_')
        and os.path.isdir(os.path.join(base_dir, d))
        and os.path.exists(os.path.join(base_dir, d, 'model_config.json'))
    ])

    if not model_dirs:
        raise FileNotFoundError(
            f"No trained models found in {base_dir}\n"
            "Train a model first: python lstm_train.py --data_path <path>"
        )

    latest = os.path.join(base_dir, model_dirs[-1])
    print(f"✓ Auto-selected latest model: {latest}")
    return latest


# =========================================================================
# AUTO-DETECTION: TEST DATA
# =========================================================================

def auto_detect_test_path(config):
    """
    Auto-detect test data path based on model config.

    Strategy:
      1. Model's training directory (train_path; relative values are resolved
         against the repo root) — test.csv or sym.csv
      2. Search data_preparation/prepared_data/ by data_id (canonical location)
      3. Sibling test CSV

    Returns:
        (test_path, test_file_type) or (None, None)
    """
    train_path = resolve_repo_path(config.get('train_path', ''))
    data_id = config.get('data_id', '')
    val_source = config.get('val_source', 'split')

    target_csv = 'sym.csv' if val_source == 'test' else 'test.csv'
    test_file_type = 'sym' if val_source == 'test' else 'test'

    # Strategy 1: model's training directory
    if train_path and os.path.isdir(train_path):
        if os.path.exists(os.path.join(train_path, target_csv)):
            print(f"✓ Auto-detected: {target_csv} in training directory")
            return train_path, test_file_type

        fallback_csv = 'test.csv' if target_csv == 'sym.csv' else 'sym.csv'
        fallback_type = 'test' if test_file_type == 'sym' else 'sym'
        if os.path.exists(os.path.join(train_path, fallback_csv)):
            print(f"⚠ {target_csv} not found, falling back to {fallback_csv}")
            return train_path, fallback_type

    # Strategy 2: search by data_id
    if data_id and data_id != 'UNKNOWN':
        search_dirs = [
            os.path.join(LSTMConfig.BASE_DIR, '..', 'data_preparation', 'prepared_data'),
        ]
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for c in sorted(os.listdir(search_dir), reverse=True):
                if not c.startswith(data_id):
                    continue
                cpath = os.path.join(search_dir, c)
                if os.path.isdir(cpath) and os.path.exists(
                        os.path.join(cpath, target_csv)):
                    print(f"✓ Auto-detected by data_id ({data_id}): {cpath}")
                    return cpath, test_file_type

    # Strategy 3: sibling test CSV
    if train_path and train_path.endswith('.csv'):
        test_csv = os.path.join(
            os.path.dirname(train_path) or '.',
            os.path.basename(train_path).replace('train', 'test'),
        )
        if os.path.exists(test_csv):
            print(f"✓ Auto-detected sibling test CSV: {test_csv}")
            return test_csv, 'test'

    return None, None


# =========================================================================
# INFERENCE
# =========================================================================

def run_inference(model, X_test, device, batch_size=1024):
    """Generate predictions in batches."""
    model.eval()
    predictions = []

    n_batches = (len(X_test) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), batch_size),
                      desc="Predicting", total=n_batches):
            X_batch = torch.FloatTensor(X_test[i:i + batch_size]).to(device)
            out = model(X_batch).squeeze(-1).cpu().numpy()
            if out.ndim == 0:
                predictions.append(out.item())
            else:
                predictions.extend(out.tolist())

    return np.array(predictions, dtype=np.float32)


# =========================================================================
# MAIN TEST FUNCTION
# =========================================================================

def test_model(model_dir=None, test_path=None, test_file_type=None,
               batch_size=1024, feature_round_decimals=None,
               target_round_decimals=None, checkpoint='best',
               multi_window=0, fee_bps=None, use_wandb=False,
               permutation_importance=True,
               threshold_mode='absolute', percentile_cutoff=5.0,
               zscore_cutoff=1.6, position_mode='sticky',
               simulation_profile='v2', auxiliary_fee_bps=2.0,
               results_format=None, results_compression=None):
    """Test a trained LSTM model on held-out data."""
    results_format = results_format or getattr(LSTMConfig, 'DEFAULT_RESULTS_FORMAT', 'parquet')
    results_compression = results_compression or getattr(LSTMConfig, 'DEFAULT_RESULTS_COMPRESSION', 'snappy')

    # ── Auto-select model ──
    if model_dir is None:
        model_dir = auto_select_model_dir()
    elif os.path.sep not in model_dir and not os.path.exists(model_dir):
        model_dir = os.path.join(LSTMConfig.MODELS_DIR, model_dir)

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # ── Load model ──
    log_run_header("LSTM LOG-RETURN MODEL TEST")
    print(f"\nLoading model: {model_dir}")

    model, config, feature_scaler, target_scaler, device = load_model(
        model_dir, checkpoint=checkpoint)
    model_id = extract_model_id(model_dir)
    data_id = extract_data_id(config)
    horizon = config['horizon']

    print(f"Model ID:   {model_id}")
    print(f"Data ID:    {data_id}")
    print(f"Horizon:    t+{horizon}, Lookback: {config['lookback']}")
    print(f"Features:   {config['n_features']}, Device: {device}")
    ts_str = f"ENABLED ({target_scaler.method})" if target_scaler else "DISABLED"
    print(f"Target scl: {ts_str}")

    # ── Auto-detect test data ──
    if test_file_type is None:
        test_file_type = 'test'
    if test_path is None:
        test_path, detected_type = auto_detect_test_path(config)
        if test_path is None:
            raise FileNotFoundError(
                f"Cannot auto-detect test data.\n"
                f"Model trained on: {config.get('train_path', 'UNKNOWN')}\n"
                f"Data ID: {data_id}, VAL_SOURCE: {config.get('val_source')}\n"
                "Provide: --test_path <prepared_data_directory>"
            )
        if test_file_type == 'test':
            test_file_type = detected_type
    elif test_file_type == 'test' and os.path.isdir(test_path):
        val_source = config.get('val_source', 'split')
        if val_source == 'test' and os.path.exists(
                os.path.join(test_path, 'sym.csv')):
            test_file_type = 'sym'

    print(f"\nTest data:  {test_path} ({test_file_type}.csv)")

    # ── Load test data ──
    X_test, y_test, close_full = load_test_data(
        test_path, config, feature_scaler, test_file_type,
    )

    # F1.I4b: pre-inference rounding (CLI override > model config > 0=off)
    _feat_round = (feature_round_decimals if feature_round_decimals is not None
                   else config.get('feature_round_decimals', 0))
    _tgt_round  = (target_round_decimals  if target_round_decimals  is not None
                   else config.get('target_round_decimals', 0))
    if _feat_round and _feat_round > 0:
        X_test = np.round(X_test, decimals=int(_feat_round))
    if _tgt_round and _tgt_round > 0:
        y_test = np.round(y_test, decimals=int(_tgt_round))

    # ── Output directory ──
    LSTMConfig.ensure_dirs()
    test_id = get_next_test_id()
    date_str = datetime.now().strftime('%d%m')
    run_name = f"{test_id}_test_results_{model_id}_{data_id}_{date_str}"
    output_dir = os.path.join(LSTMConfig.DATA_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Test ID:    {test_id}\nOutput:     {output_dir}")

    # ── Inference ──
    print(f"\nPredicting (batch_size={batch_size})...")
    y_pred_raw = run_inference(model, X_test, device, batch_size)

    # Inverse transform predictions to raw log-return space
    y_pred = (target_scaler.inverse_transform(y_pred_raw)
              if target_scaler else y_pred_raw)
    y_true = y_test

    # CP-175: If model was trained with TARGET_MODE=vol_norm, denormalise predictions
    # y_pred = y_pred_norm * sigma, so y_pred is back in real log-return space.
    if config.get('target_mode') == 'vol_norm' and close_full is not None:
        from common.simulation import compute_sigma_series, align_sigma
        from common.target_utils import denormalize_target_by_volatility
        window_sigma = int(getattr(LSTMConfig, 'SIGNAL_WINDOW_SIGMA', 100))
        sigma_series = compute_sigma_series(close_full, window_sigma)
        sigma_test = align_sigma(sigma_series, config['lookback'], len(y_test)).astype(np.float32)
        finite_mask = np.isfinite(sigma_test)
        if np.any(finite_mask):
            sigma_test = np.where(finite_mask, sigma_test, float(np.median(sigma_test[finite_mask])))
        y_pred = denormalize_target_by_volatility(y_pred, sigma_test)
        print("  ✓ Denormalized predictions from vol_norm space to raw log-returns")

    # ── Metrics ──
    log_run_header("EVALUATION METRICS")
    signal_threshold = config.get('signal_threshold')
    # F1.I4: post-hoc rounding to match the precision used in training metrics.
    # Reads from saved config so test results match training metrics exactly.
    y_pred = _apply_precision(y_pred, config.get('prediction_precision', 0))
    y_true = _apply_precision(y_true, config.get('target_precision', 0))
    metrics = calculate_metrics(y_true, y_pred, threshold=signal_threshold)

    # ── Signal + market simulation (usefulness for trading) ──
    # Standard regression metrics reward prediction collapse; the signal
    # metrics + $100 long/short simulation expose whether the model can
    # actually drive BUY/SELL signals. Requires the raw close series.
    _fee_bps = float(fee_bps) if fee_bps is not None else (
        2.0 if simulation_profile == 'v2' else 0.0)
    metrics.update(
        _evaluate_signal_metrics(
            y_true, y_pred, close_full, config,
            signal_threshold, fee_bps=_fee_bps,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            position_mode=position_mode,
            simulation_profile=simulation_profile,
            auxiliary_fee_bps=float(auxiliary_fee_bps),
        )
    )
    metrics.update(
        _posthoc_calibrator_metrics(
            y_true, y_pred, close_full, config, model_dir, signal_threshold)
    )
    if multi_window and int(multi_window) >= 2:
        mw = _multi_window_eval(
            y_true, y_pred, close_full, config, signal_threshold,
            int(multi_window), fee_bps=_fee_bps)
        if mw is not None:
            metrics['multi_window'] = mw
    if _fee_bps > 0:
        metrics['sim_fee_bps'] = float(_fee_bps)
    if config.get('epochs_trained') is not None:
        metrics['epochs_trained'] = config['epochs_trained']
    if config.get('best_epoch') is not None:
        metrics['best_epoch'] = config['best_epoch']
    metrics['test_file_type'] = test_file_type
    print_metrics_summary(metrics)
    interpret_metrics(metrics)

    # ── W&B (Faza 8, opt-in) ──
    tracker = ExperimentTracker.create(
        enabled=bool(use_wandb), config=config, run_name=f"{model_id}_test")
    tracker.log_test_metrics(metrics)

    feature_names = config.get('features', [])
    if getattr(model, 'feature_attention', None) is not None:
        fa_weights = model.feature_attention.last_weights
        if fa_weights is not None and feature_names:
            tracker.log_feature_attention_weights(
                fa_weights.cpu().numpy(), feature_names)
    if getattr(model, 'attention', None) is not None:
        ta_weights = model.attention.last_weights
        if ta_weights is not None:
            tracker.log_temporal_attention_weights(ta_weights.cpu().numpy())

    if permutation_importance and feature_names:
        print("\nComputing permutation importance...")
        y_true_scaled = (target_scaler.transform(y_true)
                         if target_scaler else y_true)
        importances = compute_permutation_importance(
            model, X_test, y_true_scaled, feature_names,
            device=device, batch_size=batch_size)
        tracker.log_permutation_importance(importances)
        sorted_importances = sorted(
            importances.items(), key=lambda kv: kv[1][0], reverse=True)
        print("Permutation importance (mean_delta_mae, std):")
        for name, (mean, std) in sorted_importances:
            print(f"  {name:30s}: {mean:+.6f} (+/- {std:.6f})")
        with open(os.path.join(output_dir, 'permutation_importance.json'),
                 'w') as f:
            json.dump({'model_id': model_id,
                      'importances': {k: list(v) for k, v in importances.items()}},
                     f, indent=2)
        print("  permutation_importance.json")
        plot_feature_importance(importances, output_dir)

    # ── Visualize + Save ──
    plot_test_results(y_true, y_pred, output_dir, horizon)
    print("\nSaving results...")
    save_results(y_true, y_pred, metrics, config, test_id, model_id,
                 data_id, output_dir, horizon, test_file_type,
                 results_format=results_format, compression=results_compression)
    log_test_complete(test_id, model_id, len(y_true), metrics, output_dir)
    tracker.finish()

    metrics['_result_dir'] = output_dir
    return metrics


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test LSTM log-return prediction model"
    )
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Path to model dir (auto-selects latest if omitted)")
    parser.add_argument('--test_path', type=str, default=None,
                        help="Path to test data (auto-detects if omitted)")
    parser.add_argument('--test_file_type', type=str, default=None,
                        choices=['test', 'sym', 'agent'],
                        help="CSV split to evaluate (default: test or auto sym)")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--feature_round_decimals', type=int, default=None,
                        help="Override feature rounding for test (0=off; None=from model config).")
    parser.add_argument('--target_round_decimals', type=int, default=None,
                        help="Override target rounding for test (0=off; None=from model config).")
    parser.add_argument('--checkpoint', type=str, default='best',
                        choices=['best', 'swa'],
                        help="Load lstm_model.pth (best) or lstm_model_swa.pth (swa).")
    parser.add_argument('--multi_window', type=int, default=0,
                        help="Split test chronologically into N sub-windows (0=off).")
    parser.add_argument('--fee_bps', type=float, default=None,
                        help="Transaction fee in bps per position change leg "
                             "(v2 default 2.0, single default 0.0).")
    parser.add_argument('--auxiliary_fee_bps', type=float, default=2.0,
                        help="Net fee for auxiliary v2 variants (default 2.0).")
    parser.add_argument('--simulation_profile', type=str, default='v2',
                        choices=['v2', 'single'],
                        help="v2 = four causal variants; single = legacy one-mode.")
    parser.add_argument('--use_wandb', type=int, default=0, choices=[0, 1],
                        help="Faza 8: log test run to Weights & Biases (opt-in).")
    parser.add_argument('--permutation_importance', type=int, default=1,
                        choices=[0, 1],
                        help="Faza 8: compute + log per-feature permutation importance.")
    parser.add_argument('--threshold_mode', type=str, default='absolute',
                        choices=['absolute', 'percentile', 'zscore'],
                        help="Signal thresholding mode (default: absolute 60%%).")
    parser.add_argument('--percentile_cutoff', type=float, default=5.0,
                        help="Cutoff for percentile mode (e.g. 5.0 -> top 5%%).")
    parser.add_argument('--zscore_cutoff', type=float, default=1.6,
                        help="Cutoff for zscore mode (e.g. 1.6 sigma for ETH, 1.5 for BTC).")
    parser.add_argument('--position_mode', type=str, default='sticky',
                        choices=['sticky', 'reset'],
                        help="Position holding strategy (sticky=hold until opposite, reset=flat on neutral).")
    parser.add_argument('--results_format', type=str, default='parquet',
                        choices=['parquet', 'csv'],
                        help="Format to save results: 'parquet' (default, compressed) or 'csv'.")
    parser.add_argument('--results_compression', type=str, default='snappy',
                        choices=['snappy', 'zstd', 'gzip', 'none'],
                        help="Compression codec for parquet results (default: snappy).")

    args = parser.parse_args()

    try:
        test_model(
            model_dir=args.model_dir,
            test_path=args.test_path,
            test_file_type=args.test_file_type,
            batch_size=args.batch_size,
            feature_round_decimals=args.feature_round_decimals,
            target_round_decimals=args.target_round_decimals,
            checkpoint=args.checkpoint,
            multi_window=args.multi_window,
            fee_bps=args.fee_bps,
            use_wandb=bool(args.use_wandb),
            permutation_importance=bool(args.permutation_importance),
            threshold_mode=args.threshold_mode,
            percentile_cutoff=args.percentile_cutoff,
            zscore_cutoff=args.zscore_cutoff,
            position_mode=args.position_mode,
            simulation_profile=args.simulation_profile,
            auxiliary_fee_bps=args.auxiliary_fee_bps,
            results_format=args.results_format,
            results_compression=args.results_compression,
        )
    except FileNotFoundError as e:
        print(f"\n{'='*70}\nERROR\n{'='*70}\n{e}\n{'='*70}")
    except Exception as e:
        print(f"\n{'='*70}\nERROR\n{'='*70}\n{e}")
        import traceback
        traceback.print_exc()
        raise
