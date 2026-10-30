"""
LSTM Signal Model — Testing / Evaluation

Loads a trained signal model, runs inference on test data,
computes signal metrics + P&L simulation, and saves results.
"""
import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from signal_config import SignalConfig
from common.model import SignalModel, build_signal_model
from common.data_utils import load_signal_data
from common.metrics import calculate_signal_metrics, compute_signal_rank_score
from common.simulation import (
    evaluate_signal_simulation,
    evaluate_signal_simulation_v2,
)
from common.test_data_utils import (
    resolve_model_dir,
    auto_detect_data_path,
    extract_signal_model_id,
    run_inference,
    save_signal_results,
)
from common.visualization import (
    plot_signal_test_results,
    plot_simulation_equity,
    plot_feature_importance,
)
from common.wandb_utils import ExperimentTracker
from common.feature_importance import compute_permutation_importance
from common.path_utils import resolve_repo_path

from common.torch_utils import set_seed, get_device
from common.id_utils import (
    get_next_id, extract_data_id, detect_symbol,
)


def _build_model_from_config(config, device):
    """Reconstruct model from saved model_config.json dict using canonical factory."""
    return build_signal_model(config, input_size=config['n_features'], device=device)


def test_model(model_dir=None, data_path=None, seed=None, batch_size=4096,
               min_signal=None, use_wandb=False,
               permutation_importance=True, val_source=None,
               fee_bps=None, multi_window=False, checkpoint='best',
               threshold_mode='absolute', percentile_cutoff=5.0, zscore_cutoff=2.0,
               position_mode='sticky', oracle_threshold_mode=None,
               oracle_percentile_cutoff=None, oracle_zscore_cutoff=None,
               submodel_mode_override=None, allow_partial_feature_set=False,
               simulation_profile='v2', auxiliary_fee_bps=2.0,
               results_format=None, results_compression=None):
    """
    Evaluate a trained signal model on test data.

    Args:
        model_dir:  Path or SIG### id (auto-select latest if None)
        data_path:  Override data path (auto-detect from config if None)
        seed:       Override seed
        batch_size: Inference batch size
        min_signal: Override MIN_SIGNAL threshold for evaluation/simulation
        use_wandb:  Enable opt-in W&B logging (no-op if wandb unavailable)
        permutation_importance: Compute + log per-feature permutation importance
        val_source: Override CSV split to evaluate: 'test' (default) | 'sym'
        fee_bps:    Primary net fee in bps (one-way per trade, two-leg flips).
                    Default: 2 for v2, 0 for single. Explicit values still apply.
        multi_window: When True, sweep window_sigma values defined in
                    SignalConfig.MULTI_WINDOW_SIGMAS and report per-window
                    metrics alongside main results.
        checkpoint: 'best' (signal_model.pth) | 'swa' (signal_model_swa.pth)
        submodel_mode_override: override the ablation mode stored in
                    model_config.json ('real' | 'shuffle' | 'noise' | 'zero').
                    None = use the mode the model was trained with, so the test
                    stays consistent with training. Used by ablation scripts to
                    re-evaluate an existing model under a different mode.
        allow_partial_feature_set: allow loading data that is missing some of the
                    features stored in model_config.json (old truncating
                    behaviour; normally a hard error)

    Returns:
        dict with all metrics + simulation results + rank_score
    """
    SignalConfig.ensure_dirs()
    results_format = results_format or getattr(SignalConfig, 'DEFAULT_RESULTS_FORMAT', 'parquet')
    results_compression = results_compression or getattr(SignalConfig, 'DEFAULT_RESULTS_COMPRESSION', 'snappy')

    model_dir = resolve_model_dir(model_dir, SignalConfig.MODELS_DIR)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    seed_val = seed or config.get('seed', 42)
    set_seed(seed_val)
    device = get_device()

    model = _build_model_from_config(config, device)
    if checkpoint == 'swa':
        weights_path = os.path.join(model_dir, 'signal_model_swa.pth')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"SWA weights checkpoint not found at: {weights_path}")
        print(f"Loading SWA weights from: {weights_path}")
    else:
        weights_path = os.path.join(model_dir, 'signal_model.pth')
        print(f"Loading best weights from: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    if data_path is None:
        data_path = auto_detect_data_path(config, SignalConfig.BASE_DIR)
    if not data_path:
        data_path = resolve_repo_path(config.get('train_path', ''))
    if not data_path or not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data path not found: '{data_path}'. Provide --data_path.")

    lookback = config['lookback']
    horizon = config.get('horizon', 1)
    if min_signal is not None:
        _min_signal = float(min_signal)
    else:
        _min_signal = float(config.get('min_signal', SignalConfig.MIN_SIGNAL))
    start_cash = float(SignalConfig.SIM_START_CASH)
    _fee_bps = float(fee_bps) if fee_bps is not None else (
        2.0 if simulation_profile == 'v2' else 0.0)

    submodel_dirs = config.get('submodel_dirs', [])
    from common.submodel_manager import resolve_submodel_feature_set
    submodel_feat_set = resolve_submodel_feature_set(
        submodel_dirs, symbol=config.get('symbol', ''), explicit_set=config.get('submodel_feature_set')
    )

    data = load_signal_data(
        data_path,
        feature_set=config.get('feature_set'),
        custom_features=config.get('features'),
        submodel_feature_set=submodel_feat_set if submodel_dirs else None,
        lookback=lookback,
        horizon=horizon,
        val_source=val_source or 'test',
        signal_threshold=config.get('signal_threshold', 0.001),
        window_sigma=config.get('window_sigma', 20),
        canonical_oracle_params={
            'window_sigma': SignalConfig.CANONICAL_WINDOW_SIGMA,
            'signal_threshold': SignalConfig.CANONICAL_SIGNAL_THRESHOLD,
        },
        feature_round_decimals=config.get('feature_round_decimals'),
        allow_partial_feature_set=bool(allow_partial_feature_set),
    )
    X_test = data['X_val']
    target_test = data['target_val']
    y_true_lr = data['y_true_val']

    # Ablation mode must match training unless explicitly overridden, otherwise
    # a model trained on shuffled submodel inputs would be tested on real ones.
    submodel_mode = submodel_mode_override or config.get('submodel_mode', 'real')

    # Pre-compute submodel predictions if model was trained with submodels
    submodel_preds_test = None
    if submodel_dirs:
        from common.submodel_manager import SubmodelManager, apply_submodel_mode
        print(f"Loading {len(submodel_dirs)} submodel(s) for inference...")
        # Existing models pre-date the guard, so stay permissive unless the model
        # was trained with it enabled.
        strict = not bool(config.get('allow_submodel_data_mismatch', 1))
        sm = SubmodelManager(
            submodel_dirs, device,
            expected_symbol=config.get('symbol'),
            expected_data_id=config.get('data_id'),
            max_missing_ratio=config.get('submodel_max_missing_ratio', 0.30),
            expected_scaler=(data['metadata'].get('scalers') or {}).get('features'),
            strict=strict,
        )
        sub_X_test = data.get('X_val_submodel', X_test)
        sub_feats = data.get('submodel_features', data['features'])
        submodel_preds_test = sm.precompute_predictions(sub_X_test, sub_feats)
        print(f"  Submodel preds: {submodel_preds_test.shape}")
        sm.diagnose(submodel_preds_test, 'test')
        if submodel_mode != 'real':
            print(f"  Applying submodel ablation mode '{submodel_mode}'")
            submodel_preds_test = apply_submodel_mode(
                submodel_preds_test, submodel_mode, seed_val, rng_offset=0)
        del sm
    elif int(config.get('null_submodel_size', 0) or 0) > 0:
        n_null = int(config['null_submodel_size'])
        submodel_preds_test = np.zeros((len(X_test), n_null), dtype=np.float32)
        print(f"  Null submodel control: {submodel_preds_test.shape} (all zeros)")

    print(f"\nPredicting (batch_size={batch_size}, samples={len(X_test):,})...")
    y_pred = run_inference(model, X_test, device, batch_size,
                           submodel_preds=submodel_preds_test)

    # Simulation uses canonical oracle so sim_final_oracle is comparable across
    # models trained with different window_sigma / signal_threshold values.
    target_canonical = data.get('target_val_canonical', target_test)
    if simulation_profile == 'v2':
        # Primary metrics and RankScore are always absolute=20 gross. MAE/corr
        # retain the model-specific oracle, while simulation uses canonical.
        metrics = calculate_signal_metrics(
            y_pred, target_test, y_true_lr, min_signal_pct=20.0,
            threshold_mode='absolute')
        asset = config.get('symbol') or config.get('data_id') or data_path
        sim_result = evaluate_signal_simulation_v2(
            y_pred, target_canonical, y_true_lr, horizon=horizon,
            asset=asset, start_cash=start_cash,
            primary_fee_bps=_fee_bps,
            auxiliary_fee_bps=float(auxiliary_fee_bps))
        rank_threshold_mode = 'absolute'
        min_signal_eval = 20.0
    else:
        # Explicit compatibility mode for diagnostic tools.
        metrics = calculate_signal_metrics(
            y_pred, target_test, y_true_lr, min_signal_pct=_min_signal,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff)
        sim_result = evaluate_signal_simulation(
            y_pred, target_canonical, y_true_lr,
            horizon=horizon,
            min_signal=_min_signal,
            start_cash=start_cash,
            fee_bps=_fee_bps,
            threshold_mode=threshold_mode,
            percentile_cutoff=percentile_cutoff,
            zscore_cutoff=zscore_cutoff,
            oracle_threshold_mode=oracle_threshold_mode,
            oracle_percentile_cutoff=oracle_percentile_cutoff,
            oracle_zscore_cutoff=oracle_zscore_cutoff,
            position_mode=position_mode,
        )
        rank_threshold_mode = threshold_mode
        min_signal_eval = _min_signal
    metrics.update(sim_result)
    metrics['min_signal_eval'] = min_signal_eval
    metrics['submodel_mode_eval'] = submodel_mode

    rank_result = compute_signal_rank_score(
        metrics, threshold_mode=rank_threshold_mode)
    metrics.update(rank_result)

    # D2: multi_window sweep — re-run simulation with different oracle window sigmas
    multi_window_results = {}
    if multi_window:
        window_sigmas = getattr(
            SignalConfig, 'MULTI_WINDOW_SIGMAS',
            [10, 20, 50, 100])
        print(f"\nMulti-window sweep: {window_sigmas}")
        for ws in window_sigmas:
            try:
                mw_data = load_signal_data(
                    data_path,
                    feature_set=config.get('feature_set'),
        custom_features=config.get('features'),
                    lookback=lookback,
                    horizon=horizon,
                    val_source=val_source or 'test',
                    signal_threshold=config.get('signal_threshold', 0.001),
                    window_sigma=ws,
                    canonical_oracle_params={
                        'window_sigma': SignalConfig.CANONICAL_WINDOW_SIGMA,
                        'signal_threshold': SignalConfig.CANONICAL_SIGNAL_THRESHOLD,
                    },
                    feature_round_decimals=config.get('feature_round_decimals'),
                    allow_partial_feature_set=bool(allow_partial_feature_set),
                )
                mw_oracle = mw_data.get('target_val_canonical',
                                        mw_data['target_val'])
                mw_sim = evaluate_signal_simulation(
                    y_pred, mw_oracle, y_true_lr,
                    horizon=horizon, min_signal=_min_signal,
                    start_cash=start_cash, fee_bps=_fee_bps,
                )
                mw_metrics = calculate_signal_metrics(
                    y_pred, mw_data['target_val'], y_true_lr,
                    min_signal_pct=_min_signal)
                mw_metrics.update(mw_sim)
                mw_rank = compute_signal_rank_score(mw_metrics)
                mw_metrics.update(mw_rank)
                multi_window_results[ws] = {
                    'rank_score': mw_metrics.get('rank_score'),
                    'sim_capture_log': mw_sim.get('sim_capture_log'),
                    'sim_return_model_pct': mw_sim.get('sim_return_model_pct'),
                    'sim_return_oracle_pct': mw_sim.get('sim_return_oracle_pct'),
                    'signal_class': mw_metrics.get('signal_class'),
                }
                print(f"  w={ws:4d}: rank={mw_metrics.get('rank_score', 0):.1f}"
                      f"  capture_log={mw_sim.get('sim_capture_log')}"
                      f"  [{mw_metrics.get('signal_class')}]")
            except Exception as exc:
                print(f"  w={ws}: FAILED ({exc})")
                multi_window_results[ws] = {'error': str(exc)}
        metrics['multi_window_results'] = multi_window_results

    model_id = config.get('model_id', extract_signal_model_id(model_dir))
    data_id = extract_data_id(data_path)
    test_id = get_next_id(SignalConfig.DATA_DIR, 'SIGT', '_test_results_')
    timestamp = datetime.now().strftime('%d%m')
    result_dir = os.path.join(
        SignalConfig.DATA_DIR,
        f"{test_id}_test_results_{model_id}_{data_id}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    metrics['_y_pred'] = y_pred
    metrics['_target_test'] = target_test
    metrics['_target_canonical'] = target_canonical
    metrics['_y_true_lr'] = y_true_lr
    metrics['_horizon'] = horizon
    metrics['_result_dir'] = result_dir

    tracker = ExperimentTracker.create(
        enabled=bool(use_wandb), config=config, run_name=f"{model_id}_test")
    tracker.log_signal_metrics(
        pct_strong=metrics.get('pct_strong', 0.0),
        corr_buy=metrics.get('corr_buy', 0.0),
        corr_sell=metrics.get('corr_sell', 0.0),
        diversity=metrics.get('diversity', 0.0),
        sim=sim_result,
    )
    tracker.log_summary({
        'mae_buy': metrics.get('mae_buy'), 'mae_sell': metrics.get('mae_sell'),
        'rank_score': metrics.get('rank_score'),
        'signal_class': metrics.get('signal_class'),
        'sim_capture_log': metrics.get('sim_capture_log'),
        'sim_n_trades_model': metrics.get('sim_n_trades_model'),
    })

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
        submodel_names = None
        if submodel_dirs and submodel_preds_test is not None:
            submodel_names = []
            for d in submodel_dirs:
                sm_id = os.path.basename(str(d).rstrip(os.sep)).split('_')[0]
                submodel_names.append(f"submodel_{sm_id}")
            if len(submodel_names) != submodel_preds_test.shape[1]:
                submodel_names = [f"submodel_dim_{i+1}" for i in range(submodel_preds_test.shape[1])]

        pi_result = compute_permutation_importance(
            model, X_test, target_test, feature_names,
            submodel_preds=submodel_preds_test,
            submodel_names=submodel_names,
            device=device, batch_size=batch_size, seed=seed_val)
        importances = pi_result['importances']
        importances_by_metric = pi_result['importances_by_metric']
        tracker.log_permutation_importance(importances)
        sorted_importances = sorted(
            importances.items(), key=lambda kv: kv[1][0], reverse=True)
        print("Permutation importance (mean_delta_mae, std):")
        for name, (mean, std) in sorted_importances:
            print(f"  {name:30s}: {mean:+.6f} (+/- {std:.6f})")
        with open(os.path.join(result_dir, 'permutation_importance.json'),
                  'w') as f:
            json.dump({'model_id': model_id,
                      'importances': {k: list(v) for k, v in importances.items()},
                      'importances_by_metric': importances_by_metric},
                     f, indent=2)
        print("  permutation_importance.json")
        plot_feature_importance(importances, result_dir)

    plot_signal_test_results(
        y_pred, target_test, y_true_lr, result_dir, horizon, min_signal_eval)
    plot_simulation_equity(sim_result, result_dir, start_cash)

    print("\nSaving results...")
    summary_text = save_signal_results(
        y_pred, target_test, y_true_lr, metrics, config,
        test_id, model_id, data_id, result_dir, horizon,
        min_signal_pct=min_signal_eval, sim_result=sim_result,
        thresh_buy=metrics.get('thresh_buy_eval'),
        thresh_sell=metrics.get('thresh_sell_eval'),
        model_dir=model_dir, data_path=data_path,
        results_format=results_format, compression=results_compression)
    tracker.finish()

    print(f"\nResults saved to: {result_dir}")
    print(summary_text)

    return metrics


# backward-compat alias
test = test_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Signal Model — Test')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Path or SIG### id (auto-select latest if omitted)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override prepared data directory')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--min_signal', type=float, default=None,
                        help='Override MIN_SIGNAL threshold for '
                             'evaluation/simulation (does not affect training)')
    parser.add_argument('--use_wandb', type=int, default=0, choices=[0, 1],
                        help='Enable Weights & Biases logging (0/1, default 0)')
    parser.add_argument('--permutation_importance', type=int, default=1,
                        choices=[0, 1],
                        help='Compute + log per-feature permutation '
                             'importance (0/1, default 1)')
    parser.add_argument('--val_source', type=str, default=None,
                        choices=['test', 'sym'],
                        help="CSV split to evaluate: 'test' (default) | 'sym'")
    parser.add_argument('--fee_bps', type=float, default=None,
                        help='Primary net fee in bps (v2 default 2; single default 0)')
    parser.add_argument('--auxiliary_fee_bps', type=float, default=2.0,
                        help='Net fee for auxiliary v2 variants (default 2; 0 disables cost)')
    parser.add_argument('--simulation_profile', type=str, default='v2',
                        choices=['v2', 'single'],
                        help='v2 full profile (default) or compatible single mode')
    parser.add_argument('--multi_window', type=int, default=0, choices=[0, 1],
                        help='Sweep window_sigma values for oracle robustness (0/1, default 0)')
    parser.add_argument('--checkpoint', type=str, default='best',
                        choices=['best', 'swa'],
                        help="Checkpoint weights to load: 'best' (signal_model.pth) | 'swa' (signal_model_swa.pth)")
    parser.add_argument('--threshold_mode', type=str, default='absolute',
                        choices=['absolute', 'percentile', 'zscore'],
                        help="Signal threshold calculation mode (default: absolute)")
    parser.add_argument('--percentile_cutoff', type=float, default=5.0,
                        help="Top X percent cutoff for percentile mode (default: 5.0)")
    parser.add_argument('--zscore_cutoff', type=float, default=2.0,
                        help="Z-score cutoff for zscore mode (default: 2.0)")
    parser.add_argument('--position_mode', type=str, default='sticky',
                        choices=['sticky', 'reset'],
                        help="Position hold behavior when signal drops: sticky (default) | reset")
    parser.add_argument('--oracle_threshold_mode', type=str, default=None,
                        choices=['absolute', 'percentile', 'zscore'],
                        help="Oracle threshold mode (default: match threshold_mode)")
    parser.add_argument('--oracle_percentile_cutoff', type=float, default=None)
    parser.add_argument('--oracle_zscore_cutoff', type=float, default=None)
    parser.add_argument('--submodel_mode', type=str, default=None,
                        choices=['real', 'shuffle', 'noise', 'zero'],
                        help='Override the submodel ablation mode stored in '
                             'model_config.json (default: use the trained mode)')
    parser.add_argument('--allow_partial_feature_set', type=int, default=0,
                        choices=[0, 1],
                        help='Allow evaluating on a dataset that lacks some of the '
                             'features stored in model_config.json (0/1, default 0)')
    parser.add_argument('--results_format', type=str, default='parquet',
                        choices=['parquet', 'csv'],
                        help="Format to save results: 'parquet' (default, compressed) or 'csv'.")
    parser.add_argument('--results_compression', type=str, default='snappy',
                        choices=['snappy', 'zstd', 'gzip', 'none'],
                        help="Compression codec for parquet results (default: snappy).")

    args = parser.parse_args()
    test_model(
        model_dir=args.model_dir,
        data_path=args.data_path,
        seed=args.seed,
        batch_size=args.batch_size,
        min_signal=args.min_signal,
        use_wandb=bool(args.use_wandb),
        permutation_importance=bool(args.permutation_importance),
        val_source=args.val_source,
        fee_bps=args.fee_bps,
        multi_window=bool(args.multi_window),
        checkpoint=args.checkpoint,
        threshold_mode=args.threshold_mode,
        percentile_cutoff=args.percentile_cutoff,
        zscore_cutoff=args.zscore_cutoff,
        position_mode=args.position_mode,
        oracle_threshold_mode=args.oracle_threshold_mode,
        oracle_percentile_cutoff=args.oracle_percentile_cutoff,
        oracle_zscore_cutoff=args.oracle_zscore_cutoff,
        submodel_mode_override=args.submodel_mode,
        allow_partial_feature_set=bool(args.allow_partial_feature_set),
        simulation_profile=args.simulation_profile,
        auxiliary_fee_bps=args.auxiliary_fee_bps,
        results_format=args.results_format,
        results_compression=args.results_compression,
    )
