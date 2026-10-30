"""
LSTM Signal Model — Training

Main training script: loads data, builds model, trains, saves.
Slim orchestrator — all logic delegated to common/ modules.
"""
import os
import sys
import numpy as np
import torch
from datetime import datetime

# ── sys.path setup so imports resolve correctly ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from signal_config import SignalConfig
from common.data_utils import load_signal_data
from common.dataset_guards import assert_fusion_compatible_dataset
from common.model import build_signal_model
from common.losses import get_signal_loss
from common.diagnostics import SignalEpochDiagnostics
from common.metrics import calculate_signal_metrics
from common.training_utils import (
    setup_logger, train_model, save_model,
)
from common.training_checkpoint import build_val_signal_ctx
from common.log_utils import (
    log_train_session_header, log_data_summary,
    log_config_flags, log_epoch_summary, log_final_summary,
)
from common.train_setup import (
    apply_config_defaults, apply_profile_overrides,
    apply_cli_overrides, apply_optuna_overrides, build_signal_loss,
)
from common.train_config_dict import build_signal_config_dict
from common.train_cli import build_parser, dispatch_args_to_train
from common.wandb_utils import ExperimentTracker

from common.torch_utils import (
    set_seed, get_device, CryptoDataset, get_scheduler,
)
from common.augmentation import SequenceAugmenter
from common.id_utils import (
    get_next_id, extract_data_id, detect_symbol,
)


def train(data_path, use_wandb=False, load_optuna_params=None, **kwargs):
    """
    Train a Signal Model end-to-end.

    Args:
        data_path: Path to prepared data directory (E104/B104)
        use_wandb: Enable opt-in W&B logging (no-op if wandb unavailable)
        load_optuna_params: Path to Optuna best_params.json (optional)
        **kwargs:  CLI overrides for SignalConfig values
    """
    # 1. Resolve config (defaults → CLI → profile → Optuna)
    # Order: defaults first, then CLI (user intent), then profile only fills
    # keys not set by CLI (via explicit_keys guard), then Optuna wins overlaps.
    resolved = apply_config_defaults(SignalConfig)
    profile = kwargs.pop('training_profile', None) or resolved['training_profile']
    resolved, explicit_keys = apply_cli_overrides(resolved, kwargs)
    resolved = apply_profile_overrides(resolved, profile, explicit_keys)
    resolved, optuna_id = apply_optuna_overrides(resolved, load_optuna_params, explicit_keys)

    SignalConfig.validate(resolved)
    # Fail before a model directory is reserved.
    assert_fusion_compatible_dataset(data_path, resolved.get('submodel_dirs') or [])

    set_seed(resolved['seed'])
    device = get_device()

    # 2. Generate model ID
    models_dir = SignalConfig.MODELS_DIR
    model_id = get_next_id(models_dir, 'SIG', '_')
    data_id = extract_data_id(data_path)
    symbol = detect_symbol(os.path.basename(data_path))
    timestamp = datetime.now().strftime('%d%m')
    if optuna_id:
        run_name = f"{model_id}_{symbol}_model_{data_id}_{optuna_id}_{timestamp}"
    else:
        run_name = f"{model_id}_{symbol}_model_{data_id}_{timestamp}"
    output_dir = os.path.join(models_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)

    tracker = ExperimentTracker.create(
        enabled=bool(use_wandb), config=resolved, run_name=model_id)

    # 3. Log header
    log_train_session_header(logger, model_id, data_path, resolved)
    if load_optuna_params:
        logger.info(f"Optuna params: {load_optuna_params}")
        if optuna_id:
            logger.info(f"Optuna ID:   {optuna_id}")
    log_config_flags(logger, resolved)

    # 4. Load data
    data_source = resolved.get('data_source', 'train')
    submodel_dirs = resolved.get('submodel_dirs', []) or []
    if submodel_dirs and data_source == 'train':
        data_source = 'agent'
        resolved['data_source'] = 'agent'
        resolved['data_source_auto_switched'] = True
        logger.info("  data_source auto-set to 'agent' (submodel_dirs active)")

    from common.submodel_manager import resolve_submodel_feature_set
    submodel_feat_set = resolve_submodel_feature_set(
        submodel_dirs, symbol=symbol, explicit_set=resolved.get('submodel_feature_set')
    )

    data = load_signal_data(
        data_path,
        feature_set=resolved['feature_set'],
        submodel_feature_set=submodel_feat_set if submodel_dirs else None,
        lookback=resolved['lookback'],
        horizon=resolved['horizon'],
        val_source=resolved['val_source'],
        signal_threshold=resolved['signal_threshold'],
        window_sigma=resolved['window_sigma'],
        preferred_data_format=resolved.get('preferred_data_format', 'csv'),
        winsorize_percentile=resolved.get('winsorize_percentile'),
        data_source=data_source,
        feature_round_decimals=resolved.get('feature_round_decimals'),
        allow_partial_feature_set=resolved.get('allow_partial_feature_set', False),
        min_feature_coverage=resolved.get('min_feature_coverage', 1.0),
    )
    features = data['features']
    n_features = len(features)

    # Record what was actually resolved so model_config.json documents the run
    if submodel_dirs:
        resolved['submodel_feature_set'] = submodel_feat_set
    coverage = data.get('feature_coverage') or {}
    resolved['feature_set_requested'] = coverage.get(
        'requested_set', resolved.get('feature_set'))
    resolved['feature_coverage'] = coverage.get('coverage')
    resolved['feature_missing'] = coverage.get('missing', [])
    resolved['submodel_feature_coverage'] = data.get('submodel_feature_coverage')
    logger.info(
        f"Feature set '{resolved.get('feature_set')}': {n_features} feature(s) "
        f"resolved, coverage={coverage.get('coverage')}")

    target_stats = {
        'buy_mean': float(np.mean(data['target_train'][:, 0])),
        'buy_std': float(np.std(data['target_train'][:, 0])),
        'sell_mean': float(np.mean(data['target_train'][:, 1])),
        'sell_std': float(np.std(data['target_train'][:, 1])),
    }
    log_data_summary(logger, len(data['X_train']), len(data['X_val']),
                     n_features, target_stats)

    # 4b. Pre-compute submodel predictions (once, before training loop)
    submodel_preds_train = None
    submodel_preds_val = None
    has_submodel_preds = False

    submodel_mode = resolved.get('submodel_mode', 'real') or 'real'
    null_submodel_size = int(resolved.get('null_submodel_size', 0) or 0)

    if submodel_dirs:
        from common.submodel_manager import SubmodelManager, apply_submodel_mode
        allow_mismatch = bool(resolved.get('allow_submodel_data_mismatch', False))
        expected_scaler = (data['metadata'].get('scalers') or {}).get('features')
        sm = SubmodelManager(
            submodel_dirs, device,
            expected_symbol=symbol, expected_data_id=data_id,
            max_missing_ratio=resolved.get('submodel_max_missing_ratio'),
            expected_scaler=expected_scaler,
            strict=not allow_mismatch,
        )
        resolved['submodel_max_missing_ratio'] = sm.max_missing_ratio
        resolved['arch_type'] = 'submodel_fusion'
        resolved['submodel_size'] = sm.total_output_dim()
        logger.info(f"Submodel fusion: arch_type='submodel_fusion', "
                    f"submodel_size={resolved['submodel_size']}, "
                    f"submodel_norm_type={resolved.get('submodel_norm_type', 'auto')}, "
                    f"submodel_mode='{submodel_mode}'")

        logger.info(f"Pre-computing submodel predictions "
                    f"({len(submodel_dirs)} model(s))...")
        sub_X_tr = data.get('X_train_submodel', data['X_train'])
        sub_X_vl = data.get('X_val_submodel', data['X_val'])
        sub_fts = data.get('submodel_features', features)
        submodel_preds_train = sm.precompute_predictions(sub_X_tr, sub_fts)
        submodel_preds_val = sm.precompute_predictions(sub_X_vl, sub_fts)
        logger.info(f"  Train preds: {submodel_preds_train.shape}, "
                    f"Val preds: {submodel_preds_val.shape}")
        sm.diagnose(submodel_preds_train, 'train', logger=logger)
        sm.diagnose(submodel_preds_val, 'val', logger=logger)
        zero_filled = sm.zero_filled_features()
        if any(zero_filled.values()):
            logger.info(f"  Zero-filled submodel features: {zero_filled}")

        if submodel_mode != 'real':
            logger.info(f"  Applying submodel ablation mode '{submodel_mode}' "
                        f"(seed={resolved['seed']})")
            submodel_preds_train = apply_submodel_mode(
                submodel_preds_train, submodel_mode, resolved['seed'], rng_offset=0)
            submodel_preds_val = apply_submodel_mode(
                submodel_preds_val, submodel_mode, resolved['seed'], rng_offset=1)
        has_submodel_preds = True
        # Free GPU memory used by submodels
        del sm
    elif null_submodel_size > 0:
        # Architecture control: same fusion head, zero information.
        resolved['arch_type'] = 'submodel_fusion'
        resolved['submodel_size'] = null_submodel_size
        submodel_preds_train = np.zeros(
            (len(data['X_train']), null_submodel_size), dtype=np.float32)
        submodel_preds_val = np.zeros(
            (len(data['X_val']), null_submodel_size), dtype=np.float32)
        has_submodel_preds = True
        logger.info(f"Null submodel control: arch_type='submodel_fusion', "
                    f"submodel_size={null_submodel_size} (all-zero channels)")

    # 5. Build dataloaders
    train_ds = CryptoDataset(data['X_train'], data['target_train'],
                             submodel_preds=submodel_preds_train)
    val_ds = CryptoDataset(data['X_val'], data['target_val'],
                           submodel_preds=submodel_preds_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=resolved['batch_size'], shuffle=True,
        num_workers=resolved['num_workers'],
        pin_memory=resolved['use_pin_memory'])
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=resolved['batch_size'], shuffle=False,
        num_workers=resolved['num_workers'],
        pin_memory=resolved['use_pin_memory'])

    # 6. Build model
    model = build_signal_model(resolved, n_features, device)
    n_params = model.count_parameters()
    logger.info(f"Model parameters: {n_params:,}")

    # 7. Build loss + optimizer + scheduler
    criterion = build_signal_loss(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved['learning_rate'],
        weight_decay=resolved['weight_decay'])
    total_steps = resolved['epochs'] * len(train_loader)
    sched_params = dict(getattr(SignalConfig, 'SCHEDULER_PARAMS', {}))
    sched_params.update(resolved.get('scheduler_params') or {})
    if 'scheduler_patience' in resolved:
        sched_params['patience'] = resolved['scheduler_patience']
    if 'scheduler_factor' in resolved:
        sched_params['factor'] = resolved['scheduler_factor']
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=resolved['scheduler_type'],
        total_steps=total_steps,
        lr_warmup_epochs=resolved.get('lr_warmup_epochs', 0),
        **sched_params)

    # 8. Setup diagnostics + augmenter (optional)
    diagnostics = None
    if resolved.get('log_signal_diagnostics'):
        diagnostics = SignalEpochDiagnostics(logger=logger)

    augmenter = SequenceAugmenter(
        jitter=resolved.get('aug_jitter', False),
        jitter_std=resolved.get('aug_jitter_std', 0.01),
        temporal_dropout=resolved.get('aug_temporal_dropout', False),
        temporal_dropout_rate=resolved.get('aug_temporal_dropout_rate', 0.05),
        subsequence=resolved.get('aug_subsequence', False),
        subsequence_offset=resolved.get('aug_subsequence_offset', 2),
    )

    # 9. Train
    val_signal_ctx = None
    if resolved.get('best_model_metric') == 'val_signal':
        val_signal_ctx = build_val_signal_ctx(
            val_targets=data['target_val'],
            y_true_val_lr=data['y_true_val'],
            horizon=resolved['horizon'],
            min_signal=resolved['min_signal'],
            diversity_weight=resolved['best_model_diversity_weight'],
        )

    def _log_epoch(epoch, train_loss, val_loss, diag_entry, grad_norms, loss_components=None):
        log_epoch_summary(logger, epoch, train_loss, val_loss,
                          diag_entry, grad_norms, loss_components)
        wandb_metrics = {}
        if diag_entry:
            wandb_metrics['val_diversity'] = diag_entry.get('signal_diversity')
            wandb_metrics['val_cr_mean'] = diag_entry.get(
                'compression_ratio_mean')
        if grad_norms:
            wandb_metrics.update(
                {f'grad_norm_{k}': v for k, v in grad_norms.items()})
        tracker.log_epoch(epoch, train_loss, val_loss, wandb_metrics)

    result = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, device,
        epochs=resolved['epochs'],
        patience=resolved['early_stopping_patience'],
        grad_clip=resolved['gradient_clip'],
        best_metric=resolved['best_model_metric'],
        diversity_weight=resolved['best_model_diversity_weight'],
        min_epochs_before_es=resolved.get('min_epochs_before_es', 0),
        early_stopping_min_delta=resolved.get('early_stopping_min_delta', 0.0),
        diagnostics=diagnostics,
        logger=logger,
        augmenter=augmenter,
        log_grad_norms=resolved.get('log_gradient_norms', False),
        log_epoch_fn=_log_epoch,
        scheduler_type=resolved['scheduler_type'],
        use_swa=resolved.get('use_swa', False),
        swa_start_epoch=resolved.get('swa_start_epoch', 30),
        swa_every=resolved.get('swa_every', 1),
        val_signal_ctx=val_signal_ctx,
        val_signal_min_epoch=resolved.get('val_signal_min_epoch', 10),
        has_submodel_preds=has_submodel_preds,
    )

    # 9b. Calibrate (optional)
    if resolved.get('calibrate', False):
        logger.info("Fitting DualSignalCalibrator on validation predictions...")
        model.eval()
        with torch.no_grad():
            val_preds_list = []
            for batch in val_loader:
                X_b = batch[0].to(device)
                sub_p = batch[-1].to(device) if has_submodel_preds else None
                val_preds_list.append(
                    model(X_b, submodel_inputs=sub_p).cpu().numpy())
            val_preds_arr = np.concatenate(val_preds_list, axis=0)
        from common.calibrator_fit import DualSignalCalibrator
        calibrator = DualSignalCalibrator(method='isotonic').fit(val_preds_arr, data['target_val'])
        import joblib
        joblib.dump(calibrator, os.path.join(output_dir, 'calibrator.pkl'))
        logger.info("  Saved calibrator.pkl")

    # 10. Save
    swa_state = result.get('swa_state')
    if swa_state is not None:
        logger.info("SWA weights collected and will be saved alongside best checkpoint.")

    config_dict = build_signal_config_dict(
        resolved=resolved,
        model_id=model_id,
        data_id=data_id,
        symbol=symbol,
        features=features,
        n_params=n_params,
        train_losses=result['train_losses'],
        val_losses=result['val_losses'],
        best_epoch=result['best_epoch'],
        diagnostics_history=result['diagnostics_history'],
        data_path=data_path,
    )
    save_model(model, output_dir, config_dict,
               feature_scaler=data.get('feature_scaler'),
               swa_state=swa_state)


    log_final_summary(logger, result['best_epoch'],
                      result['best_score'], len(result['train_losses']))

    tracker.log_summary({
        'best_epoch': result['best_epoch'],
        'best_score': result['best_score'],
        'n_epochs_trained': len(result['train_losses']),
    })
    tracker.finish()

    logger.info(f"Model saved to: {output_dir}")
    return output_dir


if __name__ == '__main__':
    args = build_parser().parse_args()
    kw = dispatch_args_to_train(args)
    use_wandb_flag = bool(getattr(args, 'use_wandb', 0))
    train(use_wandb=use_wandb_flag, **kw)
