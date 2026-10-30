"""
Optuna objective closure for Signal Model hyperparameter tuning.

The `create_objective` factory keeps a per-study data cache (keyed by
lookback/horizon/window_sigma/signal_threshold — all four affect the
oracle target) and dispatches training using the configuration assembled
by `suggest_hyperparameters`.
"""
from __future__ import annotations

import os
import optuna
import torch

from signal_config import SignalConfig
from common.data_utils import load_signal_data, resolve_submodel_feature_set
from common.losses import get_signal_loss
from common.metrics import signal_diversity, compute_signal_rank_score
from common.model import build_signal_model
from common.training_utils import train_epoch, val_epoch
from common.tuning_search_space import suggest_hyperparameters
from common.id_utils import detect_symbol

from common.torch_utils import (
    CryptoDataset, get_scheduler, set_seed,
)
from common.augmentation import SequenceAugmenter

# Composite score: penalise collapse toward neutral (0.5) predictions.
TARGET_DIVERSITY = 0.30
COMPOSITE_DIVERSITY_WEIGHT = 0.5


def compute_composite_score(val_loss, val_preds, tuning_cfg,
                             val_targets=None, y_true_lr=None,
                             horizon=1, min_signal=60.0):
    """
    Combine raw val_loss / diversity penalty / signal rank score.

    score_type values:
        'val_loss'   — raw val_loss (default, minimised)
        'composite'  — val_loss + w * max(0, TARGET_DIVERSITY - diversity)
        'signal_rank'— negated compute_signal_rank_score (0..100 → -100..0,
                        minimised by Optuna so higher rank = lower score)

    Returns float (minimised by Optuna).
    """
    score_type = getattr(tuning_cfg, 'OPTUNA_SCORE_TYPE', 'val_loss')

    if score_type == 'composite':
        if val_preds is None or len(val_preds) < 2:
            return float(val_loss)
        diversity = signal_diversity(val_preds)
        penalty = COMPOSITE_DIVERSITY_WEIGHT * max(0.0, TARGET_DIVERSITY - diversity)
        return float(val_loss) + penalty

    if score_type == 'signal_rank':
        if (val_preds is None or val_targets is None
                or y_true_lr is None or len(val_preds) < 10):
            return float(val_loss)
        try:
            from common.simulation import evaluate_signal_simulation
            from common.metrics import calculate_signal_metrics
            fallback_mode = getattr(tuning_cfg, 'RANK_FALLBACK_MODE', 'none')
            
            thresh_mode = 'percentile' if fallback_mode == 'dynamic_threshold' else 'absolute'
            p_cutoff = getattr(tuning_cfg, 'PERCENTILE_CUTOFF', 5.0)

            mets = calculate_signal_metrics(
                val_preds, val_targets, y_true_lr,
                min_signal_pct=min_signal,
                threshold_mode=thresh_mode,
                percentile_cutoff=p_cutoff)
            sim = evaluate_signal_simulation(
                val_preds, val_targets, y_true_lr,
                horizon=horizon, min_signal=min_signal,
                threshold_mode=thresh_mode,
                percentile_cutoff=p_cutoff)
            mets.update(sim)
            rank = compute_signal_rank_score(mets, threshold_mode=thresh_mode)
            rank_val = float(rank.get('rank_score', 0.0))

            if rank_val > 0.0:
                return -rank_val

            # Model is DEAD (rank_val == 0.0) -> Fallback according to RANK_FALLBACK_MODE
            if fallback_mode == 'gradient':
                corr_buy = mets.get('corr_buy', 0.0) or 0.0
                corr_sell = mets.get('corr_sell', 0.0) or 0.0
                corr_avg = (corr_buy + corr_sell) / 2.0
                div = mets.get('diversity', 0.0) or 0.0
                mae_buy = mets.get('mae_buy', 0.25)
                mae_sell = mets.get('mae_sell', 0.25)
                mae_avg = (mae_buy + mae_sell) / 2.0
                mae_health = max(0.0, 1.0 - mae_avg / 0.25)
                # Proxy continuous score in (-100, 0)
                proxy = corr_avg * 50.0 + div * 30.0 + mae_health * 20.0
                return -float(proxy)
            else:  # 'none' or unhandled
                return float(val_loss)
        except Exception:
            return float(val_loss)

    return float(val_loss)


def create_objective(tuning_cfg, device):
    """
    Create the Optuna objective closure.

    Returns a callable objective(trial) -> float.
    """
    _cache = {'data': None, 'key': None, 'submodel_preds': None}

    # Load submodels once (not per trial)
    submodel_dirs = getattr(tuning_cfg, 'SUBMODEL_DIRS', []) or []
    sm_manager = None
    if submodel_dirs:
        from common.submodel_manager import SubmodelManager
        sm_manager = SubmodelManager(
            submodel_dirs, device,
            max_missing_ratio=getattr(tuning_cfg, 'SUBMODEL_MAX_MISSING_RATIO', 0.30),
        )
        print(f"  Optuna: SubmodelManager loaded ({len(submodel_dirs)} model(s), "
              f"output_dim={sm_manager.total_output_dim()})")

    def objective(trial):
        params = suggest_hyperparameters(trial, tuning_cfg)
        set_seed(tuning_cfg.SEED + trial.number)

        cache_key = (params['lookback'], params['horizon'],
                     params['window_sigma'], params['signal_threshold'])
        if _cache['key'] != cache_key:
            data_source = 'agent' if submodel_dirs else 'train'
            sm_feat_set = getattr(tuning_cfg, 'SUBMODEL_FEATURE_SET', None)
            if sm_feat_set is None and submodel_dirs:
                sym_str = detect_symbol(os.path.basename(str(tuning_cfg.TRAIN_PATH)))
                sm_feat_set = resolve_submodel_feature_set(submodel_dirs, symbol=sym_str)

            # Load with superset 'max' to allow in-memory column sub-indexing per trial
            data = load_signal_data(
                tuning_cfg.TRAIN_PATH,
                feature_set='max',
                submodel_feature_set=sm_feat_set,
                lookback=params['lookback'],
                horizon=params['horizon'],
                val_source=tuning_cfg.VAL_SOURCE,
                signal_threshold=params['signal_threshold'],
                window_sigma=params['window_sigma'],
                data_source=data_source,
            )
            _cache['data'] = data
            _cache['key'] = cache_key

            # Pre-compute submodel preds once per cache key using dedicated submodel features
            if sm_manager:
                sub_X_tr = data.get('X_train_submodel', data['X_train'])
                sub_X_vl = data.get('X_val_submodel', data['X_val'])
                sub_fts = data.get('submodel_features', data['features'])
                _cache['submodel_preds'] = {
                    'train': sm_manager.precompute_predictions(sub_X_tr, sub_fts),
                    'val': sm_manager.precompute_predictions(sub_X_vl, sub_fts),
                }
            else:
                _cache['submodel_preds'] = None

        data = _cache['data']

        # Resolve features for this specific trial
        trial_feat_set = params.get('feature_set', getattr(tuning_cfg, 'FEATURE_SET', 'max'))
        from feature_engineering import get_feature_sets
        has_bp = 'buy_pressure' in data['features']
        f_sets = get_feature_sets(include_buy_pressure=has_bp)
        if trial_feat_set in f_sets:
            target_feat_names = f_sets[trial_feat_set]
        else:
            target_feat_names = data['features']

        feat_indices = [i for i, f in enumerate(data['features']) if f in target_feat_names]
        if not feat_indices:
            feat_indices = list(range(data['X_train'].shape[2]))

        X_train_trial = data['X_train'][:, :, feat_indices]
        X_val_trial = data['X_val'][:, :, feat_indices]
        n_features = len(feat_indices)
        has_submodel_preds = sm_manager is not None

        # Submodel fusion overrides
        submodel_size = 0
        arch_type = params.get('arch_type', 'legacy')
        if sm_manager:
            submodel_size = sm_manager.total_output_dim()
            arch_type = 'submodel_fusion'

        resolved_model = {
            'hidden_size': params['hidden_size'],
            'num_layers': params['num_layers'],
            'dropout': params['dropout'],
            'bidirectional': bool(params.get('bidirectional', False)),
            'use_attention': bool(params['use_attention']),
            'fc_layers': params['fc_layers'],
            'activation': params['activation'],
            'fc_norm': params['fc_norm'],
            'rnn_type': params.get('rnn_type', 'lstm'),
            'use_input_norm': bool(params.get('use_input_norm', True)),
            'use_submodel_norm': bool(params.get('use_submodel_norm', getattr(SignalConfig, 'USE_SUBMODEL_NORM', True))),
            'submodel_norm_type': params.get('submodel_norm_type', 'auto'),
            'attention_heads': params.get('attention_heads', 1),
            'lstm_dropout': None,
            'fc_dropout': None,
            'use_feature_attention': bool(params.get('use_feature_attention', 0)),
            'output_mode': 'dual',
            'weight_init': params.get('weight_init', getattr(SignalConfig, 'WEIGHT_INIT', 'default')),
            # F6
            'use_conv1d_front': bool(params.get('use_conv1d_front', 0)),
            'conv1d_kernel': int(params.get('conv1d_kernel', 5)),
            'pooling_mode': params.get('pooling_mode', 'last'),
            'use_fc_residual': bool(params.get('use_fc_residual', 0)),
            'use_rnn_norm': bool(params.get('use_rnn_norm', 0)),
            'use_learnable_output_scale': bool(
                params.get('use_learnable_output_scale', 0)),
            'freeze_rnn': False,
            # Submodel fusion
            'arch_type': arch_type,
            'submodel_size': submodel_size,
        }
        model = build_signal_model(resolved_model, n_features, device)

        loss_type = params.get('loss_type', getattr(tuning_cfg, 'LOSS_TYPE', 'bce'))
        criterion = get_signal_loss(
            loss_type,
            ramp_epochs=params.get('loss_schedule_ramp_epochs', getattr(tuning_cfg, 'LOSS_SCHEDULE_RAMP_EPOCHS', 0)),
            diversity_weight=params.get('loss_diversity_weight', getattr(tuning_cfg, 'LOSS_DIVERSITY_WEIGHT', 1.0)),
            diversity_target=params.get('loss_diversity_target', getattr(tuning_cfg, 'LOSS_DIVERSITY_TARGET', 0.30)),
            focal_gamma=params.get('focal_gamma', getattr(tuning_cfg, 'FOCAL_GAMMA', 2.0)),
            w_buy=params.get('w_buy', getattr(tuning_cfg, 'W_BUY', 1.0)),
            w_sell=params.get('w_sell', getattr(tuning_cfg, 'W_SELL', 1.0)),
            std_match_weight=params.get('std_match_weight', getattr(tuning_cfg, 'STD_MATCH_WEIGHT', 0.1)),
            corr_penalty_weight=params.get('corr_penalty_weight', getattr(tuning_cfg, 'CORR_PENALTY_WEIGHT', 0.5)),
            label_smoothing_alpha=params.get('label_smoothing_alpha', getattr(tuning_cfg, 'LABEL_SMOOTHING_ALPHA', 0.0)),
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params['learning_rate'],
            weight_decay=params['weight_decay'])

        # Build dataloaders with optional submodel preds
        sub_train = _cache['submodel_preds']['train'] if has_submodel_preds else None
        sub_val = _cache['submodel_preds']['val'] if has_submodel_preds else None

        train_loader = torch.utils.data.DataLoader(
            CryptoDataset(X_train_trial, data['target_train'],
                          submodel_preds=sub_train),
            batch_size=params['batch_size'], shuffle=True,
            num_workers=SignalConfig.NUM_WORKERS,
            pin_memory=(device.type == 'cuda' and SignalConfig.USE_PIN_MEMORY))
        val_loader = torch.utils.data.DataLoader(
            CryptoDataset(X_val_trial, data['target_val'],
                          submodel_preds=sub_val),
            batch_size=params['batch_size'], shuffle=False,
            num_workers=SignalConfig.NUM_WORKERS,
            pin_memory=(device.type == 'cuda' and SignalConfig.USE_PIN_MEMORY))

        total_steps = tuning_cfg.TUNING_EPOCHS * max(len(train_loader), 1)
        scheduler = get_scheduler(
            optimizer, scheduler_type=params['scheduler_type'],
            total_steps=total_steps,
            lr_warmup_epochs=params.get('lr_warmup_epochs', 0),
            patience=5, factor=0.5)
        is_onecycle = params['scheduler_type'] == 'onecycle'

        augmenter = SequenceAugmenter(
            jitter=bool(params.get('aug_jitter', 0)),
            jitter_std=params.get('aug_jitter_std', 0.01),
            temporal_dropout=bool(params.get('aug_temporal_dropout', 0)),
            temporal_dropout_rate=params.get('aug_temporal_dropout_rate', 0.05),
            subsequence=bool(params.get('aug_subsequence', 0)),
            subsequence_offset=params.get('aug_subsequence_offset', 2),
        )

        best_val_loss = float('inf')
        best_preds = None
        best_targets = None
        patience_ctr = 0
        epoch = 0

        for epoch in range(tuning_cfg.TUNING_EPOCHS):
            if hasattr(criterion, 'set_epoch'):
                criterion.set_epoch(epoch + 1)
            train_epoch(
                model, train_loader, criterion, optimizer, device,
                grad_clip=tuning_cfg.GRADIENT_CLIP,
                batch_scheduler=scheduler if is_onecycle else None,
                augmenter=augmenter,
                has_submodel_preds=has_submodel_preds,
            )
            val_loss, val_preds, val_targets = val_epoch(
                model, val_loader, criterion, device,
                has_submodel_preds=has_submodel_preds)

            if scheduler is not None and not is_onecycle:
                if params['scheduler_type'] == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_preds = val_preds
                best_targets = val_targets
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= tuning_cfg.TUNING_PATIENCE:
                    break

        y_true_lr = data.get('y_true_val')
        score = compute_composite_score(
            best_val_loss, best_preds, tuning_cfg,
            val_targets=best_targets,
            y_true_lr=y_true_lr,
            horizon=int(params.get('horizon', 1)),
            min_signal=float(getattr(tuning_cfg, 'MIN_SIGNAL',
                                     SignalConfig.MIN_SIGNAL)),
        )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Trial {trial.number:3d}: score={score:.6f} "
              f"(val_loss={best_val_loss:.6f}, params={n_params:,}, "
              f"epochs={epoch + 1})")

        return score

    return objective
