"""
Optuna objective closure for LSTM hyperparameter tuning.

Extracted from `lstm_hyperparameter_tuning.py` (FAZA 0.R2 refactor). The
`create_objective` factory keeps a per-study data cache and dispatches
training using the configuration assembled by `suggest_hyperparameters`.

`compute_composite_score` is a FAZA 0 placeholder; the full implementation
combining `val_loss` with collapse / R² penalties is added in FAZA 2.V1-5.
"""
from __future__ import annotations

import numpy as np
import optuna
import torch

from lstm_config import LSTMConfig
from common.data_utils import load_data
from common.losses import (
    LSTMCustomLoss, get_loss_function, resolve_signal_threshold,
)
from common.signal_loss import SignalLoss
from common.augmentation import SequenceAugmenter
from common.model import LSTMLogReturnModel, _init_weights
from common.torch_utils import (
    CryptoDataset, get_scheduler, set_seed,
)
from common.training_utils import predict_all, train_epoch, validate_epoch
from common.tuning_search_space import suggest_hyperparameters


def _trial_bool(params, key, default=0) -> bool:
    return bool(int(params.get(key, default)))


def _baseline_or_cfg(tuning_cfg, key, cfg_attr, default=None):
    baseline = getattr(tuning_cfg, 'BASELINE_PARAMS', None) or {}
    if key in baseline:
        return baseline[key]
    return getattr(LSTMConfig, cfg_attr, default)


def _compute_r2(y_true, y_pred, eps=1e-12):
    """Plain numpy R²; matches the formula used by `batch_r2`."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size < 2:
        return 0.0
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + eps)


def compute_composite_score(val_loss, y_val_pred, y_val_true, target_scaler,
                            cfg):
    """
    F2.V1-5: combine raw val_loss with collapse / R² penalties.

    Penalties are applied in RAW (un-scaled) log-return space because that
    is where the metric thresholds (compression_ratio, R²) are interpretable.
    When `target_scaler` is None the predictions / targets are already in
    raw space.

    Returns `float(val_loss)` unchanged when:
        - `cfg.OPTUNA_SCORE_TYPE` is 'val_loss' (default)
        - predictions are missing (empty array)
    """
    score = float(val_loss)
    score_type = getattr(cfg, 'OPTUNA_SCORE_TYPE', 'val_loss')
    if score_type not in ('composite', 'signal'):
        return score
    if y_val_pred is None or len(y_val_pred) < 2:
        return score

    y_pred_raw = (target_scaler.inverse_transform(np.asarray(y_val_pred))
                  if target_scaler is not None else np.asarray(y_val_pred))
    y_true_raw = (target_scaler.inverse_transform(np.asarray(y_val_true))
                  if target_scaler is not None else np.asarray(y_val_true))

    pred_std = float(np.std(y_pred_raw))
    true_std = float(np.std(y_true_raw))
    cr = pred_std / true_std if true_std > 0 else 0.0
    r2 = _compute_r2(y_true_raw, y_pred_raw)

    coll_w = getattr(cfg, 'OPTUNA_COMPOSITE_COLLAPSE_WEIGHT', 20.0)
    r2_w = getattr(cfg, 'OPTUNA_COMPOSITE_R2_WEIGHT', 10.0)

    if score_type == 'signal':
        # Signal-aware proxy (no close series available during tuning, so the
        # full market simulation / rank_score is evaluated later in lstm_test).
        # Symmetric CR penalty steers toward CR≈1 — punishing BOTH collapse and
        # explosion — while the R² term keeps predictions better than the mean.
        cr_pen = coll_w * abs(1.0 - cr)
        r2_pen = r2_w * max(0.0, -r2)
        return score + cr_pen + r2_pen

    threshold = getattr(cfg, 'OPTUNA_COMPOSITE_COLLAPSE_THRESHOLD', 0.05)
    collapse_pen = coll_w * max(0.0, threshold - cr)
    r2_pen = r2_w * max(0.0, -r2)
    return score + collapse_pen + r2_pen


def create_objective(tuning_cfg, device):
    """
    Create the Optuna objective closure.

    Returns a callable objective(trial) → float.
    """
    # Data cache — loaded once for fixed lookback/horizon, reloaded when they change
    _cache = {'data': None, 'key': None}

    def objective(trial):
        params = suggest_hyperparameters(trial, tuning_cfg)
        set_seed(tuning_cfg.SEED + trial.number)

        # Load data (cache if lookback/horizon unchanged)
        cache_key = (params['lookback'], params['horizon'])
        if _cache['key'] != cache_key:
            data, _, metadata = load_data(
                tuning_cfg.TRAIN_PATH,
                preferred_format=tuning_cfg.PREFERRED_FORMAT,
                feature_set=tuning_cfg.FEATURE_SET,
                custom_features=tuning_cfg.CUSTOM_FEATURES,
                lookback=params['lookback'],
                horizon=params['horizon'],
                scale_target=tuning_cfg.SCALE_TARGET,
                target_scaler_type=tuning_cfg.TARGET_SCALER_TYPE,
                winsorize_target=tuning_cfg.WINSORIZE_TARGET,
                winsorize_percentile=tuning_cfg.WINSORIZE_PERCENTILE,
                val_source=tuning_cfg.VAL_SOURCE,
                feature_round_decimals=getattr(tuning_cfg, 'FEATURE_ROUND_DECIMALS', 0),
                target_round_decimals=getattr(tuning_cfg, 'TARGET_ROUND_DECIMALS', 0),
            )
            _cache['data'] = data
            _cache['key'] = cache_key
            _cache['metadata'] = metadata

        data = _cache['data']
        target_scaler = data['target_scaler']

        # Signal threshold
        signal_threshold = None
        if tuning_cfg.SIGNAL_THRESHOLD_ENABLED:
            y_raw = (target_scaler.inverse_transform(data['y_train'])
                     if target_scaler else data['y_train'])
            signal_threshold = resolve_signal_threshold(
                y_raw, mode=tuning_cfg.SIGNAL_THRESHOLD_MODE,
                fixed_value=tuning_cfg.SIGNAL_THRESHOLD_VALUE,
                percentile=tuning_cfg.SIGNAL_THRESHOLD_PERCENTILE,
            )

        # Build model. Architecture-level F4 knobs default to LSTMConfig
        # values when the search space does not surface them as a tuning
        # dimension (e.g. attention_heads=1 with TUNE_PARAMS frozen).
        input_size = data['X_train'].shape[2]
        model = LSTMLogReturnModel(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            bidirectional=params['bidirectional'],
            use_attention=params['use_attention'],
            fc_layers=params['fc_layers'],
            activation=params['activation'],
            fc_norm=params['fc_norm'],
            arch_type=getattr(LSTMConfig, 'ARCH_TYPE', 'legacy'),
            use_input_norm=params.get(
                'use_input_norm',
                _baseline_or_cfg(tuning_cfg, 'use_input_norm', 'USE_INPUT_NORM', False)),
            rnn_type=params.get('rnn_type', getattr(LSTMConfig, 'RNN_TYPE', 'lstm')),
            attention_heads=params.get(
                'attention_heads',
                _baseline_or_cfg(tuning_cfg, 'attention_heads',
                                 'ATTENTION_HEADS', 1)),
            lstm_dropout=params.get(
                'lstm_dropout',
                _baseline_or_cfg(tuning_cfg, 'lstm_dropout', 'LSTM_DROPOUT', None)),
            fc_dropout=params.get(
                'fc_dropout',
                _baseline_or_cfg(tuning_cfg, 'fc_dropout', 'FC_DROPOUT', None)),
            use_feature_attention=_trial_bool(params, 'use_feature_attention'),
            # v3 arch flags (CP-046): inherited from LSTMConfig (not tunable),
            # ensures the trial model is identical to what lstm_train.py builds.
            use_rnn_norm=getattr(LSTMConfig, 'USE_RNN_NORM', False),
            pooling_mode=getattr(LSTMConfig, 'POOLING_MODE', 'last'),
            use_conv1d_front=getattr(LSTMConfig, 'USE_CONV1D_FRONT', False),
            conv1d_kernel=getattr(LSTMConfig, 'CONV1D_KERNEL', 5),
            use_fc_residual=getattr(LSTMConfig, 'USE_FC_RESIDUAL', False),
            use_learnable_output_scale=getattr(
                LSTMConfig, 'USE_LEARNABLE_OUTPUT_SCALE', False),
        ).to(device)
        init_mode = getattr(LSTMConfig, 'WEIGHT_INIT', 'default')
        if init_mode != 'default':
            _init_weights(model, init_mode,
                          activation=params.get('activation', LSTMConfig.ACTIVATION))
            # Match build_model: re-seed after weight init so per-trial
            # randomness is reproducible regardless of init_mode.
            seed = tuning_cfg.SEED + trial.number
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        if getattr(LSTMConfig, 'FREEZE_RNN', False):
            for param in model.rnn.parameters():
                param.requires_grad = False

        # Build loss. For LOSS_TYPE='custom' (and any future composite loss)
        # we inherit the component flags from LSTMConfig — tuning just
        # selects the TYPE, not the internals. Tuning internals is F2.5.6.
        _cfg = type('_Cfg', (), {
            'LOSS_TYPE': LSTMConfig.LOSS_TYPE,
            'HUBER_DELTA': params['huber_delta'],
            'QUANTILE_LOSS_Q': params['quantile_q'],
            # Custom-loss component flags: inherited from LSTMConfig so that
            # an Optuna trial with loss_type='custom' builds exactly the same
            # LSTMCustomLoss that training will use downstream.
            'USE_SAMPLE_WEIGHTS': LSTMConfig.USE_SAMPLE_WEIGHTS,
            'USE_MISDIR_PENALTY': LSTMConfig.USE_MISDIR_PENALTY,
            'USE_DEAD_ZONE': LSTMConfig.USE_DEAD_ZONE,
            'SAMPLE_WEIGHT_MODE': LSTMConfig.SAMPLE_WEIGHT_MODE,
            'SAMPLE_WEIGHT_PERCENTILE': LSTMConfig.SAMPLE_WEIGHT_PERCENTILE,
            'SAMPLE_WEIGHT_LOW_W': LSTMConfig.SAMPLE_WEIGHT_LOW_W,
            'MISDIR_PENALTY': LSTMConfig.MISDIR_PENALTY,
            'MISDIR_MIN_QUANTILE': LSTMConfig.MISDIR_MIN_QUANTILE,
            'DEAD_ZONE_WEIGHT': LSTMConfig.DEAD_ZONE_WEIGHT,
            'USE_AMPLIFIED_ZONE': getattr(
                LSTMConfig, 'USE_AMPLIFIED_ZONE', False),
            'AMPLIFIED_ZONE_THRESHOLD': params.get(
                'amplified_zone_threshold',
                getattr(LSTMConfig, 'AMPLIFIED_ZONE_THRESHOLD', 0.0015)),
            'AMPLIFIED_ZONE_WEIGHT': params.get(
                'amplified_zone_weight',
                getattr(LSTMConfig, 'AMPLIFIED_ZONE_WEIGHT', 2.0)),
            'USE_CORRELATION_PENALTY': getattr(
                LSTMConfig, 'USE_CORRELATION_PENALTY', False),
            'CORRELATION_PENALTY_WEIGHT': getattr(
                LSTMConfig, 'CORRELATION_PENALTY_WEIGHT', 0.1),
            'USE_VARIANCE_PENALTY': getattr(
                LSTMConfig, 'USE_VARIANCE_PENALTY', False),
            'VARIANCE_PENALTY_WEIGHT': getattr(
                LSTMConfig, 'VARIANCE_PENALTY_WEIGHT', 0.1),
            'USE_DIRECTIONAL_LOSS': getattr(
                LSTMConfig, 'USE_DIRECTIONAL_LOSS', False),
            'DIRECTIONAL_LOSS_WEIGHT': getattr(
                LSTMConfig, 'DIRECTIONAL_LOSS_WEIGHT', 0.5),
            'USE_LOG_VARIANCE_PENALTY': getattr(
                LSTMConfig, 'USE_LOG_VARIANCE_PENALTY', False),
            'LOG_VARIANCE_PENALTY_WEIGHT': getattr(
                LSTMConfig, 'LOG_VARIANCE_PENALTY_WEIGHT', 0.01),
            'USE_R2_PENALTY': getattr(
                LSTMConfig, 'USE_R2_PENALTY', False),
            'R2_PENALTY_WEIGHT': getattr(
                LSTMConfig, 'R2_PENALTY_WEIGHT', 0.01),
            'USE_CCC_PENALTY': getattr(
                LSTMConfig, 'USE_CCC_PENALTY', False),
            'CCC_PENALTY_WEIGHT': getattr(
                LSTMConfig, 'CCC_PENALTY_WEIGHT', 0.01),
            # FAZA 3.V2-3: SignalLoss v2 flags. The `signal_*` keys land in
            # `params` only when `LOSS_TYPE='signal'`; otherwise we fall back
            # to LSTMConfig values so the snapshot remains self-consistent.
            'SIGNAL_USE_R2': getattr(LSTMConfig, 'SIGNAL_USE_R2', True),
            'SIGNAL_R2_WEIGHT': params.get(
                'signal_r2_weight',
                getattr(LSTMConfig, 'SIGNAL_R2_WEIGHT', 0.01)),
            'SIGNAL_USE_CCC': getattr(LSTMConfig, 'SIGNAL_USE_CCC', True),
            'SIGNAL_CCC_WEIGHT': params.get(
                'signal_ccc_weight',
                getattr(LSTMConfig, 'SIGNAL_CCC_WEIGHT', 0.01)),
            'SIGNAL_USE_LOG_VAR': getattr(
                LSTMConfig, 'SIGNAL_USE_LOG_VAR', False),
            'SIGNAL_LOG_VAR_WEIGHT': params.get(
                'signal_log_var_weight',
                getattr(LSTMConfig, 'SIGNAL_LOG_VAR_WEIGHT', 0.005)),
            'SIGNAL_USE_STD_MATCH': getattr(
                LSTMConfig, 'SIGNAL_USE_STD_MATCH', False),
            'SIGNAL_STD_MATCH_WEIGHT': params.get(
                'signal_std_match_weight',
                getattr(LSTMConfig, 'SIGNAL_STD_MATCH_WEIGHT', 0.05)),
            'SIGNAL_USE_DIR_THRESHOLD': getattr(
                LSTMConfig, 'SIGNAL_USE_DIR_THRESHOLD', False),
            'SIGNAL_DIR_THRESHOLD_WEIGHT': params.get(
                'signal_dir_threshold_weight',
                getattr(LSTMConfig, 'SIGNAL_DIR_THRESHOLD_WEIGHT', 0.1)),
            'SIGNAL_USE_DIRECTION_BCE': getattr(
                LSTMConfig, 'SIGNAL_USE_DIRECTION_BCE', False),
            'SIGNAL_DIRECTION_BCE_WEIGHT': params.get(
                'signal_direction_bce_weight',
                getattr(LSTMConfig, 'SIGNAL_DIRECTION_BCE_WEIGHT', 0.01)),
            'SIGNAL_EMA_PENALTIES': _trial_bool(params, 'signal_ema_penalties'),
            'SIGNAL_EMA_ALPHA': params.get(
                'signal_ema_alpha',
                getattr(LSTMConfig, 'SIGNAL_EMA_ALPHA', 0.1)),
            'SIGNAL_LABEL_SMOOTHING': _trial_bool(
                params, 'signal_label_smoothing'),
            'SIGNAL_LABEL_SMOOTHING_ALPHA': params.get(
                'signal_label_smoothing_alpha',
                getattr(LSTMConfig, 'SIGNAL_LABEL_SMOOTHING_ALPHA', 0.5)),
            'SIGNAL_THRESHOLD_VALUE': tuning_cfg.SIGNAL_THRESHOLD_VALUE,
            # v3 (CP-046+): utility σ proxy, soft threshold, CR ceiling.
            # Without these keys get_loss_function silently falls back to
            # False/default even when the user enables them in LSTMConfig.
            'SIGNAL_USE_UTILITY': getattr(
                LSTMConfig, 'SIGNAL_USE_UTILITY', False),
            'SIGNAL_UTILITY_WEIGHT': getattr(
                LSTMConfig, 'SIGNAL_UTILITY_WEIGHT', 0.05),
            'SIGNAL_SOFT_THRESHOLD': getattr(
                LSTMConfig, 'SIGNAL_SOFT_THRESHOLD', False),
            'USE_CR_CEILING': getattr(LSTMConfig, 'USE_CR_CEILING', False),
            'CR_CEILING': getattr(LSTMConfig, 'CR_CEILING', 1.5),
            'CR_CEILING_WEIGHT': getattr(LSTMConfig, 'CR_CEILING_WEIGHT', 0.1),
        })()
        criterion = get_loss_function(_cfg, target_scaler, signal_threshold)
        # Match training: move criterion buffers to model's device. Prevents
        # future crashes with non-scalar buffers (see note in lstm_train.py).
        criterion = criterion.to(device)
        # `is_custom` controls per-batch index injection (sample weights) and
        # set_epoch_weights/threshold preprocessing. Both LSTMCustomLoss and
        # SignalLoss accept the `indices` kwarg, so we group them together.
        is_custom = isinstance(criterion, (LSTMCustomLoss, SignalLoss))

        # Pre-compute per-sample weights exactly as training does. Without
        # this, LSTMCustomLoss(use_weights=True) trials would silently run
        # without the sample-weight multiplier and therefore produce loss
        # curves that don't match training behaviour.
        # NOTE: narrowed to LSTMCustomLoss — SignalLoss has no set_epoch_weights
        # / set_misdir_threshold (mirrors lstm_train.py linia 426). is_custom
        # above still covers both for the indices kwarg in train/validate_epoch.
        if isinstance(criterion, LSTMCustomLoss):
            y_raw = (target_scaler.inverse_transform(data['y_train'])
                     if target_scaler is not None else data['y_train'])
            if _cfg.USE_SAMPLE_WEIGHTS:
                criterion.set_epoch_weights(y_raw)
            # F2.V1-2: stabilise misdir threshold by computing it once from
            # the full epoch's raw targets, not per-batch.
            if _cfg.USE_MISDIR_PENALTY:
                criterion.set_misdir_threshold(y_raw)

        # Optimizer + Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
        )

        train_loader = torch.utils.data.DataLoader(
            CryptoDataset(data['X_train'], data['y_train']),
            batch_size=params['batch_size'], shuffle=True, drop_last=True,
            num_workers=LSTMConfig.NUM_WORKERS,
            pin_memory=(device.type == 'cuda' and LSTMConfig.USE_PIN_MEMORY),
        )
        val_loader = torch.utils.data.DataLoader(
            CryptoDataset(data['X_val'], data['y_val']),
            batch_size=params['batch_size'], shuffle=False,
            num_workers=LSTMConfig.NUM_WORKERS,
            pin_memory=(device.type == 'cuda' and LSTMConfig.USE_PIN_MEMORY),
        )

        total_steps = tuning_cfg.TUNING_EPOCHS * len(train_loader)
        scheduler = get_scheduler(
            optimizer, params['scheduler_type'],
            total_steps=total_steps,
            lr_warmup_epochs=params.get('lr_warmup_epochs', 0),
            **LSTMConfig.SCHEDULER_PARAMS,
        )

        augmenter = SequenceAugmenter.from_trial_params(params)

        # Training loop (reduced epochs)
        best_val_loss = float('inf')
        best_state = None
        patience = 0
        is_onecycle = params['scheduler_type'] == 'onecycle'

        for epoch in range(tuning_cfg.TUNING_EPOCHS):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                tuning_cfg.GRADIENT_CLIP, is_custom,
                batch_scheduler=scheduler if is_onecycle else None,
                augmenter=augmenter,
            )
            val_loss = validate_epoch(
                model, val_loader, criterion, device, is_custom,
            )

            if scheduler is not None and not is_onecycle:
                if params['scheduler_type'] == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= tuning_cfg.TUNING_PATIENCE:
                    break

        # Return loss as score based on SCORE_SOURCE config
        score_source = tuning_cfg.SCORE_SOURCE
        if score_source == 'train':
            score = train_loss  # last epoch train_loss
            score_label = 'train_loss'
        else:
            score = best_val_loss
            score_label = 'val_loss'

        # F2.V1-5 composite score: extra forward pass on val to inverse-
        # transform predictions and add collapse / R² penalties. Skipped
        # when OPTUNA_SCORE_TYPE != 'composite' so default behaviour is
        # unchanged. Uses last-epoch weights; `best_val_loss` already
        # provides the early-stopping floor so this is a fair proxy for
        # final model behaviour without re-loading the best state.
        if getattr(LSTMConfig, 'OPTUNA_SCORE_TYPE', 'val_loss') in (
                'composite', 'signal'):
            if best_state is not None:
                model.load_state_dict(best_state)
                model.to(device)
            y_val_pred = predict_all(model, val_loader, device)
            y_val_true = np.concatenate(
                [y.numpy() for _, y, _ in val_loader])
            score = compute_composite_score(
                score, y_val_pred, y_val_true, target_scaler, LSTMConfig)
            score_label = getattr(LSTMConfig, 'OPTUNA_SCORE_TYPE', 'composite')

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Trial {trial.number:3d}: {score_label}={score:.6f} "
              f"(params={n_params:,}, epochs={epoch + 1})")

        return score

    return objective
