"""
Setup helpers for `lstm_train.train()`.

This module shrinks `lstm_train.py` by extracting six blocks that were
previously inlined:

    apply_config_defaults        — resolve None-valued kwargs from LSTMConfig
    apply_optuna_overrides       — overlay Optuna best_params on top of defaults
    apply_resolved_to_cfg        — mirror architecture knobs onto LSTMConfig
    apply_custom_loss_flags      — mirror CLI/kwargs onto LSTMConfig attributes
    build_scheduled_loss         — assemble (base_criterion, full_criterion,
                                   warmup) tuple for FAZA 5 loss switching

`build_train_config_dict` lives in `train_config_dict.py`.
`apply_config_defaults` and `apply_optuna_overrides` are side-effect-free.
"""
from __future__ import annotations

from common.tuning_baseline import coerce_resolved_bool
from common.train_loss_flags import apply_custom_loss_flags


# =========================================================================
# DEFAULTS
# =========================================================================

def apply_config_defaults(cfg, *,
                          lookback=None, horizon=None,
                          feature_set=None, custom_features=None, preferred_format=None,
                          val_source=None, data_source=None,
                          training_profile=None,
                          hidden_size=None, num_layers=None, dropout=None,
                          bidirectional=None, use_attention=None,
                          fc_layers=None, activation=None, fc_norm=None,
                          arch_type=None, rnn_type=None,
                          use_input_norm=None, weight_init=None,
                          attention_heads=None,
                          lstm_dropout=None, fc_dropout=None,
                          epochs=None, batch_size=None, lr=None,
                          weight_decay=None, early_stopping_patience=None,
                          gradient_clip=None,
                          best_model_metric=None, best_model_cr_weight=None,
                          prediction_precision=None, target_precision=None,
                          feature_round_decimals=None, target_round_decimals=None,
                          log_gradient_norms=None,
                          scale_target=None, target_scaler_type=None,
                          winsorize_target=None, winsorize_percentile=None,
                          loss_type=None, huber_delta=None,
                          signal_threshold_enabled=None,
                          signal_threshold_mode=None,
                          signal_threshold_value=None,
                          scheduler_type=None, seed=None,
                          signal_ema_penalties=None, signal_ema_alpha=None,
                          signal_label_smoothing=None,
                          signal_label_smoothing_alpha=None,
                          aug_jitter=None, aug_jitter_std=None,
                          aug_temporal_dropout=None,
                          aug_temporal_dropout_rate=None,
                          aug_subsequence=None, aug_subsequence_offset=None,
                          use_feature_attention=None,
                          use_rnn_norm=None,
                          pooling_mode=None,
                          use_conv1d_front=None,
                          conv1d_kernel=None,
                          use_fc_residual=None,
                          use_learnable_output_scale=None,
                          freeze_rnn=None,
                          lr_warmup_epochs=None,
                          min_epochs_before_es=None,
                          early_stopping_min_delta=None,
                          use_swa=None,
                          swa_start_epoch=None,
                          swa_every=None,
                          use_scheduled_loss=None,
                          scheduled_loss_warmup_epochs=None,
                          scheduled_loss_base=None,
                          scheduled_loss_full=None,
                          target_mode=None,
                          use_aux_volatility_head=None,
                          use_group_dro=None,
                          group_dro_n_groups=None,
                          group_dro_step_size=None):
    """
    Resolve every None kwarg from LSTMConfig and return a flat dict.

    Caller pattern in `train()`:
        resolved = apply_config_defaults(LSTMConfig, lookback=lookback, ...)
        lookback = resolved['lookback']

    Keeping the resolution out of `train()` removes ~40 lines of `x if x is
    not None else cfg.X` boilerplate while preserving the exact semantics
    (only None gets replaced).
    """
    return {
        'lookback': lookback if lookback is not None else cfg.DEFAULT_LOOKBACK,
        'horizon': horizon if horizon is not None else cfg.DEFAULT_HORIZON,
        'feature_set': feature_set if feature_set is not None else ('custom' if custom_features is not None else cfg.DEFAULT_FEATURE_SET),
        'custom_features': custom_features,
        'preferred_format': preferred_format if preferred_format is not None else cfg.PREFERRED_DATA_FORMAT,
        'val_source': val_source if val_source is not None else cfg.VAL_SOURCE,
        'data_source': data_source if data_source is not None else getattr(cfg, 'DATA_SOURCE', 'train'),
        'training_profile': (training_profile
                             if training_profile is not None
                             else getattr(cfg, 'TRAINING_PROFILE', 'baseline')),
        'hidden_size': hidden_size if hidden_size is not None else cfg.HIDDEN_SIZE,
        'num_layers': num_layers if num_layers is not None else cfg.NUM_LAYERS,
        'dropout': dropout if dropout is not None else cfg.DROPOUT,
        'bidirectional': (bidirectional if bidirectional is not None
                          else cfg.BIDIRECTIONAL),
        'use_attention': (use_attention if use_attention is not None
                          else cfg.USE_ATTENTION),
        'fc_layers': fc_layers if fc_layers is not None else cfg.FC_LAYERS,
        'activation': activation if activation is not None else cfg.ACTIVATION,
        'fc_norm': fc_norm if fc_norm is not None else getattr(cfg, 'FC_NORM', 'layer'),
        'arch_type': arch_type if arch_type is not None else getattr(cfg, 'ARCH_TYPE', 'legacy'),
        'rnn_type': rnn_type if rnn_type is not None else getattr(cfg, 'RNN_TYPE', 'lstm'),
        'use_input_norm': (use_input_norm if use_input_norm is not None
                           else getattr(cfg, 'USE_INPUT_NORM', False)),
        'weight_init': (weight_init if weight_init is not None
                        else getattr(cfg, 'WEIGHT_INIT', 'default')),
        'attention_heads': (attention_heads if attention_heads is not None
                            else getattr(cfg, 'ATTENTION_HEADS', 1)),
        # `None` is a meaningful default for split dropout — it means
        # "inherit DROPOUT". So we cannot reuse the `x if x is not None else
        # cfg.X` idiom that flips on None elsewhere; here we always read the
        # cfg attribute when the kwarg is None.
        'lstm_dropout': (lstm_dropout if lstm_dropout is not None
                         else getattr(cfg, 'LSTM_DROPOUT', None)),
        'fc_dropout': (fc_dropout if fc_dropout is not None
                       else getattr(cfg, 'FC_DROPOUT', None)),
        'prediction_precision': (prediction_precision
                                 if prediction_precision is not None
                                 else getattr(cfg, 'PREDICTION_PRECISION', 0)),
        'target_precision': (target_precision
                             if target_precision is not None
                             else getattr(cfg, 'TARGET_PRECISION', 0)),
        'feature_round_decimals': (feature_round_decimals
                                   if feature_round_decimals is not None
                                   else getattr(cfg, 'FEATURE_ROUND_DECIMALS', 0)),
        'target_round_decimals': (target_round_decimals
                                  if target_round_decimals is not None
                                  else getattr(cfg, 'TARGET_ROUND_DECIMALS', 0)),
        'log_gradient_norms': (log_gradient_norms
                               if log_gradient_norms is not None
                               else getattr(cfg, 'LOG_GRADIENT_NORMS', False)),
        'epochs': epochs if epochs is not None else cfg.EPOCHS,
        'batch_size': batch_size if batch_size is not None else cfg.BATCH_SIZE,
        'lr': lr if lr is not None else cfg.LEARNING_RATE,
        'weight_decay': (weight_decay if weight_decay is not None
                         else cfg.WEIGHT_DECAY),
        'early_stopping_patience': (early_stopping_patience
                                    if early_stopping_patience is not None
                                    else cfg.EARLY_STOPPING_PATIENCE),
        'gradient_clip': (gradient_clip if gradient_clip is not None
                          else cfg.GRADIENT_CLIP),
        'best_model_metric': (best_model_metric
                              if best_model_metric is not None
                              else getattr(cfg, 'BEST_MODEL_METRIC', 'val')),
        'best_model_cr_weight': (best_model_cr_weight
                                 if best_model_cr_weight is not None
                                 else getattr(cfg, 'BEST_MODEL_CR_WEIGHT', 0.01)),
        'scale_target': (scale_target if scale_target is not None
                         else cfg.SCALE_TARGET),
        'target_scaler_type': target_scaler_type if target_scaler_type is not None else cfg.TARGET_SCALER_TYPE,
        'target_mode': (target_mode if target_mode is not None
                        else getattr(cfg, 'TARGET_MODE', 'raw')),
        'use_aux_volatility_head': (use_aux_volatility_head
                                    if use_aux_volatility_head is not None
                                    else getattr(cfg, 'USE_AUX_VOLATILITY_HEAD', False)),
        'use_group_dro': (use_group_dro if use_group_dro is not None
                          else getattr(cfg, 'USE_GROUP_DRO', False)),
        'group_dro_n_groups': (group_dro_n_groups if group_dro_n_groups is not None
                               else getattr(cfg, 'GROUP_DRO_N_GROUPS', 4)),
        'group_dro_step_size': (group_dro_step_size if group_dro_step_size is not None
                                else getattr(cfg, 'GROUP_DRO_STEP_SIZE', 0.01)),
        'winsorize_target': (winsorize_target if winsorize_target is not None
                             else cfg.WINSORIZE_TARGET),
        'winsorize_percentile': (winsorize_percentile
                                 if winsorize_percentile is not None
                                 else cfg.WINSORIZE_PERCENTILE),
        'loss_type': loss_type if loss_type is not None else cfg.LOSS_TYPE,
        'huber_delta': (huber_delta if huber_delta is not None
                        else cfg.HUBER_DELTA),
        'signal_threshold_enabled': (signal_threshold_enabled
                                     if signal_threshold_enabled is not None
                                     else cfg.SIGNAL_THRESHOLD_ENABLED),
        'signal_threshold_mode': (signal_threshold_mode
                                  if signal_threshold_mode is not None
                                  else cfg.SIGNAL_THRESHOLD_MODE),
        'signal_threshold_value': (signal_threshold_value
                                   if signal_threshold_value is not None
                                   else cfg.SIGNAL_THRESHOLD_VALUE),
        'scheduler_type': scheduler_type if scheduler_type is not None else cfg.SCHEDULER_TYPE,
        'seed': seed if seed is not None else cfg.SEED,
        'signal_ema_penalties': (signal_ema_penalties
                                 if signal_ema_penalties is not None
                                 else getattr(cfg, 'SIGNAL_EMA_PENALTIES', False)),
        'signal_ema_alpha': (signal_ema_alpha
                             if signal_ema_alpha is not None
                             else getattr(cfg, 'SIGNAL_EMA_ALPHA', 0.1)),
        'signal_label_smoothing': (signal_label_smoothing
                                   if signal_label_smoothing is not None
                                   else getattr(cfg, 'SIGNAL_LABEL_SMOOTHING',
                                                False)),
        'signal_label_smoothing_alpha': (
            signal_label_smoothing_alpha
            if signal_label_smoothing_alpha is not None
            else getattr(cfg, 'SIGNAL_LABEL_SMOOTHING_ALPHA', 0.5)),
        'aug_jitter': (aug_jitter if aug_jitter is not None
                       else getattr(cfg, 'AUG_JITTER', False)),
        'aug_jitter_std': (aug_jitter_std if aug_jitter_std is not None
                           else getattr(cfg, 'AUG_JITTER_STD', 0.01)),
        'aug_temporal_dropout': (aug_temporal_dropout
                                 if aug_temporal_dropout is not None
                                 else getattr(cfg, 'AUG_TEMPORAL_DROPOUT', False)),
        'aug_temporal_dropout_rate': (
            aug_temporal_dropout_rate
            if aug_temporal_dropout_rate is not None
            else getattr(cfg, 'AUG_TEMPORAL_DROPOUT_RATE', 0.05)),
        'aug_subsequence': (aug_subsequence if aug_subsequence is not None
                            else getattr(cfg, 'AUG_SUBSEQUENCE', False)),
        'aug_subsequence_offset': (
            aug_subsequence_offset
            if aug_subsequence_offset is not None
            else getattr(cfg, 'AUG_SUBSEQUENCE_OFFSET', 2)),
        'use_feature_attention': (use_feature_attention
                                  if use_feature_attention is not None
                                  else getattr(cfg, 'USE_FEATURE_ATTENTION',
                                               False)),
        'use_rnn_norm': (use_rnn_norm if use_rnn_norm is not None
                         else getattr(cfg, 'USE_RNN_NORM', False)),
        'pooling_mode': (pooling_mode if pooling_mode is not None
                         else getattr(cfg, 'POOLING_MODE', 'last')),
        'use_conv1d_front': (use_conv1d_front
                             if use_conv1d_front is not None
                             else getattr(cfg, 'USE_CONV1D_FRONT', False)),
        'conv1d_kernel': (conv1d_kernel if conv1d_kernel is not None
                          else getattr(cfg, 'CONV1D_KERNEL', 5)),
        'use_fc_residual': (use_fc_residual if use_fc_residual is not None
                            else getattr(cfg, 'USE_FC_RESIDUAL', False)),
        'use_learnable_output_scale': (
            use_learnable_output_scale
            if use_learnable_output_scale is not None
            else getattr(cfg, 'USE_LEARNABLE_OUTPUT_SCALE', False)),
        'freeze_rnn': (freeze_rnn if freeze_rnn is not None
                       else getattr(cfg, 'FREEZE_RNN', False)),
        'lr_warmup_epochs': (lr_warmup_epochs
                             if lr_warmup_epochs is not None
                             else getattr(cfg, 'LR_WARMUP_EPOCHS', 0)),
        'min_epochs_before_es': (min_epochs_before_es
                                 if min_epochs_before_es is not None
                                 else getattr(cfg, 'MIN_EPOCHS_BEFORE_ES', 0)),
        'early_stopping_min_delta': (
            early_stopping_min_delta
            if early_stopping_min_delta is not None
            else getattr(cfg, 'EARLY_STOPPING_MIN_DELTA', 0.0)),
        'use_swa': (use_swa if use_swa is not None
                    else getattr(cfg, 'USE_SWA', False)),
        'swa_start_epoch': (swa_start_epoch
                            if swa_start_epoch is not None
                            else getattr(cfg, 'SWA_START_EPOCH', 30)),
        'swa_every': (swa_every if swa_every is not None
                      else getattr(cfg, 'SWA_EVERY', 1)),
        'use_scheduled_loss': (use_scheduled_loss
                               if use_scheduled_loss is not None
                               else getattr(cfg, 'USE_SCHEDULED_LOSS', False)),
        'scheduled_loss_warmup_epochs': (
            scheduled_loss_warmup_epochs
            if scheduled_loss_warmup_epochs is not None
            else getattr(cfg, 'SCHEDULED_LOSS_WARMUP_EPOCHS', 10)),
        'scheduled_loss_base': (scheduled_loss_base
                                if scheduled_loss_base is not None
                                else getattr(cfg, 'SCHEDULED_LOSS_BASE', 'mse')),
        'scheduled_loss_full': (scheduled_loss_full
                                if scheduled_loss_full is not None
                                else getattr(cfg, 'SCHEDULED_LOSS_FULL', 'custom')),
    }


# =========================================================================
# OPTUNA OVERRIDES
# =========================================================================

_OPTUNA_BOOL_RESOLVED = frozenset({
    'aug_jitter', 'aug_temporal_dropout', 'aug_subsequence',
    'signal_ema_penalties', 'signal_label_smoothing', 'use_feature_attention',
    'use_input_norm',
})

_OPTUNA_DIRECT_RESOLVED = (
    'lr_warmup_epochs',
    'aug_jitter_std', 'aug_temporal_dropout_rate', 'aug_subsequence_offset',
    'signal_ema_alpha', 'signal_label_smoothing_alpha',
    'signal_r2_weight', 'signal_ccc_weight', 'signal_std_match_weight',
    'signal_log_var_weight', 'signal_direction_bce_weight',
    'signal_dir_threshold_weight',
    'amplified_zone_threshold', 'amplified_zone_weight',
)


def apply_optuna_overrides(resolved, cfg, load_optuna_params, *,
                           quantile_q=None):
    """
    Overlay Optuna's best_params on top of CLI/defaults.

    Returns (updated_resolved, optuna_config_name, quantile_q). The mutation
    style mirrors the original `train()` block exactly: each `params.get(k,
    fallback)` falls back to whatever was already resolved by
    `apply_config_defaults`.

    For LOSS_TYPE='custom' trials, Phase A snapshots loss component flags in
    `metadata['loss_config']`; this function pushes those onto `cfg` so the
    downstream LSTMCustomLoss factory builds exactly the trial's loss.
    """
    if load_optuna_params is None:
        return resolved, None, quantile_q

    updated = dict(resolved)
    params, metadata = cfg.load_optuna_params(load_optuna_params)

    updated['hidden_size'] = params.get('hidden_size', updated['hidden_size'])
    updated['num_layers'] = params.get('num_layers', updated['num_layers'])
    updated['dropout'] = params.get('dropout', updated['dropout'])
    updated['lr'] = params.get('learning_rate', updated['lr'])
    updated['batch_size'] = params.get('batch_size', updated['batch_size'])
    updated['weight_decay'] = params.get('weight_decay',
                                         updated['weight_decay'])
    updated['lookback'] = params.get('lookback', updated['lookback'])
    updated['horizon'] = params.get('horizon', updated['horizon'])
    updated['scheduler_type'] = params.get('scheduler_type',
                                           updated['scheduler_type'])
    # loss_type is stored in metadata (Phase A), not in trial.params.
    # huber_delta and quantile_q remain in trial.params (tuned by TPE).
    updated['huber_delta'] = params.get('huber_delta', updated['huber_delta'])
    quantile_q = params.get('quantile_q', quantile_q)
    updated['bidirectional'] = params.get('bidirectional',
                                          updated['bidirectional'])
    updated['use_attention'] = params.get('use_attention',
                                          updated['use_attention'])
    updated['fc_layers'] = params.get('fc_layers', updated['fc_layers'])
    updated['activation'] = params.get('activation', updated['activation'])
    updated['fc_norm'] = params.get('fc_norm', updated['fc_norm'])
    # F4.M3 / M4: architecture knobs that Optuna may tune.
    updated['rnn_type'] = params.get('rnn_type', updated['rnn_type'])
    updated['attention_heads'] = params.get(
        'attention_heads', updated['attention_heads'])
    updated['lstm_dropout'] = params.get(
        'lstm_dropout', updated['lstm_dropout'])
    updated['fc_dropout'] = params.get(
        'fc_dropout', updated['fc_dropout'])
    if 'use_feature_attention' in params:
        updated['use_feature_attention'] = coerce_resolved_bool(
            params['use_feature_attention'])

    for key in _OPTUNA_DIRECT_RESOLVED:
        if key in params:
            updated[key] = params[key]
    for key in _OPTUNA_BOOL_RESOLVED:
        if key in params:
            updated[key] = coerce_resolved_bool(params[key])

    optuna_config_name = None
    if metadata:
        updated['feature_set'] = metadata.get('feature_set',
                                              updated['feature_set'])
        optuna_config_name = metadata.get('optuna_id')
        meta_loss = metadata.get('loss_type')
        if meta_loss:
            updated['loss_type'] = meta_loss
        loss_cfg = metadata.get('loss_config', {})
        if updated['loss_type'] == 'custom' and loss_cfg:
            for key, val in loss_cfg.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
        elif updated['loss_type'] == 'signal' and loss_cfg:
            # Mirror signal loss flags from tuning snapshot when present.
            _signal_map = {
                'SIGNAL_R2_WEIGHT': 'signal_r2_weight',
                'SIGNAL_CCC_WEIGHT': 'signal_ccc_weight',
                'SIGNAL_STD_MATCH_WEIGHT': 'signal_std_match_weight',
                'SIGNAL_LOG_VAR_WEIGHT': 'signal_log_var_weight',
                'SIGNAL_DIRECTION_BCE_WEIGHT': 'signal_direction_bce_weight',
                'SIGNAL_DIR_THRESHOLD_WEIGHT': 'signal_dir_threshold_weight',
                'SIGNAL_EMA_PENALTIES': 'signal_ema_penalties',
                'SIGNAL_EMA_ALPHA': 'signal_ema_alpha',
                'SIGNAL_LABEL_SMOOTHING': 'signal_label_smoothing',
                'SIGNAL_LABEL_SMOOTHING_ALPHA': 'signal_label_smoothing_alpha',
            }
            for cfg_key, resolved_key in _signal_map.items():
                if cfg_key in loss_cfg and resolved_key not in updated:
                    val = loss_cfg[cfg_key]
                    if resolved_key in _OPTUNA_BOOL_RESOLVED:
                        val = coerce_resolved_bool(val)
                    updated[resolved_key] = val

    return updated, optuna_config_name, quantile_q


def merge_train_kwargs_from_resolved(resolved, **kwargs):
    """Prefer Optuna-loaded values in ``resolved`` over CLI kwargs."""
    merged = dict(kwargs)
    for key, value in merged.items():
        if key in resolved:
            merged[key] = resolved[key]
    return merged


# =========================================================================
# ARCHITECTURE FLAG MIRRORING (build_model reads from cfg)
# =========================================================================

# Mapping of `resolved` dict keys → LSTMConfig attribute names. These are the
# architecture/training knobs that `build_model()` and `train_model()` read
# from the live config class, so they must be pushed onto cfg before the
# model is built. Kept here (rather than inlined in `train()`) so adding a
# new knob means touching exactly one location.
_RESOLVED_TO_CFG = {
    'hidden_size': 'HIDDEN_SIZE',
    'num_layers': 'NUM_LAYERS',
    'dropout': 'DROPOUT',
    'bidirectional': 'BIDIRECTIONAL',
    'use_attention': 'USE_ATTENTION',
    'fc_layers': 'FC_LAYERS',
    'activation': 'ACTIVATION',
    'fc_norm': 'FC_NORM',
    'arch_type': 'ARCH_TYPE',
    'rnn_type': 'RNN_TYPE',
    'use_input_norm': 'USE_INPUT_NORM',
    'weight_init': 'WEIGHT_INIT',
    'attention_heads': 'ATTENTION_HEADS',
    'lstm_dropout': 'LSTM_DROPOUT',
    'fc_dropout': 'FC_DROPOUT',
    'training_profile': 'TRAINING_PROFILE',
    'prediction_precision': 'PREDICTION_PRECISION',
    'target_precision': 'TARGET_PRECISION',
    'feature_round_decimals': 'FEATURE_ROUND_DECIMALS',
    'target_round_decimals': 'TARGET_ROUND_DECIMALS',
    'log_gradient_norms': 'LOG_GRADIENT_NORMS',
    'best_model_metric': 'BEST_MODEL_METRIC',
    'best_model_cr_weight': 'BEST_MODEL_CR_WEIGHT',
    'use_feature_attention': 'USE_FEATURE_ATTENTION',
    'use_rnn_norm': 'USE_RNN_NORM',
    'pooling_mode': 'POOLING_MODE',
    'use_conv1d_front': 'USE_CONV1D_FRONT',
    'conv1d_kernel': 'CONV1D_KERNEL',
    'use_fc_residual': 'USE_FC_RESIDUAL',
    'use_learnable_output_scale': 'USE_LEARNABLE_OUTPUT_SCALE',
    'freeze_rnn': 'FREEZE_RNN',
    'min_epochs_before_es': 'MIN_EPOCHS_BEFORE_ES',
    'early_stopping_min_delta': 'EARLY_STOPPING_MIN_DELTA',
    'use_swa': 'USE_SWA',
    'swa_start_epoch': 'SWA_START_EPOCH',
    'swa_every': 'SWA_EVERY',
}

_AUG_LR_TO_CFG = {
    'aug_jitter': 'AUG_JITTER',
    'aug_jitter_std': 'AUG_JITTER_STD',
    'aug_temporal_dropout': 'AUG_TEMPORAL_DROPOUT',
    'aug_temporal_dropout_rate': 'AUG_TEMPORAL_DROPOUT_RATE',
    'aug_subsequence': 'AUG_SUBSEQUENCE',
    'aug_subsequence_offset': 'AUG_SUBSEQUENCE_OFFSET',
    'lr_warmup_epochs': 'LR_WARMUP_EPOCHS',
    'signal_ema_penalties': 'SIGNAL_EMA_PENALTIES',
    'signal_ema_alpha': 'SIGNAL_EMA_ALPHA',
    'signal_label_smoothing': 'SIGNAL_LABEL_SMOOTHING',
    'signal_label_smoothing_alpha': 'SIGNAL_LABEL_SMOOTHING_ALPHA',
}


def apply_resolved_to_cfg(cfg, resolved, *, rnn_type=None):
    """
    Push resolved architecture / training values onto `cfg`.

    `build_model()` reads architecture knobs directly off the LSTMConfig
    class (instead of receiving them as arguments), so any value that
    influences model construction has to live on the class by the time
    `build_model` is called. `rnn_type` is now part of `resolved` and is
    applied via `_RESOLVED_TO_CFG`; the optional `rnn_type` kwarg is kept
    for backward compatibility only and is applied only when not None.
    """
    for key, attr in _RESOLVED_TO_CFG.items():
        if key in resolved:
            setattr(cfg, attr, resolved[key])
    for key, attr in _AUG_LR_TO_CFG.items():
        if key in resolved:
            setattr(cfg, attr, resolved[key])
    if rnn_type is not None:
        cfg.RNN_TYPE = rnn_type


# apply_custom_loss_flags — see train_loss_flags.py


# =========================================================================
# SCHEDULED LOSS BUILDER (FAZA 5)
# =========================================================================

def build_scheduled_loss(cfg, target_scaler, signal_threshold, device, *,
                         use_scheduled_loss=None,
                         scheduled_loss_warmup_epochs=None,
                         scheduled_loss_base=None,
                         scheduled_loss_full=None,
                         y_train_raw_for_full=None):
    """
    Build the (base_criterion, full_criterion, warmup) tuple for FAZA 5.

    Returns the tuple when scheduled-loss switching is active, otherwise
    `None`. Each kwarg follows the same `None → fall back to cfg` pattern
    used elsewhere. `y_train_raw_for_full` is used to mirror the LSTMCustomLoss
    epoch hooks (`set_epoch_weights`, `set_misdir_threshold`) onto the full
    criterion when it qualifies. SignalLoss has no such hooks.
    """
    # Lazy imports — train_setup is imported very early during package init,
    # and `losses.py` pulls in the heavy `tuning_*` chain transitively.
    from common.losses import LSTMCustomLoss, get_loss_function
    sched_on = (use_scheduled_loss if use_scheduled_loss is not None
                else getattr(cfg, 'USE_SCHEDULED_LOSS', False))
    if not sched_on:
        return None

    warmup = (scheduled_loss_warmup_epochs
              if scheduled_loss_warmup_epochs is not None
              else getattr(cfg, 'SCHEDULED_LOSS_WARMUP_EPOCHS', 30))
    base_type = (scheduled_loss_base
                 if scheduled_loss_base is not None
                 else getattr(cfg, 'SCHEDULED_LOSS_BASE', 'mae'))
    full_type = (scheduled_loss_full
                 if scheduled_loss_full is not None
                 else getattr(cfg, 'SCHEDULED_LOSS_FULL', 'custom'))

    # Temporarily swap cfg.LOSS_TYPE so each criterion picks the right
    # branch in `get_loss_function`. Component flags (sample weights,
    # misdir, SIGNAL_*) remain on cfg and apply to whichever is built.
    original_loss_type = cfg.LOSS_TYPE
    try:
        cfg.LOSS_TYPE = base_type
        base_criterion = get_loss_function(
            cfg, target_scaler, signal_threshold).to(device)
        cfg.LOSS_TYPE = full_type
        full_criterion = get_loss_function(
            cfg, target_scaler, signal_threshold).to(device)
    finally:
        cfg.LOSS_TYPE = original_loss_type

    # Replicate the per-epoch hooks lstm_train normally runs on the active
    # criterion — the scheduler does not re-call them at the switch.
    if isinstance(full_criterion, LSTMCustomLoss) and y_train_raw_for_full is not None:
        full_criterion.set_epoch_weights(y_train_raw_for_full)
        if getattr(cfg, 'USE_MISDIR_PENALTY', False):
            full_criterion.set_misdir_threshold(y_train_raw_for_full)

    return base_criterion, full_criterion, int(warmup)


# Keys that must not be overridden by training_profile when set on CLI/kwargs.
EXPLICIT_PROFILE_FIELDS = (
    'lookback', 'horizon', 'feature_set', 'preferred_format', 'val_source',
    'hidden_size', 'num_layers', 'dropout', 'bidirectional', 'use_attention',
    'fc_layers', 'activation', 'fc_norm', 'arch_type', 'use_input_norm',
    'weight_init', 'rnn_type', 'attention_heads', 'lstm_dropout', 'fc_dropout',
    'epochs', 'batch_size', 'lr', 'weight_decay', 'early_stopping_patience',
    'gradient_clip', 'best_model_metric', 'best_model_cr_weight',
    'min_epochs_before_es', 'early_stopping_min_delta',
    'use_swa', 'swa_start_epoch', 'swa_every',
    'loss_type', 'use_correlation_penalty', 'correlation_penalty_weight',
    'use_variance_penalty', 'variance_penalty_weight',
    'use_amplified_zone', 'amplified_zone_weight',
    'signal_use_r2', 'signal_use_ccc', 'signal_use_log_var',
    'signal_use_std_match', 'signal_std_match_weight',
    'signal_use_dir_threshold', 'signal_dir_threshold_weight',
    'signal_use_direction_bce', 'signal_direction_bce_weight',
    'signal_use_utility', 'signal_utility_weight', 'signal_soft_threshold',
    'use_cr_ceiling', 'cr_ceiling', 'cr_ceiling_weight',
    'use_rnn_norm', 'pooling_mode', 'use_conv1d_front', 'conv1d_kernel',
    'use_fc_residual', 'use_learnable_output_scale', 'freeze_rnn',
    'scheduler_type', 'lr_warmup_epochs',
)


def collect_explicit_profile_keys(bindings: dict) -> set[str]:
    return {f for f in EXPLICIT_PROFILE_FIELDS
            if bindings.get(f) is not None}
