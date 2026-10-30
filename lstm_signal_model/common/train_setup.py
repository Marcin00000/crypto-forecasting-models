"""
Config resolution and loss construction for Signal Model training.

Separated from signal_train.py to keep the orchestrator slim.
"""
from __future__ import annotations

from .losses import get_signal_loss
from .training_profiles import SIGNAL_PROFILES, normalize_profile_name


def apply_config_defaults(config_class) -> dict:
    """Resolve all SignalConfig class attributes → flat dict."""
    return {
        'lookback': config_class.DEFAULT_LOOKBACK,
        'horizon': config_class.DEFAULT_HORIZON,
        'feature_set': config_class.DEFAULT_FEATURE_SET,
        'preferred_data_format': config_class.PREFERRED_DATA_FORMAT,
        'val_source': config_class.VAL_SOURCE,
        'hidden_size': config_class.HIDDEN_SIZE,
        'num_layers': config_class.NUM_LAYERS,
        'dropout': config_class.DROPOUT,
        'bidirectional': config_class.BIDIRECTIONAL,
        'use_attention': config_class.USE_ATTENTION,
        'fc_layers': list(config_class.FC_LAYERS),
        'activation': config_class.ACTIVATION,
        'fc_norm': config_class.FC_NORM,
        'rnn_type': config_class.RNN_TYPE,
        'attention_heads': config_class.ATTENTION_HEADS,
        'lstm_dropout': config_class.LSTM_DROPOUT,
        'fc_dropout': config_class.FC_DROPOUT,
        'use_input_norm': config_class.USE_INPUT_NORM,
        'use_submodel_norm': getattr(config_class, 'USE_SUBMODEL_NORM', True),
        'submodel_norm_type': getattr(config_class, 'SUBMODEL_NORM_TYPE', 'auto'),
        'use_feature_attention': config_class.USE_FEATURE_ATTENTION,
        'weight_init': config_class.WEIGHT_INIT,
        # F6 flags
        'use_conv1d_front': getattr(config_class, 'USE_CONV1D_FRONT', False),
        'conv1d_kernel': getattr(config_class, 'CONV1D_KERNEL', 5),
        'pooling_mode': getattr(config_class, 'POOLING_MODE', 'last'),
        'use_fc_residual': getattr(config_class, 'USE_FC_RESIDUAL', False),
        'use_rnn_norm': getattr(config_class, 'USE_RNN_NORM', False),
        'use_learnable_output_scale': getattr(
            config_class, 'USE_LEARNABLE_OUTPUT_SCALE', False),
        'freeze_rnn': getattr(config_class, 'FREEZE_RNN', False),
        'output_mode': config_class.OUTPUT_MODE,
        'training_profile': config_class.TRAINING_PROFILE,
        'epochs': config_class.EPOCHS,
        'batch_size': config_class.BATCH_SIZE,
        'learning_rate': config_class.LEARNING_RATE,
        'weight_decay': config_class.WEIGHT_DECAY,
        'early_stopping_patience': config_class.EARLY_STOPPING_PATIENCE,
        'min_epochs_before_es': config_class.MIN_EPOCHS_BEFORE_ES,
        'early_stopping_min_delta': config_class.EARLY_STOPPING_MIN_DELTA,
        'gradient_clip': config_class.GRADIENT_CLIP,
        'seed': config_class.SEED,
        'scheduler_type': config_class.SCHEDULER_TYPE,
        'scheduler_patience': config_class.SCHEDULER_PATIENCE,
        'scheduler_factor': config_class.SCHEDULER_FACTOR,
        'lr_warmup_epochs': config_class.LR_WARMUP_EPOCHS,
        'num_workers': config_class.NUM_WORKERS,
        'use_pin_memory': config_class.USE_PIN_MEMORY,
        'loss_type': config_class.LOSS_TYPE,
        'loss_schedule_ramp_epochs': config_class.LOSS_SCHEDULE_RAMP_EPOCHS,
        'loss_diversity_weight': getattr(config_class, 'LOSS_DIVERSITY_WEIGHT', 1.0),
        'loss_diversity_target': getattr(config_class, 'LOSS_DIVERSITY_TARGET', 0.30),
        'focal_gamma': getattr(config_class, 'FOCAL_GAMMA', 2.0),
        'w_buy': getattr(config_class, 'ASYMMETRIC_W_BUY', 1.0),
        'w_sell': getattr(config_class, 'ASYMMETRIC_W_SELL', 1.0),
        'use_buy_sell_corr_penalty': getattr(config_class, 'USE_BUY_SELL_CORR_PENALTY', False),
        'std_match_weight': getattr(config_class, 'STD_MATCH_WEIGHT', 0.1),
        'corr_penalty_weight': getattr(config_class, 'CORR_PENALTY_WEIGHT', 0.5),
        'label_smoothing_alpha': getattr(config_class, 'LABEL_SMOOTHING_ALPHA', 0.0),
        'calibrate': getattr(config_class, 'CALIBRATE', False),
        'submodel_size': getattr(config_class, 'SUBMODEL_SIZE', 0),
        'feature_round_decimals': getattr(config_class, 'FEATURE_ROUND_DECIMALS', None),
        'target_round_decimals': getattr(config_class, 'TARGET_ROUND_DECIMALS', None),
        'arch_type': getattr(config_class, 'ARCH_TYPE', 'legacy'),
        'signal_threshold': config_class.SIGNAL_THRESHOLD,
        'window_sigma': config_class.WINDOW_SIGMA,
        'winsorize_percentile': getattr(config_class, 'WINSORIZE_PERCENTILE', None),
        'min_signal': config_class.MIN_SIGNAL,
        'log_signal_diagnostics': config_class.LOG_SIGNAL_DIAGNOSTICS,
        'log_gradient_norms': config_class.LOG_GRADIENT_NORMS,
        'best_model_metric': config_class.BEST_MODEL_METRIC,
        'best_model_diversity_weight': config_class.BEST_MODEL_DIVERSITY_WEIGHT,
        'val_signal_min_epoch': getattr(config_class, 'VAL_SIGNAL_MIN_EPOCH', 10),
        'use_swa': getattr(config_class, 'USE_SWA', False),
        'swa_start_epoch': getattr(config_class, 'SWA_START_EPOCH', 30),
        'swa_every': getattr(config_class, 'SWA_EVERY', 1),
        'aug_jitter': config_class.AUG_JITTER,
        'aug_jitter_std': config_class.AUG_JITTER_STD,
        'aug_temporal_dropout': config_class.AUG_TEMPORAL_DROPOUT,
        'aug_temporal_dropout_rate': config_class.AUG_TEMPORAL_DROPOUT_RATE,
        'aug_subsequence': config_class.AUG_SUBSEQUENCE,
        'aug_subsequence_offset': config_class.AUG_SUBSEQUENCE_OFFSET,
    }


def apply_profile_overrides(resolved, profile_name, explicit_keys=None):
    """Override resolved dict with values from the named profile.

    Args:
        resolved:      Current resolved config dict.
        profile_name:  Profile name to apply.
        explicit_keys: Set of keys explicitly provided by the caller (e.g.
                       from CLI). Profile values for these keys are ignored
                       so that CLI always wins over profile defaults.
    """
    profile_name = normalize_profile_name(profile_name)
    resolved['training_profile'] = profile_name
    profile = SIGNAL_PROFILES[profile_name]
    for key, val in profile.get('resolved', {}).items():
        if explicit_keys is None or key not in explicit_keys:
            resolved[key] = val
    return resolved


def apply_cli_overrides(resolved, cli_kwargs):
    """CLI keyword args override resolved values (None = skip).

    Returns (updated_resolved, explicit_keys) where explicit_keys is the set
    of keys that were actually provided (non-None) by the caller — used to
    protect those keys from being overwritten by apply_profile_overrides.
    """
    explicit_keys = set()
    for key, val in cli_kwargs.items():
        if val is not None:
            resolved[key] = val
            explicit_keys.add(key)
            if key == 'aug_subsequence_min_frac':
                # CLI alias — SequenceAugmenter uses offset (int timesteps).
                resolved['aug_subsequence_offset'] = int(val) if val >= 1 else 1
                explicit_keys.add('aug_subsequence_offset')
    return resolved, explicit_keys


def apply_optuna_overrides(resolved, load_optuna_params, explicit_keys=None):
    """
    Overlay Optuna best_params.json onto resolved config.

    Explicit CLI keys (in explicit_keys) take precedence over Optuna values.
    Returns (updated_resolved, optuna_id_or_None).
    """
    if not load_optuna_params:
        return resolved, None

    try:
        from signal_config import SignalConfig
    except ImportError:
        from lstm_signal_model.signal_config import SignalConfig
    params, metadata = SignalConfig.load_optuna_params(load_optuna_params)
    updated = dict(resolved)

    direct_keys = (
        'hidden_size', 'num_layers', 'dropout', 'batch_size', 'weight_decay',
        'lookback', 'horizon', 'scheduler_type', 'activation', 'fc_norm',
        'attention_heads', 'fc_layers', 'signal_threshold', 'window_sigma',
        'aug_jitter_std', 'aug_temporal_dropout_rate', 'aug_subsequence_offset',
        'lstm_dropout', 'fc_dropout', 'rnn_type', 'output_mode',
        'best_model_metric', 'weight_init', 'loss_type', 'lr_warmup_epochs',
        # F6
        'conv1d_kernel', 'pooling_mode',
    )
    for key in direct_keys:
        if key in params and (explicit_keys is None or key not in explicit_keys):
            updated[key] = params[key]

    if 'learning_rate' in params and (explicit_keys is None or 'learning_rate' not in explicit_keys):
        updated['learning_rate'] = float(params['learning_rate'])

    bool_keys = (
        'use_attention', 'use_feature_attention', 'bidirectional',
        'use_input_norm', 'use_submodel_norm', 'aug_jitter', 'aug_temporal_dropout',
        'aug_subsequence',
        # F6
        'use_conv1d_front', 'use_fc_residual', 'use_rnn_norm',
        'use_learnable_output_scale',
    )
    for key in bool_keys:
        if key in params and (explicit_keys is None or key not in explicit_keys):
            updated[key] = bool(params[key])

    optuna_id = None
    if metadata:
        optuna_id = metadata.get('optuna_id')
        if metadata.get('feature_set') and (explicit_keys is None or 'feature_set' not in explicit_keys):
            updated['feature_set'] = metadata['feature_set']
        profile = metadata.get('training_profile')
        if profile and (explicit_keys is None or 'training_profile' not in explicit_keys):
            updated['training_profile'] = profile

    return updated, optuna_id


def build_signal_loss(resolved):
    """Factory: resolved dict → loss module."""
    kwargs = {
        'loss_type': resolved['loss_type'],
        'ramp_epochs': resolved.get('loss_schedule_ramp_epochs', 0),
        'diversity_weight': resolved.get('loss_diversity_weight', 1.0),
        'diversity_target': resolved.get('loss_diversity_target', 0.30),
        'focal_gamma': resolved.get('focal_gamma', 2.0),
        'w_buy': resolved.get('w_buy', 1.0),
        'w_sell': resolved.get('w_sell', 1.0),
        'std_match_weight': resolved.get('std_match_weight', 0.1),
        'corr_penalty_weight': resolved.get('corr_penalty_weight', 0.5),
        'label_smoothing_alpha': resolved.get('label_smoothing_alpha', 0.0),
    }
    # Only forwarded when explicitly enabled, so the default call stays
    # byte-identical with the historical one (get_signal_loss decides the
    # penalty per loss_type on its own).
    if resolved.get('use_buy_sell_corr_penalty'):
        kwargs['use_corr_penalty'] = True

    # CP-173 optional loss flags
    for flag in ('use_signal_utility', 'signal_utility_weight',
                 'use_r2_loss', 'r2_loss_weight',
                 'use_ccc_loss', 'ccc_loss_weight',
                 'use_cr_ceiling', 'cr_ceiling_weight', 'cr_ceiling'):
        if flag in resolved:
            kwargs[flag] = resolved[flag]

    return get_signal_loss(**kwargs)
