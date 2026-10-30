"""
Optuna search space and per-parameter freeze logic for LSTM tuning.

Public API:
    _should_tune(tuning_cfg, name) -> bool
    suggest_hyperparameters(trial, tuning_cfg) -> dict
"""
from __future__ import annotations

from lstm_config import LSTMConfig
from common.tuning_baseline import baseline_value


def _should_tune(tuning_cfg, name):
    """True if parameter is NOT frozen via TUNE_PARAMS={name: False} or FREEZE_HYPERPARAMS."""
    if getattr(tuning_cfg, 'FREEZE_HYPERPARAMS', False):
        return name == 'feature_set'
    flags = getattr(tuning_cfg, 'TUNE_PARAMS', {}) or {}
    return flags.get(name, True)


def _frozen(tuning_cfg, name):
    """Value used when a dimension is frozen (CLI / profile baseline > LSTMConfig)."""
    if name == 'feature_set':
        fs = getattr(tuning_cfg, 'FEATURE_SET', None)
        if fs:
            return fs
    return baseline_value(tuning_cfg, name)


def suggest_hyperparameters(trial, tuning_cfg):
    """
    Suggest hyperparameters for a trial.

    Frozen params use ``tuning_cfg.BASELINE_PARAMS`` (from --training_profile
    / --baseline_params) instead of bare LSTMConfig defaults.
    """
    params = {}

    # Feature Set Tuning
    if _should_tune(tuning_cfg, 'feature_set'):
        default_candidates = [
            'lr_core', 'lr_optimal', 'technical', 'technical_bp',
            'technical_mtf', 'max', 'trend_momentum', 'volatility_regime',
            'full_extended'
        ]
        candidates = getattr(tuning_cfg, 'FEATURE_SET_CANDIDATES', None) or default_candidates
        params['feature_set'] = trial.suggest_categorical('feature_set', candidates)
    else:
        params['feature_set'] = _frozen(tuning_cfg, 'feature_set')

    # Architecture
    if _should_tune(tuning_cfg, 'hidden_size'):
        params['hidden_size'] = trial.suggest_categorical(
            'hidden_size', [32, 64, 128, 256])
    else:
        params['hidden_size'] = _frozen(tuning_cfg, 'hidden_size')

    if _should_tune(tuning_cfg, 'num_layers'):
        params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
    else:
        params['num_layers'] = _frozen(tuning_cfg, 'num_layers')

    if _should_tune(tuning_cfg, 'dropout'):
        params['dropout'] = trial.suggest_float('dropout', 0.05, 0.5)
    else:
        params['dropout'] = _frozen(tuning_cfg, 'dropout')

    if _should_tune(tuning_cfg, 'bidirectional'):
        params['bidirectional'] = trial.suggest_categorical(
            'bidirectional', [False, True])
    else:
        params['bidirectional'] = _frozen(tuning_cfg, 'bidirectional')

    if _should_tune(tuning_cfg, 'use_attention'):
        params['use_attention'] = trial.suggest_categorical(
            'use_attention', [False, True])
    else:
        params['use_attention'] = _frozen(tuning_cfg, 'use_attention')

    if (params['use_attention']
            and _should_tune(tuning_cfg, 'attention_heads')):
        params['attention_heads'] = trial.suggest_categorical(
            'attention_heads', [1, 2, 4])
    else:
        params['attention_heads'] = _frozen(tuning_cfg, 'attention_heads')

    if _should_tune(tuning_cfg, 'fc_layers'):
        n_fc = trial.suggest_int('n_fc_layers', 1, 3)
        fc_sizes = []
        prev = params['hidden_size'] * (2 if params['bidirectional'] else 1)
        for i in range(n_fc):
            fc_size = trial.suggest_categorical(
                f'fc_size_{i}', [16, 32, 64, 128, 256])
            clamped = min(fc_size, prev)
            fc_sizes.append(clamped)
            prev = clamped
        params['fc_layers'] = fc_sizes
    else:
        params['fc_layers'] = list(_frozen(tuning_cfg, 'fc_layers'))

    if _should_tune(tuning_cfg, 'activation'):
        params['activation'] = trial.suggest_categorical(
            'activation', ['relu', 'gelu', 'leaky_relu'])
    else:
        params['activation'] = _frozen(tuning_cfg, 'activation')

    if _should_tune(tuning_cfg, 'fc_norm'):
        params['fc_norm'] = trial.suggest_categorical(
            'fc_norm', ['batch', 'layer', 'none'])
    else:
        params['fc_norm'] = _frozen(tuning_cfg, 'fc_norm')

    if _should_tune(tuning_cfg, 'use_input_norm'):
        params['use_input_norm'] = trial.suggest_categorical(
            'use_input_norm', [0, 1])
    else:
        params['use_input_norm'] = _frozen(tuning_cfg, 'use_input_norm')

    # Training
    if _should_tune(tuning_cfg, 'learning_rate'):
        params['learning_rate'] = trial.suggest_float(
            'learning_rate', 5e-5, 1e-3, log=True)
    else:
        params['learning_rate'] = _frozen(tuning_cfg, 'learning_rate')

    if _should_tune(tuning_cfg, 'batch_size'):
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', [64, 128, 256, 512])
    else:
        params['batch_size'] = _frozen(tuning_cfg, 'batch_size')

    if _should_tune(tuning_cfg, 'weight_decay'):
        params['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-3, log=True)
    else:
        params['weight_decay'] = _frozen(tuning_cfg, 'weight_decay')

    if _should_tune(tuning_cfg, 'scheduler_type'):
        params['scheduler_type'] = trial.suggest_categorical(
            'scheduler_type', ['plateau', 'cosine', 'onecycle', 'none'])
    else:
        params['scheduler_type'] = _frozen(tuning_cfg, 'scheduler_type')

    if LSTMConfig.LOSS_TYPE in ('huber', 'pseudo_huber', 'custom'):
        if _should_tune(tuning_cfg, 'huber_delta'):
            params['huber_delta'] = trial.suggest_float(
                'huber_delta', 0.3, 1.5, log=True)
        else:
            params['huber_delta'] = _frozen(tuning_cfg, 'huber_delta')
    else:
        params['huber_delta'] = _frozen(tuning_cfg, 'huber_delta')

    if LSTMConfig.LOSS_TYPE == 'quantile':
        if _should_tune(tuning_cfg, 'quantile_q'):
            params['quantile_q'] = trial.suggest_float('quantile_q', 0.3, 0.7)
        else:
            params['quantile_q'] = _frozen(tuning_cfg, 'quantile_q')
    else:
        params['quantile_q'] = _frozen(tuning_cfg, 'quantile_q')

    if LSTMConfig.LOSS_TYPE == 'signal':
        if (getattr(LSTMConfig, 'SIGNAL_USE_R2', True)
                and _should_tune(tuning_cfg, 'signal_r2_weight')):
            params['signal_r2_weight'] = trial.suggest_float(
                'signal_r2_weight', 1e-3, 1e-1, log=True)
        else:
            params['signal_r2_weight'] = _frozen(tuning_cfg, 'signal_r2_weight')

        if (getattr(LSTMConfig, 'SIGNAL_USE_CCC', True)
                and _should_tune(tuning_cfg, 'signal_ccc_weight')):
            params['signal_ccc_weight'] = trial.suggest_float(
                'signal_ccc_weight', 1e-3, 1e-1, log=True)
        else:
            params['signal_ccc_weight'] = _frozen(tuning_cfg, 'signal_ccc_weight')

        if (getattr(LSTMConfig, 'SIGNAL_USE_LOG_VAR', False)
                and _should_tune(tuning_cfg, 'signal_log_var_weight')):
            params['signal_log_var_weight'] = trial.suggest_float(
                'signal_log_var_weight', 1e-3, 5e-2, log=True)
        else:
            params['signal_log_var_weight'] = _frozen(
                tuning_cfg, 'signal_log_var_weight')

        if (getattr(LSTMConfig, 'SIGNAL_USE_DIRECTION_BCE', False)
                and _should_tune(tuning_cfg, 'signal_direction_bce_weight')):
            params['signal_direction_bce_weight'] = trial.suggest_float(
                'signal_direction_bce_weight', 1e-3, 5e-2, log=True)
        else:
            params['signal_direction_bce_weight'] = _frozen(
                tuning_cfg, 'signal_direction_bce_weight')

        if (getattr(LSTMConfig, 'SIGNAL_USE_STD_MATCH', False)
                and _should_tune(tuning_cfg, 'signal_std_match_weight')):
            params['signal_std_match_weight'] = trial.suggest_float(
                'signal_std_match_weight', 0.02, 0.1, log=True)
        else:
            params['signal_std_match_weight'] = _frozen(
                tuning_cfg, 'signal_std_match_weight')

        if (getattr(LSTMConfig, 'SIGNAL_USE_DIR_THRESHOLD', False)
                and _should_tune(tuning_cfg, 'signal_dir_threshold_weight')):
            params['signal_dir_threshold_weight'] = trial.suggest_float(
                'signal_dir_threshold_weight', 0.05, 0.5, log=True)
        else:
            params['signal_dir_threshold_weight'] = _frozen(
                tuning_cfg, 'signal_dir_threshold_weight')

    if (LSTMConfig.LOSS_TYPE == 'custom'
            and getattr(LSTMConfig, 'USE_AMPLIFIED_ZONE', False)):
        if _should_tune(tuning_cfg, 'amplified_zone_threshold'):
            params['amplified_zone_threshold'] = trial.suggest_float(
                'amplified_zone_threshold', 0.001, 0.0025)
        else:
            params['amplified_zone_threshold'] = _frozen(
                tuning_cfg, 'amplified_zone_threshold')

        if _should_tune(tuning_cfg, 'amplified_zone_weight'):
            params['amplified_zone_weight'] = trial.suggest_float(
                'amplified_zone_weight', 1.2, 3.5)
        else:
            params['amplified_zone_weight'] = _frozen(
                tuning_cfg, 'amplified_zone_weight')

    if tuning_cfg.LOOKBACK is not None:
        params['lookback'] = tuning_cfg.LOOKBACK
    elif _should_tune(tuning_cfg, 'lookback'):
        params['lookback'] = trial.suggest_categorical(
            'lookback', [24, 36, 48, 72, 96])
    else:
        params['lookback'] = _frozen(tuning_cfg, 'lookback')

    if tuning_cfg.HORIZON is not None:
        params['horizon'] = tuning_cfg.HORIZON
    elif _should_tune(tuning_cfg, 'horizon'):
        params['horizon'] = trial.suggest_categorical(
            'horizon', [1, 3, 6, 12, 24])
    else:
        params['horizon'] = _frozen(tuning_cfg, 'horizon')

    # F7 — training enhancements (LRO020+)
    if _should_tune(tuning_cfg, 'lr_warmup_epochs'):
        params['lr_warmup_epochs'] = trial.suggest_categorical(
            'lr_warmup_epochs', [0, 3, 5, 8])
    else:
        params['lr_warmup_epochs'] = _frozen(tuning_cfg, 'lr_warmup_epochs')

    if _should_tune(tuning_cfg, 'aug_jitter'):
        params['aug_jitter'] = trial.suggest_categorical('aug_jitter', [0, 1])
    else:
        params['aug_jitter'] = _frozen(tuning_cfg, 'aug_jitter')

    if params['aug_jitter'] and _should_tune(tuning_cfg, 'aug_jitter_std'):
        params['aug_jitter_std'] = trial.suggest_float(
            'aug_jitter_std', 0.005, 0.03, log=True)
    else:
        params['aug_jitter_std'] = _frozen(tuning_cfg, 'aug_jitter_std')

    if _should_tune(tuning_cfg, 'aug_temporal_dropout'):
        params['aug_temporal_dropout'] = trial.suggest_categorical(
            'aug_temporal_dropout', [0, 1])
    else:
        params['aug_temporal_dropout'] = _frozen(
            tuning_cfg, 'aug_temporal_dropout')

    if (params['aug_temporal_dropout']
            and _should_tune(tuning_cfg, 'aug_temporal_dropout_rate')):
        params['aug_temporal_dropout_rate'] = trial.suggest_float(
            'aug_temporal_dropout_rate', 0.02, 0.08)
    else:
        params['aug_temporal_dropout_rate'] = _frozen(
            tuning_cfg, 'aug_temporal_dropout_rate')

    if _should_tune(tuning_cfg, 'aug_subsequence'):
        params['aug_subsequence'] = trial.suggest_categorical(
            'aug_subsequence', [0, 1])
    else:
        params['aug_subsequence'] = _frozen(tuning_cfg, 'aug_subsequence')

    if (params['aug_subsequence']
            and _should_tune(tuning_cfg, 'aug_subsequence_offset')):
        params['aug_subsequence_offset'] = trial.suggest_int(
            'aug_subsequence_offset', 1, 4)
    else:
        params['aug_subsequence_offset'] = _frozen(
            tuning_cfg, 'aug_subsequence_offset')

    if _should_tune(tuning_cfg, 'signal_ema_penalties'):
        params['signal_ema_penalties'] = trial.suggest_categorical(
            'signal_ema_penalties', [0, 1])
    else:
        params['signal_ema_penalties'] = _frozen(
            tuning_cfg, 'signal_ema_penalties')

    if (params['signal_ema_penalties']
            and _should_tune(tuning_cfg, 'signal_ema_alpha')):
        params['signal_ema_alpha'] = trial.suggest_float(
            'signal_ema_alpha', 0.05, 0.2)
    else:
        params['signal_ema_alpha'] = _frozen(tuning_cfg, 'signal_ema_alpha')

    if _should_tune(tuning_cfg, 'signal_label_smoothing'):
        params['signal_label_smoothing'] = trial.suggest_categorical(
            'signal_label_smoothing', [0, 1])
    else:
        params['signal_label_smoothing'] = _frozen(
            tuning_cfg, 'signal_label_smoothing')

    if (params['signal_label_smoothing']
            and _should_tune(tuning_cfg, 'signal_label_smoothing_alpha')):
        params['signal_label_smoothing_alpha'] = trial.suggest_float(
            'signal_label_smoothing_alpha', 0.3, 0.7)
    else:
        params['signal_label_smoothing_alpha'] = _frozen(
            tuning_cfg, 'signal_label_smoothing_alpha')

    if _should_tune(tuning_cfg, 'use_feature_attention'):
        params['use_feature_attention'] = trial.suggest_categorical(
            'use_feature_attention', [0, 1])
    else:
        params['use_feature_attention'] = _frozen(
            tuning_cfg, 'use_feature_attention')

    return params
