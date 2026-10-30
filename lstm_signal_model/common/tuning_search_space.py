"""
Optuna search space and per-parameter freeze logic for Signal Model tuning.

Public API:
    _should_tune(tuning_cfg, name) -> bool
    suggest_hyperparameters(trial, tuning_cfg) -> dict
"""
from __future__ import annotations

from common.tuning_baseline import baseline_value


def _should_tune(tuning_cfg, name):
    """True if parameter is NOT frozen via TUNE_PARAMS={name: False} or FREEZE_HYPERPARAMS."""
    if getattr(tuning_cfg, 'FREEZE_HYPERPARAMS', False):
        return name == 'feature_set'
    flags = getattr(tuning_cfg, 'TUNE_PARAMS', {}) or {}
    return flags.get(name, True)


def _frozen(tuning_cfg, name):
    """Value used when a dimension is frozen (profile baseline > SignalConfig)."""
    return baseline_value(tuning_cfg, name)


def suggest_hyperparameters(trial, tuning_cfg):
    """
    Suggest hyperparameters for a trial.

    Frozen params use ``tuning_cfg.BASELINE_PARAMS`` (from --training_profile
    / --baseline_params) instead of bare SignalConfig defaults.
    """
    params = {}

    # ---------------------------------------------------------------
    # Feature Set Tuning (Tryb A / Tryb B)
    # ---------------------------------------------------------------
    if _should_tune(tuning_cfg, 'feature_set'):
        default_candidates = [
            'signal_core', 'signal_optimal', 'technical', 'technical_bp',
            'technical_mtf', 'max', 'trend_momentum', 'volatility_regime',
            'candle_plus_bp', 'full_extended', 'aggregator_ext', 'aggregator_disjoint'
        ]
        candidates = getattr(tuning_cfg, 'FEATURE_SET_CANDIDATES', None) or default_candidates
        params['feature_set'] = trial.suggest_categorical('feature_set', candidates)
    else:
        params['feature_set'] = _frozen(tuning_cfg, 'feature_set')

    # ---------------------------------------------------------------
    # Architecture
    # ---------------------------------------------------------------
    if _should_tune(tuning_cfg, 'hidden_size'):
        params['hidden_size'] = trial.suggest_categorical(
            'hidden_size', [32, 64, 128])
    else:
        params['hidden_size'] = _frozen(tuning_cfg, 'hidden_size')

    if _should_tune(tuning_cfg, 'num_layers'):
        params['num_layers'] = trial.suggest_int('num_layers', 1, 2)
    else:
        params['num_layers'] = _frozen(tuning_cfg, 'num_layers')

    if _should_tune(tuning_cfg, 'dropout'):
        params['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
    else:
        params['dropout'] = _frozen(tuning_cfg, 'dropout')

    if _should_tune(tuning_cfg, 'bidirectional'):
        params['bidirectional'] = trial.suggest_categorical(
            'bidirectional', [0, 1])
    else:
        params['bidirectional'] = _frozen(tuning_cfg, 'bidirectional')

    if _should_tune(tuning_cfg, 'rnn_type'):
        params['rnn_type'] = trial.suggest_categorical(
            'rnn_type', ['lstm', 'gru'])
    else:
        params['rnn_type'] = _frozen(tuning_cfg, 'rnn_type')

    if _should_tune(tuning_cfg, 'use_attention'):
        params['use_attention'] = trial.suggest_categorical(
            'use_attention', [0, 1])
    else:
        params['use_attention'] = _frozen(tuning_cfg, 'use_attention')

    if (params['use_attention']
            and _should_tune(tuning_cfg, 'attention_heads')):
        # {32,64,128} all divide by {1,2,4} — no clamping needed.
        params['attention_heads'] = trial.suggest_categorical(
            'attention_heads', [1, 2, 4])
    else:
        params['attention_heads'] = _frozen(tuning_cfg, 'attention_heads')

    if _should_tune(tuning_cfg, 'fc_layers'):
        n_fc = trial.suggest_int('n_fc_layers', 1, 3)
        fc_sizes = []
        for i in range(n_fc):
            fc_sizes.append(trial.suggest_categorical(
                f'fc_units_{i}', [16, 32, 64, 128]))
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

    if _should_tune(tuning_cfg, 'use_feature_attention'):
        params['use_feature_attention'] = trial.suggest_categorical(
            'use_feature_attention', [0, 1])
    else:
        params['use_feature_attention'] = _frozen(
            tuning_cfg, 'use_feature_attention')

    # ---------------------------------------------------------------
    # F6 architecture flags (Faza C)
    # freeze_rnn NOT tuneable — manual only (sub-model fine-tuning)
    # ---------------------------------------------------------------
    if _should_tune(tuning_cfg, 'use_conv1d_front'):
        params['use_conv1d_front'] = trial.suggest_categorical(
            'use_conv1d_front', [0, 1])
    else:
        params['use_conv1d_front'] = _frozen(tuning_cfg, 'use_conv1d_front')

    if params['use_conv1d_front'] and _should_tune(tuning_cfg, 'conv1d_kernel'):
        params['conv1d_kernel'] = trial.suggest_categorical(
            'conv1d_kernel', [3, 5, 7])
    else:
        params['conv1d_kernel'] = _frozen(tuning_cfg, 'conv1d_kernel')

    # pooling_mode only relevant when attention is disabled
    if not params['use_attention'] and _should_tune(tuning_cfg, 'pooling_mode'):
        params['pooling_mode'] = trial.suggest_categorical(
            'pooling_mode', ['last', 'mean', 'max'])
    else:
        params['pooling_mode'] = _frozen(tuning_cfg, 'pooling_mode')

    if _should_tune(tuning_cfg, 'use_fc_residual'):
        params['use_fc_residual'] = trial.suggest_categorical(
            'use_fc_residual', [0, 1])
    else:
        params['use_fc_residual'] = _frozen(tuning_cfg, 'use_fc_residual')

    if _should_tune(tuning_cfg, 'use_rnn_norm'):
        params['use_rnn_norm'] = trial.suggest_categorical(
            'use_rnn_norm', [0, 1])
    else:
        params['use_rnn_norm'] = _frozen(tuning_cfg, 'use_rnn_norm')

    if _should_tune(tuning_cfg, 'use_learnable_output_scale'):
        params['use_learnable_output_scale'] = trial.suggest_categorical(
            'use_learnable_output_scale', [0, 1])
    else:
        params['use_learnable_output_scale'] = _frozen(
            tuning_cfg, 'use_learnable_output_scale')

    if _should_tune(tuning_cfg, 'weight_init'):
        params['weight_init'] = trial.suggest_categorical(
            'weight_init', ['default', 'kaiming', 'xavier'])
    else:
        params['weight_init'] = _frozen(tuning_cfg, 'weight_init')

    # ---------------------------------------------------------------
    # Submodel fusion (only when HAS_SUBMODELS is set)
    # submodel_size is fixed by loaded submodel total output dim
    # ---------------------------------------------------------------
    if getattr(tuning_cfg, 'HAS_SUBMODELS', False):
        params['arch_type'] = 'submodel_fusion'
        if _should_tune(tuning_cfg, 'use_submodel_norm'):
            params['use_submodel_norm'] = trial.suggest_categorical(
                'use_submodel_norm', [0, 1])
        else:
            params['use_submodel_norm'] = _frozen(tuning_cfg, 'use_submodel_norm')
        if _should_tune(tuning_cfg, 'submodel_norm_type'):
            params['submodel_norm_type'] = trial.suggest_categorical(
                'submodel_norm_type', ['auto', 'batch', 'zscore', 'layer', 'none'])
        else:
            params['submodel_norm_type'] = _frozen(tuning_cfg, 'submodel_norm_type')
    else:
        params['use_submodel_norm'] = _frozen(tuning_cfg, 'use_submodel_norm')
        params['submodel_norm_type'] = _frozen(tuning_cfg, 'submodel_norm_type')

    if _should_tune(tuning_cfg, 'loss_type'):
        params['loss_type'] = trial.suggest_categorical(
            'loss_type', ['bce', 'asymmetric', 'bce_diversity', 'focal'])
    else:
        params['loss_type'] = _frozen(tuning_cfg, 'loss_type')

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    if _should_tune(tuning_cfg, 'learning_rate'):
        params['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-4, 1e-3, log=True)
    else:
        params['learning_rate'] = _frozen(tuning_cfg, 'learning_rate')

    if _should_tune(tuning_cfg, 'batch_size'):
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', [256, 512, 1024])
    else:
        params['batch_size'] = _frozen(tuning_cfg, 'batch_size')

    if _should_tune(tuning_cfg, 'weight_decay'):
        params['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-6, 1e-3, log=True)
    else:
        params['weight_decay'] = _frozen(tuning_cfg, 'weight_decay')

    if _should_tune(tuning_cfg, 'scheduler_type'):
        params['scheduler_type'] = trial.suggest_categorical(
            'scheduler_type', ['plateau', 'cosine', 'onecycle'])
    else:
        params['scheduler_type'] = _frozen(tuning_cfg, 'scheduler_type')

    if _should_tune(tuning_cfg, 'lr_warmup_epochs'):
        params['lr_warmup_epochs'] = trial.suggest_categorical(
            'lr_warmup_epochs', [0, 3, 5, 10])
    else:
        params['lr_warmup_epochs'] = _frozen(tuning_cfg, 'lr_warmup_epochs')

    # ---------------------------------------------------------------
    # Signal-specific target (oracle) parameters
    # Note: Sweeping signal_threshold alters target entropy floor.
    # Freeze via TUNE_PARAMS={'signal_threshold': False} during arch tuning.
    # ---------------------------------------------------------------
    if tuning_cfg.LOOKBACK is not None:
        params['lookback'] = tuning_cfg.LOOKBACK
    elif _should_tune(tuning_cfg, 'lookback'):
        params['lookback'] = trial.suggest_categorical(
            'lookback', [24, 48, 72])
    else:
        params['lookback'] = _frozen(tuning_cfg, 'lookback')

    if tuning_cfg.HORIZON is not None:
        params['horizon'] = tuning_cfg.HORIZON
    else:
        params['horizon'] = _frozen(tuning_cfg, 'horizon')

    if _should_tune(tuning_cfg, 'signal_threshold'):
        params['signal_threshold'] = trial.suggest_float(
            'signal_threshold', 0.0003, 0.003, log=True)
    else:
        params['signal_threshold'] = _frozen(tuning_cfg, 'signal_threshold')

    if _should_tune(tuning_cfg, 'window_sigma'):
        params['window_sigma'] = trial.suggest_categorical(
            'window_sigma', [10, 20, 50, 100])
    else:
        params['window_sigma'] = _frozen(tuning_cfg, 'window_sigma')

    # ---------------------------------------------------------------
    # Augmentation
    # ---------------------------------------------------------------
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
            'aug_temporal_dropout_rate', 0.02, 0.15)
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

    return params
