"""
Baseline parameter resolution for Signal Model Optuna tuning.

Frozen dimensions and profile defaults resolve through ``BASELINE_PARAMS``
instead of raw ``SignalConfig`` values (e.g. conservative_signal threshold).
"""
from __future__ import annotations

from signal_config import SignalConfig
from common.training_profiles import SIGNAL_PROFILES, normalize_profile_name

# Optuna trial.param name -> SignalConfig attribute (fallback when not in baseline)
_SIGNALCONFIG_ATTR: dict[str, str] = {
    'feature_set': 'DEFAULT_FEATURE_SET',
    'hidden_size': 'HIDDEN_SIZE',
    'num_layers': 'NUM_LAYERS',
    'dropout': 'DROPOUT',
    'bidirectional': 'BIDIRECTIONAL',
    'rnn_type': 'RNN_TYPE',
    'use_attention': 'USE_ATTENTION',
    'attention_heads': 'ATTENTION_HEADS',
    'fc_layers': 'FC_LAYERS',
    'activation': 'ACTIVATION',
    'fc_norm': 'FC_NORM',
    'use_feature_attention': 'USE_FEATURE_ATTENTION',
    'learning_rate': 'LEARNING_RATE',
    'batch_size': 'BATCH_SIZE',
    'weight_decay': 'WEIGHT_DECAY',
    'scheduler_type': 'SCHEDULER_TYPE',
    'lr_warmup_epochs': 'LR_WARMUP_EPOCHS',
    'lookback': 'DEFAULT_LOOKBACK',
    'horizon': 'DEFAULT_HORIZON',
    'signal_threshold': 'SIGNAL_THRESHOLD',
    'window_sigma': 'WINDOW_SIGMA',
    'aug_jitter': 'AUG_JITTER',
    'aug_jitter_std': 'AUG_JITTER_STD',
    'aug_temporal_dropout': 'AUG_TEMPORAL_DROPOUT',
    'aug_temporal_dropout_rate': 'AUG_TEMPORAL_DROPOUT_RATE',
    'aug_subsequence': 'AUG_SUBSEQUENCE',
    'aug_subsequence_offset': 'AUG_SUBSEQUENCE_OFFSET',
    # F6 flags (Faza C)
    'use_conv1d_front': 'USE_CONV1D_FRONT',
    'conv1d_kernel': 'CONV1D_KERNEL',
    'pooling_mode': 'POOLING_MODE',
    'use_fc_residual': 'USE_FC_RESIDUAL',
    'use_rnn_norm': 'USE_RNN_NORM',
    'use_learnable_output_scale': 'USE_LEARNABLE_OUTPUT_SCALE',
    # Submodel & loss flags
    'use_submodel_norm': 'USE_SUBMODEL_NORM',
    'submodel_norm_type': 'SUBMODEL_NORM_TYPE',
    'weight_init': 'WEIGHT_INIT',
    'loss_type': 'LOSS_TYPE',
}

# Profile ``resolved`` keys that differ from Optuna param names
_PROFILE_KEY_MAP = {'lr': 'learning_rate'}

# Stored as 0/1 in trial.params for TPE compatibility
_BOOL_OPTUNA_KEYS = frozenset({
    'aug_jitter', 'aug_temporal_dropout', 'aug_subsequence',
    'use_attention', 'use_feature_attention', 'bidirectional',
    # F6
    'use_conv1d_front', 'use_fc_residual', 'use_rnn_norm',
    'use_learnable_output_scale', 'use_submodel_norm',
})


def coerce_optuna_bool(val) -> int:
    """Normalize bool / 0/1 to Optuna int flag."""
    if isinstance(val, bool):
        return 1 if val else 0
    return int(val)


def coerce_resolved_bool(val) -> bool:
    """Normalize Optuna 0/1 or bool to Python bool for training."""
    if isinstance(val, bool):
        return val
    return bool(int(val))


def _profile_to_baseline(profile_name: str) -> dict:
    profile = SIGNAL_PROFILES[normalize_profile_name(profile_name)]
    baseline: dict = {}
    for key, value in profile.get('resolved', {}).items():
        baseline[_PROFILE_KEY_MAP.get(key, key)] = value
    return baseline


def build_tuning_baseline(training_profile=None, baseline_params_path=None) -> dict:
    """
    Merge training profile and optional Optuna JSON into a baseline dict.

    Keys use Optuna ``trial.params`` naming (``learning_rate``, not ``lr``).
    """
    baseline: dict = {}
    if training_profile and normalize_profile_name(training_profile) != 'baseline':
        baseline.update(_profile_to_baseline(training_profile))
    if baseline_params_path:
        params, _ = SignalConfig.load_optuna_params(baseline_params_path)
        baseline.update(params)
    for key in _BOOL_OPTUNA_KEYS:
        if key in baseline:
            baseline[key] = coerce_optuna_bool(baseline[key])
    return baseline


def baseline_value(tuning_cfg, name: str):
    """Return frozen/default value: BASELINE_PARAMS > tuning_cfg attr > SignalConfig."""
    baseline = getattr(tuning_cfg, 'BASELINE_PARAMS', None) or {}
    if name in baseline:
        return baseline[name]
    cfg_val = getattr(tuning_cfg, name.upper(), None)
    if cfg_val is not None:
        return cfg_val
    cfg_val_lower = getattr(tuning_cfg, name, None)
    if cfg_val_lower is not None:
        return cfg_val_lower
    attr = _SIGNALCONFIG_ATTR.get(name)
    if attr is not None:
        val = getattr(SignalConfig, attr, None)
        if name in _BOOL_OPTUNA_KEYS and val is not None:
            return coerce_optuna_bool(val)
        return val
    raise KeyError(f"No baseline mapping for Optuna param: {name!r}")
