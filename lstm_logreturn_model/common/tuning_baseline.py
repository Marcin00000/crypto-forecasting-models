"""
Baseline parameter resolution for Optuna tuning.

Frozen dimensions and profile defaults resolve through ``BASELINE_PARAMS``
instead of raw ``LSTMConfig`` values (e.g. rank_focused hidden_size=256).
"""
from __future__ import annotations

from lstm_config import LSTMConfig
from common.training_profiles import TRAINING_PROFILES, normalize_profile_name

# Optuna trial.param name -> LSTMConfig attribute (fallback when not in baseline)
_LSTMCONFIG_ATTR: dict[str, str] = {
    'hidden_size': 'HIDDEN_SIZE',
    'num_layers': 'NUM_LAYERS',
    'dropout': 'DROPOUT',
    'bidirectional': 'BIDIRECTIONAL',
    'use_attention': 'USE_ATTENTION',
    'attention_heads': 'ATTENTION_HEADS',
    'fc_layers': 'FC_LAYERS',
    'activation': 'ACTIVATION',
    'fc_norm': 'FC_NORM',
    'learning_rate': 'LEARNING_RATE',
    'batch_size': 'BATCH_SIZE',
    'weight_decay': 'WEIGHT_DECAY',
    'scheduler_type': 'SCHEDULER_TYPE',
    'huber_delta': 'HUBER_DELTA',
    'quantile_q': 'QUANTILE_LOSS_Q',
    'lookback': 'DEFAULT_LOOKBACK',
    'horizon': 'DEFAULT_HORIZON',
    'feature_set': 'DEFAULT_FEATURE_SET',
    'lstm_dropout': 'LSTM_DROPOUT',
    'fc_dropout': 'FC_DROPOUT',
    'use_input_norm': 'USE_INPUT_NORM',
    'lr_warmup_epochs': 'LR_WARMUP_EPOCHS',
    'aug_jitter': 'AUG_JITTER',
    'aug_jitter_std': 'AUG_JITTER_STD',
    'aug_temporal_dropout': 'AUG_TEMPORAL_DROPOUT',
    'aug_temporal_dropout_rate': 'AUG_TEMPORAL_DROPOUT_RATE',
    'aug_subsequence': 'AUG_SUBSEQUENCE',
    'aug_subsequence_offset': 'AUG_SUBSEQUENCE_OFFSET',
    'signal_ema_penalties': 'SIGNAL_EMA_PENALTIES',
    'signal_ema_alpha': 'SIGNAL_EMA_ALPHA',
    'signal_label_smoothing': 'SIGNAL_LABEL_SMOOTHING',
    'signal_label_smoothing_alpha': 'SIGNAL_LABEL_SMOOTHING_ALPHA',
    'use_feature_attention': 'USE_FEATURE_ATTENTION',
    'signal_r2_weight': 'SIGNAL_R2_WEIGHT',
    'signal_ccc_weight': 'SIGNAL_CCC_WEIGHT',
    'signal_log_var_weight': 'SIGNAL_LOG_VAR_WEIGHT',
    'signal_std_match_weight': 'SIGNAL_STD_MATCH_WEIGHT',
    'signal_direction_bce_weight': 'SIGNAL_DIRECTION_BCE_WEIGHT',
    'signal_dir_threshold_weight': 'SIGNAL_DIR_THRESHOLD_WEIGHT',
    'amplified_zone_threshold': 'AMPLIFIED_ZONE_THRESHOLD',
    'amplified_zone_weight': 'AMPLIFIED_ZONE_WEIGHT',
}

# Profile ``resolved`` keys that differ from Optuna param names
_PROFILE_KEY_MAP = {'lr': 'learning_rate'}

# Stored as 0/1 in trial.params for TPE compatibility
_BOOL_OPTUNA_KEYS = frozenset({
    'aug_jitter', 'aug_temporal_dropout', 'aug_subsequence',
    'signal_ema_penalties', 'signal_label_smoothing', 'use_feature_attention',
    'bidirectional', 'use_attention',
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
    profile = TRAINING_PROFILES[normalize_profile_name(profile_name)]
    baseline: dict = {}
    for key, value in profile['resolved'].items():
        baseline[_PROFILE_KEY_MAP.get(key, key)] = value
    baseline.update(profile['loss'])
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
        params, _ = LSTMConfig.load_optuna_params(baseline_params_path)
        baseline.update(params)
    for key in _BOOL_OPTUNA_KEYS:
        if key in baseline:
            baseline[key] = coerce_optuna_bool(baseline[key])
    return baseline


def baseline_value(tuning_cfg, name: str):
    """Return frozen/default value: BASELINE_PARAMS > LSTMConfig."""
    baseline = getattr(tuning_cfg, 'BASELINE_PARAMS', None) or {}
    if name in baseline:
        return baseline[name]
    attr = _LSTMCONFIG_ATTR.get(name)
    if attr is not None:
        val = getattr(LSTMConfig, attr, None)
        if name in _BOOL_OPTUNA_KEYS and val is not None:
            return coerce_optuna_bool(val)
        return val
    raise KeyError(f"No baseline mapping for Optuna param: {name!r}")
