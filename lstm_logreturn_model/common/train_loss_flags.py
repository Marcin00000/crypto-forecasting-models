"""Mirror CLI/kwargs loss flags onto LSTMConfig."""
from __future__ import annotations


def apply_custom_loss_flags(cfg, *,
                            loss_type=None, huber_delta=None,
                            quantile_q=None,
                            use_sample_weights=None, use_misdir_penalty=None,
                            use_dead_zone=None, dead_zone_threshold=None,
                            use_amplified_zone=None,
                            amplified_zone_threshold=None,
                            amplified_zone_weight=None,
                            use_correlation_penalty=None,
                            correlation_penalty_weight=None,
                            use_variance_penalty=None,
                            variance_penalty_weight=None,
                            use_directional_loss=None,
                            directional_loss_weight=None,
                            use_log_variance_penalty=None,
                            log_variance_penalty_weight=None,
                            use_r2_penalty=None, r2_penalty_weight=None,
                            use_ccc_penalty=None, ccc_penalty_weight=None,
                            signal_use_r2=None, signal_r2_weight=None,
                            signal_use_ccc=None, signal_ccc_weight=None,
                            signal_use_log_var=None,
                            signal_log_var_weight=None,
                            signal_use_std_match=None,
                            signal_std_match_weight=None,
                            signal_use_dir_threshold=None,
                            signal_dir_threshold_weight=None,
                            signal_use_direction_bce=None,
                            signal_direction_bce_weight=None,
                            signal_ema_penalties=None,
                            signal_ema_alpha=None,
                            signal_label_smoothing=None,
                            signal_label_smoothing_alpha=None,
                            signal_use_utility=None,
                            signal_utility_weight=None,
                            signal_soft_threshold=None,
                            use_cr_ceiling=None,
                            cr_ceiling=None,
                            cr_ceiling_weight=None):
    """Mutate LSTMConfig with resolved loss flags (None kwargs skipped)."""
    if loss_type is not None:
        cfg.LOSS_TYPE = loss_type
    if huber_delta is not None:
        cfg.HUBER_DELTA = huber_delta
    if quantile_q is not None:
        cfg.QUANTILE_LOSS_Q = quantile_q
    if use_sample_weights is not None:
        cfg.USE_SAMPLE_WEIGHTS = use_sample_weights
    if use_misdir_penalty is not None:
        cfg.USE_MISDIR_PENALTY = use_misdir_penalty
    if use_dead_zone is not None:
        cfg.USE_DEAD_ZONE = use_dead_zone
    if dead_zone_threshold is not None:
        cfg.DEAD_ZONE_THRESHOLD = dead_zone_threshold
    if use_amplified_zone is not None:
        cfg.USE_AMPLIFIED_ZONE = use_amplified_zone
    if amplified_zone_threshold is not None:
        cfg.AMPLIFIED_ZONE_THRESHOLD = amplified_zone_threshold
    if amplified_zone_weight is not None:
        cfg.AMPLIFIED_ZONE_WEIGHT = amplified_zone_weight
    if use_correlation_penalty is not None:
        cfg.USE_CORRELATION_PENALTY = use_correlation_penalty
    if correlation_penalty_weight is not None:
        cfg.CORRELATION_PENALTY_WEIGHT = correlation_penalty_weight
    if use_variance_penalty is not None:
        cfg.USE_VARIANCE_PENALTY = use_variance_penalty
    if variance_penalty_weight is not None:
        cfg.VARIANCE_PENALTY_WEIGHT = variance_penalty_weight
    if use_directional_loss is not None:
        cfg.USE_DIRECTIONAL_LOSS = use_directional_loss
    if directional_loss_weight is not None:
        cfg.DIRECTIONAL_LOSS_WEIGHT = directional_loss_weight
    if use_log_variance_penalty is not None:
        cfg.USE_LOG_VARIANCE_PENALTY = use_log_variance_penalty
    if log_variance_penalty_weight is not None:
        cfg.LOG_VARIANCE_PENALTY_WEIGHT = log_variance_penalty_weight
    if use_r2_penalty is not None:
        cfg.USE_R2_PENALTY = use_r2_penalty
    if r2_penalty_weight is not None:
        cfg.R2_PENALTY_WEIGHT = r2_penalty_weight
    if use_ccc_penalty is not None:
        cfg.USE_CCC_PENALTY = use_ccc_penalty
    if ccc_penalty_weight is not None:
        cfg.CCC_PENALTY_WEIGHT = ccc_penalty_weight
    if signal_use_r2 is not None:
        cfg.SIGNAL_USE_R2 = signal_use_r2
    if signal_r2_weight is not None:
        cfg.SIGNAL_R2_WEIGHT = signal_r2_weight
    if signal_use_ccc is not None:
        cfg.SIGNAL_USE_CCC = signal_use_ccc
    if signal_ccc_weight is not None:
        cfg.SIGNAL_CCC_WEIGHT = signal_ccc_weight
    if signal_use_log_var is not None:
        cfg.SIGNAL_USE_LOG_VAR = signal_use_log_var
    if signal_log_var_weight is not None:
        cfg.SIGNAL_LOG_VAR_WEIGHT = signal_log_var_weight
    if signal_use_std_match is not None:
        cfg.SIGNAL_USE_STD_MATCH = signal_use_std_match
    if signal_std_match_weight is not None:
        cfg.SIGNAL_STD_MATCH_WEIGHT = signal_std_match_weight
    if signal_use_dir_threshold is not None:
        cfg.SIGNAL_USE_DIR_THRESHOLD = signal_use_dir_threshold
    if signal_dir_threshold_weight is not None:
        cfg.SIGNAL_DIR_THRESHOLD_WEIGHT = signal_dir_threshold_weight
    if signal_use_direction_bce is not None:
        cfg.SIGNAL_USE_DIRECTION_BCE = signal_use_direction_bce
    if signal_direction_bce_weight is not None:
        cfg.SIGNAL_DIRECTION_BCE_WEIGHT = signal_direction_bce_weight
    if signal_ema_penalties is not None:
        cfg.SIGNAL_EMA_PENALTIES = signal_ema_penalties
    if signal_ema_alpha is not None:
        cfg.SIGNAL_EMA_ALPHA = signal_ema_alpha
    if signal_label_smoothing is not None:
        cfg.SIGNAL_LABEL_SMOOTHING = signal_label_smoothing
    if signal_label_smoothing_alpha is not None:
        cfg.SIGNAL_LABEL_SMOOTHING_ALPHA = signal_label_smoothing_alpha
    if signal_use_utility is not None:
        cfg.SIGNAL_USE_UTILITY = signal_use_utility
    if signal_utility_weight is not None:
        cfg.SIGNAL_UTILITY_WEIGHT = signal_utility_weight
    if signal_soft_threshold is not None:
        cfg.SIGNAL_SOFT_THRESHOLD = signal_soft_threshold
    if use_cr_ceiling is not None:
        cfg.USE_CR_CEILING = use_cr_ceiling
    if cr_ceiling is not None:
        cfg.CR_CEILING = cr_ceiling
    if cr_ceiling_weight is not None:
        cfg.CR_CEILING_WEIGHT = cr_ceiling_weight
