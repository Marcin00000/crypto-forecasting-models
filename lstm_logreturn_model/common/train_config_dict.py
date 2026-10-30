"""Assemble model_config.json payload for lstm_train.save_model."""
from __future__ import annotations

from datetime import datetime

from .path_utils import to_repo_relative


def build_train_config_dict(*, cfg, resolved, quantile_q,
                            model_id, data_id, symbol, features,
                            n_params, train_losses, val_losses,
                            train_metrics, val_metrics, best_epoch,
                            diagnostics_history,
                            signal_threshold, is_prepared,
                            data_path, load_optuna_params):
    """Build the serialisable training snapshot written to model_config.json."""
    return {
        'model_id': model_id,
        'data_id': data_id,
        'symbol': symbol,
        'features': features,
        'n_features': len(features),
        'training_profile': resolved.get('training_profile', 'baseline'),
        'lookback': resolved['lookback'],
        'horizon': resolved['horizon'],
        'hidden_size': resolved['hidden_size'],
        'num_layers': resolved['num_layers'],
        'dropout': resolved['dropout'],
        'bidirectional': resolved['bidirectional'],
        'use_attention': resolved['use_attention'],
        'fc_layers': resolved['fc_layers'],
        'activation': resolved['activation'],
        'fc_norm': resolved['fc_norm'],
        'arch_type': resolved.get('arch_type', 'legacy'),
        'use_input_norm': resolved['use_input_norm'],
        'weight_init': resolved['weight_init'],
        'attention_heads': resolved['attention_heads'],
        'lstm_dropout': resolved['lstm_dropout'],
        'fc_dropout': resolved['fc_dropout'],
        'rnn_type': getattr(cfg, 'RNN_TYPE', 'lstm'),
        'prediction_precision': resolved['prediction_precision'],
        'target_precision': resolved['target_precision'],
        'feature_round_decimals': resolved['feature_round_decimals'],
        'target_round_decimals': resolved['target_round_decimals'],
        'log_gradient_norms': resolved['log_gradient_norms'],
        'best_model_metric': resolved['best_model_metric'],
        'best_model_cr_weight': resolved['best_model_cr_weight'],
        'n_params': n_params,
        'loss_type': resolved['loss_type'],
        'huber_delta': resolved['huber_delta'],
        'quantile_q': getattr(cfg, 'QUANTILE_LOSS_Q', 0.5),
        'use_sample_weights': getattr(cfg, 'USE_SAMPLE_WEIGHTS', False),
        'use_misdir_penalty': getattr(cfg, 'USE_MISDIR_PENALTY', False),
        'use_dead_zone': getattr(cfg, 'USE_DEAD_ZONE', False),
        'use_amplified_zone': getattr(cfg, 'USE_AMPLIFIED_ZONE', False),
        'amplified_zone_threshold': getattr(
            cfg, 'AMPLIFIED_ZONE_THRESHOLD', 0.002),
        'amplified_zone_weight': getattr(
            cfg, 'AMPLIFIED_ZONE_WEIGHT', 2.0),
        'use_correlation_penalty': getattr(
            cfg, 'USE_CORRELATION_PENALTY', False),
        'correlation_penalty_weight': getattr(
            cfg, 'CORRELATION_PENALTY_WEIGHT', 0.1),
        'use_variance_penalty': getattr(
            cfg, 'USE_VARIANCE_PENALTY', False),
        'variance_penalty_weight': getattr(
            cfg, 'VARIANCE_PENALTY_WEIGHT', 0.1),
        'use_directional_loss': getattr(
            cfg, 'USE_DIRECTIONAL_LOSS', False),
        'directional_loss_weight': getattr(
            cfg, 'DIRECTIONAL_LOSS_WEIGHT', 0.5),
        'use_log_variance_penalty': getattr(
            cfg, 'USE_LOG_VARIANCE_PENALTY', False),
        'log_variance_penalty_weight': getattr(
            cfg, 'LOG_VARIANCE_PENALTY_WEIGHT', 0.01),
        'use_r2_penalty': getattr(cfg, 'USE_R2_PENALTY', False),
        'r2_penalty_weight': getattr(cfg, 'R2_PENALTY_WEIGHT', 0.01),
        'use_ccc_penalty': getattr(cfg, 'USE_CCC_PENALTY', False),
        'ccc_penalty_weight': getattr(cfg, 'CCC_PENALTY_WEIGHT', 0.01),
        'signal_use_r2': getattr(cfg, 'SIGNAL_USE_R2', True),
        'signal_r2_weight': getattr(cfg, 'SIGNAL_R2_WEIGHT', 0.01),
        'signal_use_ccc': getattr(cfg, 'SIGNAL_USE_CCC', True),
        'signal_ccc_weight': getattr(cfg, 'SIGNAL_CCC_WEIGHT', 0.01),
        'signal_use_log_var': getattr(cfg, 'SIGNAL_USE_LOG_VAR', False),
        'signal_log_var_weight': getattr(cfg, 'SIGNAL_LOG_VAR_WEIGHT', 0.005),
        'signal_use_std_match': getattr(cfg, 'SIGNAL_USE_STD_MATCH', False),
        'signal_std_match_weight': getattr(
            cfg, 'SIGNAL_STD_MATCH_WEIGHT', 0.05),
        'signal_use_dir_threshold': getattr(
            cfg, 'SIGNAL_USE_DIR_THRESHOLD', False),
        'signal_dir_threshold_weight': getattr(
            cfg, 'SIGNAL_DIR_THRESHOLD_WEIGHT', 0.1),
        'signal_use_direction_bce': getattr(
            cfg, 'SIGNAL_USE_DIRECTION_BCE', False),
        'signal_direction_bce_weight': getattr(
            cfg, 'SIGNAL_DIRECTION_BCE_WEIGHT', 0.01),
        'signal_ema_penalties': getattr(cfg, 'SIGNAL_EMA_PENALTIES', False),
        'signal_ema_alpha': getattr(cfg, 'SIGNAL_EMA_ALPHA', 0.1),
        'signal_label_smoothing': getattr(cfg, 'SIGNAL_LABEL_SMOOTHING', False),
        'signal_label_smoothing_alpha': getattr(
            cfg, 'SIGNAL_LABEL_SMOOTHING_ALPHA', 0.5),
        'aug_jitter': resolved.get('aug_jitter', False),
        'aug_jitter_std': resolved.get('aug_jitter_std', 0.01),
        'aug_temporal_dropout': resolved.get('aug_temporal_dropout', False),
        'aug_temporal_dropout_rate': resolved.get(
            'aug_temporal_dropout_rate', 0.05),
        'aug_subsequence': resolved.get('aug_subsequence', False),
        'aug_subsequence_offset': resolved.get('aug_subsequence_offset', 2),
        'use_feature_attention': resolved.get('use_feature_attention', False),
        'use_rnn_norm': resolved.get('use_rnn_norm', False),
        'pooling_mode': resolved.get('pooling_mode', 'last'),
        'use_conv1d_front': resolved.get('use_conv1d_front', False),
        'conv1d_kernel': resolved.get('conv1d_kernel', 5),
        'use_fc_residual': resolved.get('use_fc_residual', False),
        'use_learnable_output_scale': resolved.get(
            'use_learnable_output_scale', False),
        'freeze_rnn': resolved.get('freeze_rnn', False),
        'lr_warmup_epochs': resolved.get('lr_warmup_epochs', 0),
        'use_scheduled_loss': getattr(cfg, 'USE_SCHEDULED_LOSS', False),
        'scheduled_loss_warmup_epochs': getattr(
            cfg, 'SCHEDULED_LOSS_WARMUP_EPOCHS', 30),
        'scheduled_loss_base': getattr(cfg, 'SCHEDULED_LOSS_BASE', 'mae'),
        'scheduled_loss_full': getattr(cfg, 'SCHEDULED_LOSS_FULL', 'custom'),
        'epochs_trained': len(train_losses),
        'best_epoch': best_epoch,
        'batch_size': resolved['batch_size'],
        'learning_rate': resolved['lr'],
        'weight_decay': resolved['weight_decay'],
        'scheduler_type': resolved['scheduler_type'],
        'scale_target': resolved['scale_target'],
        'target_scaler_type': (resolved['target_scaler_type']
                               if resolved['scale_target'] else None),
        'target_mode': resolved.get('target_mode', 'raw'),
        'use_aux_volatility_head': resolved.get('use_aux_volatility_head', False),
        'use_group_dro': getattr(cfg, 'USE_GROUP_DRO', False),
        'group_dro_n_groups': getattr(cfg, 'GROUP_DRO_N_GROUPS', 4),
        'group_dro_step_size': getattr(cfg, 'GROUP_DRO_STEP_SIZE', 0.01),
        'winsorize_target': resolved['winsorize_target'],
        'signal_threshold': signal_threshold,
        'val_source': resolved['val_source'],
        'preferred_format': resolved['preferred_format'],
        'feature_set': resolved['feature_set'],
        'custom_features': resolved.get('custom_features'),
        'train_path': to_repo_relative(data_path),
        'data_source': resolved.get('data_source', 'train'),
        'data_prep_type': 'prepared' if is_prepared else 'raw_csv',
        'optuna_params': load_optuna_params,
        'seed': resolved['seed'],
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'diagnostics_history': diagnostics_history,
    }
