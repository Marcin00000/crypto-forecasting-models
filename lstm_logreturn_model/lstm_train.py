"""
LSTM Log-Return Prediction — Training

Main training script: loads data, builds model, trains, saves.
Supports prepared CSV/NPY directories, raw CSV, and Optuna params.
"""
import os
from datetime import datetime

import torch

from lstm_config import LSTMConfig
from common.data_utils import load_data
from common.model import build_model
from common.losses import get_loss_function, resolve_signal_threshold, LSTMCustomLoss
from common.signal_loss import SignalLoss
from common.torch_utils import set_seed, get_device, CryptoDataset, get_scheduler
from common.training_utils import setup_logger, train_model
from common.model_saving import save_model
from common.log_utils import (
    log_train_session_header, log_data_summary, log_config_flags,
    log_signal_threshold, log_model_architecture, log_train_val_metrics,
    log_final_summary,
)
from common.visualization import plot_training_curves
from common.id_utils import get_next_model_id, reserve_next_model_dir, extract_data_id, detect_symbol
from common.train_setup import (
    apply_config_defaults, apply_optuna_overrides,
    apply_resolved_to_cfg, apply_custom_loss_flags,
    build_scheduled_loss, collect_explicit_profile_keys,
)
from common.training_checkpoint import build_val_signal_ctx
from common.train_config_dict import build_train_config_dict
from common.training_profiles import apply_training_profile, normalize_profile_name
from common.train_cli import build_parser, dispatch_args_to_train
from common.augmentation import SequenceAugmenter
from common.calibrator_fit import fit_and_save_calibrator
from common.wandb_utils import ExperimentTracker


def train(data_path,
          # data
          lookback=None, horizon=None, feature_set=None, custom_features=None,
          preferred_format=None, val_source=None, data_source=None,
          training_profile=None,
          # model
          hidden_size=None, num_layers=None, dropout=None,
          bidirectional=None, use_attention=None,
          fc_layers=None, activation=None, fc_norm=None,
          arch_type=None,
          # F1.I2/I3: stability flags
          use_input_norm=None, weight_init=None,
          # F4.M3/M4/M6: architecture knobs (multi-head + split dropout + GRU)
          rnn_type=None, attention_heads=None,
          lstm_dropout=None, fc_dropout=None,
          # training
          epochs=None, batch_size=None, lr=None, weight_decay=None,
          early_stopping_patience=None, gradient_clip=None,
          best_model_metric=None, best_model_cr_weight=None,
          # F1.I4/I5: precision + grad monitoring
          prediction_precision=None, target_precision=None,
          feature_round_decimals=None, target_round_decimals=None,
          log_gradient_norms=None,
          # target
          target_mode=None, use_aux_volatility_head=None,
          use_group_dro=None, group_dro_n_groups=None, group_dro_step_size=None,
          scale_target=None, target_scaler_type=None,
          winsorize_target=None, winsorize_percentile=None,
          # loss
          loss_type=None, huber_delta=None, quantile_q=None,
          use_sample_weights=None, use_misdir_penalty=None, use_dead_zone=None,
          dead_zone_threshold=None,
          use_amplified_zone=None, amplified_zone_threshold=None,
          amplified_zone_weight=None,
          use_correlation_penalty=None, correlation_penalty_weight=None,
          use_variance_penalty=None, variance_penalty_weight=None,
          use_directional_loss=None, directional_loss_weight=None,
          use_log_variance_penalty=None, log_variance_penalty_weight=None,
          use_r2_penalty=None, r2_penalty_weight=None,
          use_ccc_penalty=None, ccc_penalty_weight=None,
          # FAZA 3.V2-3 — SignalLoss v2 component flags
          signal_use_r2=None, signal_r2_weight=None,
          signal_use_ccc=None, signal_ccc_weight=None,
          signal_use_log_var=None, signal_log_var_weight=None,
          signal_use_std_match=None, signal_std_match_weight=None,
          signal_use_dir_threshold=None, signal_dir_threshold_weight=None,
          signal_use_direction_bce=None, signal_direction_bce_weight=None,
          signal_ema_penalties=None, signal_ema_alpha=None,
          signal_label_smoothing=None, signal_label_smoothing_alpha=None,
          # F7 — augmentation / architecture extras
          aug_jitter=None, aug_jitter_std=None,
          aug_temporal_dropout=None, aug_temporal_dropout_rate=None,
          aug_subsequence=None, aug_subsequence_offset=None,
          use_feature_attention=None, lr_warmup_epochs=None,
          use_rnn_norm=None, pooling_mode=None,
          use_conv1d_front=None, conv1d_kernel=None,
          use_fc_residual=None, use_learnable_output_scale=None,
          freeze_rnn=None,
          min_epochs_before_es=None, early_stopping_min_delta=None,
          use_swa=None, swa_start_epoch=None, swa_every=None,
          signal_use_utility=None, signal_utility_weight=None,
          signal_soft_threshold=None,
          use_cr_ceiling=None, cr_ceiling=None, cr_ceiling_weight=None,
          # signal threshold
          signal_threshold_enabled=None, signal_threshold_mode=None,
          signal_threshold_value=None,
          # scheduler
          scheduler_type=None,
          # FAZA 5 — scheduled loss switching
          use_scheduled_loss=None,
          scheduled_loss_warmup_epochs=None,
          scheduled_loss_base=None,
          scheduled_loss_full=None,
          # optuna
          load_optuna_params=None,
          calibrate=None,
          # misc
          seed=None, verbose=True, use_wandb=None):
    """
    Train LSTM for log-return prediction.

    Parameters default to LSTMConfig values when None.
    Optuna params override when load_optuna_params is provided.
    """
    # ── Apply defaults from config ──
    cfg = LSTMConfig
    explicit_profile_keys = collect_explicit_profile_keys(locals())
    training_profile = normalize_profile_name(training_profile)

    resolved = apply_config_defaults(
        cfg,
        lookback=lookback, horizon=horizon,
        feature_set=feature_set, custom_features=custom_features, preferred_format=preferred_format,
        val_source=val_source, data_source=data_source,
        training_profile=training_profile,
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
        bidirectional=bidirectional, use_attention=use_attention,
        fc_layers=fc_layers, activation=activation, fc_norm=fc_norm,
        arch_type=arch_type, rnn_type=rnn_type,
        use_input_norm=use_input_norm, weight_init=weight_init,
        attention_heads=attention_heads,
        lstm_dropout=lstm_dropout, fc_dropout=fc_dropout,
        epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        gradient_clip=gradient_clip,
        best_model_metric=best_model_metric,
        best_model_cr_weight=best_model_cr_weight,
        prediction_precision=prediction_precision,
        target_precision=target_precision,
        feature_round_decimals=feature_round_decimals,
        target_round_decimals=target_round_decimals,
        log_gradient_norms=log_gradient_norms,
        target_mode=target_mode,
        use_aux_volatility_head=use_aux_volatility_head,
        use_group_dro=use_group_dro,
        group_dro_n_groups=group_dro_n_groups,
        group_dro_step_size=group_dro_step_size,
        scale_target=scale_target, target_scaler_type=target_scaler_type,
        winsorize_target=winsorize_target,
        winsorize_percentile=winsorize_percentile,
        loss_type=loss_type, huber_delta=huber_delta,
        signal_threshold_enabled=signal_threshold_enabled,
        signal_threshold_mode=signal_threshold_mode,
        signal_threshold_value=signal_threshold_value,
        scheduler_type=scheduler_type, seed=seed,
        signal_ema_penalties=signal_ema_penalties,
        signal_ema_alpha=signal_ema_alpha,
        signal_label_smoothing=signal_label_smoothing,
        signal_label_smoothing_alpha=signal_label_smoothing_alpha,
        aug_jitter=aug_jitter, aug_jitter_std=aug_jitter_std,
        aug_temporal_dropout=aug_temporal_dropout,
        aug_temporal_dropout_rate=aug_temporal_dropout_rate,
        aug_subsequence=aug_subsequence,
        aug_subsequence_offset=aug_subsequence_offset,
        use_feature_attention=use_feature_attention,
        use_rnn_norm=use_rnn_norm,
        pooling_mode=pooling_mode,
        use_conv1d_front=use_conv1d_front,
        conv1d_kernel=conv1d_kernel,
        use_fc_residual=use_fc_residual,
        use_learnable_output_scale=use_learnable_output_scale,
        freeze_rnn=freeze_rnn,
        lr_warmup_epochs=lr_warmup_epochs,
        min_epochs_before_es=min_epochs_before_es,
        early_stopping_min_delta=early_stopping_min_delta,
        use_swa=use_swa,
        swa_start_epoch=swa_start_epoch,
        swa_every=swa_every,
    )

    # ── Apply named recipe profile (explicit kwargs still win) ──
    resolved, profile_loss_overrides = apply_training_profile(
        resolved, training_profile, explicit_profile_keys)
    loss_type = profile_loss_overrides.get('loss_type', resolved['loss_type'])
    signal_use_std_match = profile_loss_overrides.get(
        'signal_use_std_match', signal_use_std_match)
    signal_std_match_weight = profile_loss_overrides.get(
        'signal_std_match_weight', signal_std_match_weight)
    signal_use_dir_threshold = profile_loss_overrides.get(
        'signal_use_dir_threshold', signal_use_dir_threshold)
    signal_use_direction_bce = profile_loss_overrides.get(
        'signal_use_direction_bce', signal_use_direction_bce)
    use_correlation_penalty = profile_loss_overrides.get(
        'use_correlation_penalty', use_correlation_penalty)
    correlation_penalty_weight = profile_loss_overrides.get(
        'correlation_penalty_weight', correlation_penalty_weight)
    use_variance_penalty = profile_loss_overrides.get(
        'use_variance_penalty', use_variance_penalty)
    variance_penalty_weight = profile_loss_overrides.get(
        'variance_penalty_weight', variance_penalty_weight)
    use_amplified_zone = profile_loss_overrides.get(
        'use_amplified_zone', use_amplified_zone)
    amplified_zone_weight = profile_loss_overrides.get(
        'amplified_zone_weight', amplified_zone_weight)

    # ── Load Optuna params (override) ──
    resolved, optuna_config_name, quantile_q = apply_optuna_overrides(
        resolved, cfg, load_optuna_params, quantile_q=quantile_q,
    )

    def _pick(key, current):
        return resolved[key] if key in resolved else current

    signal_r2_weight = _pick('signal_r2_weight', signal_r2_weight)
    signal_ccc_weight = _pick('signal_ccc_weight', signal_ccc_weight)
    signal_std_match_weight = _pick('signal_std_match_weight', signal_std_match_weight)
    signal_log_var_weight = _pick('signal_log_var_weight', signal_log_var_weight)
    signal_direction_bce_weight = _pick('signal_direction_bce_weight', signal_direction_bce_weight)
    signal_ema_penalties = _pick('signal_ema_penalties', signal_ema_penalties)
    signal_ema_alpha = _pick('signal_ema_alpha', signal_ema_alpha)
    signal_label_smoothing = _pick('signal_label_smoothing', signal_label_smoothing)
    signal_label_smoothing_alpha = _pick('signal_label_smoothing_alpha', signal_label_smoothing_alpha)
    aug_jitter = _pick('aug_jitter', aug_jitter)
    aug_jitter_std = _pick('aug_jitter_std', aug_jitter_std)
    aug_temporal_dropout = _pick('aug_temporal_dropout', aug_temporal_dropout)
    aug_temporal_dropout_rate = _pick('aug_temporal_dropout_rate', aug_temporal_dropout_rate)
    aug_subsequence = _pick('aug_subsequence', aug_subsequence)
    aug_subsequence_offset = _pick('aug_subsequence_offset', aug_subsequence_offset)
    use_feature_attention = _pick('use_feature_attention', use_feature_attention)
    lr_warmup_epochs = _pick('lr_warmup_epochs', lr_warmup_epochs)
    amplified_zone_threshold = _pick('amplified_zone_threshold', amplified_zone_threshold)
    amplified_zone_weight = _pick(
        'amplified_zone_weight', amplified_zone_weight)

    # CP-166: Validate configuration consistency before starting training / loading data
    LSTMConfig.validate(resolved)

    # Keep a small set of locals only for spots that read them more than
    # once (logging strings, ID prefix). All other values are accessed via
    # `resolved['key']` to avoid drift between dict and local copies.
    lookback = resolved['lookback']
    horizon = resolved['horizon']
    hidden_size = resolved['hidden_size']
    num_layers = resolved['num_layers']
    dropout = resolved['dropout']
    bidirectional = resolved['bidirectional']
    use_attention = resolved['use_attention']
    fc_layers = resolved['fc_layers']
    fc_norm = resolved['fc_norm']
    epochs = resolved['epochs']
    batch_size = resolved['batch_size']
    lr = resolved['lr']
    weight_decay = resolved['weight_decay']
    gradient_clip = resolved['gradient_clip']
    loss_type = resolved['loss_type']
    scheduler_type = resolved['scheduler_type']
    seed = resolved['seed']
    do_calibrate = bool(calibrate) if calibrate is not None else False

    # ── Setup ──
    set_seed(seed)
    device = get_device()
    cfg.ensure_dirs()

    # ── Load data ──
    data, is_prepared, metadata = load_data(
        data_path,
        preferred_format=resolved['preferred_format'],
        feature_set=resolved['feature_set'],
        custom_features=custom_features,
        lookback=lookback,
        horizon=horizon,
        scale_target=resolved['scale_target'],
        target_scaler_type=resolved['target_scaler_type'],
        winsorize_target=resolved['winsorize_target'],
        winsorize_percentile=resolved['winsorize_percentile'],
        val_source=resolved['val_source'],
        data_source=resolved.get('data_source'),
        target_mode=resolved.get('target_mode'),
        use_aux_volatility_head=resolved.get('use_aux_volatility_head'),
        feature_round_decimals=resolved['feature_round_decimals'],
        target_round_decimals=resolved['target_round_decimals'],
    )

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    feature_scaler = data['feature_scaler']
    target_scaler = data['target_scaler']
    sigma_train = data.get('sigma_train')
    sigma_val = data.get('sigma_val')
    sigma_val_raw = data.get('sigma_val_raw')
    val_close_prices = data.get('val_close_prices')
    y_val_raw = data.get('y_val_raw')
    features = metadata['features']
    data_id = extract_data_id(metadata)

    # Detect symbol
    if is_prepared:
        symbol = metadata.get('symbol', detect_symbol(os.path.basename(data_path)))
    else:
        symbol = detect_symbol(os.path.basename(data_path))

    # ── Model ID & output dir ──
    date_short = datetime.now().strftime('%d%m')
    if optuna_config_name:
        run_name_template = (
            f"{{model_id}}_{symbol}_model_{data_id}_{optuna_config_name}_{date_short}"
        )
    else:
        run_name_template = f"{{model_id}}_{symbol}_model_{data_id}_{date_short}"
    model_id, output_dir = reserve_next_model_dir(
        run_name_template, base_dir=cfg.MODELS_DIR
    )

    logger = setup_logger(output_dir)

    tracker = ExperimentTracker.create(
        enabled=bool(use_wandb), config={
            'model_id': model_id, 'data_id': data_id,
            'hidden_size': hidden_size, 'num_layers': num_layers,
            'loss_type': loss_type, 'scheduler_type': scheduler_type,
        }, run_name=f"{model_id}_train")

    # ── Log session ──
    log_train_session_header(logger, symbol, horizon, model_id, data_id, output_dir)
    logger.info(f"Device:     {device}")
    logger.info(f"Seed:       {seed}")
    # D3: make it obvious which params file drove this run (was previously
    # only visible via run_name suffix and in model_config.json).
    if load_optuna_params is not None:
        logger.info(f"Optuna params: {load_optuna_params}")
        if optuna_config_name:
            logger.info(f"Optuna ID:   {optuna_config_name}")
    log_data_summary(features, lookback, horizon, len(X_train), len(X_val),
                     resolved['val_source'], logger)

    # ── Signal threshold ──
    signal_threshold = None
    if resolved['signal_threshold_enabled']:
        y_raw_for_threshold = (target_scaler.inverse_transform(y_train)
                               if target_scaler else y_train)
        signal_threshold = resolve_signal_threshold(
            y_raw_for_threshold, mode=resolved['signal_threshold_mode'],
            fixed_value=resolved['signal_threshold_value'],
            percentile=cfg.SIGNAL_THRESHOLD_PERCENTILE,
        )
    log_signal_threshold(resolved['signal_threshold_enabled'], signal_threshold,
                         resolved['signal_threshold_mode'], logger)

    # ── Build model ──
    # Push resolved architecture knobs onto cfg so build_model picks them up.
    apply_resolved_to_cfg(cfg, resolved, rnn_type=rnn_type)

    input_size = X_train.shape[2]  # n_features
    model = build_model(cfg, input_size, device)
    n_params = model.count_parameters()

    log_model_architecture(model, logger)
    logger.info(f"Model:      Hidden={hidden_size}, Layers={num_layers}, "
                f"Dropout={dropout}")
    logger.info(f"Bidir:      {bidirectional}, Attention: {use_attention}")
    logger.info(f"FC layers:  {fc_layers}")
    logger.info(f"FC norm:    {fc_norm}")

    # ── Build loss ──
    apply_custom_loss_flags(
        cfg,
        loss_type=loss_type, huber_delta=resolved['huber_delta'],
        quantile_q=quantile_q,
        use_sample_weights=use_sample_weights,
        use_misdir_penalty=use_misdir_penalty,
        use_dead_zone=use_dead_zone,
        dead_zone_threshold=dead_zone_threshold,
        use_amplified_zone=use_amplified_zone,
        amplified_zone_threshold=amplified_zone_threshold,
        amplified_zone_weight=amplified_zone_weight,
        use_correlation_penalty=use_correlation_penalty,
        correlation_penalty_weight=correlation_penalty_weight,
        use_variance_penalty=use_variance_penalty,
        variance_penalty_weight=variance_penalty_weight,
        use_directional_loss=use_directional_loss,
        directional_loss_weight=directional_loss_weight,
        use_log_variance_penalty=use_log_variance_penalty,
        log_variance_penalty_weight=log_variance_penalty_weight,
        use_r2_penalty=use_r2_penalty,
        r2_penalty_weight=r2_penalty_weight,
        use_ccc_penalty=use_ccc_penalty,
        ccc_penalty_weight=ccc_penalty_weight,
        signal_use_r2=signal_use_r2,
        signal_r2_weight=signal_r2_weight,
        signal_use_ccc=signal_use_ccc,
        signal_ccc_weight=signal_ccc_weight,
        signal_use_log_var=signal_use_log_var,
        signal_log_var_weight=signal_log_var_weight,
        signal_use_std_match=signal_use_std_match,
        signal_std_match_weight=signal_std_match_weight,
        signal_use_dir_threshold=signal_use_dir_threshold,
        signal_dir_threshold_weight=signal_dir_threshold_weight,
        signal_use_direction_bce=signal_use_direction_bce,
        signal_direction_bce_weight=signal_direction_bce_weight,
        signal_ema_penalties=signal_ema_penalties,
        signal_ema_alpha=signal_ema_alpha,
        signal_label_smoothing=signal_label_smoothing,
        signal_label_smoothing_alpha=signal_label_smoothing_alpha,
        signal_use_utility=signal_use_utility,
        signal_utility_weight=signal_utility_weight,
        signal_soft_threshold=signal_soft_threshold,
        use_cr_ceiling=use_cr_ceiling,
        cr_ceiling=cr_ceiling,
        cr_ceiling_weight=cr_ceiling_weight,
    )

    criterion = get_loss_function(cfg, target_scaler, signal_threshold)
    # Move criterion buffers/parameters (e.g. _target_scale, _target_center
    # in LSTMCustomLoss) to the same device as the model. 0-D CPU scalars
    # currently cross-promote with GPU tensors, but future non-scalar buffers
    # (e.g. per-feature statistics) would crash without this.
    criterion = criterion.to(device)
    # `is_custom_loss` decides:
    #   - whether `indices` are forwarded to criterion (needed for
    #     LSTMCustomLoss's sample weights; harmless for SignalLoss),
    #   - whether `_last_components` is aggregated for breakdown logging.
    # SignalLoss qualifies on both counts, so include it.
    is_custom_loss = isinstance(criterion, (LSTMCustomLoss, SignalLoss))

    # Pre-compute LSTMCustomLoss-only inputs (SignalLoss has none).
    if isinstance(criterion, LSTMCustomLoss):
        y_raw = (target_scaler.inverse_transform(y_train)
                 if target_scaler else y_train)
        criterion.set_epoch_weights(y_raw)
        # F2.V1-2: epoch-level misdir threshold (stable, see set_misdir_threshold).
        if getattr(cfg, 'USE_MISDIR_PENALTY', False):
            criterion.set_misdir_threshold(y_raw)

    # FAZA 5 — scheduled loss switching. Built lazily inside `train_setup`
    # so this function stays orchestration-only.
    y_raw_for_full = (target_scaler.inverse_transform(y_train)
                      if target_scaler is not None else y_train)
    scheduled_loss_arg = build_scheduled_loss(
        cfg, target_scaler, signal_threshold, device,
        use_scheduled_loss=use_scheduled_loss,
        scheduled_loss_warmup_epochs=scheduled_loss_warmup_epochs,
        scheduled_loss_base=scheduled_loss_base,
        scheduled_loss_full=scheduled_loss_full,
        y_train_raw_for_full=y_raw_for_full,
    )
    if scheduled_loss_arg is not None:
        # Swap to the full criterion so the metric pipeline + breakdown
        # logger match the training's final behaviour.
        _, criterion, _ = scheduled_loss_arg
        is_custom_loss = isinstance(
            criterion, (LSTMCustomLoss, SignalLoss))

    config_flags = {
        'loss_type': loss_type,
        'quantile_q': getattr(cfg, 'QUANTILE_LOSS_Q', 0.5),
        'winsorize_target': resolved['winsorize_target'],
        'feature_round_decimals': resolved['feature_round_decimals'],
        'target_round_decimals': resolved['target_round_decimals'],
        'use_sample_weights': getattr(cfg, 'USE_SAMPLE_WEIGHTS', False),
        'use_misdir_penalty': getattr(cfg, 'USE_MISDIR_PENALTY', False),
        'use_dead_zone': getattr(cfg, 'USE_DEAD_ZONE', False),
        'dead_zone_threshold': getattr(cfg, 'DEAD_ZONE_THRESHOLD', 0.001),
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
        'scheduler_type': scheduler_type,
    }
    log_config_flags(config_flags, target_scaler, logger)

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    logger.info(f"Training:   Epochs={epochs}, Batch={batch_size}, "
                f"LR={lr}, WD={weight_decay}")

    # DataLoaders — optional per-sample sigma for SignalLoss utility term.
    train_sigma = sigma_train if getattr(cfg, 'SIGNAL_USE_UTILITY', False) else None
    val_sigma = sigma_val if getattr(cfg, 'SIGNAL_USE_UTILITY', False) else None
    train_loader = torch.utils.data.DataLoader(
        CryptoDataset(X_train, y_train, sigma=train_sigma),
        batch_size=batch_size,
        shuffle=True, drop_last=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == 'cuda' and cfg.USE_PIN_MEMORY),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )
    val_loader = torch.utils.data.DataLoader(
        CryptoDataset(X_val, y_val, sigma=val_sigma),
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=(device.type == 'cuda' and cfg.USE_PIN_MEMORY),
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    val_signal_ctx = build_val_signal_ctx(
        val_close_prices, lookback, horizon, signal_threshold, cfg,
        sigma=sigma_val_raw)

    total_steps = epochs * len(train_loader)
    scheduler = get_scheduler(
        optimizer, scheduler_type,
        total_steps=total_steps,
        lr_warmup_epochs=resolved['lr_warmup_epochs'],
        **cfg.SCHEDULER_PARAMS)

    augmenter = SequenceAugmenter.from_config(cfg)

    # ── Train ──
    (best_state, train_losses, val_losses, train_metrics, val_metrics,
     best_epoch, diagnostics_history, swa_state) = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, device, epochs, resolved['early_stopping_patience'],
        gradient_clip, logger, target_scaler, signal_threshold,
        scheduler_type, is_custom_loss,
        scheduled_loss=scheduled_loss_arg,
        augmenter=augmenter,
        val_signal_ctx=val_signal_ctx,
        min_epochs_before_es=resolved['min_epochs_before_es'],
        early_stopping_min_delta=resolved['early_stopping_min_delta'],
        use_swa=resolved['use_swa'],
        swa_start_epoch=resolved['swa_start_epoch'],
        swa_every=resolved['swa_every'],
        tracker=tracker,
    )

    # ── Log & save ──
    log_train_val_metrics(logger, train_metrics, val_metrics)

    config_dict = build_train_config_dict(
        cfg=cfg, resolved=resolved, quantile_q=quantile_q,
        model_id=model_id, data_id=data_id, symbol=symbol,
        features=features, n_params=n_params,
        train_losses=train_losses, val_losses=val_losses,
        train_metrics=train_metrics, val_metrics=val_metrics,
        best_epoch=best_epoch, diagnostics_history=diagnostics_history,
        signal_threshold=signal_threshold, is_prepared=is_prepared,
        data_path=data_path, load_optuna_params=load_optuna_params,
    )

    save_model(best_state, feature_scaler, config_dict, output_dir,
               logger, target_scaler, swa_state=swa_state)

    if do_calibrate and val_close_prices is not None and y_val_raw is not None:
        fit_and_save_calibrator(
            model, val_loader, device, target_scaler,
            y_val_raw, val_close_prices, lookback, horizon,
            signal_threshold, cfg, output_dir, logger)

    plot_training_curves(
        train_losses, val_losses, best_epoch,
        os.path.join(output_dir, 'training_curves.png'),
    )
    logger.info("  ✓ training_curves.png")

    log_final_summary(model_id, output_dir, train_metrics, val_metrics, logger)

    tracker.log_summary({
        'final_train_mae': train_metrics.get('mae'),
        'final_val_mae': val_metrics.get('mae'),
        'final_val_dir_acc': val_metrics.get('dir_acc'),
        'best_epoch': best_epoch,
    })
    tracker.finish()

    return model, feature_scaler, config_dict


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    _args = build_parser().parse_args()
    dispatch_args_to_train(train, _args)
