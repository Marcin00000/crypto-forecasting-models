"""
CLI argument parser + dispatcher for `lstm_train.py`.

The argparse block in `lstm_train.py` grew past the 500-LOC ceiling once
FAZA 3 (SignalLoss flags) and FAZA 5 (scheduled-loss flags) landed. Moving
the parser into its own module keeps the training orchestration file
focused on the actual training pipeline. The mapping of `args` →
`train()` kwargs lives in `dispatch_args_to_train` so the schema is kept
in lockstep with the parser.

Each kwarg flows through `train(**kwargs)` as `None` when the user did
not specify it on the command line; `train()` then resolves the value
from LSTMConfig (or the loaded Optuna params).
"""
from __future__ import annotations

import argparse


def _maybe_bool(value):
    """argparse `int(0|1)` → optional `bool` (None passes through)."""
    return None if value is None else bool(value)


def build_parser():
    """Return the argparse.ArgumentParser used by lstm_train.py CLI."""
    parser = argparse.ArgumentParser(
        description="Train LSTM for log-return prediction"
    )

    # ── Data / paths ──
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to prepared data directory or raw CSV")
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--custom_features', type=str, nargs='+', default=None,
                        help="Explicit list of feature names to use (overrides --feature_set).")
    parser.add_argument('--preferred_format', type=str, default=None,
                        choices=['csv', 'npy'])
    parser.add_argument('--val_source', type=str, default=None,
                        choices=['split', 'test', 'agent'],
                        help=("'split' = 80/20 chronological split of "
                              "training data; 'test' = use test.csv as val; "
                              "'agent' = use agent.csv as val."))
    parser.add_argument('--data_source', type=str, default=None,
                        choices=['train', 'agent'],
                        help="Data source: 'train' (default, train.csv) or 'agent' (agent.csv)")
    parser.add_argument('--training_profile', type=str, default=None,
                        choices=['baseline', 'rank_focused', 'sim_focused',
                                 'custom_sim',
                                 'rank-focused', 'sim-focused'],
                        help="Optional named recipe overlay for rank/sim lines.")

    # ── Architecture ──
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--bidirectional', type=int, default=None,
                        choices=[0, 1])
    parser.add_argument('--use_attention', type=int, default=None,
                        choices=[0, 1])
    parser.add_argument('--fc_layers', type=int, nargs='+', default=None,
                        help="FC head layer sizes, e.g. --fc_layers 64 32")
    parser.add_argument('--fc_norm', type=str, default=None,
                        choices=['batch', 'layer', 'none'])
    parser.add_argument('--arch_type', type=str, default=None,
                        choices=['legacy', 'dual_head'],
                        help="Optional architecture variant; legacy is baseline.")
    parser.add_argument('--rnn_type', type=str, default=None,
                        choices=['lstm', 'gru'], help="F4.M6: RNN cell type.")
    parser.add_argument('--attention_heads', type=int, default=None,
                        help="F4.M3: number of attention heads.")
    parser.add_argument('--lstm_dropout', type=float, default=None,
                        help="F4.M4: dropout between stacked LSTM/GRU layers.")
    parser.add_argument('--fc_dropout', type=float, default=None,
                        help="F4.M4: dropout inside FC head.")
    parser.add_argument('--activation', type=str, default=None,
                        choices=['relu', 'gelu', 'leaky_relu'],
                        help="FC/projection activation function.")

    # ── Training ──
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--scheduler_type', type=str, default=None,
                        choices=['plateau', 'cosine', 'onecycle', 'step',
                                 'none'])
    parser.add_argument('--best_model_metric', type=str, default=None,
                        choices=['val', 'train', 'val_composite', 'val_signal'],
                        help="Metric used for checkpoint/early stopping.")
    parser.add_argument('--best_model_cr_weight', type=float, default=None,
                        help="Weight for |1-CR| when best_model_metric=val_composite.")
    parser.add_argument('--min_epochs_before_es', type=int, default=None,
                        help="Minimum epochs before early stopping can trigger.")
    parser.add_argument('--early_stopping_min_delta', type=float, default=None,
                        help="Min improvement to reset ES patience (0=any).")
    parser.add_argument('--use_swa', type=int, default=None, choices=[0, 1],
                        help="Stochastic Weight Averaging (default off).")
    parser.add_argument('--swa_start_epoch', type=int, default=None,
                        help="First epoch for SWA accumulation.")
    parser.add_argument('--swa_every', type=int, default=None,
                        help="Update SWA every N epochs.")

    # ── Loss core ──
    _LOSS_CHOICES = ['mse', 'mae', 'huber', 'smooth_l1', 'pseudo_huber',
                     'quantile', 'custom', 'signal']
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=_LOSS_CHOICES)
    parser.add_argument('--quantile_q', type=float, default=None,
                        help="F2.4: quantile q in (0,1) for LOSS_TYPE=quantile.")
    parser.add_argument('--huber_delta', type=float, default=None,
                        help="Shape parameter for huber/pseudo_huber loss.")
    parser.add_argument('--signal_threshold_enabled', type=int, default=None,
                        choices=[0, 1],
                        help="Enable economic threshold T for signal scoring.")
    parser.add_argument('--signal_threshold_mode', type=str, default=None,
                        choices=['fixed', 'empirical'],
                        help="'fixed'=use signal_threshold_value; "
                             "'empirical'=derive from training data percentile.")
    parser.add_argument('--signal_threshold_value', type=float, default=None,
                        help="Fixed threshold value T when "
                             "signal_threshold_mode=fixed.")

    # ── LSTMCustomLoss component flags ──
    parser.add_argument('--use_amplified_zone', type=int, default=None,
                        choices=[0, 1], help="F2.3.")
    parser.add_argument('--amplified_zone_threshold', type=float, default=None)
    parser.add_argument('--amplified_zone_weight', type=float, default=None)
    parser.add_argument('--use_directional_loss', type=int, default=None,
                        choices=[0, 1], help="F2.2.")
    parser.add_argument('--directional_loss_weight', type=float, default=None)
    parser.add_argument('--use_correlation_penalty', type=int, default=None,
                        choices=[0, 1], help="F2.1a.")
    parser.add_argument('--correlation_penalty_weight', type=float, default=None)
    parser.add_argument('--use_variance_penalty', type=int, default=None,
                        choices=[0, 1], help="F2.1b.")
    parser.add_argument('--variance_penalty_weight', type=float, default=None)

    # ── SignalLoss v2 (LOSS_TYPE='signal') ──
    parser.add_argument('--signal_use_r2', type=int, default=None,
                        choices=[0, 1], help="F3.V2: R² penalty.")
    parser.add_argument('--signal_r2_weight', type=float, default=None)
    parser.add_argument('--signal_use_ccc', type=int, default=None,
                        choices=[0, 1], help="F3.V2: CCC penalty.")
    parser.add_argument('--signal_ccc_weight', type=float, default=None)
    parser.add_argument('--signal_use_log_var', type=int, default=None,
                        choices=[0, 1], help="F3.V2: log-variance penalty.")
    parser.add_argument('--signal_log_var_weight', type=float, default=None)
    parser.add_argument('--signal_use_std_match', type=int, default=None,
                        choices=[0, 1],
                        help="Anti-collapse std-match (targets CR≈1).")
    parser.add_argument('--signal_std_match_weight', type=float, default=None)
    parser.add_argument('--signal_use_dir_threshold', type=int, default=None,
                        choices=[0, 1],
                        help="Directional penalty on |y|>=signal_threshold.")
    parser.add_argument('--signal_dir_threshold_weight', type=float,
                        default=None)
    parser.add_argument('--signal_use_direction_bce', type=int, default=None,
                        choices=[0, 1],
                        help="BCE direction calibration on |y|>=threshold.")
    parser.add_argument('--signal_direction_bce_weight', type=float,
                        default=None)
    parser.add_argument('--signal_use_utility', type=int, default=None,
                        choices=[0, 1],
                        help="F1b: utility loss aligned with signal formula.")
    parser.add_argument('--signal_utility_weight', type=float, default=None)
    parser.add_argument('--signal_soft_threshold', type=int, default=None,
                        choices=[0, 1],
                        help="F1b: soft sigmoid masks at economic threshold.")
    parser.add_argument('--use_cr_ceiling', type=int, default=None,
                        choices=[0, 1],
                        help="Penalise std(pred)/std(true) above ceiling.")
    parser.add_argument('--cr_ceiling', type=float, default=None)
    parser.add_argument('--cr_ceiling_weight', type=float, default=None)

    # ── F7 — SignalLoss extras + augmentation + feature attention ──
    parser.add_argument('--signal_ema_penalties', type=int, default=None,
                        choices=[0, 1], help="F7.A: EMA-smooth batch penalties.")
    parser.add_argument('--signal_ema_alpha', type=float, default=None)
    parser.add_argument('--signal_label_smoothing', type=int, default=None,
                        choices=[0, 1], help="F7.D: dead-zone label smoothing.")
    parser.add_argument('--signal_label_smoothing_alpha', type=float,
                        default=None)
    parser.add_argument('--aug_jitter', type=int, default=None,
                        choices=[0, 1], help="F7.B: Gaussian feature jitter.")
    parser.add_argument('--aug_jitter_std', type=float, default=None)
    parser.add_argument('--aug_temporal_dropout', type=int, default=None,
                        choices=[0, 1], help="F7.B: zero random timesteps.")
    parser.add_argument('--aug_temporal_dropout_rate', type=float,
                        default=None)
    parser.add_argument('--aug_subsequence', type=int, default=None,
                        choices=[0, 1], help="F7.B: circular time shift.")
    parser.add_argument('--aug_subsequence_offset', type=int, default=None)
    parser.add_argument('--use_feature_attention', type=int, default=None,
                        choices=[0, 1], help="F7.F: SE gate before RNN.")
    parser.add_argument('--use_rnn_norm', type=int, default=None,
                        choices=[0, 1], help="F6: LayerNorm on RNN output.")
    parser.add_argument('--pooling_mode', type=str, default=None,
                        choices=['last', 'mean', 'max'],
                        help="Pooling when use_attention=False.")
    parser.add_argument('--use_conv1d_front', type=int, default=None,
                        choices=[0, 1], help="F6: Conv1d front before RNN.")
    parser.add_argument('--conv1d_kernel', type=int, default=None,
                        help="F6: Conv1d kernel size.")
    parser.add_argument('--use_fc_residual', type=int, default=None,
                        choices=[0, 1], help="F6: residual FC blocks.")
    parser.add_argument('--use_learnable_output_scale', type=int, default=None,
                        choices=[0, 1], help="F6: learnable output scale/bias.")
    parser.add_argument('--freeze_rnn', type=int, default=None,
                        choices=[0, 1], help="F6: freeze RNN weights.")
    parser.add_argument('--lr_warmup_epochs', type=int, default=None,
                        help="F7.E: linear LR warmup epochs (0=off).")

    # ── Scheduled loss switching (FAZA 5) ──
    parser.add_argument('--use_scheduled_loss', type=int, default=None,
                        choices=[0, 1], help="F5: scheduled loss switching.")
    parser.add_argument('--scheduled_loss_warmup_epochs', type=int,
                        default=None)
    parser.add_argument('--scheduled_loss_base', type=str, default=None,
                        choices=_LOSS_CHOICES)
    parser.add_argument('--scheduled_loss_full', type=str, default=None,
                        choices=_LOSS_CHOICES)

    # ── FAZA 1: infrastructure flags ──
    parser.add_argument('--gradient_clip', type=float, default=None,
                        help="F1.I1: max grad norm (0=disabled).")
    parser.add_argument('--use_input_norm', type=int, default=None,
                        choices=[0, 1], help="F1.I2: LayerNorm on input.")
    parser.add_argument('--weight_init', type=str, default=None,
                        choices=['default', 'kaiming', 'xavier'],
                        help="F1.I3: weight init strategy.")
    parser.add_argument('--prediction_precision', type=int, default=None,
                        help="F1.I4: round predictions to N decimals (0=off).")
    parser.add_argument('--target_precision', type=int, default=None,
                        help="F1.I4: round targets to N decimals (0=off).")
    parser.add_argument('--feature_round_decimals', type=int, default=None,
                        help="F1.I4b: round features to N decimals before training (0=off).")
    parser.add_argument('--target_round_decimals', type=int, default=None,
                        help="F1.I4b: round targets to N decimals before training (0=off).")
    parser.add_argument('--log_gradient_norms', type=int, default=None,
                        choices=[0, 1],
                        help="F1.I5: log gradient norms per layer-group.")

    # ── Target / Optuna / misc ──
    parser.add_argument('--target_mode', type=str, default=None,
                        choices=['raw', 'vol_norm'],
                        help="CP-175: raw returns or volatility-normalized targets.")
    parser.add_argument('--use_aux_volatility_head', type=int, default=None,
                        choices=[0, 1], help="CP-028: predict sigma alongside returns.")
    parser.add_argument('--use_group_dro', type=int, default=None,
                        choices=[0, 1], help="CP-183: Group DRO for market regimes.")
    parser.add_argument('--group_dro_n_groups', type=int, default=None,
                        help="CP-183: number of volatility groups.")
    parser.add_argument('--group_dro_step_size', type=float, default=None,
                        help="CP-183: group weight update step size.")
    parser.add_argument('--scale_target', type=int, default=None,
                        help="1=enable, 0=disable")
    parser.add_argument('--target_scaler_type', type=str, default=None,
                        choices=['standard', 'robust'])
    parser.add_argument('--winsorize_target', type=int, default=None,
                        choices=[0, 1], help="Enable target winsorization.")
    parser.add_argument('--winsorize_percentile', type=float, default=None,
                        help="Winsorization percentile (e.g. 99.0)")
    parser.add_argument('--use_sample_weights', type=int, default=None,
                        choices=[0, 1], help="Enable sample weighting.")
    parser.add_argument('--use_misdir_penalty', type=int, default=None,
                        choices=[0, 1],
                        help="Enable misdirection penalty in custom loss.")
    parser.add_argument('--use_dead_zone', type=int, default=None,
                        choices=[0, 1], help="Enable dead zone in custom loss.")
    parser.add_argument('--dead_zone_threshold', type=float, default=None,
                        help="Threshold for dead zone.")
    parser.add_argument('--use_log_variance_penalty', type=int, default=None,
                        choices=[0, 1],
                        help="Enable log-variance penalty in custom loss.")
    parser.add_argument('--use_r2_penalty', type=int, default=None,
                        choices=[0, 1], help="Enable R2 penalty in custom loss.")
    parser.add_argument('--use_ccc_penalty', type=int, default=None,
                        choices=[0, 1], help="Enable CCC penalty in custom loss.")
    parser.add_argument('--load_optuna_params', type=str, default=None,
                        help="Path to Optuna best_params.json")
    parser.add_argument('--calibrate', type=int, default=None, choices=[0, 1],
                        help="Fit isotonic calibrator on val after training.")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--use_wandb', type=int, default=0, choices=[0, 1],
                        help="Faza 8: log run to Weights & Biases (opt-in).")

    return parser


def dispatch_args_to_train(train_fn, args):
    """
    Translate parsed argparse args into a `train_fn(**kwargs)` call.

    The kwarg names mirror the `train()` signature in `lstm_train.py`. Bool
    flags coming from `type=int, choices=[0, 1]` are converted via
    `_maybe_bool` so they remain tri-state (None / True / False).
    """
    return train_fn(
        data_path=args.data_path,
        # data
        lookback=args.lookback, horizon=args.horizon,
        feature_set=args.feature_set,
        custom_features=args.custom_features,
        preferred_format=args.preferred_format,
        val_source=args.val_source,
        data_source=args.data_source,
        training_profile=args.training_profile,
        # architecture
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        dropout=args.dropout,         bidirectional=_maybe_bool(args.bidirectional),
        use_attention=_maybe_bool(args.use_attention), fc_norm=args.fc_norm,
        fc_layers=args.fc_layers,
        arch_type=args.arch_type,
        rnn_type=args.rnn_type, attention_heads=args.attention_heads,
        lstm_dropout=args.lstm_dropout, fc_dropout=args.fc_dropout,
        activation=args.activation,
        # training
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_type=args.scheduler_type,
        best_model_metric=args.best_model_metric,
        best_model_cr_weight=args.best_model_cr_weight,
        min_epochs_before_es=args.min_epochs_before_es,
        early_stopping_min_delta=args.early_stopping_min_delta,
        use_swa=_maybe_bool(args.use_swa),
        swa_start_epoch=args.swa_start_epoch,
        swa_every=args.swa_every,
        # loss core
        loss_type=args.loss_type, quantile_q=args.quantile_q,
        huber_delta=args.huber_delta,
        signal_threshold_enabled=_maybe_bool(args.signal_threshold_enabled),
        signal_threshold_mode=args.signal_threshold_mode,
        signal_threshold_value=args.signal_threshold_value,
        # LSTMCustomLoss components
        use_amplified_zone=_maybe_bool(args.use_amplified_zone),
        amplified_zone_threshold=args.amplified_zone_threshold,
        amplified_zone_weight=args.amplified_zone_weight,
        use_correlation_penalty=_maybe_bool(args.use_correlation_penalty),
        correlation_penalty_weight=args.correlation_penalty_weight,
        use_variance_penalty=_maybe_bool(args.use_variance_penalty),
        variance_penalty_weight=args.variance_penalty_weight,
        use_directional_loss=_maybe_bool(args.use_directional_loss),
        directional_loss_weight=args.directional_loss_weight,
        # SignalLoss
        signal_use_r2=_maybe_bool(args.signal_use_r2),
        signal_r2_weight=args.signal_r2_weight,
        signal_use_ccc=_maybe_bool(args.signal_use_ccc),
        signal_ccc_weight=args.signal_ccc_weight,
        signal_use_log_var=_maybe_bool(args.signal_use_log_var),
        signal_log_var_weight=args.signal_log_var_weight,
        signal_use_std_match=_maybe_bool(args.signal_use_std_match),
        signal_std_match_weight=args.signal_std_match_weight,
        signal_use_dir_threshold=_maybe_bool(args.signal_use_dir_threshold),
        signal_dir_threshold_weight=args.signal_dir_threshold_weight,
        signal_use_direction_bce=_maybe_bool(args.signal_use_direction_bce),
        signal_direction_bce_weight=args.signal_direction_bce_weight,
        signal_use_utility=_maybe_bool(args.signal_use_utility),
        signal_utility_weight=args.signal_utility_weight,
        signal_soft_threshold=_maybe_bool(args.signal_soft_threshold),
        use_cr_ceiling=_maybe_bool(args.use_cr_ceiling),
        cr_ceiling=args.cr_ceiling,
        cr_ceiling_weight=args.cr_ceiling_weight,
        signal_ema_penalties=_maybe_bool(args.signal_ema_penalties),
        signal_ema_alpha=args.signal_ema_alpha,
        signal_label_smoothing=_maybe_bool(args.signal_label_smoothing),
        signal_label_smoothing_alpha=args.signal_label_smoothing_alpha,
        aug_jitter=_maybe_bool(args.aug_jitter),
        aug_jitter_std=args.aug_jitter_std,
        aug_temporal_dropout=_maybe_bool(args.aug_temporal_dropout),
        aug_temporal_dropout_rate=args.aug_temporal_dropout_rate,
        aug_subsequence=_maybe_bool(args.aug_subsequence),
        aug_subsequence_offset=args.aug_subsequence_offset,
        use_feature_attention=_maybe_bool(args.use_feature_attention),
        use_rnn_norm=_maybe_bool(args.use_rnn_norm),
        pooling_mode=args.pooling_mode,
        use_conv1d_front=_maybe_bool(args.use_conv1d_front),
        conv1d_kernel=args.conv1d_kernel,
        use_fc_residual=_maybe_bool(args.use_fc_residual),
        use_learnable_output_scale=_maybe_bool(
            args.use_learnable_output_scale),
        freeze_rnn=_maybe_bool(args.freeze_rnn),
        lr_warmup_epochs=args.lr_warmup_epochs,
        # Scheduled loss
        use_scheduled_loss=_maybe_bool(args.use_scheduled_loss),
        scheduled_loss_warmup_epochs=args.scheduled_loss_warmup_epochs,
        scheduled_loss_base=args.scheduled_loss_base,
        scheduled_loss_full=args.scheduled_loss_full,
        # FAZA 1 infrastructure
        gradient_clip=args.gradient_clip,
        use_input_norm=_maybe_bool(args.use_input_norm),
        weight_init=args.weight_init,
        prediction_precision=args.prediction_precision,
        target_precision=args.target_precision,
        feature_round_decimals=args.feature_round_decimals,
        target_round_decimals=args.target_round_decimals,
        log_gradient_norms=_maybe_bool(args.log_gradient_norms),
        # target / optuna / misc
        target_mode=args.target_mode,
        use_aux_volatility_head=_maybe_bool(args.use_aux_volatility_head),
        use_group_dro=_maybe_bool(args.use_group_dro),
        group_dro_n_groups=args.group_dro_n_groups,
        group_dro_step_size=args.group_dro_step_size,
        scale_target=_maybe_bool(args.scale_target),
        target_scaler_type=args.target_scaler_type,
        winsorize_target=_maybe_bool(args.winsorize_target),
        winsorize_percentile=args.winsorize_percentile,
        use_sample_weights=_maybe_bool(args.use_sample_weights),
        use_misdir_penalty=_maybe_bool(args.use_misdir_penalty),
        use_dead_zone=_maybe_bool(args.use_dead_zone),
        dead_zone_threshold=args.dead_zone_threshold,
        use_log_variance_penalty=_maybe_bool(args.use_log_variance_penalty),
        use_r2_penalty=_maybe_bool(args.use_r2_penalty),
        use_ccc_penalty=_maybe_bool(args.use_ccc_penalty),
        load_optuna_params=args.load_optuna_params,
        calibrate=_maybe_bool(args.calibrate),
        seed=args.seed,
        use_wandb=bool(args.use_wandb),
    )
