"""
CLI argument parser for Signal Model training.

Separated from training logic so signal_train.py stays a slim orchestrator.
"""
import argparse

from .training_profiles import PROFILE_CHOICES


def _parse_fc_layers(val):
    if val is None:
        return None
    res = []
    if isinstance(val, (list, tuple)):
        for item in val:
            if isinstance(item, int):
                res.append(item)
            elif isinstance(item, str):
                res.extend([int(x) for x in item.split(',') if x.strip()])
    elif isinstance(val, str):
        res.extend([int(x) for x in val.split(',') if x.strip()])
    elif isinstance(val, int):
        res.append(val)
    return res if res else None


def build_parser():
    """Build argparse parser with key SignalConfig overrides."""
    p = argparse.ArgumentParser(
        description='LSTM Signal Model — Training')

    # Data
    p.add_argument('--data_path', type=str, required=True,
                   help='Path to prepared data directory')
    p.add_argument('--lookback', type=int, default=None)
    p.add_argument('--horizon', type=int, default=None)
    p.add_argument('--feature_set', type=str, default=None)
    p.add_argument('--allow_partial_feature_set', type=int, default=0,
                   choices=[0, 1],
                   help='Allow a feature set to resolve to fewer features than '
                        'requested (0/1, default 0 = hard error)')
    p.add_argument('--min_feature_coverage', type=float, default=1.0,
                   help='Minimum fraction of the requested feature set that must '
                        'exist in the dataset (default 1.0)')
    # --preferred_format mirrors the LR family flag name so one CLI recipe works for both.
    p.add_argument('--preferred_data_format', '--preferred_format',
                   dest='preferred_data_format', type=str, default=None,
                   choices=['csv', 'npy'])
    p.add_argument('--val_source', type=str, default=None,
                   choices=['split', 'test', 'sym', 'agent'])

    # Model Architecture
    p.add_argument('--arch_type', type=str, default=None,
                   choices=['legacy', 'dual_head', 'submodel_fusion'],
                   help='Architecture variant')
    p.add_argument('--submodel_size', type=int, default=None,
                   help='Input dimension for submodel fusion head')
    p.add_argument('--hidden_size', type=int, default=None)
    p.add_argument('--num_layers', type=int, default=None)
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--fc_dropout', type=float, default=None)
    p.add_argument('--lstm_dropout', type=float, default=None)
    p.add_argument('--fc_layers', type=str, nargs='+', default=None,
                   help='FC layer sizes e.g. 64 32 or 64,32')
    p.add_argument('--activation', type=str, default=None,
                   choices=['relu', 'gelu', 'leaky_relu', 'silu', 'swish', 'mish'])
    p.add_argument('--fc_norm', type=str, default=None,
                   choices=['batch', 'layer', 'none'])
    p.add_argument('--weight_init', type=str, default=None,
                   help='Weight initialization strategy (e.g. kaiming, default)')
    p.add_argument('--no_attention', action='store_true')
    p.add_argument('--attention_heads', type=int, default=None,
                   help='Number of temporal attention heads')
    p.add_argument('--bidirectional', type=int, default=None, choices=[0, 1],
                   help='Bidirectional LSTM layer (0/1)')
    p.add_argument('--rnn_type', type=str, default=None, choices=['lstm', 'gru'],
                   help="RNN architecture type ('lstm' or 'gru')")
    p.add_argument('--use_feature_attention', type=int, default=None,
                   choices=[0, 1],
                   help='Squeeze-and-excitation gate on input features (0/1)')
    p.add_argument('--output_mode', type=str, default=None,
                   choices=['dual', 'single'])

    # F6 architecture flags
    p.add_argument('--use_conv1d_front', type=int, default=None,
                   choices=[0, 1],
                   help='1D conv front-end before RNN (0/1)')
    p.add_argument('--conv1d_kernel', type=int, default=None,
                   help='Kernel size for conv1d_front (default 5)')
    p.add_argument('--pooling_mode', type=str, default=None,
                   choices=['last', 'mean', 'max'],
                   help='RNN output pooling when attention disabled')
    p.add_argument('--use_fc_residual', type=int, default=None,
                   choices=[0, 1],
                   help='Residual skip in FC blocks (0/1)')
    p.add_argument('--use_rnn_norm', type=int, default=None,
                   choices=[0, 1],
                   help='LayerNorm on RNN output (0/1)')
    p.add_argument('--use_learnable_output_scale', type=int, default=None,
                   choices=[0, 1],
                   help='Learnable per-head scale+bias before sigmoid (0/1)')
    p.add_argument('--freeze_rnn', type=int, default=None,
                   choices=[0, 1],
                   help='Freeze RNN weights (0/1, for FC fine-tuning)')

    # Submodel integration
    p.add_argument('--submodel_dirs', type=str, nargs='+', default=None,
                   help='LR, XGB, RF or Price-LSTM model directories (or IDs) for dynamic '
                        'submodel fusion, e.g. LR1107 X240 R057 L012. LR/XGB/RF channels '
                        'carry mu, the Price channel the predicted price in USD.')
    p.add_argument('--use_submodel_norm', type=int, default=None,
                   choices=[0, 1],
                   help='Apply normalization to submodel inputs in SubmodelFusionHead (0/1, default 1)')
    p.add_argument('--submodel_norm_type', type=str, default=None,
                   choices=['auto', 'batch', 'zscore', 'layer', 'none'],
                   help="Normalization type for submodel fusion: 'auto' (always BatchNorm1d, D=1 and D>1), "
                        "'batch' (BatchNorm1d), 'zscore' (fixed scale), 'layer' (LayerNorm), 'none'")
    p.add_argument('--submodel_feature_set', type=str, default=None,
                   help='Feature set used for feeding submodel predictions (e.g. submodel_compat_eth, '
                        'submodel_compat_btc, technical_bp, full). Defaults to auto-detection from symbol; '
                        'with XGB/RF/Price submodels the default is the union of their features.')
    p.add_argument('--submodel_max_missing_ratio', type=float, default=None,
                   help='Max fraction of LR features allowed to be missing in Signal '
                        'data (default 0.30). Set higher (e.g. 0.9) for disjoint '
                        'feature set experiments where Signal and LR use different features.')
    p.add_argument('--data_source', type=str, default=None,
                   choices=['train', 'agent'],
                   help="Training data source: 'train' (default) or 'agent' "
                        "(auto-set when --submodel_dirs provided)")
    p.add_argument('--allow_submodel_data_mismatch', type=int, default=0,
                   choices=[0, 1],
                   help='Allow feeding submodels data with a different symbol / '
                        'data_id / feature scaler than they were trained on '
                        '(0/1, default 0 = hard error)')

    # Submodel ablation controls
    p.add_argument('--submodel_mode', type=str, default='real',
                   choices=['real', 'shuffle', 'noise', 'zero'],
                   help="Ablation mode for submodel predictions: 'real' (default), "
                        "'shuffle' (row permutation), 'noise' (N(mean, std) per "
                        "channel), 'zero' (all zeros)")
    p.add_argument('--null_submodel_size', type=int, default=0,
                   help='Without --submodel_dirs: build the submodel_fusion head '
                        'with N all-zero input channels — architecture control '
                        '("same fusion head, no information"). Default 0 = off.')

    # Optuna Feature Tuning
    p.add_argument('--tune_feature_set', type=int, default=None, choices=[0, 1],
                   help='Enable Optuna feature set selection tuning (0/1)')
    p.add_argument('--feature_set_candidates', type=str, default=None,
                   help='Comma-separated feature sets for Optuna tuning candidates')
    p.add_argument('--freeze_hyperparams', type=int, default=None, choices=[0, 1],
                   help='Freeze model hyperparameters in Optuna to isolate feature selection (Mode A)')

    # Training
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--training_profile', type=str, default=None,
                   choices=list(PROFILE_CHOICES))
    p.add_argument('--scheduler_type', type=str, default=None,
                   choices=['plateau', 'cosine', 'onecycle', 'step', 'none'])

    # Signal target
    p.add_argument('--signal_threshold', type=float, default=None)
    p.add_argument('--window_sigma', type=int, default=None)
    p.add_argument('--winsorize_percentile', type=float, default=None,
                   help='Percentile for extreme target winsorization (e.g. 99.0 for 1 percent trimming)')

    # Augmentation
    p.add_argument('--aug_jitter', type=int, default=None, choices=[0, 1],
                   help='Enable jitter augmentation (0/1)')
    p.add_argument('--aug_jitter_std', type=float, default=None)
    p.add_argument('--aug_temporal_dropout', type=int, default=None,
                   choices=[0, 1],
                   help='Enable temporal dropout augmentation (0/1)')
    p.add_argument('--aug_temporal_dropout_rate', type=float, default=None)
    p.add_argument('--aug_subsequence', type=int, default=None, choices=[0, 1],
                   help='Enable subsequence augmentation (0/1)')
    p.add_argument('--aug_subsequence_min_frac', type=float, default=None)

    # Checkpoint / best model
    p.add_argument('--best_model_metric', type=str, default=None,
                   choices=['val', 'val_composite', 'val_signal'])

    # Min epochs before early stopping
    p.add_argument('--min_epochs_before_es', type=int, default=None,
                   help='Minimum epochs before early stopping kicks in')
    p.add_argument('--early_stopping_min_delta', type=float, default=None,
                   help='Minimum delta change for early stopping')

    # LR warmup
    p.add_argument('--lr_warmup_epochs', type=int, default=None,
                   help='Linear LR warmup epochs (0=disabled)')

    # SWA (Stochastic Weight Averaging)
    p.add_argument('--use_swa', type=int, default=None, choices=[0, 1],
                   help='Enable SWA (0/1)')
    p.add_argument('--swa_start_epoch', type=int, default=None,
                   help='First epoch included in SWA average')
    p.add_argument('--swa_every', type=int, default=None,
                   help='Update SWA every N epochs')
    p.add_argument('--swa_lr', type=float, default=None,
                   help='Learning rate for SWA phase')

    p.add_argument('--use_attention', type=int, default=None, choices=[0, 1],
                   help='Enable temporal attention (0/1)')
    p.add_argument('--use_input_norm', type=int, default=None, choices=[0, 1],
                   help='Enable input normalization (0/1)')

    # Loss
    p.add_argument('--loss_type', type=str, default=None,
                   choices=['bce', 'bce_diversity', 'focal', 'asymmetric', 'custom', 'custom_signal'])
    p.add_argument('--focal_gamma', type=float, default=None)
    p.add_argument('--w_buy', type=float, default=None)
    p.add_argument('--w_sell', type=float, default=None)
    p.add_argument('--use_buy_sell_corr_penalty', type=int, default=None, choices=[0, 1])
    p.add_argument('--diversity_weight', type=float, default=None,
                   help='Weight for diversity loss penalty')
    p.add_argument('--loss_diversity_weight', type=float, default=None,
                   help='Alias for --diversity_weight')
    p.add_argument('--loss_diversity_target', type=float, default=None,
                   help='Target diversity value (default 0.30)')
    p.add_argument('--loss_schedule_ramp_epochs', type=int, default=None,
                   help='Ramp up epochs for diversity loss penalty')
    p.add_argument('--label_smoothing', type=float, default=None,
                   help='Alias for --label_smoothing_alpha')
    p.add_argument('--label_smoothing_alpha', type=float, default=None,
                   help='Label smoothing alpha parameter (default 0.0)')

    # CP-173: Economic utility & metric parity penalties
    p.add_argument('--use_signal_utility', type=int, default=None, choices=[0, 1],
                   help='Enable signal economic utility penalty (0/1)')
    p.add_argument('--signal_utility_weight', type=float, default=None,
                   help='Weight for signal utility penalty (default 0.05)')
    p.add_argument('--use_r2_loss', type=int, default=None, choices=[0, 1],
                   help='Enable batch R2 correlation loss (0/1)')
    p.add_argument('--r2_loss_weight', type=float, default=None,
                   help='Weight for batch R2 loss (default 0.1)')
    p.add_argument('--use_ccc_loss', type=int, default=None, choices=[0, 1],
                   help='Enable batch CCC concordance loss (0/1)')
    p.add_argument('--ccc_loss_weight', type=float, default=None,
                   help='Weight for batch CCC loss (default 0.1)')
    p.add_argument('--use_cr_ceiling', type=int, default=None, choices=[0, 1],
                   help='Enable CR ceiling penalty on signal variance (0/1)')
    p.add_argument('--cr_ceiling_weight', type=float, default=None,
                   help='Weight for CR ceiling penalty (default 0.1)')
    p.add_argument('--cr_ceiling', type=float, default=None,
                   help='Ceiling ratio for CR penalty (default 1.5)')

    # CP-183: Regime-robust Group DRO
    p.add_argument('--use_group_dro', type=int, default=None, choices=[0, 1],
                   help='Enable Group Distributionally Robust Optimization (0/1)')
    p.add_argument('--group_dro_eta', type=float, default=None,
                   help='Learning rate for Group DRO weight updates (default 0.01)')
    p.add_argument('--group_dro_groups', type=int, default=None,
                   help='Number of volatility regime groups for DRO (default 3)')

    # Diagnostics & Calibration
    p.add_argument('--gradient_clip', type=float, default=None)
    p.add_argument('--log_gradient_norms', type=int, default=None, choices=[0, 1])
    p.add_argument('--feature_round_decimals', type=int, default=None)
    p.add_argument('--target_round_decimals', type=int, default=None)
    p.add_argument('--calibrate', type=int, default=None, choices=[0, 1],
                   help='Fit probability calibrator on validation set after training (0/1)')

    # Experiment tracking
    p.add_argument('--use_wandb', type=int, default=0, choices=[0, 1],
                   help='Enable Weights & Biases logging (0/1, default 0)')
    p.add_argument('--load_optuna_params', type=str, default=None,
                   help='Path to Optuna best_params.json (overrides HP)')

    return p


def dispatch_args_to_train(args) -> dict:
    """Convert parsed Namespace → kwargs dict for signal_train.train()."""
    kwargs = {'data_path': args.data_path}
    _optional = [
        'lookback', 'horizon', 'feature_set', 'preferred_data_format', 'val_source',
        'arch_type', 'submodel_size', 'hidden_size', 'num_layers', 'dropout', 'fc_dropout', 'lstm_dropout',
        'activation', 'fc_norm', 'weight_init', 'attention_heads', 'output_mode',
        'rnn_type', 'pooling_mode',
        'epochs', 'batch_size', 'seed', 'training_profile',
        'scheduler_type', 'lr_warmup_epochs',
        'signal_threshold', 'window_sigma', 'winsorize_percentile',
        'loss_type', 'focal_gamma', 'w_buy', 'w_sell', 'weight_decay', 'best_model_metric',
        'min_epochs_before_es', 'early_stopping_min_delta',
        'loss_diversity_target', 'loss_schedule_ramp_epochs', 'label_smoothing_alpha',
        'gradient_clip', 'feature_round_decimals', 'target_round_decimals',
        'load_optuna_params',
        'use_swa', 'swa_start_epoch', 'swa_every', 'swa_lr',
    ]
    for key in _optional:
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = val

    if getattr(args, 'log_gradient_norms', None) is not None:
        kwargs['log_gradient_norms'] = bool(args.log_gradient_norms)

    if getattr(args, 'calibrate', None) is not None:
        kwargs['calibrate'] = bool(args.calibrate)

    if getattr(args, 'use_buy_sell_corr_penalty', None) is not None:
        kwargs['use_buy_sell_corr_penalty'] = bool(args.use_buy_sell_corr_penalty)

    if getattr(args, 'fc_layers', None) is not None:
        parsed_fc = _parse_fc_layers(args.fc_layers)
        if parsed_fc is not None:
            kwargs['fc_layers'] = parsed_fc

    div_w = getattr(args, 'diversity_weight', None)
    if div_w is None:
        div_w = getattr(args, 'loss_diversity_weight', None)
    if div_w is not None:
        kwargs['loss_diversity_weight'] = div_w

    ls_a = getattr(args, 'label_smoothing', None)
    if ls_a is None:
        ls_a = getattr(args, 'label_smoothing_alpha', None)
    if ls_a is not None:
        kwargs['label_smoothing_alpha'] = ls_a

    if getattr(args, 'lr', None) is not None:
        kwargs['learning_rate'] = args.lr
    if getattr(args, 'use_attention', None) is not None:
        kwargs['use_attention'] = bool(args.use_attention)
    elif getattr(args, 'no_attention', False):
        kwargs['use_attention'] = False
    if getattr(args, 'use_input_norm', None) is not None:
        kwargs['use_input_norm'] = bool(args.use_input_norm)
    if getattr(args, 'bidirectional', None) is not None:
        kwargs['bidirectional'] = bool(args.bidirectional)
    if getattr(args, 'use_feature_attention', None) is not None:
        kwargs['use_feature_attention'] = bool(args.use_feature_attention)

    # F6 bool flags
    _f6_bool = ['use_conv1d_front', 'use_fc_residual', 'use_rnn_norm',
                'use_learnable_output_scale', 'freeze_rnn']
    for key in _f6_bool:
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = bool(val)

    # CP-173 & CP-183 loss flags
    _cp_flags = [
        ('use_signal_utility', bool), ('signal_utility_weight', float),
        ('use_r2_loss', bool), ('r2_loss_weight', float),
        ('use_ccc_loss', bool), ('ccc_loss_weight', float),
        ('use_cr_ceiling', bool), ('cr_ceiling_weight', float), ('cr_ceiling', float),
        ('use_group_dro', bool), ('group_dro_eta', float), ('group_dro_groups', int),
    ]
    for key, cast in _cp_flags:
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = cast(val)

    # F6 non-bool
    if getattr(args, 'conv1d_kernel', None) is not None:
        kwargs['conv1d_kernel'] = args.conv1d_kernel
    if getattr(args, 'pooling_mode', None) is not None:
        kwargs['pooling_mode'] = args.pooling_mode

    _aug_flags = [
        ('aug_jitter', bool), ('aug_jitter_std', float),
        ('aug_temporal_dropout', bool),
        ('aug_temporal_dropout_rate', float),
        ('aug_subsequence', bool),
        ('aug_subsequence_min_frac', float),
    ]
    for key, cast in _aug_flags:
        val = getattr(args, key, None)
        if val is not None:
            kwargs[key] = cast(val)

    # Submodel integration
    kwargs['data_source_auto_switched'] = False
    if getattr(args, 'submodel_dirs', None):
        kwargs['submodel_dirs'] = args.submodel_dirs
        # Auto-set data_source to 'agent' (agent split (agent.csv)) unless explicitly overridden
        if getattr(args, 'data_source', None) is None:
            kwargs['data_source'] = 'agent'
            kwargs['data_source_auto_switched'] = True
    if getattr(args, 'data_source', None) is not None:
        kwargs['data_source'] = args.data_source
    if getattr(args, 'submodel_max_missing_ratio', None) is not None:
        kwargs['submodel_max_missing_ratio'] = args.submodel_max_missing_ratio
    if getattr(args, 'use_submodel_norm', None) is not None:
        kwargs['use_submodel_norm'] = bool(args.use_submodel_norm)
    if getattr(args, 'submodel_norm_type', None) is not None:
        kwargs['submodel_norm_type'] = args.submodel_norm_type
    if getattr(args, 'submodel_feature_set', None) is not None:
        kwargs['submodel_feature_set'] = args.submodel_feature_set

    # Feature-set coverage guards + submodel ablation controls (always set —
    # these flags have concrete defaults rather than None)
    kwargs['allow_partial_feature_set'] = bool(
        getattr(args, 'allow_partial_feature_set', 0))
    kwargs['min_feature_coverage'] = float(
        getattr(args, 'min_feature_coverage', 1.0))
    kwargs['allow_submodel_data_mismatch'] = bool(
        getattr(args, 'allow_submodel_data_mismatch', 0))
    kwargs['submodel_mode'] = getattr(args, 'submodel_mode', 'real') or 'real'
    kwargs['null_submodel_size'] = int(getattr(args, 'null_submodel_size', 0) or 0)

    # Optuna Feature Tuning
    if getattr(args, 'tune_feature_set', None) is not None:
        kwargs['tune_feature_set'] = bool(args.tune_feature_set)
    if getattr(args, 'feature_set_candidates', None) is not None:
        kwargs['feature_set_candidates'] = args.feature_set_candidates
    if getattr(args, 'freeze_hyperparams', None) is not None:
        kwargs['freeze_hyperparams'] = bool(args.freeze_hyperparams)

    return kwargs
