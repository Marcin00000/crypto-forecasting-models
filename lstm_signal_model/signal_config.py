"""
Configuration for LSTM Signal Model.

Central configuration — all training, model, loss, signal target,
and diagnostic parameters. Analogous to lstm_config.py but tailored
for direct buy/sell signal prediction.
"""
import os
import json
from pathlib import Path


class SignalConfig:
    """Master configuration for LSTM Signal prediction model."""

    # ===================================================================
    # ID PREFIXES
    # ===================================================================
    MODEL_ID_PREFIX = 'SIG'
    TEST_ID_PREFIX = 'SIGT'
    OPTUNA_ID_PREFIX = 'SIGO'

    # ===================================================================
    # DATA
    # ===================================================================
    DEFAULT_LOOKBACK = 48
    DEFAULT_HORIZON = 1
    DEFAULT_FEATURE_SET = 'technical_bp'

    PREFERRED_DATA_FORMAT = 'csv'

    # Test results save format:
    # 'parquet' — compressed parquet format (DEFAULT, fast I/O, compact disk usage)
    # 'csv'     — uncompressed CSV format (alternative)
    DEFAULT_RESULTS_FORMAT = 'parquet'
    DEFAULT_RESULTS_COMPRESSION = 'snappy'  # 'snappy' | 'zstd' | 'gzip' | 'none'

    # 'split' — train.csv split 80/20 chronologically → train/val
    # 'test'  — test.csv used as val
    VAL_SOURCE = 'split'

    # ===================================================================
    # MODEL ARCHITECTURE
    # ===================================================================
    ARCH_TYPE = 'legacy'  # 'legacy' | 'dual_head' | 'submodel_fusion'
    HIDDEN_SIZE = 32
    NUM_LAYERS = 1
    DROPOUT = 0.35

    BIDIRECTIONAL = False
    USE_ATTENTION = True
    FC_LAYERS = [32]
    ACTIVATION = 'gelu'
    FC_NORM = 'layer'
    RNN_TYPE = 'lstm'
    ATTENTION_HEADS = 1

    LSTM_DROPOUT = None
    FC_DROPOUT = None

    USE_INPUT_NORM = True
    USE_SUBMODEL_NORM = True
    SUBMODEL_NORM_TYPE = 'auto'
    USE_FEATURE_ATTENTION = False

    WEIGHT_INIT = 'default'

    # F6 architecture flags (Faza C — parity with LR)
    USE_CONV1D_FRONT = False
    CONV1D_KERNEL = 5
    POOLING_MODE = 'last'       # 'last' | 'mean' | 'max' (when attention off)
    USE_FC_RESIDUAL = False
    USE_RNN_NORM = False
    USE_LEARNABLE_OUTPUT_SCALE = False
    FREEZE_RNN = False

    # Output mode: 'dual' — (p_buy, p_sell), 'single' — placeholder
    OUTPUT_MODE = 'dual'

    # ===================================================================
    # TRAINING
    # ===================================================================
    TRAINING_PROFILE = 'baseline'
    EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 25
    MIN_EPOCHS_BEFORE_ES = 0       # blokuje early stopping przez N pierwszych epok
    EARLY_STOPPING_MIN_DELTA = 0.0 # minimalna poprawa score przed resetem patience
    GRADIENT_CLIP = 1.0
    SEED = 42

    SCHEDULER_TYPE = 'plateau'
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    LR_WARMUP_EPOCHS = 0
    SCHEDULER_PARAMS = {
        'patience': 5,      # plateau
        'factor': 0.5,      # plateau, step
        'T_0': 10,          # cosine
        'T_mult': 2,        # cosine
        'step_size': 20,    # step
        'gamma': 0.5,       # step
        'max_lr': 0.01,     # onecycle
    }

    NUM_WORKERS = 0
    USE_PIN_MEMORY = True

    # ===================================================================
    # LOSS
    # ===================================================================
    # 'bce', 'bce_diversity', 'focal', 'asymmetric', 'custom_signal'
    LOSS_TYPE = 'bce'
    FOCAL_GAMMA = 2.0
    ASYMMETRIC_W_BUY = 1.0
    ASYMMETRIC_W_SELL = 1.0
    USE_BUY_SELL_CORR_PENALTY = False

    # CP-173: Economic utility & metric parity penalties (opt-in)
    USE_SIGNAL_UTILITY = False
    SIGNAL_UTILITY_WEIGHT = 0.05
    USE_R2_LOSS = False
    R2_LOSS_WEIGHT = 0.1
    USE_CCC_LOSS = False
    CCC_LOSS_WEIGHT = 0.1
    USE_CR_CEILING = False
    CR_CEILING_WEIGHT = 0.1
    CR_CEILING = 1.5

    # CP-183: Regime-robust optimization (opt-in)
    USE_GROUP_DRO = False
    GROUP_DRO_ETA = 0.01
    GROUP_DRO_GROUPS = 3

    # Scheduled loss: ramp-up weight for auxiliary terms over N epochs.
    # 0 = disabled (ScheduledLossWrapper is no-op).
    LOSS_SCHEDULE_RAMP_EPOCHS = 0

    # ===================================================================
    # OPTUNA
    # ===================================================================
    OPTUNA_SCORE_SOURCE = 'val'     # 'val' | 'train'
    OPTUNA_SCORE_TYPE = 'val_loss'   # 'val_loss' | 'composite' | 'signal'
    OPTUNA_N_BEST_RESULTS = 3
    TUNE_PARAMS: dict = {}

    # ===================================================================
    # SIGNAL TARGET & EVALUATION
    # ===================================================================
    SIGNAL_THRESHOLD = 0.001
    WINDOW_SIGMA = 20
    MIN_SIGNAL = 20.0
    SIM_START_CASH = 100.0

    WINSORIZE_PERCENTILE = None  # None or float (e.g. 99.0 for 1% extreme data trimming)
    THRESHOLD_MODE = 'absolute'       # 'absolute' | 'percentile' | 'zscore'
    PERCENTILE_CUTOFF = 5.0           # Top X% (mode='percentile')
    ZSCORE_CUTOFF = 2.0               # Mean + Z*std (mode='zscore')
    POSITION_MODE = 'sticky'          # 'sticky' | 'reset'
    ORACLE_THRESHOLD_MODE = None      # None = match THRESHOLD_MODE
    RANK_FALLBACK_MODE = 'none'       # 'none' | 'gradient' | 'dynamic_threshold'

    # Canonical oracle — stałe referencyjne do symulacji porównawczej.
    # Każdy model trenowany z innymi (window_sigma, signal_threshold) dostaje
    # innego oracle treningowego — dlatego do sim benchmarku używamy stałych
    # kanonicznych, żeby wyniki sim_final_oracle były porównywalne między
    # modelami. Wartości te NIE wpływają na trening ani metryki MAE/corr.
    CANONICAL_WINDOW_SIGMA = 100
    CANONICAL_SIGNAL_THRESHOLD = 0.001

    # D2: window_sigma values for --multi_window sweep in signal_test.py
    MULTI_WINDOW_SIGMAS = [10, 20, 50, 100]

    # ===================================================================
    # AUGMENTATION
    # ===================================================================
    AUG_JITTER = False
    AUG_JITTER_STD = 0.01
    AUG_TEMPORAL_DROPOUT = False
    AUG_TEMPORAL_DROPOUT_RATE = 0.05
    AUG_SUBSEQUENCE = False
    AUG_SUBSEQUENCE_OFFSET = 2

    # ===================================================================
    # DIAGNOSTICS
    # ===================================================================
    # Per-epoch signal collapse monitoring (diversity, compression ratio).
    LOG_SIGNAL_DIAGNOSTICS = True

    # Per-group gradient norm logging (rnn/attention/fc).
    LOG_GRADIENT_NORMS = False

    # ===================================================================
    # CHECKPOINT / BEST MODEL
    # ===================================================================
    # 'val'           — lowest val_loss
    # 'val_composite' — val_loss + DIVERSITY_WEIGHT * (1 - signal_diversity)
    # 'val_signal'    — -score(corr_avg, pct_strong, sim_capture_log)
    BEST_MODEL_METRIC = 'val'
    BEST_MODEL_DIVERSITY_WEIGHT = 0.01
    # Earliest epoch at which val_signal checkpoint activates (epochs 1..N-1
    # fall back to val_loss to avoid thrashing before model stabilises).
    VAL_SIGNAL_MIN_EPOCH = 10

    # ===================================================================
    # SWA (Stochastic Weight Averaging)
    # ===================================================================
    USE_SWA = False
    SWA_START_EPOCH = 30
    SWA_EVERY = 1

    # ===================================================================
    # PATHS
    # ===================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'signal_models')
    DATA_DIR = os.path.join(BASE_DIR, 'signal_data')

    # ===================================================================
    # ALLOWED VALUES (for validation)
    # ===================================================================
    _ALLOWED_LOSS_TYPES = {'bce', 'bce_diversity', 'focal', 'asymmetric', 'custom', 'custom_signal'}
    _ALLOWED_ARCH_TYPES = {'legacy', 'dual_head', 'submodel_fusion'}
    _ALLOWED_OUTPUT_MODES = {'dual', 'single'}
    _ALLOWED_RNN_TYPES = {'lstm', 'gru'}
    _ALLOWED_ACTIVATIONS = {'relu', 'gelu', 'leaky_relu', 'silu', 'swish', 'mish'}
    _ALLOWED_FC_NORMS = {'batch', 'layer', 'none'}
    _ALLOWED_SCHEDULER_TYPES = {'plateau', 'cosine', 'onecycle', 'step', 'none'}
    _ALLOWED_BEST_METRICS = {'val', 'val_composite', 'val_signal'}
    _ALLOWED_POOLING_MODES = {'last', 'mean', 'max'}
    _ALLOWED_RESULTS_FORMATS = {'parquet', 'csv'}
    _ALLOWED_RESULTS_COMPRESSIONS = {'snappy', 'zstd', 'gzip', 'none', None}

    @classmethod
    def ensure_dirs(cls):
        """Create output directories if they do not exist."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)

    @classmethod
    def validate(cls, resolved: dict | None = None):
        """
        Validate configuration values. Works on a resolved dict
        (from train_setup.apply_config_defaults) or directly on class attrs.

        Raises ValueError with a descriptive message on failure.
        """
        def _get(key, default=None):
            if resolved is not None:
                return resolved.get(key, default)
            return getattr(cls, key.upper(), default)

        errors = []

        loss_type = _get('loss_type', 'bce')
        if loss_type not in cls._ALLOWED_LOSS_TYPES:
            errors.append(
                f"loss_type='{loss_type}' not in {cls._ALLOWED_LOSS_TYPES}")

        output_mode = _get('output_mode', 'dual')
        if output_mode not in cls._ALLOWED_OUTPUT_MODES:
            errors.append(
                f"output_mode='{output_mode}' not in {cls._ALLOWED_OUTPUT_MODES}")

        rnn_type = _get('rnn_type', 'lstm')
        if rnn_type not in cls._ALLOWED_RNN_TYPES:
            errors.append(
                f"rnn_type='{rnn_type}' not in {cls._ALLOWED_RNN_TYPES}")

        activation = _get('activation', 'gelu')
        if activation not in cls._ALLOWED_ACTIVATIONS:
            errors.append(
                f"activation='{activation}' not in {cls._ALLOWED_ACTIVATIONS}")

        fc_norm = _get('fc_norm', 'layer')
        if fc_norm not in cls._ALLOWED_FC_NORMS:
            errors.append(
                f"fc_norm='{fc_norm}' not in {cls._ALLOWED_FC_NORMS}")

        scheduler_type = _get('scheduler_type', 'plateau')
        if scheduler_type not in cls._ALLOWED_SCHEDULER_TYPES:
            errors.append(
                f"scheduler_type='{scheduler_type}' "
                f"not in {cls._ALLOWED_SCHEDULER_TYPES}")

        best_metric = _get('best_model_metric', 'val')
        if best_metric not in cls._ALLOWED_BEST_METRICS:
            errors.append(
                f"best_model_metric='{best_metric}' "
                f"not in {cls._ALLOWED_BEST_METRICS}")

        pooling_mode = _get('pooling_mode', 'last')
        if pooling_mode not in cls._ALLOWED_POOLING_MODES:
            errors.append(
                f"pooling_mode='{pooling_mode}' "
                f"not in {cls._ALLOWED_POOLING_MODES}")

        hidden_size = _get('hidden_size', 32)
        use_attention = _get('use_attention', True)
        attention_heads = _get('attention_heads', 1)
        if use_attention and hidden_size % attention_heads != 0:
            errors.append(
                f"attention_heads={attention_heads} must divide "
                f"hidden_size={hidden_size}")

        dropout = _get('dropout', 0.35)
        if not (0.0 <= dropout <= 1.0):
            errors.append(f"dropout={dropout} must be in [0, 1]")

        min_epochs_before_es = _get('min_epochs_before_es', 0)
        if min_epochs_before_es is None or min_epochs_before_es < 0:
            errors.append(
                f"min_epochs_before_es={min_epochs_before_es} must be >= 0")
        early_stopping_min_delta = _get('early_stopping_min_delta', 0.0)
        if early_stopping_min_delta is None or early_stopping_min_delta < 0.0:
            errors.append(
                f"early_stopping_min_delta={early_stopping_min_delta} must be >= 0")

        epochs = _get('epochs', 100)
        if epochs is None or epochs <= 0:
            errors.append(f"epochs={epochs} must be > 0")

        batch_size = _get('batch_size', 512)
        if batch_size is None or batch_size <= 0:
            errors.append(f"batch_size={batch_size} must be > 0")

        lr = _get('learning_rate', 3e-4)
        if lr is None or lr <= 0:
            errors.append(f"learning_rate={lr} must be > 0")

        signal_threshold = _get('signal_threshold', 0.001)
        if signal_threshold is None or signal_threshold <= 0:
            errors.append(
                f"signal_threshold={signal_threshold} must be > 0")

        window_sigma = _get('window_sigma', 20)
        if window_sigma is None or window_sigma < 2:
            errors.append(f"window_sigma={window_sigma} must be >= 2")

        min_signal = _get('min_signal', 60.0)
        if not (0.0 <= min_signal <= 100.0):
            errors.append(f"min_signal={min_signal} must be in [0, 100]")

        lookback = _get('lookback', 48)
        if lookback is None or lookback <= 0:
            errors.append(f"lookback={lookback} must be > 0")

        horizon = _get('horizon', 1)
        if horizon is None or horizon < 1:
            errors.append(f"horizon={horizon} must be >= 1")

        fc_layers = _get('fc_layers', [32])
        if not fc_layers or not isinstance(fc_layers, list):
            errors.append("fc_layers must be a non-empty list")
        elif not all(isinstance(x, int) and x > 0 for x in fc_layers):
            errors.append("fc_layers entries must be positive integers")

        results_format = _get('results_format', cls.DEFAULT_RESULTS_FORMAT)
        if results_format not in cls._ALLOWED_RESULTS_FORMATS:
            errors.append(
                f"results_format='{results_format}' not in {cls._ALLOWED_RESULTS_FORMATS}")

        results_compression = _get('results_compression', cls.DEFAULT_RESULTS_COMPRESSION)
        if results_compression not in cls._ALLOWED_RESULTS_COMPRESSIONS:
            errors.append(
                f"results_compression='{results_compression}' not in {cls._ALLOWED_RESULTS_COMPRESSIONS}")

        if errors:
            raise ValueError(
                "SignalConfig validation failed:\n  - "
                + "\n  - ".join(errors))

    @staticmethod
    def load_optuna_params(json_path: str) -> tuple[dict, dict]:
        """
        Load Optuna best_params.json and reconstruct fc_layers.

        Supports both formats produced by ``tuning_utils.save_params``:
          - nested ``_metadata`` (canonical, as in SIGO###/best_params.json)
          - flat metadata keys (legacy / unit tests)

        Returns:
            (params, metadata) — params dict ready to override resolved,
            metadata contains optuna_id, study info etc.
        """
        p = Path(json_path)
        if not p.exists():
            repo_root = Path(__file__).resolve().parents[1]
            signal_root = Path(__file__).resolve().parent
            candidates = [
                repo_root / json_path,
                signal_root / json_path,
                signal_root / "signal_data" / json_path,
            ]
            for c in candidates:
                if c.exists():
                    p = c
                    break

        with open(p, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        metadata = raw.pop('_metadata', None) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        meta_keys = {'optuna_id', 'study_name', 'trial_number',
                     'score', 'timestamp', 'loss_config',
                     'training_profile', 'data_id', 'symbol',
                     'feature_set', 'direction', 'score_type',
                     'baseline_params', 'best_value', 'rank',
                     'n_completed', 'n_total'}
        params = {}
        for k, v in raw.items():
            if k in meta_keys:
                metadata.setdefault(k, v)
            else:
                params[k] = v

        n_fc = params.pop('n_fc_layers', None)
        if n_fc is not None and 'fc_layers' not in params:
            fc_layers = []
            for i in range(int(n_fc)):
                units = params.pop(f'fc_units_{i}', 32)
                fc_layers.append(int(units))
            params['fc_layers'] = fc_layers
        else:
            for key in list(params.keys()):
                if key.startswith('fc_units_'):
                    params.pop(key)

        _skip_bool = {
            'aug_jitter_std', 'aug_temporal_dropout_rate',
            'aug_subsequence_offset', 'attention_heads',
        }
        for k in list(params.keys()):
            if isinstance(params[k], int) and k.startswith(('use_', 'aug_')):
                if k not in _skip_bool:
                    params[k] = bool(params[k])

        return params, metadata
