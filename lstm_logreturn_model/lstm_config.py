"""
Configuration for LSTM Log-Return model.

Central configuration — all training, model, loss, and data parameters.
Analogous to xgb_config.py but tailored for LSTM.
python lstm_logreturn_model/lstm_hyperparameter_tuning.py --data_path data_preparation/prepared_data/B104_prep_btc_universal_full_1m --n_trials 7 --lookback 48 --horizon 12
"""
import os


class LSTMConfig:
    """Master configuration for LSTM Log-Return prediction model."""

    # ===================================================================
    # DATA
    # ===================================================================
    DEFAULT_LOOKBACK = 48
    DEFAULT_HORIZON = 12
    DEFAULT_FEATURE_SET = 'technical_bp'

    # Preferred data format when prepared directory contains both CSV and NPY.
    # 'csv' — load train.csv/test.csv (universal format from data_preparation)
    # 'npy' — load X_train.npy / y_train.npy (3D sequences)
    PREFERRED_DATA_FORMAT = 'csv'

    # Test results save format:
    # 'parquet' — compressed parquet format (DEFAULT, fast I/O, compact disk usage)
    # 'csv'     — uncompressed CSV format (alternative)
    DEFAULT_RESULTS_FORMAT = 'parquet'
    DEFAULT_RESULTS_COMPRESSION = 'snappy'  # 'snappy' | 'zstd' | 'gzip' | 'none'

    # Validation data source:
    # 'split' — train.csv split 80/20 chronologically → train/val (DEFAULT,
    #            avoids selection bias: test.csv stays untouched for final
    #            evaluation via lstm_test.py).
    # 'test'  — test.csv used as val (faster iteration, but introduces
    #            selection bias because Optuna/early-stopping optimize on
    #            the same set that lstm_test.py reports metrics on; in this
    #            mode lstm_test.py evaluates on sym.csv instead).
    # 'agent' — agent split (agent.csv) used as val (real fresh OOS validation).
    VAL_SOURCE = 'split'

    # Training data source:
    # 'train' — load train.csv (DEFAULT, 172k samples)
    # 'agent' — load agent split (agent.csv) (fresh OOS sequence, 258k samples)
    DATA_SOURCE = 'train'

    # ===================================================================
    # MODEL ARCHITECTURE
    # ===================================================================
    # 'legacy' keeps the historical LSTM/GRU -> attention/last -> FC -> Linear
    # state_dict intact. New variants must be opt-in.
    ARCH_TYPE = 'legacy'  # 'legacy' | 'dual_head'
    HIDDEN_SIZE = 32
    NUM_LAYERS = 1
    DROPOUT = 0.35

    # Bidirectional LSTM — sees lookback window in both directions.
    # Not future leakage: bidirectional applies within the historical window.
    BIDIRECTIONAL = False

    # Self-attention on LSTM output — weighted aggregation of all timesteps
    # instead of taking only the last hidden state.
    USE_ATTENTION = True

    # Fully connected layers after LSTM (list of hidden sizes).
    FC_LAYERS = [32]

    # Activation: 'relu' | 'gelu' | 'leaky_relu'
    ACTIVATION = 'gelu'

    # Normalization inside FC head: 'batch' | 'layer' | 'none'
    # 'batch' — BatchNorm1d (legacy; may suppress amplitude gradient when
    #           predictions have very small variance — flattening collapse)
    # 'layer' — LayerNorm (default; per-sample, stable for small-variance
    #           regression outputs)
    # 'none'  — no normalization (research: test if norm itself hurts)
    FC_NORM = 'layer'

    # RNN cell type: 'lstm' | 'gru'
    # GRU is fully supported (FAZA 4.M6). LSTM remains the default because
    # historical experiments (LR075/LR179) were tuned against it.
    RNN_TYPE = 'lstm'

    # F4.M3: multi-head temporal attention. `attention_heads=1` is functionally
    # close to the legacy single-head attention; values > 1 split the RNN
    # output dimension into independent heads (`hidden_size` must therefore be
    # divisible by `attention_heads`; bidirectional doubles `hidden_size` to
    # the value that needs to be divisible).
    ATTENTION_HEADS = 1

    # F4.M4: split dropout. `None` falls back to `DROPOUT` for that location,
    # so a single LSTMConfig.DROPOUT value still works as a baseline. Set
    # `LSTM_DROPOUT` (between stacked LSTM/GRU layers; effective only when
    # NUM_LAYERS > 1) and `FC_DROPOUT` (inside the FC head) independently
    # when you want different regularisation pressures on each block.
    LSTM_DROPOUT = None
    FC_DROPOUT = None

    # F1.I2: LayerNorm on the LSTM input. Per-feature normalisation across
    # the lookback window — stabilises gradients when raw features have very
    # different scales (volume vs price-derived). Cheap (O(B·T·F)).
    USE_INPUT_NORM = True

    # F1.I3: Weight initialisation strategy for Linear layers in the FC head.
    # 'default'  — PyTorch defaults (Kaiming uniform for Linear, sensible for LSTM)
    # 'kaiming'  — Kaiming normal (He) — best with ReLU/GELU activations
    # 'xavier'   — Xavier/Glorot normal — best with tanh/sigmoid heads
    # LSTM/GRU weights are left alone (PyTorch defaults are well-tuned).
    WEIGHT_INIT = 'default'

    # ===================================================================
    # TRAINING
    # ===================================================================
    # Named recipe overlays. 'baseline' leaves all values unchanged; other
    # profiles encode proven rank/sim recipes and are opt-in via CLI/kwargs.
    TRAINING_PROFILE = 'baseline'  # 'baseline' | 'rank_focused' | 'sim_focused' | 'custom_sim'
    EPOCHS = 100
    BATCH_SIZE = 512
    LEARNING_RATE = 3.43e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 25
    GRADIENT_CLIP = 1.0  # F1.I1: max grad norm. 0 (or negative) = disabled.
    SEED = 42

    # F1.I4: post-hoc rounding of predictions/targets before metric computation.
    # Used to test whether decimal noise affects R² / directional accuracy.
    # 0 = disabled (no rounding). Otherwise number of decimal places.
    PREDICTION_PRECISION = 0
    TARGET_PRECISION = 0

    # F1.I4b: pre-training rounding of features and targets (before gradients).
    # Applied after feature scaling / target winsorization+scaling.
    # 0 = disabled. Otherwise number of decimal places.
    # Typical ranges: features (normalized ~[-3,+3]): 3–5; target (log-return ~0.001): 4–6.
    FEATURE_ROUND_DECIMALS = 0
    TARGET_ROUND_DECIMALS = 0

    # F1.I5: log gradient norms per layer-group during training. Adds an
    # extra mean/max scan over `param.grad` for every parameter — cheap.
    # Values are aggregated into `diagnostics_history[epoch]['grad_norms']`.
    LOG_GRADIENT_NORMS = False

    # Which metric to monitor for early stopping and best model checkpoint:
    # 'val'           = standard — best model = lowest val_loss (baseline)
    # 'train'         = research — best model = lowest train_loss
    # 'val_composite' = val_loss + BEST_MODEL_CR_WEIGHT * |1 - val_CR|.
    #                   Uses the per-epoch validation compression ratio from
    #                   EpochDiagnostics; it is a cheap signal-aware proxy when
    #                   split validation has no close series for full rank/sim.
    # 'val_signal'    = per-epoch signal score on val (needs val_close).
    BEST_MODEL_METRIC = 'val'       # 'val' | 'train' | 'val_composite' | 'val_signal'
    BEST_MODEL_CR_WEIGHT = 0.01
    VAL_SIGNAL_MIN_EPOCH = 5

    # Minimum epochs before early stopping can trigger (0 = disabled).
    MIN_EPOCHS_BEFORE_ES = 0
    # Improvement must exceed this delta to reset patience (0 = any strict improvement).
    EARLY_STOPPING_MIN_DELTA = 0.0

    # Stochastic Weight Averaging (default off).
    USE_SWA = False
    SWA_START_EPOCH = 30
    SWA_EVERY = 1

    # DataLoader workers: 0 = single-thread (safe for Windows + ROCm);
    # 2-4 = faster on Linux with ROCm 5.x+ / CUDA.
    NUM_WORKERS = 0

    # Pinned memory for GPU transfer. Disable if ROCm < 5.0 crashes with OOM.
    USE_PIN_MEMORY = True

    # ===================================================================
    # TARGET SCALING
    # ===================================================================
    SCALE_TARGET = False
    TARGET_SCALER_TYPE = 'robust'   # 'standard' | 'robust'

    # ===================================================================
    # TARGET WINSORISATION
    # ===================================================================
    WINSORIZE_TARGET = False
    WINSORIZE_PERCENTILE = 99.0

    # ===================================================================
    # TARGET VOLATILITY NORMALISATION (CP-175)
    # ===================================================================
    # 'raw' — unnormalised log-returns y = log(close_{t+H} / close_t)
    # 'vol_norm' — target divided by rolling volatility: y_norm = y / sigma
    TARGET_MODE = 'raw'  # 'raw' | 'vol_norm'

    # Auxiliary volatility head (CP-028 / CP-175)
    USE_AUX_VOLATILITY_HEAD = False
    AUX_VOLATILITY_WEIGHT = 0.1

    # CP-183: Regime-robust Group DRO
    USE_GROUP_DRO = False
    GROUP_DRO_ETA = 0.01
    GROUP_DRO_GROUPS = 3

    # ===================================================================
    # LOSS FUNCTION
    # ===================================================================
    # Native PyTorch losses: 'mse', 'mae', 'huber', 'smooth_l1'
    # Custom losses:         'pseudo_huber', 'quantile', 'custom'
    LOSS_TYPE = 'custom'
    HUBER_DELTA = 1.0

    # Quantile (pinball) loss — q in (0, 1). q=0.5 ≡ MAE / 2. q>0.5 penalises
    # under-prediction (biases model to predict higher), q<0.5 the opposite.
    # Used only when LOSS_TYPE='quantile'.
    QUANTILE_LOSS_Q = 0.5

    # --- Custom loss components (active when LOSS_TYPE='custom') ---
    USE_SAMPLE_WEIGHTS = False
    SAMPLE_WEIGHT_MODE = 'percentile'   # 'abs' | 'abs_clip' | 'rank' | 'percentile'
    SAMPLE_WEIGHT_PERCENTILE = 70.0
    SAMPLE_WEIGHT_LOW_W = 0.1

    USE_MISDIR_PENALTY = False
    MISDIR_PENALTY = 2.0
    MISDIR_MIN_QUANTILE = 0.7

    USE_DEAD_ZONE = False
    DEAD_ZONE_WEIGHT = 0.1

    # Anti-collapse: Amplified Zone (F2.3)
    # Counterpart to dead_zone: |y_raw| >= AMPLIFIED_ZONE_THRESHOLD gets its
    # loss multiplied by AMPLIFIED_ZONE_WEIGHT (> 1.0). Complementary to
    # dead_zone — together they push gradients away from small moves and
    # toward large ones. Operates in RAW log-return space (un-scales y_true
    # via registered buffers, same as dead_zone).
    USE_AMPLIFIED_ZONE = True
    AMPLIFIED_ZONE_THRESHOLD = 0.0015
    AMPLIFIED_ZONE_WEIGHT = 2.0

    # Anti-collapse: Directional Loss (F2.2)
    # Additive batch-level term that penalizes wrong-sign predictions.
    # Uses relu(-sign(y_true) * y_pred) normalised by mean(|y_true|) → bounded
    # roughly in [0, weight] when pred magnitude matches true. Zero when pred
    # collapses to 0 (correlation/variance penalties handle that case).
    USE_DIRECTIONAL_LOSS = False
    DIRECTIONAL_LOSS_WEIGHT = 0.5

    # Anti-collapse: Correlation Penalty (F2.1a)
    # Adds weight * (1 - Pearson(pred, true)) at batch level. Scale-invariant,
    # diverges from 0 when predictions collapse to a constant. Additive — does
    # NOT replace MAE/Huber pressure, only complements it.
    USE_CORRELATION_PENALTY = False
    CORRELATION_PENALTY_WEIGHT = 0.1

    # Anti-collapse: Variance Penalty (F2.1b)
    # Adds weight * relu(1 - var(pred)/var(true)) at batch level. Normalized
    # to [0, weight]. Directly penalizes prediction variance below target
    # variance — complementary to Correlation Penalty which only targets
    # direction. Useful as a stronger anti-flattening signal.
    USE_VARIANCE_PENALTY = False
    VARIANCE_PENALTY_WEIGHT = 0.1

    # Anti-collapse: Log-variance Penalty (F2.V1-1)
    # Adds weight * relu(log_var(true) - log_var(pred)) at batch level.
    # Unlike the linear variance penalty, the log form keeps a non-zero
    # gradient at the collapse point (var(pred) ≈ 0), so the optimiser can
    # still escape the degenerate solution. Complementary to USE_VARIANCE_PENALTY.
    USE_LOG_VARIANCE_PENALTY = False
    LOG_VARIANCE_PENALTY_WEIGHT = 0.01

    # Anti-collapse: R² Penalty (F2.V1-3)
    # Adds weight * relu(1 - R²(pred, true)). R² near 1 → ~0 penalty; R² ≤ 0
    # (constant predictions, or worse than the mean) → ~weight. Combines
    # naturally with USE_CORRELATION_PENALTY (which is bias-insensitive).
    USE_R2_PENALTY = False
    R2_PENALTY_WEIGHT = 0.01

    # Anti-collapse: CCC Penalty (F2.V1-4)
    # Adds weight * (1 - CCC(pred, true)). CCC penalises low correlation AND
    # scale/mean mismatch, so it is strictly stronger than pure Pearson when
    # the model has the freedom to scale predictions (e.g. unscaled targets).
    USE_CCC_PENALTY = False
    CCC_PENALTY_WEIGHT = 0.01

    # ===================================================================
    # SCHEDULED LOSS SWITCHING (FAZA 5)
    # ===================================================================
    # Train with a simple base loss (e.g. MAE) during the early "warm-up"
    # phase to obtain a non-degenerate starting point, then switch to the
    # full loss (e.g. LSTMCustomLoss or SignalLoss) for the rest of training.
    # Mitigates the prediction-collapse failure mode where the custom loss
    # immediately drives variance to ~0 before the model learns the signal.
    USE_SCHEDULED_LOSS = False
    SCHEDULED_LOSS_WARMUP_EPOCHS = 30
    SCHEDULED_LOSS_BASE = 'mae'
    SCHEDULED_LOSS_FULL = 'custom'

    # ===================================================================
    # SIGNAL LOSS V2 (LOSS_TYPE='signal')
    # ===================================================================
    # FAZA 3 — minimalist MAE-based loss with optional R²/CCC/log-var penalties.
    # Independent of the LSTMCustomLoss flags above so the two variants can be
    # tuned separately without flag pollution.
    SIGNAL_USE_R2 = True
    SIGNAL_R2_WEIGHT = 0.01
    SIGNAL_USE_CCC = True
    SIGNAL_CCC_WEIGHT = 0.01
    SIGNAL_USE_LOG_VAR = False
    SIGNAL_LOG_VAR_WEIGHT = 0.005

    # Anti-collapse std-match: penalises |1 - std(pred)/std(true)|, steering the
    # compression ratio toward ~1 (punishes BOTH collapse and explosion, unlike
    # the one-sided variance/log-variance penalties). Recommended primary
    # anti-flattening term for signal models.
    SIGNAL_USE_STD_MATCH = True
    SIGNAL_STD_MATCH_WEIGHT = 0.05
    # Threshold-gated directional penalty: wrong-sign predictions on samples
    # with |y_true| >= SIGNAL_THRESHOLD_VALUE are penalised (small noise moves
    # are ignored). Focuses learning on economically meaningful direction.
    SIGNAL_USE_DIR_THRESHOLD = False
    SIGNAL_DIR_THRESHOLD_WEIGHT = 0.1

    # Direction calibration term for economically meaningful moves. Interprets
    # y_pred / SIGNAL_THRESHOLD_VALUE as a logit for positive direction and
    # applies BCE only where |y_true| crosses the threshold. Default OFF so
    # historical SignalLoss behaviour is unchanged.
    SIGNAL_USE_DIRECTION_BCE = False
    SIGNAL_DIRECTION_BCE_WEIGHT = 0.01

    # F7.A — EMA smoothing for batch-level SignalLoss penalties (reduces
    # per-batch noise when batch_size is small). Default OFF = legacy.
    SIGNAL_EMA_PENALTIES = False
    SIGNAL_EMA_ALPHA = 0.1          # blend: alpha*raw + (1-alpha)*ema_buffer

    # F7.D — Label smoothing for dead-zone targets (|y_raw| < threshold).
    SIGNAL_LABEL_SMOOTHING = False
    SIGNAL_LABEL_SMOOTHING_ALPHA = 0.5  # y_smooth = alpha*y + (1-alpha)*0

    # Faza 1b — utility loss aligned with trading signal (default OFF).
    SIGNAL_USE_UTILITY = False
    SIGNAL_UTILITY_WEIGHT = 0.05
    # Soft threshold masks for dir_threshold / direction_bce (default OFF).
    SIGNAL_SOFT_THRESHOLD = False

    # Optional CR ceiling penalty in loss (default OFF).
    USE_CR_CEILING = False
    CR_CEILING = 1.5
    CR_CEILING_WEIGHT = 0.1

    # ===================================================================
    # DATA AUGMENTATION (F7.B — training only, default OFF)
    # ===================================================================
    AUG_JITTER = False
    AUG_JITTER_STD = 0.01           # noise scale relative to per-feature std
    AUG_TEMPORAL_DROPOUT = False
    AUG_TEMPORAL_DROPOUT_RATE = 0.05
    AUG_SUBSEQUENCE = False         # circular time shift within lookback
    AUG_SUBSEQUENCE_OFFSET = 2      # max candles to shift (+/-)

    # ===================================================================
    # ARCHITECTURE EXTRAS (F7.F)
    # ===================================================================
    USE_FEATURE_ATTENTION = False   # squeeze-and-excitation before RNN

    # Faza 6 — optional architecture toggles (all default OFF).
    USE_RNN_NORM = False
    POOLING_MODE = 'last'           # 'last' | 'mean' | 'max' (when use_attention=False)
    USE_CONV1D_FRONT = False
    CONV1D_KERNEL = 5
    USE_FC_RESIDUAL = False
    USE_LEARNABLE_OUTPUT_SCALE = False
    FREEZE_RNN = False

    # ===================================================================
    # SCHEDULER EXTRAS (F7.E)
    # ===================================================================
    LR_WARMUP_EPOCHS = 0            # 0 = disabled; linear warmup before main sched

    # ===================================================================
    # SIGNAL THRESHOLD
    # ===================================================================
    SIGNAL_THRESHOLD_ENABLED = True
    SIGNAL_THRESHOLD_MODE = 'fixed'     # 'fixed' | 'empirical'
    SIGNAL_THRESHOLD_VALUE = 0.001
    SIGNAL_THRESHOLD_PERCENTILE = 50.0
    USE_THRESHOLD_PENALTIES = True

    # ===================================================================
    # SIGNAL / SIMULATION EVALUATION (lstm_test, common.simulation)
    # ===================================================================
    # These describe how a trained model is *judged* for trading usefulness
    # (BUY/SELL signal quality + a naive $100 long/short simulation), not how
    # it is trained. Mirror of xgb_model_package.SignalConfig.
    SIGNAL_WINDOW_SIGMA = 100   # rolling window for market volatility (sigma)
    SIGNAL_MIN_STRONG = 60.0    # signal >= this == a "strong"/tradable signal
    SIM_START_CASH = 100.0      # starting capital for the simulation

    # CP-168: explicit simulation threshold configuration & constants
    THRESHOLD_MODE = 'absolute'       # 'absolute' | 'percentile' | 'zscore'
    ZSCORE_CUTOFF = 1.6               # Mean + Z*std (mode='zscore', ETH default)
    PERCENTILE_CUTOFF = 5.0           # Top X% (mode='percentile')
    POSITION_MODE = 'sticky'          # 'sticky' | 'reset'
    ORACLE_THRESHOLD_MODE = None      # None = match THRESHOLD_MODE

    THRESHOLD_MODE_ABSOLUTE = 'absolute'
    THRESHOLD_MODE_PERCENTILE = 'percentile'
    THRESHOLD_MODE_ZSCORE = 'zscore'
    THRESHOLD_MODE_EXPANDING_Z = 'expanding_z'
    THRESHOLD_MODE_EXPANDING_PCT = 'expanding_pct'
    VALID_THRESHOLD_MODES = [
        THRESHOLD_MODE_ABSOLUTE,
        THRESHOLD_MODE_PERCENTILE,
        THRESHOLD_MODE_ZSCORE,
        THRESHOLD_MODE_EXPANDING_Z,
        THRESHOLD_MODE_EXPANDING_PCT,
    ]

    # ===================================================================
    # SCHEDULER
    # ===================================================================
    # 'plateau' | 'cosine' | 'onecycle' | 'step' | 'none'
    SCHEDULER_TYPE = 'cosine'
    SCHEDULER_PARAMS = {
        'patience': 5,      # plateau
        'factor': 0.5,      # plateau, step
        'T_0': 10,          # cosine
        'T_mult': 2,        # cosine
        'step_size': 30,    # step
        'max_lr': 0.01,     # onecycle (absolute max LR)
    }

    # ===================================================================
    # OPTUNA
    # ===================================================================

    # Which loss to use as Optuna score: 'val' (default) or 'train'.
    # 'val'   = standard — minimize val_loss (recommended for production)
    # 'train' = research — minimize train_loss (for overfitting diagnostics,
    #           architecture capacity analysis, etc.)
    # In both cases direction='minimize' and score_type is the chosen loss.
    OPTUNA_SCORE_SOURCE = 'val'     # 'val' | 'train'

    # F2.V1-5: composite score (alternative to raw val/train loss).
    # 'val_loss'  — score = val_loss (current behaviour)
    # 'composite' — score = val_loss
    #             + OPTUNA_COMPOSITE_COLLAPSE_WEIGHT
    #                 * relu(threshold - compression_ratio)
    #             + OPTUNA_COMPOSITE_R2_WEIGHT
    #                 * relu(-r2_val)
    # The composite penalises trials whose predictions collapse below the
    # configured compression ratio threshold or have negative R² (worse
    # than the mean) — both classic failure modes for log-return models.
    # 'signal'    — signal-aware proxy: val_loss
    #             + OPTUNA_COMPOSITE_COLLAPSE_WEIGHT * |1 - compression_ratio|
    #             + OPTUNA_COMPOSITE_R2_WEIGHT * relu(-r2_val)
    #             Symmetric CR term targets CR≈1 (punishes collapse AND
    #             explosion). The full market simulation / rank_score needs the
    #             close series and is therefore evaluated in lstm_test, not here.
    OPTUNA_SCORE_TYPE = 'val_loss'
    OPTUNA_COMPOSITE_COLLAPSE_THRESHOLD = 0.05
    OPTUNA_COMPOSITE_COLLAPSE_WEIGHT = 20.0
    OPTUNA_COMPOSITE_R2_WEIGHT = 10.0

    # How many top-N trials are persisted as `*_params.json`. Ranks 1..3 use
    # the legacy names (`best`, `second_best`, `third_best`) for backward
    # compatibility with existing loaders; ranks 4+ use `top04_params.json`
    # etc. Set e.g. 5 to get best + second_best + third_best + top04 + top05.
    OPTUNA_N_BEST_RESULTS = 3

    # F2.5.5 — Selective tuning: keys map to parameter names as they appear
    # in Optuna `trial.params`. When a key maps to False, the corresponding
    # parameter is NOT suggested and instead takes its value from LSTMConfig.
    # Default {} (empty) → tune everything (backward compatible). Special
    # case: setting 'fc_layers' to False freezes the whole FC head group
    # (n_fc_layers + fc_size_*).
    #
    # Note: loss_type is NOT a tunable parameter — it is set via
    # LSTMConfig.LOSS_TYPE (or --optimize_target in CLI). All trials
    # always use the same loss.
    #
    # Example:  TUNE_PARAMS = {
    #     'fc_layers': False,        # always use LSTMConfig.FC_LAYERS
    #     'hidden_size': False,      # always use LSTMConfig.HIDDEN_SIZE
    # }
    TUNE_PARAMS: dict = {}

    # ===================================================================
    # PATHS — derived from module location
    # ===================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'lstm_lr_models')
    DATA_DIR = os.path.join(BASE_DIR, 'lstm_lr_data')

    # ===================================================================
    # ID PREFIXES
    # ===================================================================
    MODEL_ID_PREFIX = 'LR'
    OPTUNA_ID_PREFIX = 'LRO'
    TEST_ID_PREFIX = 'LRT'
    DATA_ID_PREFIX = 'LRB'

    # ===================================================================
    # METHODS
    # ===================================================================

    @classmethod
    def validate(cls, resolved: dict | None = None):
        """
        Validate configuration consistency. Works on a resolved dict
        (e.g. from train_setup) or directly on class attributes.
        """
        def _get(key, default=None):
            if resolved is not None:
                val = resolved.get(key.lower(), resolved.get(key.upper(), default))
                return val if val is not None else default
            return getattr(cls, key.upper(), default)

        errors = []

        rnn_type = _get('rnn_type', cls.RNN_TYPE)
        if rnn_type not in ('lstm', 'gru'):
            errors.append(f"RNN_TYPE must be 'lstm' or 'gru', got: {rnn_type}")

        arch_type = _get('arch_type', cls.ARCH_TYPE)
        if arch_type not in ('legacy', 'dual_head'):
            errors.append(f"ARCH_TYPE must be legacy/dual_head, got: {arch_type}")

        activation = _get('activation', cls.ACTIVATION)
        if activation not in ('relu', 'gelu', 'leaky_relu'):
            errors.append(f"ACTIVATION must be relu/gelu/leaky_relu, got: {activation}")

        weight_init = _get('weight_init', cls.WEIGHT_INIT)
        if weight_init not in ('default', 'kaiming', 'xavier'):
            errors.append(f"WEIGHT_INIT must be default/kaiming/xavier, got: {weight_init}")

        attention_heads = _get('attention_heads', cls.ATTENTION_HEADS)
        if attention_heads < 1:
            errors.append(f"ATTENTION_HEADS must be >= 1, got: {attention_heads}")
        if attention_heads > 1:
            hidden_size = _get('hidden_size', cls.HIDDEN_SIZE)
            bidirectional = _get('bidirectional', cls.BIDIRECTIONAL)
            rnn_out = hidden_size * (2 if bidirectional else 1)
            if rnn_out % attention_heads != 0:
                errors.append(
                    "ATTENTION_HEADS does not divide rnn_output_size "
                    f"(hidden_size * dirs = {rnn_out}, attention_heads = {attention_heads})"
                )

        fc_norm = _get('fc_norm', cls.FC_NORM)
        if fc_norm not in ('batch', 'layer', 'none'):
            errors.append(f"FC_NORM must be batch/layer/none, got: {fc_norm}")

        loss_type = _get('loss_type', cls.LOSS_TYPE)
        if loss_type not in ('mse', 'mae', 'huber', 'smooth_l1',
                             'pseudo_huber', 'quantile', 'custom', 'signal'):
            errors.append(f"LOSS_TYPE invalid: {loss_type}")

        quantile_q = _get('quantile_loss_q', _get('quantile_q', cls.QUANTILE_LOSS_Q))
        if loss_type == 'quantile' and not (0.0 < quantile_q < 1.0):
            errors.append(f"QUANTILE_LOSS_Q must be in (0, 1), got: {quantile_q}")

        training_profile = _get('training_profile', cls.TRAINING_PROFILE)
        if training_profile not in ('baseline', 'rank_focused', 'sim_focused', 'custom_sim'):
            errors.append(
                "TRAINING_PROFILE must be baseline/rank_focused/sim_focused/custom_sim, "
                f"got: {training_profile}"
            )

        best_metric = _get('best_model_metric', cls.BEST_MODEL_METRIC)
        if best_metric not in ('val', 'train', 'val_composite', 'val_signal'):
            errors.append(
                "BEST_MODEL_METRIC must be val/train/val_composite/val_signal, "
                f"got: {best_metric}"
            )

        best_cr_weight = _get('best_model_cr_weight', cls.BEST_MODEL_CR_WEIGHT)
        if best_cr_weight is None or best_cr_weight < 0:
            errors.append(f"BEST_MODEL_CR_WEIGHT must be >= 0, got: {best_cr_weight}")

        scheduler_type = _get('scheduler_type', cls.SCHEDULER_TYPE)
        if scheduler_type not in ('plateau', 'cosine', 'onecycle', 'step', 'none'):
            errors.append(f"SCHEDULER_TYPE invalid: {scheduler_type}")

        target_scaler_type = _get('target_scaler_type', cls.TARGET_SCALER_TYPE)
        if target_scaler_type not in ('standard', 'robust'):
            errors.append(f"TARGET_SCALER_TYPE invalid: {target_scaler_type}")

        val_source = _get('val_source', cls.VAL_SOURCE)
        if val_source not in ('split', 'test', 'agent'):
            errors.append(f"VAL_SOURCE must be 'split', 'test' or 'agent', got: {val_source}")

        preferred_format = _get('preferred_data_format', _get('preferred_format', cls.PREFERRED_DATA_FORMAT))
        if preferred_format not in ('csv', 'npy'):
            errors.append(f"PREFERRED_DATA_FORMAT must be 'csv' or 'npy', got: {preferred_format}")

        signal_threshold_mode = _get('signal_threshold_mode', cls.SIGNAL_THRESHOLD_MODE)
        if signal_threshold_mode not in ('fixed', 'empirical'):
            errors.append(f"SIGNAL_THRESHOLD_MODE invalid: {signal_threshold_mode}")

        sample_weight_mode = _get('sample_weight_mode', cls.SAMPLE_WEIGHT_MODE)
        if sample_weight_mode not in ('abs', 'abs_clip', 'rank', 'percentile'):
            errors.append(f"SAMPLE_WEIGHT_MODE invalid: {sample_weight_mode}")

        lookback = _get('default_lookback', _get('lookback', cls.DEFAULT_LOOKBACK))
        if lookback is None or lookback < 1:
            errors.append(f"lookback must be >= 1, got: {lookback}")

        horizon = _get('default_horizon', _get('horizon', cls.DEFAULT_HORIZON))
        if horizon is None or horizon < 1:
            errors.append(f"horizon must be >= 1, got: {horizon}")

        pooling_mode = _get('pooling_mode', cls.POOLING_MODE)
        if pooling_mode not in ('last', 'mean', 'max'):
            errors.append(f"POOLING_MODE must be last/mean/max, got: {pooling_mode}")

        min_epochs_before_es = _get('min_epochs_before_es', cls.MIN_EPOCHS_BEFORE_ES)
        if min_epochs_before_es is None or min_epochs_before_es < 0:
            errors.append(f"MIN_EPOCHS_BEFORE_ES must be >= 0, got: {min_epochs_before_es}")

        epochs = _get('epochs', cls.EPOCHS)
        if epochs is None or epochs <= 0:
            errors.append(f"epochs must be > 0, got: {epochs}")

        batch_size = _get('batch_size', cls.BATCH_SIZE)
        if batch_size is None or batch_size <= 0:
            errors.append(f"batch_size must be > 0, got: {batch_size}")

        dropout = _get('dropout', cls.DROPOUT)
        if dropout is not None and not (0.0 <= dropout <= 1.0):
            errors.append(f"dropout must be in [0, 1], got: {dropout}")

        results_format = _get('default_results_format', _get('results_format', cls.DEFAULT_RESULTS_FORMAT))
        if results_format not in ('parquet', 'csv'):
            errors.append(f"DEFAULT_RESULTS_FORMAT must be 'parquet' or 'csv', got: {results_format}")

        results_compression = _get('default_results_compression', _get('results_compression', cls.DEFAULT_RESULTS_COMPRESSION))
        if results_compression not in ('snappy', 'zstd', 'gzip', 'none', None):
            errors.append(f"DEFAULT_RESULTS_COMPRESSION invalid: {results_compression}")

        threshold_mode = _get('threshold_mode', cls.THRESHOLD_MODE)
        if threshold_mode not in ('absolute', 'percentile', 'zscore', 'expanding_z', 'expanding_pct'):
            errors.append(f"threshold_mode invalid: {threshold_mode}")

        position_mode = _get('position_mode', cls.POSITION_MODE)
        if position_mode not in ('sticky', 'reset'):
            errors.append(f"position_mode invalid: {position_mode}")

        if errors:
            raise ValueError(
                "LSTMConfig validation failed:\n  " + "\n  ".join(errors)
            )

    @staticmethod
    def load_optuna_params(path):
        """Load Optuna best params from JSON file.

        Reconstructs fc_layers from fc_size_* keys if present.

        Returns:
            params: dict of hyperparameters (without _metadata)
            metadata: dict with trial info, or None
        """
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        metadata = data.pop('_metadata', None)

        # Reconstruct fc_layers from Optuna's fc_size_0, fc_size_1, ...
        # Apply the same min(fc_size, prev) clamping as suggest_hyperparameters.
        # prev tracks the CLAMPED value (funnel: sizes are non-increasing).
        # Kept consistent with the fix in lstm_hyperparameter_tuning.py.
        n_fc = data.pop('n_fc_layers', None)
        if n_fc is not None and 'fc_layers' not in data:
            hidden = data.get('hidden_size', 128)
            bidir = data.get('bidirectional', False)
            prev = hidden * (2 if bidir else 1)
            fc_layers = []
            for i in range(n_fc):
                key = f'fc_size_{i}'
                if key in data:
                    raw = data.pop(key)
                    clamped = min(raw, prev)
                    fc_layers.append(clamped)
                    prev = clamped
            if fc_layers:
                data['fc_layers'] = fc_layers

        # Clean up any remaining fc_size_* keys not consumed
        for key in list(data.keys()):
            if key.startswith('fc_size_'):
                data.pop(key)

        return data, metadata

    @classmethod
    def ensure_dirs(cls):
        """Create output directories if they don't exist."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)
