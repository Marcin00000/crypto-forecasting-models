"""
Random Forest Model Configuration
"""
import os
import json


class RFConfig:
    """Configuration for Random Forest model"""

    # -------------------------------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------------------------------
    DEFAULT_LOOKBACK        = 48
    DEFAULT_HORIZON         = 12
    DEFAULT_FLATTENING      = 'none'
    DEFAULT_FEATURE_SET     = 'technical'

    # -------------------------------------------------------------------------
    # TARGET SCALING
    # -------------------------------------------------------------------------
    SCALE_TARGET        = True
    TARGET_SCALER_TYPE  = 'robust'          # 'standard' | 'robust'

    # -------------------------------------------------------------------------
    # TARGET WINSORISATION
    # Clips extreme log-return targets to [p, 1-p] percentile range computed
    # on the TRAINING set only.  Set WINSORIZE_TARGET = False to disable.
    # Recommended: 99.0  (clips top/bottom 0.5% -- flash crashes, pump&dumps)
    # -------------------------------------------------------------------------
    WINSORIZE_TARGET        = False
    WINSORIZE_PERCENTILE    = 99.0

    # -------------------------------------------------------------------------
    # OBJECTIVE FUNCTION (sklearn criterion)
    # Compatible options for log-return regression targets:
    #   'squared_error'     -- MSE (default, mean regression)
    #   'absolute_error'    -- MAE (median regression, more robust to outliers)
    #   'friedman_mse'      -- MSE with Friedman improvement score for splits
    #   'poisson'           -- requires non-negative targets (NOT suitable for
    #                          log-returns which can be negative -- excluded)
    #
    # Recommended: 'absolute_error'
    # -------------------------------------------------------------------------
    OBJECTIVE = 'absolute_error'

    # -------------------------------------------------------------------------
    # OPTUNA OPTIMIZATION TARGET
    #   'val'   -- minimise loss on validation set (default, recommended)
    #   'train' -- minimise loss on training set
    # -------------------------------------------------------------------------
    OPTUNA_OPTIMIZE_TARGET = 'train'

    # -------------------------------------------------------------------------
    # RANDOM FOREST HYPERPARAMETERS
    # -------------------------------------------------------------------------
    # DEFAULT_PARAMS = {
    #     'n_estimators':          300,
    #     'max_depth':             5,      # None = fully grown trees
    #     'max_features':          0.33,      # fraction of features per split
    #     'max_samples':           0.8,       # bootstrap sample fraction
    #     'min_samples_leaf':      5,
    #     'min_impurity_decrease': 0.0,
    #     'bootstrap':             True,
    #     'oob_score':             True,      # enables OOB R2 monitoring
    #     'n_jobs':                -1,
    #     'random_state':          42,
    # }
    DEFAULT_PARAMS = {
        'n_estimators': 862,
        'max_depth': 15,  # None = fully grown trees
        'max_features': 0.33,  # fraction of features per split
        'max_samples': 0.85,  # bootstrap sample fraction
        'min_samples_leaf': 9,
        'min_samples_split': 2,
        'min_impurity_decrease':  0.0001796,
        'min_weight_fraction_leaf': 0.0,
        'bootstrap': True,
        'oob_score': True,  # enables OOB R2 monitoring
        'n_jobs': -1,
        'random_state': 42,
    }

    # -------------------------------------------------------------------------
    # SAMPLE WEIGHTS
    # Assigns per-sample weights to fit() so the model focuses more on samples
    # with large absolute log-returns (extreme moves) and less on near-zero
    # returns (noise / flat market).
    #
    # SAMPLE_WEIGHT_MODE controls the weighting scheme:
    #   None              -- no weighting (uniform, default sklearn behaviour)
    #   'abs'             -- w = |y|^alpha  (plain absolute value)
    #   'abs_clip'        -- w = clip(|y|, min_w, inf)^alpha  (floor prevents
    #                        near-zero samples from getting weight=0)
    #   'rank'            -- w = rank(|y|) / N  (rank-based, scale-free)
    #   'percentile'      -- w = 1 if |y| >= p-th percentile else low_w
    #                        (binary: high-return vs. low-return samples)
    #
    # SAMPLE_WEIGHT_ALPHA exponent for 'abs' and 'abs_clip' modes.
    #   alpha = 1.0  -- linear scaling with |y|
    #   alpha = 0.5  -- sqrt scaling (gentler)
    #   alpha = 2.0  -- quadratic (aggressive, emphasises big moves strongly)
    #   Recommended starting point: 1.0
    #
    # SAMPLE_WEIGHT_MIN_W minimum weight floor for 'abs_clip' mode.
    #   Prevents samples with |y| ~= 0 from being completely ignored.
    #   Expressed as a fraction of max weight (e.g. 0.1 = floor at 10%).
    #
    # SAMPLE_WEIGHT_PERCENTILE threshold for 'percentile' mode (0-100).
    #   e.g. 70 -> top 30% by |y| get weight=1.0, bottom 70% get low_w.
    #
    # SAMPLE_WEIGHT_LOW_W weight assigned to below-threshold samples in
    #   'percentile' mode. 0.0 = exclude entirely, 0.1 = small influence.
    #
    # NOTE: weights are computed from y_train BEFORE target scaling so they
    # always reflect raw log-return magnitude regardless of scaling settings.
    # -------------------------------------------------------------------------
    SAMPLE_WEIGHT_MODE        = 'percentile'        # None | 'abs' | 'abs_clip' | 'rank' | 'percentile'
    SAMPLE_WEIGHT_ALPHA       = 1.0         # exponent for 'abs' / 'abs_clip' modes
    SAMPLE_WEIGHT_MIN_W       = 0.1         # floor fraction for 'abs_clip' mode
    SAMPLE_WEIGHT_PERCENTILE  = 70.0        # threshold percentile for 'percentile' mode
    SAMPLE_WEIGHT_LOW_W       = 0.1         # weight for below-threshold samples in 'percentile' mode

    # -------------------------------------------------------------------------
    # MISDIR PENALTY — asymmetric reweighting after first fit
    #
    # When REWEIGHT_MISDIR = True, a second fit is performed after an initial
    # fit on the training data.  Samples where the initial model predicted the
    # wrong direction (sign(y_pred) != sign(y_true)) on large moves are given
    # a multiplied weight, then the model is re-trained on these new weights.
    #
    # MISDIR_PENALTY   -- constant multiplier applied to mis-directional
    #                     samples.  Kept intentionally moderate (default 2.0)
    #                     to avoid flattening predictions toward zero.
    #                     A value of 1.0 disables the effect (no penalty).
    #
    # MISDIR_MIN_QUANTILE -- only samples whose |y_true| >= this quantile of
    #                     |y_train| are eligible for the penalty.  This limits
    #                     the correction to genuinely large moves and avoids
    #                     penalising noise (near-zero returns where direction
    #                     is inherently uncertain).
    #                     Range: 0.0–1.0  (default 0.7 = top 30% by |y|).
    #
    # NOTE: weights passed to the second fit are always >= original weights
    # (the penalty only increases weights; it never zeroes anything out).
    # The total training time roughly doubles when this is enabled.
    # -------------------------------------------------------------------------
    REWEIGHT_MISDIR       = False   # default OFF — enable explicitly when needed
    MISDIR_PENALTY        = 2.0     # multiplier for mis-directional large-move samples
    MISDIR_MIN_QUANTILE   = 0.7     # |y| quantile threshold to qualify for penalty

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    VERBOSE     = 1     # 0 = silent, 1 = progress dots, 2 = per-tree (noisy)

    USE_OOB_SCORE = True    # log OOB R2 (only valid when bootstrap=True)

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------
    BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
    PREPARED_DATA_DIR   = os.path.join(BASE_DIR, 'rf_prepared_data')
    RAW_DATA_DIR        = os.path.join(BASE_DIR, 'data')
    MODELS_DIR          = os.path.join(BASE_DIR, 'rf_models')
    LOGS_DIR            = os.path.join(BASE_DIR, 'rf_data')

    # -------------------------------------------------------------------------
    # ID PREFIXES
    # -------------------------------------------------------------------------
    DATA_ID_PREFIX      = 'RB'
    MODEL_ID_PREFIX     = 'R'
    OPTUNA_ID_PREFIX    = 'RO'
    TEST_ID_PREFIX      = 'RT'

    # -------------------------------------------------------------------------
    # FEATURE SETS: resolved at runtime by feature_engineering.get_feature_sets()
    # (a local name list here would go stale whenever a set is added there).
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # SIMULATION v2 CONSTANTS
    # -------------------------------------------------------------------------
    SIMULATION_PROFILE      = 'rf_simulation_v2'
    SIM_PRIMARY_FEE_BPS     = 2.0
    SIM_AUXILIARY_FEE_BPS   = 2.0
    RESULTS_FORMAT          = 'parquet'

    # -------------------------------------------------------------------------
    # FLATTENING METHODS
    # -------------------------------------------------------------------------
    FLATTENING_METHODS  = ['lag', 'agg', 'hybrid', 'signal', 'none']
    HYBRID_LAG_POINTS   = [1, 2, 3, 5, 10, 24]
    HYBRID_WINDOWS      = [6, 12, 24, 48]
    # signal flattening windows — short vs long for trend, and momentum windows
    SIGNAL_SHORT_WINDOWS = [3, 6, 12]
    SIGNAL_LONG_WINDOWS  = [24, 48]
    SIGNAL_DIFF_LAGS     = [1, 3, 5, 12]

    # -------------------------------------------------------------------------
    # THRESHOLD METRICS
    # Metrics computed only for samples where |y_pred| >= threshold.
    # Mirrors the signal generation logic in rf_model_package (SignalConfig):
    # predictions below threshold are treated as 'no signal' and excluded
    # from directional accuracy and other threshold metrics.
    #
    # DIR_ACC_THRESHOLD:
    #   None  -- disabled, show standard dir_acc only (current behaviour)
    #   float -- absolute log-return threshold, same unit as model output.
    #            Use the same value as SignalConfig.threshold in rf_model_package.
    #            Example: 0.001 = 0.1% minimum log-return to count as a signal.
    # -------------------------------------------------------------------------
    DIR_ACC_THRESHOLD = 0.001    # None | float -- set to None to disable

    # -------------------------------------------------------------------------
    # OBJECTIVE VALIDATION
    # -------------------------------------------------------------------------
    VALID_OBJECTIVES        = {'squared_error', 'absolute_error', 'friedman_mse'}
    INCOMPATIBLE_OBJECTIVES = {'poisson'}   # requires non-negative targets

    @classmethod
    def get_rf_criterion(cls):
        """
        Returns the sklearn criterion string.
        Raises ValueError for objectives incompatible with log-return targets.
        """
        if cls.OBJECTIVE in cls.INCOMPATIBLE_OBJECTIVES:
            raise ValueError(
                f"Objective '{cls.OBJECTIVE}' is incompatible with log-return targets.\n"
                f"  poisson -- requires non-negative labels (log-returns can be negative)\n"
                f"Use one of: {sorted(cls.VALID_OBJECTIVES)}"
            )
        if cls.OBJECTIVE not in cls.VALID_OBJECTIVES:
            raise ValueError(
                f"Unknown objective '{cls.OBJECTIVE}'. "
                f"Valid: {sorted(cls.VALID_OBJECTIVES)}"
            )
        return cls.OBJECTIVE

    @staticmethod
    def load_optuna_params(params_path_or_dir, params_name='best'):
        """Load parameters from Optuna results"""
        if os.path.isdir(params_path_or_dir):
            params_path = os.path.join(params_path_or_dir, f'{params_name}_params.json')
        else:
            params_path = params_path_or_dir

        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Optuna params not found: {params_path}")

        with open(params_path, 'r') as f:
            data = json.load(f)

        params   = {k: v for k, v in data.items() if not k.startswith('_')}
        metadata = data.get('_metadata', {})
        return params, metadata

    @staticmethod
    def train_with_optuna(data_path, optuna_path_or_dir, params_name='best',
                          seed=42, verbose=True):
        """Train model with Optuna parameters"""
        from rf_train import train

        print("=" * 80)
        print("TRAINING WITH OPTUNA PARAMETERS")
        print("=" * 80)

        model, feature_scaler, config = train(
            data_path=data_path,
            load_optuna_params=(
                optuna_path_or_dir
                if not os.path.isdir(optuna_path_or_dir)
                else os.path.join(optuna_path_or_dir, f'{params_name}_params.json')
            ),
            seed=seed,
            verbose=verbose,
        )

        print(f"\nModel ID:  {config['model_id']}")
        print(f"Output:    {config['output_dir']}")
        print(f"Val MAE:   {config['val_metrics']['mae']:.6f}")
        print(f"Val R2:    {config['val_metrics']['r2']:.4f}")
        print(f"Dir Acc:   {config['val_metrics']['dir_acc']:.2f}%")

        return model, feature_scaler, config


# Create directories on import
for _dir in [RFConfig.PREPARED_DATA_DIR, RFConfig.MODELS_DIR,
             RFConfig.LOGS_DIR, RFConfig.RAW_DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)
