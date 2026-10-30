"""
XGBoost Model Configuration and Optuna Integration
"""
import os
import json
import argparse


class XGBConfig:
    """Configuration for XGBoost model"""
    # -------------------------------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------------------------------
    DEFAULT_LOOKBACK = 48
    DEFAULT_HORIZON = 12
    DEFAULT_FLATTENING = 'hybrid'
    DEFAULT_FEATURE_SET = 'full'

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    EARLY_STOPPING_ROUNDS = 1000
    DEFAULT_EARLY_STOPPING_METRIC = 't_rmse'   # 't_rmse' | 'rmse' | 'pseudo_huber' | 'corr' | 't_dir_acc' | 'none'
    VERBOSE_EVAL = 60
    # USE_BEST_ITERATION:
    #   False (default) - save model from last iteration (all n_estimators trees)
    #   True            - save model from best_iteration (early stopping checkpoint)
    USE_BEST_ITERATION = False

    # -------------------------------------------------------------------------
    # STORAGE FORMATS (CP-161)
    # -------------------------------------------------------------------------
    DEFAULT_RESULTS_FORMAT = 'parquet'         # 'parquet' | 'csv'
    DEFAULT_RESULTS_COMPRESSION = 'snappy'     # 'snappy' | 'gzip' | 'zstd' | 'none'

    # -------------------------------------------------------------------------
    # TARGET SCALING
    # -------------------------------------------------------------------------
    SCALE_TARGET = True
    TARGET_SCALER_TYPE = 'robust'               # 'standard' | 'robust'


    # -------------------------------------------------------------------------
    # OPTUNA OPTIMIZATION TARGET
    # -------------------------------------------------------------------------
    # Controls which dataset Optuna uses to compute the loss it minimises.
    #   'val'   - minimise loss on validation set (default, recommended)
    #             → selects params that generalise best
    #   'train' - minimise loss on training set
    #             → useful to diagnose underfitting, not recommended for production
    # The loss function used is always the one defined by OBJECTIVE above
    # (e.g. Pseudo-Huber), NOT sklearn MAE.
    OPTUNA_OPTIMIZE_TARGET = 'val'          # 'val' | 'train'
    OPTUNA_OPTIMIZE_METRIC = 't_rmse'          # 'mae' | 'rmse' | 'pseudo_huber'
                                            # 't_mae' | 't_rmse' | 't_pseudo_huber' | 't_dir_acc'
                                            # prefiks t_ = metryki liczone na próbkach |y| >= threshold

    # -------------------------------------------------------------------------
    # SIGNAL THRESHOLD — próg sygnałowy
    # -------------------------------------------------------------------------
    # SIGNAL_THRESHOLD_ENABLED:
    #   False (domyślne) — próg nieaktywny, brak metryk progowych
    #   True             — próg aktywny: metryki progowe (t_mae, t_dir_acc, ...)
    #                      w train, test, tuning. Penalties kontrolowane osobno
    #                      przez USE_THRESHOLD_PENALTIES.
    #
    # Centralny próg sygnałowy — wartości |log_return| poniżej tego progu
    # są traktowane jako „szum" (brak realnego zysku po opłatach).
    # Używany w metrykach (Thresholded Dir Acc, Thresholded MAE, ...)
    # i opcjonalnie w karach objective / sample weights.
    #
    # SIGNAL_THRESHOLD_MODE:
    #   'fixed'      — stała wartość SIGNAL_THRESHOLD_VALUE
    #   'empirical'  — obliczana z |y_train| jako percentyl
    #                  SIGNAL_THRESHOLD_PERCENTILE (np. 50.0 = mediana |y|)
    #
    # USE_THRESHOLD_PENALTIES:
    #   False (domyślne) — próg używany tylko do metryk
    #   True             — próg wpływa na kary w objective / sample weights
    #
    # THRESHOLD_SMOOTH_TRANSITION:
    #   False — twarde odcięcie: w=1.0 powyżej, w=DEAD_ZONE_WEIGHT poniżej
    #   True  — sigmoid: gładkie przejście wokół progu
    SIGNAL_THRESHOLD_ENABLED     = True          # master switch (jak REWEIGHT_MISDIR)
    SIGNAL_THRESHOLD_MODE        = 'fixed'        # 'fixed' | 'empirical'
    SIGNAL_THRESHOLD_VALUE       = 0.001          # log-return threshold (fixed mode)
    SIGNAL_THRESHOLD_PERCENTILE  = 50.0           # percentyl |y_train| (empirical mode)
    USE_THRESHOLD_PENALTIES      = True          # próg wpływa na penalties
    THRESHOLD_SMOOTH_TRANSITION  = False          # gładkie przejście sigmoid
    THRESHOLD_SMOOTH_WIDTH       = 2.0            # szerokość sigmoidy (1=ostry, 5=łagodny)

    # -------------------------------------------------------------------------
    # SAMPLE WEIGHTS — podejście A (natywne DMatrix weight)
    # -------------------------------------------------------------------------
    # Wagi per-próbka obliczane z |log_return| i przekazywane do
    # xgb.DMatrix(weight=...). XGBoost mnoży hessjan każdej próbki przez
    # jej wagę, co zwiększa wpływ próbek z dużymi ruchami na budowę drzew.
    # Wagi obliczane z surowego (nieskalowanego) y_train.
    #
    # SAMPLE_WEIGHT_MODE:  None        -- brak ważenia (domyślne XGBoost)
    #                      'abs'       -- w = |y|^alpha
    #                      'abs_clip'  -- w = clip(|y|^alpha, floor, inf)
    #                      'rank'      -- w = rank(|y|) / N (scale-free)
    #                      'percentile'-- w = 1.0 if |y| >= pct else low_w
    SAMPLE_WEIGHT_MODE       = None          # domyslnie wylaczone
    SAMPLE_WEIGHT_ALPHA      = 1.0           # wykladnik dla 'abs' / 'abs_clip'
    SAMPLE_WEIGHT_MIN_W      = 0.1           # floor fraction dla 'abs_clip'
    SAMPLE_WEIGHT_PERCENTILE = 70.0          # prog percentyla dla 'percentile'
    SAMPLE_WEIGHT_LOW_W      = 0.1           # waga ponizej progu dla 'percentile'

    # -------------------------------------------------------------------------
    # MISDIR PENALTY — podejście A (drugi trening z zaktualizowanymi wagami)
    # -------------------------------------------------------------------------
    # Po pierwszym treningu identyfikuje próbki gdzie model przewidział zly
    # kierunek dla duzych ruchow (|y| >= MISDIR_MIN_QUANTILE). Tym probkom
    # podnosi wagi przez MISDIR_PENALTY, nastepnie trenuje ponownie.
    #
    # MISDIR_APPROACH_A_VARIANT:
    #   'retrain' -- pelny restart od zera z nowymi wagami (czas x2)
    #   'boost'   -- kontynuacja od checkpointu (xgb_model=model),
    #               nowe drzewa dodawane z korekta mis-directional
    REWEIGHT_MISDIR            = False        # domyslnie wylaczone
    MISDIR_APPROACH_A_VARIANT  = 'boost'    # 'retrain' | 'boost'
    MISDIR_PENALTY             = 2.0          # mnoznik dla mis-directional probek
    MISDIR_MIN_QUANTILE        = 0.7          # prog |y| kwalifikujacy do kary

    # -------------------------------------------------------------------------
    # CUSTOM OBJECTIVE -- podejście B
    # -------------------------------------------------------------------------
    # Zastepuje standardowy params['objective'] -- XGBoost ignoruje
    # 'objective' w params gdy obj= jest podane w xgb.train().
    #
    # Parametry use_sample_weights i use_misdir_penalty sa niezalezne,
    # co umozliwia porownanie skutecznosci kazdego mechanizmu osobno:
    #
    #   CUSTOM_OBJ_SAMPLE_WEIGHTS=True,  CUSTOM_OBJ_MISDIR_PENALTY=False
    #       -> samo wazenie probek przez |y_raw|
    #
    #   CUSTOM_OBJ_SAMPLE_WEIGHTS=False, CUSTOM_OBJ_MISDIR_PENALTY=True
    #       -> sama kara za mis-direction
    #
    #   CUSTOM_OBJ_SAMPLE_WEIGHTS=True,  CUSTOM_OBJ_MISDIR_PENALTY=True
    #       -> laczone (domyslne gdy USE_CUSTOM_OBJECTIVE=True)
    #
    # Parametry wag i misdir w custom obj sa wspoldzielone z podejsciem A
    # (SAMPLE_WEIGHT_* i MISDIR_*) -- nie duplikujemy konfiguracji.
    # Jedyny dodatkowy parametr to CUSTOM_OBJ_HUBER_DELTA (delta pseudo-Hubera
    # uzywana w custom obj zamiast XGBoost-natywnego huber_slope).
    USE_CUSTOM_OBJECTIVE       = True        # False = standard obj string (etap 1)
    CUSTOM_OBJ_SAMPLE_WEIGHTS  = True         # skladnik wag w custom obj
    CUSTOM_OBJ_MISDIR_PENALTY  = True         # skladnik misdir w custom obj
    CUSTOM_OBJ_HUBER_DELTA     = 'auto'        # 'auto' = oblicz z danych gdy brak scalera
                                              # float  = konkretna wartosc (1.0 ok dla skalowanych)
                                              # (oddzielna od HUBER_SLOPE ktory
                                              #  dotyczy natywnego XGBoost obj)

    # -------------------------------------------------------------------------
    # DEAD-ZONE OBJECTIVE — alternatywna funkcja celu
    # -------------------------------------------------------------------------
    # Tłumi gradienty dla próbek poniżej progu sygnałowego.
    # Model uczy się tych wartości, ale z niskim priorytetem.
    # Powyżej progu: pełny gradient + opcjonalne kary misdir.
    #
    # CUSTOM_OBJ_TYPE:
    #   'weighted_huber' — obecna implementacja (Pseudo-Huber + wagi + misdir)
    #   'dead_zone'      — Pseudo-Huber z dead-zone poniżej progu
    CUSTOM_OBJ_TYPE              = 'weighted_huber'   # 'weighted_huber' | 'dead_zone'
    DEAD_ZONE_WEIGHT             = 0.1                # mnożnik gradientu |y| < threshold


    # -------------------------------------------------------------------------
    # OBJECTIVE FUNCTION
    # -------------------------------------------------------------------------
    # Compatible options (log-return targets, values can be negative):
    #   'reg:pseudohubererror'          - Pseudo-Huber (default, robust to outliers)
    #   'reg:pseudohubererror_large'    - Pseudo-Huber with large delta (huber_slope=10, closer to MSE)
    #   'reg:absoluteerror'             - MAE (median regression, prone to flattening) 0
    #   'reg:squarederror'              - MSE (mean regression, sensitive to outliers)
    #
    # INCOMPATIBLE — do not use with log-return targets:
    #   'reg:squaredlogerror'           - requires label > -1 (fails with scaled log-returns)
    #   'rank:ndcg'                     - requires non-negative integer labels
    OBJECTIVE = 'reg:pseudohubererror_large'

    # Huber delta — only used when OBJECTIVE == 'reg:pseudohubererror_large'
    # Larger value = more MSE-like behaviour, better at capturing extremes
    HUBER_SLOPE = 10.0


    # -------------------------------------------------------------------------
    # TARGET WINSORISATION
    # -------------------------------------------------------------------------
    # Clips extreme log-return targets to [p, 1-p] percentile range computed
    # on the TRAINING set only. Val/test are clipped with the same bounds
    # (no leakage). Set WINSORIZE_TARGET = False to disable entirely.
    #
    # Recommended: 99.0 (clips top/bottom 0.5% — flash crashes, pump&dumps)
    # Conservative: 99.5 (clips top/bottom 0.25%)
    # Disabled:     set WINSORIZE_TARGET = False
    WINSORIZE_TARGET = False
    WINSORIZE_PERCENTILE = 99.0                 # e.g. 99.0 → clips at [0.5%, 99.5%]


    # -------------------------------------------------------------------------
    # ZABEZPIECZENIE PRZED DUBLOWANIEM PODEJSC
    # -------------------------------------------------------------------------
    # Blokuje jednoczesne uzycie podejscia A i B dla tego samego mechanizmu.
    # Ustaw ALLOW_COMBINED_APPROACHES=True aby pominac zabezpieczenie
    # (przydatne do eksperymentowania i porownan).
    ALLOW_COMBINED_APPROACHES  = False


    # -------------------------------------------------------------------------
    # MARKET SIMULATION & SIGNAL EVALUATION (Simulation v2 Profile)
    # -------------------------------------------------------------------------
    SIGNAL_WINDOW_SIGMA     = 100        # okno zmienności kroczącej (std dev log returns)
    SIGNAL_MIN_STRONG       = 60.0       # minimalna siła sygnału (0..100) do otwarcia pozycji
    SIM_START_CASH          = 100.0      # kapitał początkowy symulacji ($)
    SIM_PRIMARY_FEE_BPS     = 2.0        # prowizja bazowa w punktach bazowych (0.02% = maker/VIP)
    SIM_AUXILIARY_FEE_BPS   = 2.0        # prowizja pomocnicza w punktach bazowych (do weryfikacji wrażliwości)
    SIMULATION_PROFILE      = 'v2'       # domyślny profil ewaluacji rynkowej ('v2')


    # =========================================================================
    # FIXED / SYSTEM SETTINGS (RARELY MODIFIED)
    # =========================================================================

    # -------------------------------------------------------------------------
    # PATHS
    # -------------------------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PREPARED_DATA_DIR = os.path.join(BASE_DIR, 'prepared_data')
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'xgb_models')
    LOGS_DIR = os.path.join(BASE_DIR, 'xgb_data')

    # -------------------------------------------------------------------------
    # ID PREFIXES
    # -------------------------------------------------------------------------
    DATA_ID_PREFIX = 'XB'
    MODEL_ID_PREFIX = 'X'
    OPTUNA_ID_PREFIX = 'XO'
    TEST_ID_PREFIX = 'XT'

    # -------------------------------------------------------------------------
    # FEATURE SETS
    # -------------------------------------------------------------------------
    # The authoritative list of feature sets is get_feature_sets() in the root
    # feature_engineering.py (full, technical, technical_bp, minimal, max,
    # lr_optimal, signal_optimal, candle_plus_bp, xgb_*, ...); data_utils resolves
    # the name at load time. A hand-maintained copy here went stale (it lacked
    # max / lr_optimal / signal_optimal / candle_plus_bp) and had no readers.
    #
    # Policy (strict): the chosen set must be a subset of the dataset columns,
    # otherwise training raises instead of silently training on fewer features.
    # The default 'full' is the 23-feature legacy set; use --feature_set max to
    # train on all 39 FE 3.0 features.

    # -------------------------------------------------------------------------
    # FLATTENING METHODS
    # -------------------------------------------------------------------------
    FLATTENING_METHODS = ['lag', 'agg', 'hybrid', 'signal', 'none']
    HYBRID_LAG_POINTS = [1, 2, 3, 5, 10, 24]
    HYBRID_WINDOWS = [6, 12, 24, 48]
    # signal flattening windows -- short vs long for trend, and momentum windows
    SIGNAL_SHORT_WINDOWS = [3, 6, 12]
    SIGNAL_LONG_WINDOWS  = [24, 48]
    SIGNAL_DIFF_LAGS     = [1, 3, 5, 12]


    # -------------------------------------------------------------------------
    # XGBOOST HYPERPARAMETERS
    # -------------------------------------------------------------------------
    DEFAULT_PARAMS = {
        'n_estimators': 3222,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'min_child_weight': 3,
        'gamma': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 1,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
    }


    # =========================================================================
    # CLASS METHODS
    # =========================================================================

    @classmethod
    def get_xgb_objective(cls):
        """
        Returns (objective_str, extra_params_dict).
        'reg:pseudohubererror_large' is a virtual name handled here —
        XGBoost receives 'reg:pseudohubererror' + huber_slope kwarg.
        """
        if cls.OBJECTIVE == 'reg:pseudohubererror_large':
            return 'reg:pseudohubererror', {'huber_slope': cls.HUBER_SLOPE}
        return cls.OBJECTIVE, {}

    @classmethod
    def validate_approach_config(cls):
        """
        Validates that native (A) and custom objective (B) approaches
        are not active simultaneously for the same mechanism.

        Raises ValueError if a conflict is detected and
        ALLOW_COMBINED_APPROACHES is False.
        """
        if cls.ALLOW_COMBINED_APPROACHES:
            return  # eksperyment -- brak blokady

        conflicts = []

        if (cls.SAMPLE_WEIGHT_MODE is not None
                and cls.USE_CUSTOM_OBJECTIVE
                and cls.CUSTOM_OBJ_SAMPLE_WEIGHTS):
            conflicts.append(
                "Sample weights: SAMPLE_WEIGHT_MODE aktywny (podejscie A) "
                "oraz USE_CUSTOM_OBJECTIVE+CUSTOM_OBJ_SAMPLE_WEIGHTS aktywne "
                "(podejscie B) jednoczesnie."
            )

        if (cls.REWEIGHT_MISDIR
                and cls.USE_CUSTOM_OBJECTIVE
                and cls.CUSTOM_OBJ_MISDIR_PENALTY):
            conflicts.append(
                "Misdir penalty: REWEIGHT_MISDIR aktywny (podejscie A) "
                "oraz USE_CUSTOM_OBJECTIVE+CUSTOM_OBJ_MISDIR_PENALTY aktywne "
                "(podejscie B) jednoczesnie."
            )

        if conflicts:
            raise ValueError(
                "Konflikt konfiguracji -- aktywne oba podejscia jednoczesnie:\n"
                + "\n".join(f"  - {c}" for c in conflicts)
                + "\nUstaw ALLOW_COMBINED_APPROACHES=True "
                  "aby pominac to zabezpieczenie."
            )

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

        params = {k: v for k, v in data.items() if not k.startswith('_')}
        metadata = data.get('_metadata', {})
        return params, metadata

    @staticmethod
    def train_with_optuna(data_path, optuna_path_or_dir, params_name='best',
                          seed=42, verbose=True):
        """Train model with Optuna parameters"""
        from xgb_train import train

        print("=" * 80)
        print("TRAINING WITH OPTUNA PARAMETERS")
        print("=" * 80)

        model, feature_scaler, config = train(
            data_path=data_path,
            load_optuna_params=optuna_path_or_dir
            if not os.path.isdir(optuna_path_or_dir)
            else os.path.join(optuna_path_or_dir, f'{params_name}_params.json'),
            seed=seed,
            verbose=verbose,
        )

        print(f"\nModel ID:  {config['model_id']}")
        print(f"Output:    {config['output_dir']}")
        print(f"Val MAE:   {config['val_metrics']['mae']:.6f}")
        print(f"Val R²:    {config['val_metrics']['r2']:.4f}")
        print(f"Dir Acc:   {config['val_metrics']['dir_acc']:.2f}%")

        return model, feature_scaler, config


# Create directories
for _dir in [XGBConfig.PREPARED_DATA_DIR, XGBConfig.MODELS_DIR,
             XGBConfig.LOGS_DIR, XGBConfig.RAW_DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost with Optuna parameters')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--optuna_dir', type=str, default=None)
    parser.add_argument('--optuna_params', type=str, default=None)
    parser.add_argument('--params_name', type=str, default='best',
                        choices=['best', 'second_best', 'third_best'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    if args.optuna_params:
        optuna_path = args.optuna_params
    elif args.optuna_dir:
        optuna_path = args.optuna_dir
    else:
        raise ValueError("Provide either --optuna_dir or --optuna_params")

    XGBConfig.train_with_optuna(
        data_path=args.data_path,
        optuna_path_or_dir=optuna_path,
        params_name=args.params_name,
        seed=args.seed,
        verbose=not args.quiet,
    )
