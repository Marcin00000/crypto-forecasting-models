"""
Random Forest Hyperparameter Tuning using Optuna
"""
import os
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib

from rf_config import RFConfig
from common.data_utils import load_data
from common.id_utils import get_next_optuna_id, extract_data_id_from_path
from common.visualization import plot_optuna_results

warnings.filterwarnings('ignore')


# =============================================================================
# TUNING CONFIGURATION
# =============================================================================

class TuningConfig:
    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    TRAIN_PATH       = 'datasets/prepared'
    FEATURE_SET      = RFConfig.DEFAULT_FEATURE_SET     # 'full'
    CUSTOM_FEATURES  = None
    USE_SUBSET       = False
    SUBSET_SIZE      = 200_000

    # ------------------------------------------------------------------
    # OPTUNA
    # ------------------------------------------------------------------
    N_TRIALS        = 100
    TIMEOUT         = None          # seconds; None = no time limit
    N_JOBS          = 1             # parallel trials (>1 only if thread-safe)
    ENABLE_PRUNING  = False         # RF has no native intermediate reporting;
                                    # pruning via MedianPruner on reported MAE
    PRUNING_WARMUP  = 5             # min completed trials before pruning starts
    SEED            = 42

    # ------------------------------------------------------------------
    # FIXED / TUNED AXES
    # ------------------------------------------------------------------
    FLATTENING_METHOD        = RFConfig.DEFAULT_FLATTENING  # 'agg'
    FIXED_FLATTENING_METHOD  = None     # None = tune; str = fix to that method
    LOOKBACK                 = RFConfig.DEFAULT_LOOKBACK    # 48
    HORIZON                  = RFConfig.DEFAULT_HORIZON     # 12
    TUNE_LOOKBACK            = False
    TUNE_HORIZON             = False

    # ------------------------------------------------------------------
    # TARGET PIPELINE (fixed during tuning)
    # ------------------------------------------------------------------
    SCALE_TARGET        = RFConfig.SCALE_TARGET
    TARGET_SCALER_TYPE  = RFConfig.TARGET_SCALER_TYPE
    WINSORIZE_TARGET    = RFConfig.WINSORIZE_TARGET
    WINSORIZE_PERCENTILE = RFConfig.WINSORIZE_PERCENTILE

    # ------------------------------------------------------------------
    # SAMPLE WEIGHTS (mirrors RFConfig -- fixed during the whole study)
    # ------------------------------------------------------------------
    SAMPLE_WEIGHT_MODE        = RFConfig.SAMPLE_WEIGHT_MODE
    SAMPLE_WEIGHT_ALPHA       = RFConfig.SAMPLE_WEIGHT_ALPHA
    SAMPLE_WEIGHT_MIN_W       = RFConfig.SAMPLE_WEIGHT_MIN_W
    SAMPLE_WEIGHT_PERCENTILE  = RFConfig.SAMPLE_WEIGHT_PERCENTILE
    SAMPLE_WEIGHT_LOW_W       = RFConfig.SAMPLE_WEIGHT_LOW_W

    # ------------------------------------------------------------------
    # OBJECTIVE / OPTIMIZE TARGET
    # ------------------------------------------------------------------
    # CRITERION is fixed per tuning run (from RFConfig.OBJECTIVE).
    # Changing criterion mid-study would make trials incomparable.
    OPTIMIZE_TARGET  = RFConfig.OPTUNA_OPTIMIZE_TARGET  # 'val' | 'train'

    # ------------------------------------------------------------------
    # OPTIMIZE METRIC
    # Controls which scalar Optuna minimises each trial.
    #
    #   'mae'       -- Mean Absolute Error on log-returns (default)
    #                  Safe, stable, comparable across all trials.
    #
    #   'dir_acc'   -- 1 - directional_accuracy
    #                  Directly optimises sign prediction.
    #                  More noisy than MAE -- recommend >= 50 trials.
    #
    #   'composite' -- w * MAE_norm + (1-w) * (1 - dir_acc)
    #                  MAE is normalised by std(y_ref) so both terms
    #                  are on the same 0..1 scale before weighting.
    #                  Weight controlled by COMPOSITE_MAE_WEIGHT.
    #
    #   'sharpe'    -- MAE / std(errors)
    #                  Penalises noisy predictors; favours consistent
    #                  small errors over occasional accurate ones.
    # ------------------------------------------------------------------
    OPTIMIZE_METRIC       = 'composite'   # 'mae' | 'dir_acc' | 'composite' | 'sharpe'
    COMPOSITE_MAE_WEIGHT  = 0.7     # MAE weight in composite (dir_acc weight = 1 - this)


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

def suggest_hyperparameters(trial):
    """Define hyperparameter search space for Random Forest"""

    criterion = RFConfig.get_rf_criterion()   # fixed for the whole study

    params = {
        # Trees
        'n_estimators':          trial.suggest_int('n_estimators', 200, 900),
        # Tree structure
        'max_depth':             trial.suggest_int('max_depth', 3, 20),
        'min_samples_leaf':      trial.suggest_int('min_samples_leaf', 1, 20),
        'min_samples_split':     trial.suggest_int('min_samples_split', 2, 20),
        'min_impurity_decrease': trial.suggest_float('min_impurity_decrease',
                                                     0.0, 0.01),
        # Randomisation / regularisation
        'max_features':          trial.suggest_float('max_features', 0.1, 1.0),
        'max_samples':           trial.suggest_float('max_samples', 0.5, 1.0),
        # min_weight_fraction_leaf: active only when sample weights are used
        # (min_samples_leaf counts raw samples; min_weight_fraction_leaf
        # accounts for weighted sums and is more meaningful with sample_weight)
        'min_weight_fraction_leaf': (
            trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.01)
            if TuningConfig.SAMPLE_WEIGHT_MODE is not None
            else 0.0
        ),
        # Fixed for all trials
        'criterion':             criterion,
        'bootstrap':             True,
        'oob_score':             False,    # disabled during tuning (saves memory)
        'n_jobs':                -1,
        'random_state':          TuningConfig.SEED,
    }

    # Lookback / horizon
    params['lookback'] = (
        trial.suggest_categorical('lookback', [24, 36, 48, 72, 96, 120])
        if TuningConfig.TUNE_LOOKBACK else TuningConfig.LOOKBACK
    )
    params['horizon'] = (
        trial.suggest_categorical('horizon', [1, 3, 6, 12, 24])
        if TuningConfig.TUNE_HORIZON else TuningConfig.HORIZON
    )

    # Flattening method
    if TuningConfig.FIXED_FLATTENING_METHOD is not None:
        params['flattening_method'] = TuningConfig.FIXED_FLATTENING_METHOD
    else:
        params['flattening_method'] = trial.suggest_categorical(
            'flattening_method', ['lag', 'agg', 'hybrid', 'signal', 'none']
        )

    return params


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def objective(trial):
    """Optuna objective: returns MAE (lower = better)"""
    try:
        params = suggest_hyperparameters(trial)

        data, _, _ = load_data(
            TuningConfig.TRAIN_PATH,
            flattening_method=params.get('flattening_method',
                                         TuningConfig.FLATTENING_METHOD),
            feature_set=TuningConfig.FEATURE_SET,
            custom_features=TuningConfig.CUSTOM_FEATURES,
            lookback=params.get('lookback', TuningConfig.LOOKBACK),
            horizon=params.get('horizon',  TuningConfig.HORIZON),
            scale_target=TuningConfig.SCALE_TARGET,
            target_scaler_type=TuningConfig.TARGET_SCALER_TYPE,
            winsorize_target=TuningConfig.WINSORIZE_TARGET,
            winsorize_percentile=TuningConfig.WINSORIZE_PERCENTILE,
        )

        X_train = data['X_train']
        y_train = data['y_train']
        X_val   = data['X_val']
        y_val   = data['y_val']
        target_scaler = data.get('target_scaler')

        if len(X_train) < 100 or len(X_val) < 50:
            raise optuna.TrialPruned("Insufficient data")

        # Strip non-RF keys before passing to evaluator
        train_params = {
            k: v for k, v in params.items()
            if k not in ('lookback', 'horizon', 'flattening_method')
        }

        # Compute sample weights (same logic as rf_train.py)
        from common.training_utils import evaluate_for_tuning, compute_sample_weights
        if TuningConfig.SAMPLE_WEIGHT_MODE is not None:
            if target_scaler is not None:
                y_train_raw = target_scaler.inverse_transform(y_train)
            else:
                y_train_raw = y_train
            sw = compute_sample_weights(
                y_train_raw,
                mode=TuningConfig.SAMPLE_WEIGHT_MODE,
                alpha=TuningConfig.SAMPLE_WEIGHT_ALPHA,
                min_w=TuningConfig.SAMPLE_WEIGHT_MIN_W,
                percentile=TuningConfig.SAMPLE_WEIGHT_PERCENTILE,
                low_w=TuningConfig.SAMPLE_WEIGHT_LOW_W,
            )
        else:
            sw = None

        score = evaluate_for_tuning(
            X_train, y_train, X_val, y_val,
            train_params,
            target_scaler=target_scaler,
            optimize_target=TuningConfig.OPTIMIZE_TARGET,
            optimize_metric=TuningConfig.OPTIMIZE_METRIC,
            composite_mae_weight=TuningConfig.COMPOSITE_MAE_WEIGHT,
            sample_weight=sw,
        )

        # Report intermediate value for pruner
        # RF has no native per-round reporting, so we report the final MAE
        # at step 0 â€” MedianPruner can still prune based on this value
        trial.report(score, step=0)
        if TuningConfig.ENABLE_PRUNING and trial.should_prune():
            raise optuna.TrialPruned()

        return score

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


# =============================================================================
# MAIN TUNING FUNCTION
# =============================================================================

def run_tuning(n_trials=None, timeout=None, study_name=None):
    """Run Optuna hyperparameter tuning for Random Forest"""

    if n_trials is None:
        n_trials = TuningConfig.N_TRIALS
    if timeout is None:
        timeout  = TuningConfig.TIMEOUT

    data_id    = extract_data_id_from_path(TuningConfig.TRAIN_PATH)
    os.makedirs(RFConfig.LOGS_DIR, exist_ok=True)

    optuna_id  = get_next_optuna_id(RFConfig.LOGS_DIR)
    date_short = datetime.now().strftime('%d%m')

    if study_name is None:
        study_name = f"{optuna_id}_rf_optuna_{data_id}_{date_short}"

    output_dir = os.path.join(RFConfig.LOGS_DIR, study_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                os.path.join(output_dir, 'tuning.log'), encoding='utf-8'
            ),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    criterion = RFConfig.get_rf_criterion()

    logger.info("=" * 80)
    logger.info("RANDOM FOREST HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Optuna ID:   {optuna_id}")
    logger.info(f"Data ID:     {data_id}")
    logger.info(f"Output:      {output_dir}")
    logger.info(f"Trials:      {n_trials}")
    logger.info(f"Criterion:   {criterion}")
    logger.info(f"Minimises:   {TuningConfig.OPTIMIZE_TARGET.upper()} {TuningConfig.OPTIMIZE_METRIC.upper()}"
                + (f"  (mae_weight={TuningConfig.COMPOSITE_MAE_WEIGHT})"
                   if TuningConfig.OPTIMIZE_METRIC == 'composite' else ""))
    logger.info(f"Target scaler: {TuningConfig.SCALE_TARGET} ({TuningConfig.TARGET_SCALER_TYPE})")
    logger.info(f"Feature set: {TuningConfig.FEATURE_SET}")
    logger.info(f"Lookback:    {'TUNING' if TuningConfig.TUNE_LOOKBACK else f'FIXED ({TuningConfig.LOOKBACK})'}")
    logger.info(f"Horizon:     {'TUNING' if TuningConfig.TUNE_HORIZON  else f'FIXED ({TuningConfig.HORIZON})'}")
    logger.info(f"Flattening:  {'TUNING' if not TuningConfig.FIXED_FLATTENING_METHOD else f'FIXED ({TuningConfig.FIXED_FLATTENING_METHOD})'}")
    logger.info(f"Pruning:     {'ENABLED' if TuningConfig.ENABLE_PRUNING else 'DISABLED'}")

    # ------------------------------------------------------------------
    # Study
    # ------------------------------------------------------------------
    sampler = TPESampler(seed=TuningConfig.SEED)

    if TuningConfig.ENABLE_PRUNING:
        pruner = MedianPruner(
            n_startup_trials=TuningConfig.PRUNING_WARMUP,
            n_warmup_steps=0,       # RF reports at step=0; no warmup steps needed
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION START")
    logger.info("=" * 80)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=TuningConfig.N_JOBS,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nTotal trials:    {len(study.trials)}")
    logger.info(f"Complete trials: {len(completed)}")
    logger.info(f"Pruned trials:   {len(pruned)}")

    if not completed:
        logger.error("\nNo trials completed successfully!")
        return None, None

    best = study.best_trial
    score_label = f"{TuningConfig.OPTIMIZE_TARGET.upper()} {TuningConfig.OPTIMIZE_METRIC.upper()} ({criterion})"

    logger.info("\n" + "=" * 80)
    logger.info("BEST TRIAL")
    logger.info("=" * 80)
    logger.info(f"\nTrial:     {best.number}")
    logger.info(f"{score_label}: {best.value:.6f}")
    logger.info("\nParameters:")
    for k, v in sorted(best.params.items()):
        logger.info(f"  {k:25s}: {v}")

    # ------------------------------------------------------------------
    # Save top-3
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    sorted_trials = sorted(completed, key=lambda t: t.value)
    top_trials    = sorted_trials[:min(3, len(sorted_trials))]
    names         = ['best', 'second_best', 'third_best']

    for trial, name in zip(top_trials, names):
        params_to_save = trial.params.copy()

        # Ensure horizon is always present (needed by rf_train.py)
        if 'horizon' not in params_to_save:
            params_to_save['horizon'] = TuningConfig.HORIZON

        params_to_save['_metadata'] = {
            'optuna_id':          optuna_id,
            'data_id':            data_id,
            'best_value':         float(trial.value),
            'best_value_meaning': f"{TuningConfig.OPTIMIZE_TARGET}_{TuningConfig.OPTIMIZE_METRIC} ({criterion})",
            'trial_number':       trial.number,
            'timestamp':          datetime.now().strftime('%Y%m%d_%H%M%S'),
            'feature_set':        TuningConfig.FEATURE_SET,
            'custom_features':    TuningConfig.CUSTOM_FEATURES,
            'scale_target':        TuningConfig.SCALE_TARGET,
            'target_scaler_type':  TuningConfig.TARGET_SCALER_TYPE,
            'optimize_target':     TuningConfig.OPTIMIZE_TARGET,
            'optimize_metric':     TuningConfig.OPTIMIZE_METRIC,
            'composite_mae_weight': TuningConfig.COMPOSITE_MAE_WEIGHT,
            'criterion':           criterion,
        }

        out_path = os.path.join(output_dir, f'{name}_params.json')
        with open(out_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        logger.info(
            f"  [OK] {name}_params.json  "
            f"(trial {trial.number}, {score_label}: {trial.value:.6f})"
        )

    # Persist full study object
    joblib.dump(study, os.path.join(output_dir, 'study.pkl'))
    study.trials_dataframe().to_csv(
        os.path.join(output_dir, 'trials.csv'), index=False
    )
    logger.info("  [OK] study.pkl")
    logger.info("  [OK] trials.csv")

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    plot_optuna_results(study, output_dir)

    # ------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RANDOM FOREST OPTUNA TUNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Optuna ID:   {optuna_id}\n")
        f.write(f"Data ID:     {data_id}\n")
        f.write(f"Criterion:   {criterion}\n")
        metric_line = f"{TuningConfig.OPTIMIZE_TARGET.upper()} {TuningConfig.OPTIMIZE_METRIC.upper()}"
        if TuningConfig.OPTIMIZE_METRIC == 'composite':
            metric_line += f"  (mae_weight={TuningConfig.COMPOSITE_MAE_WEIGHT})"
        f.write(f"Minimises:   {metric_line}\n")
        f.write(f"Feature set: {TuningConfig.FEATURE_SET}\n")
        f.write(f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total:       {len(study.trials)}\n")
        f.write(f"Complete:    {len(completed)}\n\n")
        f.write(f"TOP {len(top_trials)} TRIAL(S):\n\n")
        for idx, (trial, name) in enumerate(zip(top_trials, names), 1):
            f.write(
                f"{idx}. {name.upper()}: "
                f"trial={trial.number}, {score_label}={trial.value:.6f}\n"
            )
            for k, v in sorted(trial.params.items()):
                f.write(f"   {k:25s}: {v}\n")
            f.write("\n")
    logger.info("  [OK] summary.txt")

    # ------------------------------------------------------------------
    # Next-step hint
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("\nTrain with best params:")
    logger.info(
        f"  python rf_train.py --data_path {TuningConfig.TRAIN_PATH} "
        f"--load_optuna_params {output_dir}/best_params.json"
    )

    return study, best


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter tuning for Random Forest'
    )
    parser.add_argument('--data_path',          type=str, required=True)
    parser.add_argument('--n_trials',           type=int, default=None)
    parser.add_argument('--timeout',            type=int, default=None,
                        help='Time limit in seconds')
    parser.add_argument('--feature_set',        type=str, default=None)
    parser.add_argument('--custom_features',    type=str, nargs='+', default=None)
    parser.add_argument('--lookback',           type=int, default=None)
    parser.add_argument('--horizon',            type=int, default=None)
    parser.add_argument('--flattening_method',  type=str, default=None,
                        choices=['lag', 'agg', 'hybrid', 'signal', 'none'])
    parser.add_argument('--scale_target',       action='store_true', default=None)
    parser.add_argument('--no_scale_target',    dest='scale_target',
                        action='store_false')
    parser.add_argument('--target_scaler_type', type=str, default=None,
                        choices=['standard', 'robust'])
    parser.add_argument('--tune_lookback',      action='store_true')
    parser.add_argument('--tune_horizon',       action='store_true')
    parser.add_argument('--optimize_target',    type=str, default=None,
                        choices=['val', 'train'])
    parser.add_argument('--enable_pruning',     action='store_true')
    parser.add_argument('--study_name',         type=str, default=None)

    args = parser.parse_args()

    # Apply CLI overrides to TuningConfig
    TuningConfig.TRAIN_PATH = args.data_path

    if args.feature_set:
        TuningConfig.FEATURE_SET        = args.feature_set
    if args.custom_features:
        TuningConfig.CUSTOM_FEATURES    = args.custom_features
    if args.lookback is not None:
        TuningConfig.LOOKBACK           = args.lookback
    if args.horizon is not None:
        TuningConfig.HORIZON            = args.horizon
    if args.scale_target is not None:
        TuningConfig.SCALE_TARGET       = args.scale_target
    if args.target_scaler_type:
        TuningConfig.TARGET_SCALER_TYPE = args.target_scaler_type
    if args.optimize_target:
        TuningConfig.OPTIMIZE_TARGET    = args.optimize_target
    if args.enable_pruning:
        TuningConfig.ENABLE_PRUNING     = True

    TuningConfig.TUNE_LOOKBACK = args.tune_lookback
    TuningConfig.TUNE_HORIZON  = args.tune_horizon

    if args.flattening_method:
        TuningConfig.FIXED_FLATTENING_METHOD = args.flattening_method
        TuningConfig.FLATTENING_METHOD       = args.flattening_method

    try:
        run_tuning(
            n_trials=args.n_trials,
            timeout=args.timeout,
            study_name=args.study_name,
        )
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
