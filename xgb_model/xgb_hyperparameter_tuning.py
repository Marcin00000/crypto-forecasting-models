"""
XGBoost Hyperparameter Tuning using Optuna
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

from xgb_config import XGBConfig
from common.data_utils import load_data
from common.id_utils import get_next_optuna_id, extract_data_id_from_path
from common.visualization import plot_optuna_results
from common.sample_weighting import (
    compute_sample_weights, make_custom_objective,
    make_dead_zone_objective, resolve_signal_threshold,
)
from common.log_utils import log_tuning_header

warnings.filterwarnings('ignore')


class TuningConfig:
    TRAIN_PATH      = 'datasets/prepared'
    FEATURE_SET     = XGBConfig.DEFAULT_FEATURE_SET     # 'full'
    CUSTOM_FEATURES = None
    USE_SUBSET      = False
    SUBSET_SIZE     = 200000
    N_TRIALS        = 100
    TIMEOUT         = None
    N_JOBS          = 1
    ENABLE_PRUNING  = True
    PRUNING_WARMUP  = 10
    SEED            = 42
    FLATTENING_METHOD       = XGBConfig.DEFAULT_FLATTENING      # 'agg'
    FIXED_FLATTENING_METHOD = None
    LOOKBACK        = XGBConfig.DEFAULT_LOOKBACK        # 48
    HORIZON         = XGBConfig.DEFAULT_HORIZON         # 12
    TUNE_LOOKBACK   = False
    TUNE_HORIZON    = False
    SCALE_TARGET       = True
    TARGET_SCALER_TYPE = 'robust'
    OPTIMIZE_TARGET    = XGBConfig.OPTUNA_OPTIMIZE_TARGET   # 'val' | 'train'
    # Podejscie A: sample weights (stosowane podczas tuningu)
    SAMPLE_WEIGHT_MODE       = XGBConfig.SAMPLE_WEIGHT_MODE
    SAMPLE_WEIGHT_ALPHA      = XGBConfig.SAMPLE_WEIGHT_ALPHA
    SAMPLE_WEIGHT_MIN_W      = XGBConfig.SAMPLE_WEIGHT_MIN_W
    SAMPLE_WEIGHT_PERCENTILE = XGBConfig.SAMPLE_WEIGHT_PERCENTILE
    SAMPLE_WEIGHT_LOW_W      = XGBConfig.SAMPLE_WEIGHT_LOW_W
    # Uwaga: REWEIGHT_MISDIR nie jest stosowany podczas tuningu --
    # podwoilby czas kazdego trialu. Wagi DMatrix sa stosowane.
    # Podejscie B: custom objective
    USE_CUSTOM_OBJECTIVE      = XGBConfig.USE_CUSTOM_OBJECTIVE
    CUSTOM_OBJ_SAMPLE_WEIGHTS = XGBConfig.CUSTOM_OBJ_SAMPLE_WEIGHTS
    CUSTOM_OBJ_MISDIR_PENALTY = XGBConfig.CUSTOM_OBJ_MISDIR_PENALTY
    CUSTOM_OBJ_HUBER_DELTA    = XGBConfig.CUSTOM_OBJ_HUBER_DELTA
    CUSTOM_OBJ_TYPE           = XGBConfig.CUSTOM_OBJ_TYPE
    # Signal threshold
    SIGNAL_THRESHOLD_ENABLED     = XGBConfig.SIGNAL_THRESHOLD_ENABLED
    SIGNAL_THRESHOLD_MODE        = XGBConfig.SIGNAL_THRESHOLD_MODE
    SIGNAL_THRESHOLD_VALUE       = XGBConfig.SIGNAL_THRESHOLD_VALUE
    SIGNAL_THRESHOLD_PERCENTILE  = XGBConfig.SIGNAL_THRESHOLD_PERCENTILE
    USE_THRESHOLD_PENALTIES      = XGBConfig.USE_THRESHOLD_PENALTIES
    THRESHOLD_SMOOTH_TRANSITION  = XGBConfig.THRESHOLD_SMOOTH_TRANSITION
    THRESHOLD_SMOOTH_WIDTH       = XGBConfig.THRESHOLD_SMOOTH_WIDTH
    DEAD_ZONE_WEIGHT             = XGBConfig.DEAD_ZONE_WEIGHT
    OPTIMIZE_METRIC              = XGBConfig.OPTUNA_OPTIMIZE_METRIC


def suggest_hyperparameters(trial):
    """Suggest hyperparameters for trial"""
    xgb_objective, extra_params = XGBConfig.get_xgb_objective()

    params = {
        'max_depth':          trial.suggest_int('max_depth', 3, 10),
        'min_child_weight':   trial.suggest_int('min_child_weight', 1, 7),
        'gamma':              trial.suggest_float('gamma', 0.0, 2.0),
        'subsample':          trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel':  trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'learning_rate':      trial.suggest_float('learning_rate', 0.00005, 0.5, log=True),
        'n_estimators':       trial.suggest_int('n_estimators', 400, 1000),
        'reg_alpha':          trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda':         trial.suggest_float('reg_lambda', 0.01, 5.0),
        'random_state':       TuningConfig.SEED,
        'n_jobs':             -1,
        'tree_method':        'hist',
        'objective':          xgb_objective,
        **extra_params,
    }

    params['lookback'] = (
        trial.suggest_categorical('lookback', [24, 36, 48, 72, 96, 120])
        if TuningConfig.TUNE_LOOKBACK else TuningConfig.LOOKBACK
    )
    params['horizon'] = (
        trial.suggest_categorical('horizon', [1, 3, 6, 12, 24])
        if TuningConfig.TUNE_HORIZON else TuningConfig.HORIZON
    )

    if TuningConfig.FIXED_FLATTENING_METHOD is not None:
        params['flattening_method'] = TuningConfig.FIXED_FLATTENING_METHOD
    else:
        params['flattening_method'] = trial.suggest_categorical(
            'flattening_method', ['lag', 'agg', 'hybrid', 'signal', 'none']
        )

    return params


def objective(trial):
    """Objective function for Optuna"""
    try:
        params = suggest_hyperparameters(trial)

        data, _, _ = load_data(
            TuningConfig.TRAIN_PATH,
            flattening_method=params.get('flattening_method', TuningConfig.FLATTENING_METHOD),
            feature_set=TuningConfig.FEATURE_SET,
            custom_features=TuningConfig.CUSTOM_FEATURES,
            lookback=params.get('lookback', TuningConfig.LOOKBACK),
            horizon=params.get('horizon', TuningConfig.HORIZON),
            scale_target=TuningConfig.SCALE_TARGET,
            target_scaler_type=TuningConfig.TARGET_SCALER_TYPE,
        )

        X_train = data['X_train']
        y_train = data['y_train']
        X_val   = data['X_val']
        y_val   = data['y_val']
        target_scaler = data.get('target_scaler')

        if len(X_train) < 100 or len(X_val) < 50:
            raise optuna.TrialPruned("Insufficient data")

        # Strip non-XGBoost keys before passing to trainer
        train_params = {k: v for k, v in params.items()
                        if k not in ('lookback', 'horizon', 'flattening_method')}

        # Oblicz wagi (podejscie A) z surowego y_train
        if target_scaler is not None:
            y_train_raw = target_scaler.inverse_transform(y_train)
        else:
            y_train_raw = y_train

        # Resolve signal threshold (None gdy wyĹ‚Ä…czony)
        resolved_threshold = None
        if TuningConfig.SIGNAL_THRESHOLD_ENABLED:
            resolved_threshold = resolve_signal_threshold(
                y_train_raw,
                mode=TuningConfig.SIGNAL_THRESHOLD_MODE,
                fixed_value=TuningConfig.SIGNAL_THRESHOLD_VALUE,
                percentile=TuningConfig.SIGNAL_THRESHOLD_PERCENTILE,
            )

        # Skip threshold overlay in DMatrix weights when custom objective is active â€”
        # it already applies the overlay internally (avoids double-penalising).
        _dmatrix_use_threshold = (
            TuningConfig.USE_THRESHOLD_PENALTIES
            and resolved_threshold is not None
            and not TuningConfig.USE_CUSTOM_OBJECTIVE
        )
        sample_weight = compute_sample_weights(
            y_train_raw,
            mode=TuningConfig.SAMPLE_WEIGHT_MODE,
            alpha=TuningConfig.SAMPLE_WEIGHT_ALPHA,
            min_w=TuningConfig.SAMPLE_WEIGHT_MIN_W,
            percentile=TuningConfig.SAMPLE_WEIGHT_PERCENTILE,
            low_w=TuningConfig.SAMPLE_WEIGHT_LOW_W,
            use_threshold=_dmatrix_use_threshold,
            threshold=resolved_threshold or 0.001,
            smooth_transition=TuningConfig.THRESHOLD_SMOOTH_TRANSITION,
            smooth_width=TuningConfig.THRESHOLD_SMOOTH_WIDTH,
            threshold_low_w=TuningConfig.DEAD_ZONE_WEIGHT,
        ) if TuningConfig.SAMPLE_WEIGHT_MODE is not None or _dmatrix_use_threshold else None

        # Stworz custom objective (podejscie B) jesli aktywna
        custom_objective = None
        if TuningConfig.USE_CUSTOM_OBJECTIVE:
            if TuningConfig.CUSTOM_OBJ_TYPE == 'dead_zone':
                custom_objective = make_dead_zone_objective(
                    target_scaler=target_scaler,
                    signal_threshold=resolved_threshold,
                    dead_zone_weight=TuningConfig.DEAD_ZONE_WEIGHT,
                    smooth_transition=TuningConfig.THRESHOLD_SMOOTH_TRANSITION,
                    smooth_width=TuningConfig.THRESHOLD_SMOOTH_WIDTH,
                    use_misdir_penalty=TuningConfig.CUSTOM_OBJ_MISDIR_PENALTY,
                    misdir_penalty=XGBConfig.MISDIR_PENALTY,
                    huber_delta=TuningConfig.CUSTOM_OBJ_HUBER_DELTA,
                )
            else:  # 'weighted_huber'
                custom_objective = make_custom_objective(
                    target_scaler=target_scaler,
                    use_sample_weights=TuningConfig.CUSTOM_OBJ_SAMPLE_WEIGHTS,
                    weight_mode=TuningConfig.SAMPLE_WEIGHT_MODE or 'percentile',
                    weight_alpha=TuningConfig.SAMPLE_WEIGHT_ALPHA,
                    weight_min_w=TuningConfig.SAMPLE_WEIGHT_MIN_W,
                    weight_percentile=TuningConfig.SAMPLE_WEIGHT_PERCENTILE,
                    weight_low_w=TuningConfig.SAMPLE_WEIGHT_LOW_W,
                    use_misdir_penalty=TuningConfig.CUSTOM_OBJ_MISDIR_PENALTY,
                    misdir_penalty=XGBConfig.MISDIR_PENALTY,
                    misdir_min_quantile=XGBConfig.MISDIR_MIN_QUANTILE,
                    huber_delta=TuningConfig.CUSTOM_OBJ_HUBER_DELTA,
                    use_threshold=TuningConfig.USE_THRESHOLD_PENALTIES and resolved_threshold is not None,
                    threshold=resolved_threshold,
                    smooth_transition=TuningConfig.THRESHOLD_SMOOTH_TRANSITION,
                    smooth_width=TuningConfig.THRESHOLD_SMOOTH_WIDTH,
                    threshold_low_w=TuningConfig.DEAD_ZONE_WEIGHT,
                )

        from common.training_utils import evaluate_for_tuning
        score = evaluate_for_tuning(
            X_train, y_train, X_val, y_val,
            train_params,
            early_stopping_rounds=XGBConfig.EARLY_STOPPING_ROUNDS,
            target_scaler=target_scaler,
            optimize_target=TuningConfig.OPTIMIZE_TARGET,
            optimize_metric=TuningConfig.OPTIMIZE_METRIC,
            sample_weight=sample_weight,
            custom_objective=custom_objective,
            signal_threshold=resolved_threshold,
        )

        return score

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()


def run_tuning(n_trials=None, timeout=None, study_name=None):
    """Run Optuna hyperparameter tuning"""
    if n_trials is None:
        n_trials = TuningConfig.N_TRIALS
    if timeout is None:
        timeout = TuningConfig.TIMEOUT

    # Guard: dead_zone objective requires signal threshold to be enabled
    if (TuningConfig.USE_CUSTOM_OBJECTIVE
            and TuningConfig.CUSTOM_OBJ_TYPE == 'dead_zone'
            and not TuningConfig.SIGNAL_THRESHOLD_ENABLED):
        raise ValueError(
            "CUSTOM_OBJ_TYPE='dead_zone' wymaga aktywnego signal threshold "
            "(SIGNAL_THRESHOLD_ENABLED=True). Ustaw SIGNAL_THRESHOLD_ENABLED=True "
            "lub zmien CUSTOM_OBJ_TYPE."
        )

    data_id = extract_data_id_from_path(TuningConfig.TRAIN_PATH)
    os.makedirs(XGBConfig.LOGS_DIR, exist_ok=True)

    optuna_id  = get_next_optuna_id(XGBConfig.LOGS_DIR)
    date_short = datetime.now().strftime('%d%m')

    if study_name is None:
        study_name = f"{optuna_id}_xgb_optuna_{data_id}_{date_short}"

    output_dir = os.path.join(XGBConfig.LOGS_DIR, study_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'tuning.log'), encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    xgb_objective, _ = XGBConfig.get_xgb_objective()

    log_tuning_header(
        logger, optuna_id, data_id, output_dir, n_trials, xgb_objective,
        XGBConfig, TuningConfig,
    )

    sampler = TPESampler(seed=TuningConfig.SEED)
    pruner  = (MedianPruner(n_startup_trials=5, n_warmup_steps=TuningConfig.PRUNING_WARMUP)
               if TuningConfig.ENABLE_PRUNING else None)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=TuningConfig.N_JOBS,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]

    logger.info(f"\nTotal: {len(study.trials)}, Complete: {len(completed)}, Pruned: {len(pruned)}")

    if not completed:
        logger.error("No trials completed!")
        return None, None

    best = study.best_trial
    optimize_metric = TuningConfig.OPTIMIZE_METRIC
    score_label = f"{TuningConfig.OPTIMIZE_TARGET.upper()} {optimize_metric}"
    # t_dir_acc and t_r2 are negated before passing to Optuna (higher=better â†’ minimise negative)
    _is_negated = optimize_metric in ('t_dir_acc', 't_r2', 'dir_acc')
    display_value = -best.value if _is_negated else best.value
    logger.info(f"\nBest trial {best.number}: {score_label}={display_value:.6f}"
                + (" (stored negated for Optuna minimisation)" if _is_negated else ""))
    for k, v in sorted(best.params.items()):
        logger.info(f"  {k:20s}: {v}")

    # Save top-3 results
    sorted_trials = sorted(completed, key=lambda t: t.value)
    top_trials    = sorted_trials[:min(3, len(sorted_trials))]
    names         = ['best', 'second_best', 'third_best']

    for trial, name in zip(top_trials, names):
        params_to_save = trial.params.copy()
        if 'horizon' not in params_to_save:
            params_to_save['horizon'] = TuningConfig.HORIZON
        if 'lookback' not in params_to_save:
            params_to_save['lookback'] = TuningConfig.LOOKBACK
        if 'flattening_method' not in params_to_save:
            params_to_save['flattening_method'] = (
                TuningConfig.FIXED_FLATTENING_METHOD or TuningConfig.FLATTENING_METHOD
            )

        _trial_display = -trial.value if _is_negated else trial.value
        params_to_save['_metadata'] = {
            'optuna_id':          optuna_id,
            'data_id':            data_id,
            'best_value':         float(trial.value),
            'best_value_display': float(_trial_display),
            'best_value_meaning': (
                f"{TuningConfig.OPTIMIZE_TARGET}_{optimize_metric}"
                + (" (negated â€” higher=better)" if _is_negated else "")
            ),
            'trial_number':       trial.number,
            'timestamp':          datetime.now().strftime('%Y%m%d_%H%M%S'),
            'feature_set':        TuningConfig.FEATURE_SET,
            'custom_features':    TuningConfig.CUSTOM_FEATURES,
            'scale_target':       TuningConfig.SCALE_TARGET,
            'target_scaler_type': TuningConfig.TARGET_SCALER_TYPE,
            'optimize_target':    TuningConfig.OPTIMIZE_TARGET,
            'optimize_metric':    optimize_metric,
            'xgb_objective':      xgb_objective,
        }

        with open(os.path.join(output_dir, f'{name}_params.json'), 'w') as f:
            json.dump(params_to_save, f, indent=2)
        logger.info(f"  [OK] {name}_params.json ({score_label}: {_trial_display:.6f})")

    import joblib
    joblib.dump(study, os.path.join(output_dir, 'study.pkl'))
    study.trials_dataframe().to_csv(os.path.join(output_dir, 'trials.csv'), index=False)
    logger.info("  [OK] study.pkl, trials.csv")

    plot_optuna_results(study, output_dir)

    huber_note = f" (huber_slope={XGBConfig.HUBER_SLOPE})" if XGBConfig.OBJECTIVE == 'reg:pseudohubererror_large' else ""
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBOOST OPTUNA TUNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Optuna ID:  {optuna_id}\n")
        f.write(f"Data ID:    {data_id}\n")
        f.write(f"Objective:  {xgb_objective}{huber_note}\n")
        f.write(
            f"Optimises:  {TuningConfig.OPTIMIZE_TARGET.upper()} {optimize_metric}"
            + (" (maximise via negation)" if _is_negated else " (minimise)") + "\n"
        )
        f.write(f"Feature set: {TuningConfig.FEATURE_SET}\n")
        f.write(f"Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total:      {len(study.trials)}\n")
        f.write(f"Complete:   {len(completed)}\n\n")
        f.write("TOP 3 TRIALS:\n\n")
        for idx, (trial, name) in enumerate(zip(top_trials, names), 1):
            _dv = -trial.value if _is_negated else trial.value
            f.write(f"{idx}. {name.upper()}: trial={trial.number}, {score_label}={_dv:.6f}\n")
            for k, v in sorted(trial.params.items()):
                f.write(f"   {k:20s}: {v}\n")
            f.write("\n")

    logger.info(f"\nNext step:")
    logger.info(f"  python xgb_train.py --data_path {TuningConfig.TRAIN_PATH} "
                f"--load_optuna_params {output_dir}/best_params.json")

    return study, best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--custom_features', type=str, nargs='+', default=None)
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--flattening_method', type=str, default=None,
                        choices=['lag', 'agg', 'hybrid', 'signal', 'none'])
    parser.add_argument('--scale_target', action='store_true', default=None)
    parser.add_argument('--no_scale_target', dest='scale_target', action='store_false')
    parser.add_argument('--target_scaler_type', type=str, default=None,
                        choices=['standard', 'robust'])
    parser.add_argument('--tune_lookback', action='store_true')
    parser.add_argument('--tune_horizon', action='store_true')
    parser.add_argument('--optimize_target', type=str, default=None,
                        choices=['val', 'train'],
                        help="Which dataset loss Optuna minimises: 'val' (default) or 'train'")
    parser.add_argument('--optimize_metric', type=str, default=None,
                        choices=['mae', 'rmse', 'pseudo_huber', 'dir_acc',
                                 't_mae', 't_rmse', 't_pseudo_huber', 't_dir_acc', 't_r2'],
                        help='Metric to optimise (default: mae)')
    parser.add_argument('--signal_threshold_mode', type=str, default=None,
                        choices=['fixed', 'empirical'])
    parser.add_argument('--signal_threshold_value', type=float, default=None)
    parser.add_argument('--use_threshold_penalties', action='store_true', default=None)
    parser.add_argument('--custom_obj_type', type=str, default=None,
                        choices=['weighted_huber', 'dead_zone'])
    parser.add_argument('--dead_zone_weight', type=float, default=None)

    args = parser.parse_args()

    TuningConfig.TRAIN_PATH = args.data_path
    if args.feature_set:        TuningConfig.FEATURE_SET = args.feature_set
    if args.custom_features:    TuningConfig.CUSTOM_FEATURES = args.custom_features
    if args.lookback is not None:   TuningConfig.LOOKBACK = args.lookback
    if args.horizon is not None:    TuningConfig.HORIZON  = args.horizon
    if args.scale_target is not None:   TuningConfig.SCALE_TARGET = args.scale_target
    if args.target_scaler_type:         TuningConfig.TARGET_SCALER_TYPE = args.target_scaler_type
    TuningConfig.TUNE_LOOKBACK = args.tune_lookback
    TuningConfig.TUNE_HORIZON  = args.tune_horizon
    if args.optimize_target:    TuningConfig.OPTIMIZE_TARGET = args.optimize_target
    if args.optimize_metric:    TuningConfig.OPTIMIZE_METRIC = args.optimize_metric
    if args.signal_threshold_mode:     TuningConfig.SIGNAL_THRESHOLD_MODE    = args.signal_threshold_mode
    if args.signal_threshold_value is not None: TuningConfig.SIGNAL_THRESHOLD_VALUE = args.signal_threshold_value
    if args.use_threshold_penalties:   TuningConfig.USE_THRESHOLD_PENALTIES  = True
    if args.custom_obj_type:           TuningConfig.CUSTOM_OBJ_TYPE          = args.custom_obj_type
    if args.dead_zone_weight is not None: TuningConfig.DEAD_ZONE_WEIGHT      = args.dead_zone_weight

    if args.flattening_method:
        TuningConfig.FIXED_FLATTENING_METHOD = args.flattening_method
        TuningConfig.FLATTENING_METHOD       = args.flattening_method

    try:
        run_tuning(n_trials=args.n_trials, timeout=args.timeout)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
