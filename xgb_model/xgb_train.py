"""
XGBoost Price Prediction Training
"""
import os
import warnings
from datetime import datetime

import numpy as np

from xgb_config import XGBConfig
from common.data_utils import load_data
from common.id_utils import get_next_model_id, extract_data_id, detect_symbol
from common.training_utils import (
    set_seed, setup_logger,
    train_model as _train_model,
    save_xgb_model, print_final_summary,
    build_training_config,
)
from common.sample_weighting import (
    compute_sample_weights, make_custom_objective,
    make_dead_zone_objective, resolve_signal_threshold,
)
from common.log_utils import (
    log_run_header, log_train_session_header, log_data_summary,
    log_config_flags, log_signal_threshold, log_hyperparameters,
    log_train_val_metrics,
)

warnings.filterwarnings('ignore')


def train(
        data_path,
        lookback=None,
        horizon=None,
        flattening_method=None,
        feature_set=None,
        custom_features=None,
        n_estimators=None,
        max_depth=None,
        learning_rate=None,
        subsample=None,
        colsample_bytree=None,
        colsample_bylevel=None,
        min_child_weight=None,
        gamma=None,
        reg_alpha=None,
        reg_lambda=None,
        objective=None,
        early_stopping_rounds=None,
        seed=42,
        verbose=True,
        scale_target=None,
        target_scaler_type=None,
        winsorize_target=None,
        winsorize_percentile=None,
        load_optuna_params=None,
        # --- podejscie A: sample weights ---
        sample_weight_mode=None,
        sample_weight_alpha=None,
        sample_weight_min_w=None,
        sample_weight_percentile=None,
        sample_weight_low_w=None,
        # --- podejscie A: misdir penalty ---
        reweight_misdir=None,
        misdir_approach_a_variant=None,
        misdir_penalty=None,
        misdir_min_quantile=None,
        # --- podejscie B: custom objective ---
        use_custom_objective=None,
        custom_obj_sample_weights=None,
        custom_obj_misdir_penalty=None,
        custom_obj_huber_delta=None,
        # --- signal threshold ---
        signal_threshold_enabled=None,
        signal_threshold_mode=None,
        signal_threshold_value=None,
        signal_threshold_percentile=None,
        use_threshold_penalties=None,
        threshold_smooth_transition=None,
        threshold_smooth_width=None,
        custom_obj_type=None,
        dead_zone_weight=None,
        early_stopping_metric=None,
        no_early_stopping=False,
        save_from=None,
        **kwargs,
):
    set_seed(seed)

    # Load Optuna params — override only values not explicitly provided
    if load_optuna_params is not None:
        op, om = XGBConfig.load_optuna_params(load_optuna_params)
        print(f"Loaded Optuna params from: {load_optuna_params}")
        lookback = lookback or op.get('lookback')
        horizon = horizon or op.get('horizon')
        flattening_method = flattening_method or op.get('flattening_method')
        n_estimators = n_estimators or op.get('n_estimators')
        max_depth = max_depth or op.get('max_depth')
        learning_rate = learning_rate or op.get('learning_rate')
        subsample = subsample or op.get('subsample')
        colsample_bytree = colsample_bytree or op.get('colsample_bytree')
        colsample_bylevel = colsample_bylevel or op.get('colsample_bylevel')
        min_child_weight = min_child_weight or op.get('min_child_weight')
        gamma = gamma or op.get('gamma')
        reg_alpha = reg_alpha or op.get('reg_alpha')
        reg_lambda = reg_lambda or op.get('reg_lambda')
        objective = objective or op.get('objective')
        scale_target = scale_target if scale_target is not None else om.get('scale_target')
        target_scaler_type = target_scaler_type or om.get('target_scaler_type')
        feature_set = feature_set or om.get('feature_set')

    # Apply config defaults
    lookback = lookback or XGBConfig.DEFAULT_LOOKBACK
    horizon = horizon or XGBConfig.DEFAULT_HORIZON
    flattening_method = flattening_method or XGBConfig.DEFAULT_FLATTENING
    feature_set = feature_set or XGBConfig.DEFAULT_FEATURE_SET
    early_stopping_rounds = None if no_early_stopping else (early_stopping_rounds or XGBConfig.EARLY_STOPPING_ROUNDS)
    early_stopping_metric = early_stopping_metric or XGBConfig.DEFAULT_EARLY_STOPPING_METRIC
    use_best_iteration = (save_from == 'best') if save_from is not None else XGBConfig.USE_BEST_ITERATION
    scale_target = scale_target if scale_target is not None else XGBConfig.SCALE_TARGET
    target_scaler_type = target_scaler_type or XGBConfig.TARGET_SCALER_TYPE
    winsorize_target = winsorize_target if winsorize_target is not None else XGBConfig.WINSORIZE_TARGET
    winsorize_percentile = winsorize_percentile if winsorize_percentile is not None else XGBConfig.WINSORIZE_PERCENTILE

    # Defaults dla podejscia A
    sample_weight_mode = sample_weight_mode or XGBConfig.SAMPLE_WEIGHT_MODE
    sample_weight_alpha = sample_weight_alpha if sample_weight_alpha is not None else XGBConfig.SAMPLE_WEIGHT_ALPHA
    sample_weight_min_w = sample_weight_min_w if sample_weight_min_w is not None else XGBConfig.SAMPLE_WEIGHT_MIN_W
    sample_weight_percentile = sample_weight_percentile if sample_weight_percentile is not None else XGBConfig.SAMPLE_WEIGHT_PERCENTILE
    sample_weight_low_w = sample_weight_low_w if sample_weight_low_w is not None else XGBConfig.SAMPLE_WEIGHT_LOW_W
    reweight_misdir = reweight_misdir if reweight_misdir is not None else XGBConfig.REWEIGHT_MISDIR
    misdir_approach_a_variant = misdir_approach_a_variant or XGBConfig.MISDIR_APPROACH_A_VARIANT
    misdir_penalty = misdir_penalty if misdir_penalty is not None else XGBConfig.MISDIR_PENALTY
    misdir_min_quantile = misdir_min_quantile if misdir_min_quantile is not None else XGBConfig.MISDIR_MIN_QUANTILE

    # Defaults dla podejscia B
    use_custom_objective = use_custom_objective if use_custom_objective is not None else XGBConfig.USE_CUSTOM_OBJECTIVE
    custom_obj_sample_weights = custom_obj_sample_weights if custom_obj_sample_weights is not None else XGBConfig.CUSTOM_OBJ_SAMPLE_WEIGHTS
    custom_obj_misdir_penalty = custom_obj_misdir_penalty if custom_obj_misdir_penalty is not None else XGBConfig.CUSTOM_OBJ_MISDIR_PENALTY
    custom_obj_huber_delta = custom_obj_huber_delta or XGBConfig.CUSTOM_OBJ_HUBER_DELTA

    # Defaults dla signal threshold
    signal_threshold_enabled = signal_threshold_enabled if signal_threshold_enabled is not None else XGBConfig.SIGNAL_THRESHOLD_ENABLED
    signal_threshold_mode = signal_threshold_mode or XGBConfig.SIGNAL_THRESHOLD_MODE
    signal_threshold_value = signal_threshold_value if signal_threshold_value is not None else XGBConfig.SIGNAL_THRESHOLD_VALUE
    signal_threshold_percentile = signal_threshold_percentile if signal_threshold_percentile is not None else XGBConfig.SIGNAL_THRESHOLD_PERCENTILE
    use_threshold_penalties = use_threshold_penalties if use_threshold_penalties is not None else XGBConfig.USE_THRESHOLD_PENALTIES
    threshold_smooth_transition = threshold_smooth_transition if threshold_smooth_transition is not None else XGBConfig.THRESHOLD_SMOOTH_TRANSITION
    threshold_smooth_width = threshold_smooth_width if threshold_smooth_width is not None else XGBConfig.THRESHOLD_SMOOTH_WIDTH
    custom_obj_type = custom_obj_type or XGBConfig.CUSTOM_OBJ_TYPE
    dead_zone_weight = dead_zone_weight if dead_zone_weight is not None else XGBConfig.DEAD_ZONE_WEIGHT

    # Walidacja -- blokada dublowania podejsc A i B
    XGBConfig.validate_approach_config()

    # Build XGBoost params
    xgb_objective, extra_params = XGBConfig.get_xgb_objective()
    params = {**XGBConfig.DEFAULT_PARAMS, 'objective': xgb_objective, **extra_params}

    overrides = {
        'n_estimators': n_estimators, 'max_depth': max_depth,
        'learning_rate': learning_rate, 'subsample': subsample,
        'colsample_bytree': colsample_bytree, 'colsample_bylevel': colsample_bylevel,
        'min_child_weight': min_child_weight, 'gamma': gamma,
        'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,
    }
    # objective override from Optuna/CLI (already a real XGBoost string, no large-delta handling)
    INCOMPATIBLE_OBJECTIVES = {'reg:squaredlogerror', 'rank:ndcg'}
    if objective is not None:
        if objective in INCOMPATIBLE_OBJECTIVES:
            raise ValueError(
                f"Objective '{objective}' is incompatible with log-return targets.\n"
                f"  reg:squaredlogerror — requires label > -1 (scaled log-returns can be negative)\n"
                f"  rank:ndcg          — requires non-negative integer labels\n"
                f"Use one of: reg:pseudohubererror, reg:squarederror, reg:absoluteerror"
            )
        # 'reg:pseudohubererror_large' is a virtual name — map to real XGBoost objective
        if objective == 'reg:pseudohubererror_large':
            params['objective'] = 'reg:pseudohubererror'
            params['huber_slope'] = XGBConfig.HUBER_SLOPE
        else:
            params['objective'] = objective
    for k, v in overrides.items():
        if v is not None:
            params[k] = v

    log_run_header("XGBOOST TRAINING")

    symbol = detect_symbol(os.path.basename(data_path))

    data, _, metadata = load_data(
        data_path,
        flattening_method=flattening_method,
        feature_set=feature_set,
        custom_features=custom_features,
        lookback=lookback,
        horizon=horizon,
        scale_target=scale_target,
        target_scaler_type=target_scaler_type,
        winsorize_target=winsorize_target,
        winsorize_percentile=winsorize_percentile,
    )

    data_id        = extract_data_id(metadata)
    features       = metadata['features']
    flat_features  = metadata['flat_features']
    lookback       = metadata['lookback']
    horizon        = metadata['horizon']
    flattening_method = metadata['flattening_method']
    clip_bounds    = data.get('clip_bounds')

    X_train        = data['X_train']
    y_train        = data['y_train']
    X_val          = data['X_val']
    y_val          = data['y_val']
    feature_scaler = data['feature_scaler']
    target_scaler  = data.get('target_scaler')

    # Raw y_train (dla threshold resolution i sample weights)
    if target_scaler is not None:
        y_train_raw = target_scaler.inverse_transform(y_train)
    else:
        y_train_raw = y_train

    # Resolve signal threshold (None gdy wyłączony)
    resolved_threshold = None
    if signal_threshold_enabled:
        resolved_threshold = resolve_signal_threshold(
            y_train_raw,
            mode=signal_threshold_mode,
            fixed_value=signal_threshold_value,
            percentile=signal_threshold_percentile,
        )

    # When USE_CUSTOM_OBJECTIVE=True the custom objective already applies the threshold
    # overlay internally — skip it here to avoid double-penalising below-threshold samples
    # (DMatrix weight × custom obj multiplier would compound to e.g. 0.1 × 0.1 = 0.01).
    _dmatrix_use_threshold = (
        use_threshold_penalties
        and resolved_threshold is not None
        and not use_custom_objective
    )
    sample_weight = compute_sample_weights(
        y_train_raw,
        mode=sample_weight_mode,
        alpha=sample_weight_alpha,
        min_w=sample_weight_min_w,
        percentile=sample_weight_percentile,
        low_w=sample_weight_low_w,
        use_threshold=_dmatrix_use_threshold,
        threshold=resolved_threshold or 0.001,
        smooth_transition=threshold_smooth_transition,
        smooth_width=threshold_smooth_width,
        threshold_low_w=dead_zone_weight,
    )

    # Stworz custom objective (podejscie B) jesli aktywna
    custom_objective = None
    if use_custom_objective:
        if custom_obj_type == 'dead_zone' and resolved_threshold is None:
            raise ValueError(
                "CUSTOM_OBJ_TYPE='dead_zone' wymaga aktywnego signal threshold "
                "(SIGNAL_THRESHOLD_ENABLED=True). Dead-zone objective nie moze dzialac "
                "bez progu — ustaw SIGNAL_THRESHOLD_ENABLED=True lub zmien CUSTOM_OBJ_TYPE."
            )
        if custom_obj_type == 'dead_zone':
            custom_objective = make_dead_zone_objective(
                target_scaler=target_scaler,
                signal_threshold=resolved_threshold,
                dead_zone_weight=dead_zone_weight,
                smooth_transition=threshold_smooth_transition,
                smooth_width=threshold_smooth_width,
                use_misdir_penalty=custom_obj_misdir_penalty,
                misdir_penalty=misdir_penalty,
                huber_delta=custom_obj_huber_delta,
            )
        else:  # 'weighted_huber'
            custom_objective = make_custom_objective(
                target_scaler=target_scaler,
                use_sample_weights=custom_obj_sample_weights,
                weight_mode=sample_weight_mode or 'percentile',
                weight_alpha=sample_weight_alpha,
                weight_min_w=sample_weight_min_w,
                weight_percentile=sample_weight_percentile,
                weight_low_w=sample_weight_low_w,
                use_misdir_penalty=custom_obj_misdir_penalty,
                misdir_penalty=misdir_penalty,
                misdir_min_quantile=misdir_min_quantile,
                huber_delta=custom_obj_huber_delta,
                use_threshold=use_threshold_penalties and resolved_threshold is not None,
                threshold=resolved_threshold,
                smooth_transition=threshold_smooth_transition,
                smooth_width=threshold_smooth_width,
                threshold_low_w=dead_zone_weight,
            )

    model_id   = get_next_model_id()
    date_short = datetime.now().strftime('%d%m')
    run_name   = f"{model_id}_{symbol}_xgb_model_{data_id}_{date_short}"
    # Absolute MODELS_DIR: output location must not depend on the working directory
    output_dir = os.path.join(XGBConfig.MODELS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    log_train_session_header(logger, symbol, horizon, model_id, data_id, output_dir)
    log_data_summary(
        features, flat_features, flattening_method, lookback, horizon,
        len(X_train), len(X_val), logger=logger,
    )
    log_config_flags(
        target_scaler, winsorize_target, winsorize_percentile,
        sample_weight_mode, reweight_misdir, misdir_approach_a_variant,
        use_custom_objective, custom_obj_type, custom_obj_sample_weights,
        custom_obj_misdir_penalty, custom_obj_huber_delta, logger=logger,
    )
    log_signal_threshold(
        signal_threshold_enabled, resolved_threshold, signal_threshold_mode,
        use_threshold_penalties, threshold_smooth_transition, threshold_smooth_width,
        clip_bounds=clip_bounds, logger=logger,
    )
    log_hyperparameters(logger, params, early_stopping_rounds, use_best_iteration)

    model, best_iteration, best_score, train_metrics, val_metrics, evals_result = _train_model(
        X_train, y_train, X_val, y_val,
        params, early_stopping_rounds,
        logger, verbose,
        target_scaler=target_scaler,
        use_best_iteration=use_best_iteration,
        sample_weight=sample_weight,
        reweight_misdir=reweight_misdir,
        misdir_approach_a_variant=misdir_approach_a_variant,
        misdir_penalty=misdir_penalty,
        misdir_min_quantile=misdir_min_quantile,
        custom_objective=custom_objective,
        signal_threshold=resolved_threshold,
        early_stopping_metric=early_stopping_metric,
    )

    log_train_val_metrics(logger, train_metrics, val_metrics)

    sw_kwargs = {
        'sample_weight_mode': sample_weight_mode,
        'sample_weight_alpha': sample_weight_alpha if sample_weight_mode else None,
        'sample_weight_min_w': sample_weight_min_w if sample_weight_mode == 'abs_clip' else None,
        'sample_weight_percentile': sample_weight_percentile if sample_weight_mode == 'percentile' else None,
        'sample_weight_low_w': sample_weight_low_w if sample_weight_mode == 'percentile' else None,
        'reweight_misdir': reweight_misdir,
        'misdir_approach_a_variant': misdir_approach_a_variant if reweight_misdir else None,
        'misdir_penalty': misdir_penalty if reweight_misdir else None,
        'misdir_min_quantile': misdir_min_quantile if reweight_misdir else None,
    }
    co_kwargs = {
        'use_custom_objective': use_custom_objective,
        'custom_obj_type': custom_obj_type if use_custom_objective else None,
        'custom_obj_sample_weights': custom_obj_sample_weights if use_custom_objective else None,
        'custom_obj_misdir_penalty': custom_obj_misdir_penalty if use_custom_objective else None,
        'custom_obj_huber_delta': custom_obj_huber_delta if use_custom_objective else None,
        'dead_zone_weight': dead_zone_weight if custom_obj_type == 'dead_zone' else None,
    }
    th_kwargs = {
        'signal_threshold_enabled': signal_threshold_enabled,
        'signal_threshold_mode': signal_threshold_mode if signal_threshold_enabled else None,
        'signal_threshold_value': signal_threshold_value if signal_threshold_enabled else None,
        'signal_threshold_resolved': resolved_threshold,
        'use_threshold_penalties': use_threshold_penalties,
        'threshold_smooth_transition': threshold_smooth_transition if use_threshold_penalties else None,
        'threshold_smooth_width': threshold_smooth_width if use_threshold_penalties else None,
    }
    config = build_training_config(
        model_id=model_id, data_id=data_id, symbol=symbol, features=features,
        flat_features=flat_features, lookback=lookback, horizon=horizon,
        flattening_method=flattening_method, feature_set=feature_set, params=params,
        best_iteration=int(best_iteration) if best_iteration is not None else None,
        best_score=float(best_score) if best_score is not None else None,
        total_rounds=int(model.num_boosted_rounds()),
        save_from='best' if use_best_iteration else 'last',
        early_stopping_metric=early_stopping_metric,
        train_metrics=train_metrics, val_metrics=val_metrics, output_dir=output_dir,
        early_stopping_rounds=early_stopping_rounds, scale_target=scale_target,
        target_scaler_type=target_scaler_type, winsorize_target=winsorize_target,
        winsorize_percentile=winsorize_percentile, clip_bounds=clip_bounds,
        train_path=data_path, feature_scaler=feature_scaler,
        sample_weight_kwargs=sw_kwargs, custom_obj_kwargs=co_kwargs, threshold_kwargs=th_kwargs,
    )

    save_xgb_model(model, feature_scaler, config, output_dir, logger,
                   target_scaler=target_scaler, evals_result=evals_result)
    print_final_summary(model_id, output_dir, train_metrics, val_metrics, logger)

    return model, feature_scaler, config


# Keep backward-compatible alias
train_model = train


def build_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--load_optuna_params', type=str, default=None)
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--flattening_method', type=str, default=None,
                        choices=['lag', 'agg', 'hybrid', 'signal', 'none'])
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--n_estimators', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--subsample', type=float, default=None)
    parser.add_argument('--colsample_bytree', type=float, default=None)
    parser.add_argument('--colsample_bylevel', type=float, default=None)
    parser.add_argument('--min_child_weight', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--reg_alpha', type=float, default=None)
    parser.add_argument('--reg_lambda', type=float, default=None)
    parser.add_argument('--objective', type=str, default=None,
                        choices=['reg:pseudohubererror', 'reg:pseudohubererror_large',
                                 'reg:squarederror', 'reg:absoluteerror'])
    parser.add_argument('--early_stopping_rounds', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale_target', action='store_true', default=None)
    parser.add_argument('--no_scale_target', dest='scale_target', action='store_false')
    parser.add_argument('--target_scaler_type', type=str, default=None, choices=['standard', 'robust'])
    parser.add_argument('--winsorize_target', action='store_true', default=None)
    parser.add_argument('--winsorize_percentile', type=float, default=None)
    parser.add_argument('--sample_weight_mode', type=str, default=None,
                        choices=['abs', 'abs_clip', 'rank', 'percentile'])
    parser.add_argument('--sample_weight_alpha', type=float, default=None)
    parser.add_argument('--sample_weight_min_w', type=float, default=None)
    parser.add_argument('--sample_weight_percentile', type=float, default=None)
    parser.add_argument('--sample_weight_low_w', type=float, default=None)
    parser.add_argument('--reweight_misdir', action='store_true', default=None)
    parser.add_argument('--misdir_approach_a_variant', type=str, default=None, choices=['retrain', 'boost'])
    parser.add_argument('--misdir_penalty', type=float, default=None)
    parser.add_argument('--misdir_min_quantile', type=float, default=None)
    parser.add_argument('--use_custom_objective', action='store_true', default=None)
    parser.add_argument('--custom_obj_sample_weights', action='store_true', default=None)
    parser.add_argument('--no_custom_obj_sample_weights', dest='custom_obj_sample_weights', action='store_false')
    parser.add_argument('--custom_obj_misdir_penalty', action='store_true', default=None)
    parser.add_argument('--no_custom_obj_misdir_penalty', dest='custom_obj_misdir_penalty', action='store_false')
    parser.add_argument('--custom_obj_huber_delta', type=str, default=None)
    parser.add_argument('--signal_threshold_enabled', action='store_true', default=None)
    parser.add_argument('--signal_threshold_mode', type=str, default=None, choices=['fixed', 'empirical'])
    parser.add_argument('--signal_threshold_value', type=float, default=None)
    parser.add_argument('--signal_threshold_percentile', type=float, default=None)
    parser.add_argument('--use_threshold_penalties', action='store_true', default=None)
    parser.add_argument('--threshold_smooth_transition', action='store_true', default=None)
    parser.add_argument('--threshold_smooth_width', type=float, default=None)
    parser.add_argument('--custom_obj_type', type=str, default=None, choices=['weighted_huber', 'dead_zone'])
    parser.add_argument('--dead_zone_weight', type=float, default=None)
    parser.add_argument('--early_stopping_metric', type=str, default=None,
                        choices=['t_rmse', 'rmse', 'pseudo_huber', 'corr', 't_dir_acc', 'none'])
    parser.add_argument('--no_early_stopping', action='store_true', default=False)
    parser.add_argument('--save_from', type=str, default=None, choices=['last', 'best'])
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Parse huber_delta: 'auto' stays as string, otherwise convert to float
    parsed_huber_delta = args.custom_obj_huber_delta
    if parsed_huber_delta is not None and parsed_huber_delta != 'auto':
        try:
            parsed_huber_delta = float(parsed_huber_delta)
        except ValueError:
            raise ValueError(
                f"--custom_obj_huber_delta must be 'auto' or a float, got: {parsed_huber_delta!r}"
            )

    resolved_n_estimators = args.n_estimators if args.n_estimators is not None else args.epochs

    train_kwargs = vars(args).copy()
    train_kwargs['custom_obj_huber_delta'] = parsed_huber_delta
    train_kwargs['n_estimators'] = resolved_n_estimators
    train_kwargs.pop('epochs', None)
    train(**train_kwargs)
