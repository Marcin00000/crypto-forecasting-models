"""
Training utilities for XGBoost models
"""
import os
import json
import logging

import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime

from common.metrics import calculate_metrics, _jsonable
from common.sample_weighting import compute_sample_weights, apply_misdir_penalty, make_custom_objective
from common.log_utils import log_train_model_setup, log_metrics_comparison
from common.eval_metrics import make_custom_eval_metric
from common.id_utils import to_repo_relative


def set_seed(seed=42):
    np.random.seed(seed)


def setup_logger(output_dir, name='xgb_training'):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(output_dir, 'training.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _inverse_if_scaled(y, target_scaler):
    """Inverse transform if scaler provided, else return as-is"""
    if target_scaler is not None:
        return target_scaler.inverse_transform(y)
    return y


def _compute_metrics(y_true_scaled, y_pred_scaled, target_scaler, threshold=None):
    """Compute metrics in original log-return space"""
    y_true = _inverse_if_scaled(y_true_scaled, target_scaler)
    y_pred = _inverse_if_scaled(y_pred_scaled, target_scaler)
    return calculate_metrics(y_true, y_pred, threshold=threshold)


def _run_xgb_train(params, dtrain, dval, early_stopping_rounds, verbose,
                   xgb_model=None, custom_objective=None, custom_eval_metric=None):
    """
    Internal helper: single xgb.train() call.

    xgb_model:          existing Booster to continue from ('boost' variant).
    custom_objective:   callable (y_pred, dtrain) -> (grad, hess) or None.
    custom_eval_metric: callable (preds, dmatrix) -> (name, score) or None.

    Returns: (model, evals_result)
        evals_result: dict with keys 'train' and 'validation', each mapping
                      metric name → list of per-round values.
    """
    evals_list = [(dtrain, 'train'), (dval, 'validation')]
    evals_result = {}

    params_for_train = dict(params)
    if custom_objective is not None:
        params_for_train.pop('objective', None)
    if custom_eval_metric is not None:
        params_for_train['disable_default_eval_metric'] = 1

    effective_es = None if (early_stopping_rounds is None or early_stopping_rounds <= 0) else early_stopping_rounds

    train_kwargs = {
        'params': params_for_train,
        'dtrain': dtrain,
        'num_boost_round': params.get('n_estimators', 1000),
        'evals': evals_list,
        'evals_result': evals_result,
        'early_stopping_rounds': effective_es,
        'verbose_eval': verbose_eval_interval(verbose),
        'xgb_model': xgb_model,
    }
    if custom_objective is not None:
        train_kwargs['obj'] = custom_objective
    if custom_eval_metric is not None:
        train_kwargs['custom_metric'] = custom_eval_metric

    model = xgb.train(**train_kwargs)
    return model, evals_result


def train_model(X_train, y_train, X_val, y_val,
                params, early_stopping_rounds,
                logger, verbose=True, target_scaler=None,
                use_best_iteration=False,
                sample_weight=None,
                reweight_misdir=False,
                misdir_approach_a_variant='retrain',
                misdir_penalty=2.0,
                misdir_min_quantile=0.7,
                custom_objective=None,
                signal_threshold=None,
                early_stopping_metric='t_rmse'):
    """
    Train XGBoost with holdout validation.
    X_train/X_val are separate -- no leakage into early stopping.
    y_train/y_val are already scaled if target_scaler is provided.

    use_best_iteration=False (default): predict and save from last iteration
    use_best_iteration=True:            predict and save from best_iteration

    Early stopping always monitors validation loss (last entry in evals list).

    Returns: (model, best_iteration, best_score, train_metrics, val_metrics, evals_result)
    """
    log_train_model_setup(
        logger, target_scaler, sample_weight, reweight_misdir,
        misdir_approach_a_variant, misdir_penalty, misdir_min_quantile, custom_objective,
    )

    # --- base_score fix for custom objective ---
    if custom_objective is not None and 'base_score' not in params:
        base_score_val = float(np.mean(y_train))
        params = {**params, 'base_score': base_score_val}
        logger.info(f"Auto base_score:  {base_score_val:.8f} (mean of y_train)")

    custom_eval_metric, _ = make_custom_eval_metric(
        early_stopping_metric,
        target_scaler=target_scaler,
        signal_threshold=signal_threshold,
    )
    if custom_eval_metric is not None:
        logger.info(f"Early stopping metric: {early_stopping_metric} (custom eval metric)")

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    # --- Trening 1 ---
    logger.info("Training (fit 1/1)..." if not reweight_misdir else "Training (fit 1/2)...")
    model, evals_result = _run_xgb_train(
        params, dtrain, dval, early_stopping_rounds, verbose,
        custom_objective=custom_objective,
        custom_eval_metric=custom_eval_metric,
    )

    # --- Opcjonalny misdir penalty (drugi trening) ---
    if reweight_misdir:
        # Predykcje z modelu 1 w raw przestrzeni (odwroc skalowanie)
        iter_range_1 = (0, model.num_boosted_rounds())
        y_pred_scaled_1 = model.predict(dtrain, iteration_range=iter_range_1)
        y_raw_1  = _inverse_if_scaled(y_train,          target_scaler)
        y_pred_1 = _inverse_if_scaled(y_pred_scaled_1,  target_scaler)

        updated_weights, n_penalised = apply_misdir_penalty(
            y_raw_1, y_pred_1, sample_weight,
            penalty=misdir_penalty, min_quantile=misdir_min_quantile,
        )
        logger.info(
            f"Misdir penalty: {n_penalised} samples penalised  "
            f"updated weights min={updated_weights.min():.4f}  "
            f"max={updated_weights.max():.4f}"
        )

        dtrain2 = xgb.DMatrix(X_train, label=y_train, weight=updated_weights)

        if misdir_approach_a_variant == 'boost':
            # Kontynuacja od checkpointu -- dodaje nowe drzewa z korekta
            logger.info("Training (fit 2/2) -- boosting from checkpoint...")
            model, evals_result = _run_xgb_train(
                params, dtrain2, dval, early_stopping_rounds, verbose,
                xgb_model=model,
                custom_objective=custom_objective,
                custom_eval_metric=custom_eval_metric,
            )
        else:
            # Pelny restart od zera z zaktualizowanymi wagami
            logger.info("Training (fit 2/2) -- full retrain...")
            model, evals_result = _run_xgb_train(
                params, dtrain2, dval, early_stopping_rounds, verbose,
                custom_objective=custom_objective,
                custom_eval_metric=custom_eval_metric,
            )

    n_total   = model.num_boosted_rounds()
    try:
        best_iter = model.best_iteration
        best_score = model.best_score
    except Exception:
        best_iter = None
        best_score = None

    if use_best_iteration and best_iter is not None:
        iter_range = (0, best_iter + 1)
        saved_at = f"best_iteration ({best_iter})"
    else:
        iter_range = (0, n_total)
        saved_at = f"last_iteration ({n_total - 1})"

    iter_str = str(best_iter) if best_iter is not None else "n/a"
    logger.info(f"Best iteration: {iter_str}, Total rounds: {n_total}, "
                f"Saving from: {saved_at}")

    # Validate X shape before predict
    if X_train.shape[1] != X_val.shape[1]:
        logger.warning(f"[SHAPE WARNING] X_train cols={X_train.shape[1]} != X_val cols={X_val.shape[1]}")

    y_train_pred = model.predict(dtrain, iteration_range=iter_range)
    y_val_pred   = model.predict(dval,   iteration_range=iter_range)

    train_metrics = _compute_metrics(y_train, y_train_pred, target_scaler, threshold=signal_threshold)
    val_metrics   = _compute_metrics(y_val,   y_val_pred,   target_scaler, threshold=signal_threshold)

    # --- Faza 5: diagnostyczny log porównawczy (scaled vs original) ---
    log_metrics_comparison(logger, target_scaler, train_metrics, val_metrics)

    if use_best_iteration and best_iter is not None:
        model.set_attr(saved_iteration=str(best_iter + 1))
    else:
        model.set_attr(saved_iteration=str(n_total))

    return model, best_iter, best_score, train_metrics, val_metrics, evals_result


def verbose_eval_interval(verbose):
    """Return verbose_eval value for xgb.train"""
    if verbose is True:
        return 60
    if isinstance(verbose, int):
        return verbose
    return False


def evaluate_for_tuning(X_train, y_train, X_val, y_val,
                        params, early_stopping_rounds,
                        target_scaler=None,
                        optimize_target='val',
                        optimize_metric='mae',
                        sample_weight=None,
                        custom_objective=None,
                        signal_threshold=None):
    """
    Evaluate hyperparameters for Optuna.
    Always returns metrics in ORIGINAL log-return space (after
    inverse_transform if target_scaler is active).

    Both code paths (custom objective and standard objective) now
    compute metrics identically via _compute_metrics(), ensuring
    consistent Optuna scores regardless of objective type.

    Args:
        optimize_target: 'val' (default) or 'train'
            'val'   -- evaluate on validation set
            'train' -- evaluate on training set
        optimize_metric: 'mae' (default), 'rmse', or 'pseudo_huber'
            Which metric to return as the Optuna score.
            All metrics are computed in original log-return space.
        sample_weight: np.ndarray | None
            Per-sample weights for DMatrix (podejscie A).
            Note: reweight_misdir is NOT applied during tuning.
        custom_objective: callable | None
            Custom objective function (podejscie B).
    """
    if optimize_target not in ('val', 'train'):
        raise ValueError(f"optimize_target must be 'val' or 'train', got: {optimize_target!r}")
    valid_metrics = ('mae', 'rmse', 'pseudo_huber', 'dir_acc',
                     't_mae', 't_rmse', 't_pseudo_huber', 't_dir_acc', 't_r2')
    if optimize_metric not in valid_metrics:
        raise ValueError(f"optimize_metric must be one of {valid_metrics}, got: {optimize_metric!r}")

    # base_score fix for custom objective (same as in train_model)
    if custom_objective is not None and 'base_score' not in params:
        params = {**params, 'base_score': float(np.mean(y_train))}

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    custom_eval, _ = make_custom_eval_metric(
        optimize_metric if optimize_metric in ('t_rmse', 'pseudo_huber', 'corr', 't_dir_acc') else None,
        target_scaler=target_scaler,
        signal_threshold=signal_threshold,
    )
    bst, _ = _run_xgb_train(
        params, dtrain, dval, early_stopping_rounds, verbose=False,
        custom_objective=custom_objective,
        custom_eval_metric=custom_eval,
    )

    # Always compute metrics in original log-return space
    # (identical logic for both custom and standard objective)
    y_pred = bst.predict(dval if optimize_target == 'val' else dtrain)
    y_ref  = y_val if optimize_target == 'val' else y_train
    metrics = _compute_metrics(y_ref, y_pred, target_scaler, threshold=signal_threshold)

    if optimize_metric in metrics:
        score = metrics[optimize_metric]
    else:
        # Thresholded metric requested but threshold not set or insufficient samples
        fallback = optimize_metric.replace('t_', '')
        score = metrics.get(fallback, metrics['mae'])

    # For accuracy/R² metrics, Optuna minimises — invert so higher is better
    if optimize_metric in ('t_dir_acc', 't_r2', 'dir_acc'):
        score = -score

    return score


def save_feature_importance(model, flat_features, output_dir, logger):
    """
    Save feature importance (gain, weight, cover) to feature_importance.json.
    Keys from get_score() may be positional ('f0','f1',...) when DMatrix was
    created without feature names — remap them to actual flat_feature names.
    """
    importance_types = ['gain', 'weight', 'cover']
    result = {}

    for imp_type in importance_types:
        raw_scores = model.get_score(importance_type=imp_type)

        # Remap positional keys (f0, f1, ...) → actual feature names
        remapped = {}
        for k, v in raw_scores.items():
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                name = flat_features[idx] if idx < len(flat_features) else k
            else:
                name = k
            remapped[name] = float(v)

        # Fill missing features with 0.0 and sort by score descending
        full_scores = {f: remapped.get(f, 0.0) for f in flat_features}
        result[imp_type] = dict(
            sorted(full_scores.items(), key=lambda x: x[1], reverse=True)
        )

    # Summary: top-20 by gain
    top20_gain = list(result['gain'].items())[:20]
    result['top20_gain'] = {k: v for k, v in top20_gain}

    importance_path = os.path.join(output_dir, 'feature_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("  ✓ feature_importance.json")
    return result


def save_xgb_model(model, feature_scaler, config_dict, output_dir, logger,
                   target_scaler=None, evals_result=None):
    """Save XGBoost model, scalers, config, feature importance, learning curves and summary"""
    logger.info("\nSAVING MODEL")

    model.save_model(os.path.join(output_dir, 'xgb_model.json'))
    logger.info("  ✓ xgb_model.json")

    joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    logger.info("  ✓ feature_scaler.pkl")

    if target_scaler is not None:
        joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
        config_dict['target_scaler_params'] = target_scaler.get_params()
        logger.info("  ✓ target_scaler.pkl")

    # Feature importance JSON
    flat_features = config_dict.get('flat_features', [])
    if flat_features:
        save_feature_importance(model, flat_features, output_dir, logger)

    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info("  ✓ model_config.json")

    if evals_result is not None:
        curves_path = os.path.join(output_dir, 'learning_curves.json')
        with open(curves_path, 'w', encoding='utf-8') as f:
            json.dump(_jsonable(evals_result), f, indent=2)
        logger.info("  ✓ learning_curves.json")

    val_metrics = config_dict['val_metrics']
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBOOST MODEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model ID:   {config_dict['model_id']}\n")
        f.write(f"Data ID:    {config_dict['data_id']}\n")
        f.write(f"Symbol:     {config_dict['symbol'].upper()}\n")
        f.write(f"Horizon:    t+{config_dict['horizon']}\n")
        f.write(f"Features:   {config_dict['n_features']} → {config_dict['n_flat_features']} flat\n")
        f.write(f"Flattening: {config_dict['flattening_method']}\n")
        f.write(f"Target scaling: {'ENABLED (' + target_scaler.method + ')' if target_scaler else 'DISABLED'}\n")
        f.write(f"Timestamp:  {config_dict['timestamp']}\n\n")
        f.write("HYPERPARAMETERS:\n")
        for key, value in sorted(config_dict['params'].items()):
            f.write(f"  {key:20s}: {value}\n")
        f.write(f"  early_stopping:      {config_dict.get('early_stopping_rounds', 'N/A')}\n\n")
        f.write("TRAIN METRICS (log return):\n")
        train_m = config_dict['train_metrics']
        f.write(f"  MAE:      {train_m['mae']:.6f}\n")
        f.write(f"  RMSE:     {train_m['rmse']:.6f}\n")
        f.write(f"  R²:       {train_m['r2']:.4f}\n")
        f.write(f"  Dir Acc:  {train_m['dir_acc']:.2f}%\n")
        if 't_mae' in train_m:
            f.write(f"\n  THRESHOLDED (|y| ≥ {train_m['t_threshold']:.6f},"
                    f" n={train_m['t_n_samples']}):\n")
            f.write(f"  t_MAE:      {train_m['t_mae']:.6f}\n")
            f.write(f"  t_RMSE:     {train_m['t_rmse']:.6f}\n")
            f.write(f"  t_R²:       {train_m['t_r2']:.4f}\n")
            f.write(f"  t_PH(δ=1):  {train_m['t_pseudo_huber']:.6f}\n")
            f.write(f"  t_PH(δ=10): {train_m['t_pseudo_huber_large']:.6f}\n")
            f.write(f"  t_Dir Acc:  {train_m['t_dir_acc']:.2f}%\n")
        f.write("\n")
        f.write("VALIDATION METRICS (log return):\n")
        f.write(f"  MAE:      {val_metrics['mae']:.6f}\n")
        f.write(f"  RMSE:     {val_metrics['rmse']:.6f}\n")
        f.write(f"  R²:       {val_metrics['r2']:.4f}\n")
        f.write(f"  Dir Acc:  {val_metrics['dir_acc']:.2f}%\n")
        if 't_mae' in val_metrics:
            f.write(f"\n  THRESHOLDED (|y| ≥ {val_metrics['t_threshold']:.6f},"
                    f" n={val_metrics['t_n_samples']}):\n")
            f.write(f"  t_MAE:      {val_metrics['t_mae']:.6f}\n")
            f.write(f"  t_RMSE:     {val_metrics['t_rmse']:.6f}\n")
            f.write(f"  t_R²:       {val_metrics['t_r2']:.4f}\n")
            f.write(f"  t_PH(δ=1):  {val_metrics['t_pseudo_huber']:.6f}\n")
            f.write(f"  t_PH(δ=10): {val_metrics['t_pseudo_huber_large']:.6f}\n")
            f.write(f"  t_Dir Acc:  {val_metrics['t_dir_acc']:.2f}%\n")
    logger.info("  ✓ summary.txt")


def print_final_summary(model_id, output_dir, train_metrics, val_metrics, logger):
    """Print final training summary"""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model ID:      {model_id}")
    logger.info(f"Output:        {output_dir}")
    logger.info(f"Train MAE:     {train_metrics['mae']:.6f}")
    logger.info(f"Train R²:      {train_metrics['r2']:.4f}")
    logger.info(f"Val MAE:       {val_metrics['mae']:.6f}")
    logger.info(f"Val R²:        {val_metrics['r2']:.4f}")
    logger.info(f"Val Dir Acc:   {val_metrics['dir_acc']:.2f}%")
    logger.info("=" * 80)


def build_training_config(
    model_id: str, data_id: str, symbol: str, features: list, flat_features: list,
    lookback: int, horizon: int, flattening_method: str, feature_set: str | None,
    params: dict, best_iteration: int, best_score: float, total_rounds: int,
    save_from: str, early_stopping_metric: str, train_metrics: dict, val_metrics: dict,
    output_dir: str, early_stopping_rounds: int | None, scale_target: bool | None,
    target_scaler_type: str | None, winsorize_target: bool | None,
    winsorize_percentile: float | None, clip_bounds: tuple | None, train_path: str,
    feature_scaler, sample_weight_kwargs: dict, custom_obj_kwargs: dict, threshold_kwargs: dict,
) -> dict:
    """
    Construct standardized model configuration dictionary for model_config.json.

    train_path and output_dir are stored relative to the repository root (see
    to_repo_relative) so a model stays usable after the repo is moved or cloned.
    """
    cols = list(feature_scaler.feature_names_in_) if hasattr(feature_scaler, 'feature_names_in_') else None
    config = {
        'model_id': model_id, 'data_id': data_id, 'symbol': symbol,
        'model_type': 'xgboost', 'target_type': 'log_return',
        'features': features, 'flat_features': flat_features,
        'n_features': len(features), 'n_flat_features': len(flat_features),
        'lookback': lookback, 'horizon': horizon, 'flattening_method': flattening_method,
        'feature_set': feature_set, 'params': params,
        'best_iteration': int(best_iteration) if best_iteration is not None else None,
        'best_score': float(best_score) if best_score is not None else None,
        'total_rounds': int(total_rounds), 'save_from': save_from,
        'early_stopping_metric': early_stopping_metric,
        'train_metrics': train_metrics, 'val_metrics': val_metrics,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'output_dir': to_repo_relative(output_dir),
        'early_stopping_rounds': early_stopping_rounds, 'scale_target': scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target': winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds': clip_bounds, 'train_path': to_repo_relative(train_path),
        'scaler_feature_cols': cols,
    }
    config.update(sample_weight_kwargs)
    config.update(custom_obj_kwargs)
    config.update(threshold_kwargs)
    return config
