"""
Training utilities for Random Forest models
"""
import os
import json
import logging

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common.metrics import calculate_metrics
from common.id_utils import to_repo_relative


def set_seed(seed=42):
    np.random.seed(seed)


def compute_sample_weights(y_raw, mode, alpha=1.0, min_w=0.1,
                           percentile=70.0, low_w=0.1):
    """
    Compute per-sample weights from raw (unscaled) log-return targets.

    Args:
        y_raw:      np.ndarray — raw log-returns (before target scaling).
        mode:       str | None — weighting scheme (see RFConfig docs).
        alpha:      float — exponent for 'abs' / 'abs_clip' modes.
        min_w:      float — floor fraction of max weight for 'abs_clip' mode.
        percentile: float — threshold percentile for 'percentile' mode.
        low_w:      float — weight for below-threshold samples in
                    'percentile' mode.

    Returns:
        np.ndarray of shape (n_samples,) with positive weights, or
        None if mode is None.
    """
    if mode is None or str(mode).lower() in ('none', 'uniform'):
        return None

    abs_y = np.abs(y_raw).astype(np.float64)

    if mode == 'abs':
        w = abs_y ** alpha

    elif mode == 'abs_clip':
        raw = abs_y ** alpha
        floor = min_w * raw.max() if raw.max() > 0 else 1.0
        w = np.clip(raw, floor, None)

    elif mode == 'rank':
        # rank by |y|, normalised to [1/N, 1]
        order = np.argsort(abs_y)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(abs_y) + 1)
        w = ranks / len(abs_y)

    elif mode == 'percentile':
        threshold = np.percentile(abs_y, percentile)
        w = np.where(abs_y >= threshold, 1.0, low_w)

    else:
        raise ValueError(
            f"Unknown sample_weight_mode: '{mode}'. "
            f"Valid: None, 'none', 'uniform', 'abs', 'abs_clip', 'rank', 'percentile'"
        )

    # Guard: replace any zero / negative weights with a tiny positive value
    w = np.where(w <= 0, 1e-9, w)
    return w.astype(np.float32)


def setup_logger(output_dir, name='rf_training'):
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


def _compute_metrics(y_true_scaled, y_pred_scaled, target_scaler):
    """Compute metrics in original log-return space"""
    y_true = _inverse_if_scaled(y_true_scaled, target_scaler)
    y_pred = _inverse_if_scaled(y_pred_scaled, target_scaler)
    return calculate_metrics(y_true, y_pred)


def apply_misdir_penalty(y_raw, y_pred_raw, weights, penalty, min_quantile):
    """
    Apply a constant multiplier to weights of samples that were mis-directed
    on large moves by a preliminary model fit.

    "Mis-directed" means sign(y_pred_raw) != sign(y_raw) for samples whose
    |y_raw| >= the min_quantile-th percentile of |y_raw|.

    The penalty is a plain multiplier (>= 1.0), so existing weights are only
    ever increased — never zeroed out.  This keeps the model from collapsing
    to a near-zero predictor.

    Args:
        y_raw:        np.ndarray — raw (unscaled) training targets.
        y_pred_raw:   np.ndarray — preliminary predictions in raw target space.
        weights:      np.ndarray — current per-sample weights (positive).
                      If None, starts from uniform weights of 1.0.
        penalty:      float — multiplier for mis-directed large-move samples.
                      Recommended: 2.0.  Values < 1.0 are clamped to 1.0.
        min_quantile: float in [0, 1) — |y_raw| quantile threshold.
                      Only samples at or above this quantile are eligible.
                      Default 0.7 = top 30% by |return|.

    Returns:
        np.ndarray of updated weights (same shape as input weights).
    """
    penalty = max(1.0, float(penalty))   # guard: penalty must be >= 1

    if weights is None:
        weights = np.ones(len(y_raw), dtype=np.float32)
    else:
        weights = weights.copy().astype(np.float64)

    abs_y     = np.abs(y_raw)
    threshold = np.quantile(abs_y, min_quantile)

    large_move   = abs_y >= threshold                     # large-move mask
    wrong_dir    = np.sign(y_pred_raw) != np.sign(y_raw)  # mis-directed mask
    penalty_mask = large_move & wrong_dir

    weights[penalty_mask] *= penalty
    n_penalised = int(np.sum(penalty_mask))
    return weights.astype(np.float32), n_penalised


def train_model(X_train, y_train, X_val, y_val,
                params, logger, verbose=True, target_scaler=None,
                sample_weight=None,
                reweight_misdir=False, misdir_penalty=2.0,
                misdir_min_quantile=0.7):
    """
    Train RandomForestRegressor.

    No early stopping -- RF trains all n_estimators trees.
    OOB score (if bootstrap=True and oob_score=True) is logged as a
    training-set quality indicator.

    Args:
        sample_weight:        np.ndarray | None -- per-sample weights for fit().
            Must have shape (len(X_train),). Computed externally via
            compute_sample_weights() from raw (unscaled) y_train.
        reweight_misdir:      bool -- if True, perform a second fit after
            applying a mis-direction penalty to large-move errors from the
            first fit.  Controlled by RFConfig.REWEIGHT_MISDIR.
        misdir_penalty:       float -- constant weight multiplier for
            mis-directed large-move samples (RFConfig.MISDIR_PENALTY).
        misdir_min_quantile:  float -- |y| quantile threshold to qualify
            for the penalty (RFConfig.MISDIR_MIN_QUANTILE).

    Returns: (model, train_metrics, val_metrics)
    """
    if target_scaler is not None:
        logger.info(f"Target scaling: ENABLED ({target_scaler.method})")

    if sample_weight is not None:
        logger.info(
            f"Sample weights: ENABLED  "
            f"min={sample_weight.min():.4f}  "
            f"max={sample_weight.max():.4f}  "
            f"mean={sample_weight.mean():.4f}"
        )
    else:
        logger.info("Sample weights: DISABLED (uniform)")

    if reweight_misdir:
        logger.info(
            f"Misdir penalty:  ENABLED  "
            f"penalty={misdir_penalty:.2f}  "
            f"min_quantile={misdir_min_quantile:.2f}"
        )
    else:
        logger.info("Misdir penalty:  DISABLED")

    # Map verbose int to sklearn verbose level
    # sklearn verbose: 0=silent, 1=progress, 2=per-tree
    sklearn_verbose = verbose if isinstance(verbose, int) else (1 if verbose else 0)

    model = RandomForestRegressor(verbose=sklearn_verbose, **params)

    logger.info(f"\nFitting RandomForest: n_estimators={params.get('n_estimators')}, "
                f"max_depth={params.get('max_depth')}, "
                f"criterion={params.get('criterion', 'squared_error')}")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # --- First fit ---
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("Fit 1/1 complete." if not reweight_misdir else "Fit 1/2 complete.")

    # --- Optional misdir-penalty second fit ---
    if reweight_misdir:
        # Predict in scaled space, then invert for raw-target penalty computation
        y_pred_scaled_1 = model.predict(X_train)
        y_raw_1   = _inverse_if_scaled(y_train,          target_scaler)
        y_pred_1  = _inverse_if_scaled(y_pred_scaled_1,  target_scaler)

        updated_weights, n_penalised = apply_misdir_penalty(
            y_raw_1, y_pred_1,
            sample_weight,
            penalty=misdir_penalty,
            min_quantile=misdir_min_quantile,
        )
        logger.info(
            f"Misdir penalty applied: {n_penalised} samples penalised  "
            f"new weights min={updated_weights.min():.4f}  "
            f"max={updated_weights.max():.4f}"
        )

        # Recreate model (same params) and re-fit with updated weights
        model = RandomForestRegressor(verbose=sklearn_verbose, **params)
        model.fit(X_train, y_train, sample_weight=updated_weights)
        logger.info("Fit 2/2 complete.")

    # OOB score (R2 on out-of-bag samples -- training-set proxy, no leakage)
    if params.get('oob_score') and params.get('bootstrap', True):
        logger.info(f"OOB R2 (training proxy): {model.oob_score_:.4f}")

    y_train_pred = model.predict(X_train)
    y_val_pred   = model.predict(X_val)

    train_metrics = _compute_metrics(y_train, y_train_pred, target_scaler)
    val_metrics   = _compute_metrics(y_val,   y_val_pred,   target_scaler)

    return model, train_metrics, val_metrics


# Valid optimize_metric values and their directions (all minimised by Optuna)
_VALID_OPTIMIZE_METRICS = {
    'mae',          # Mean Absolute Error                  (lower = better)
    'dir_acc',      # Directional accuracy  -> 1 - dir_acc (higher dir = lower score)
    'composite',    # Weighted MAE + (1 - dir_acc)         (lower = better)
    'sharpe',       # MAE / std(errors)                    (lower = better)
}


def compute_tuning_score(y_ref, y_pred, optimize_metric,
                         composite_mae_weight=0.7):
    """
    Compute the scalar score that Optuna minimises.

    All metrics are expressed as "lower is better" so Optuna always minimises.

    Args:
        y_ref:               np.ndarray -- true log-returns (original scale).
        y_pred:              np.ndarray -- predicted log-returns (original scale).
        optimize_metric:     str -- one of _VALID_OPTIMIZE_METRICS.
        composite_mae_weight: float -- weight of MAE in composite score
                             (dir_acc weight = 1 - composite_mae_weight).

    Returns:
        float -- scalar score to minimise.
    """
    if optimize_metric not in _VALID_OPTIMIZE_METRICS:
        raise ValueError(
            f"Unknown optimize_metric: '{optimize_metric}'. "
            f"Valid: {sorted(_VALID_OPTIMIZE_METRICS)}"
        )

    if optimize_metric == 'mae':
        return float(mean_absolute_error(y_ref, y_pred))

    # Directional accuracy: sign(y_pred) == sign(y_ref)
    actual_dir = np.sign(y_ref)
    pred_dir   = np.sign(y_pred)
    dir_acc    = float(np.mean(actual_dir == pred_dir))   # 0..1

    if optimize_metric == 'dir_acc':
        # Minimise 1 - dir_acc  (so higher accuracy = lower score)
        return 1.0 - dir_acc

    mae = float(mean_absolute_error(y_ref, y_pred))

    if optimize_metric == 'composite':
        # Normalise MAE to a 0..1-ish range by dividing by std of y_ref
        # so both terms are on comparable scales before weighting.
        mae_std   = float(np.std(y_ref)) or 1e-9
        mae_norm  = mae / mae_std
        w_mae     = composite_mae_weight
        w_dir     = 1.0 - composite_mae_weight
        return float(w_mae * mae_norm + w_dir * (1.0 - dir_acc))

    if optimize_metric == 'sharpe':
        # Sharpe-like: MAE / std(errors)  -- penalises noisy predictors
        # Lower MAE and lower error variance both improve the score.
        errors    = y_pred - y_ref
        err_std   = float(np.std(errors)) or 1e-9
        return float(mae / err_std)


def evaluate_for_tuning(X_train, y_train, X_val, y_val,
                        params, target_scaler=None,
                        optimize_target='val',
                        optimize_metric='mae',
                        composite_mae_weight=0.7,
                        sample_weight=None):
    """
    Evaluate hyperparameters for Optuna.
    Returns a scalar score (lower = better) on the chosen dataset.

    Args:
        optimize_target:      'val' (default) or 'train'
        optimize_metric:      'mae' | 'dir_acc' | 'composite' | 'sharpe'
        composite_mae_weight: float in (0, 1) -- MAE weight in composite score
        sample_weight:        np.ndarray | None -- per-sample weights for fit().
    """
    if optimize_target not in ('val', 'train'):
        raise ValueError(
            f"optimize_target must be 'val' or 'train', got: {optimize_target!r}"
        )

    model = RandomForestRegressor(verbose=0, **params)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    if optimize_target == 'val':
        y_pred_scaled = model.predict(X_val)
        y_ref_scaled  = y_val
    else:
        y_pred_scaled = model.predict(X_train)
        y_ref_scaled  = y_train

    y_pred = _inverse_if_scaled(y_pred_scaled, target_scaler)
    y_ref  = _inverse_if_scaled(y_ref_scaled,  target_scaler)

    return compute_tuning_score(y_ref, y_pred, optimize_metric,
                                composite_mae_weight)


def save_feature_importance(model, flat_features, output_dir, logger, base_features=None):
    """
    Save feature importance (MDI / mean decrease in impurity) to
    feature_importance.json.
    """
    importance = model.feature_importances_   # shape: (n_flat_features,)

    scores = {f: float(v) for f, v in zip(flat_features, importance)}
    scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    top20 = dict(list(scores_sorted.items())[:20])

    result = {
        'mdi':      scores_sorted,   # Mean Decrease in Impurity (all features)
        'top20_mdi': top20,
    }

    if base_features:
        from .feature_importance import aggregate_mdi_to_base
        result['base_mdi'] = aggregate_mdi_to_base(scores, base_features)

    importance_path = os.path.join(output_dir, 'feature_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("  [OK] feature_importance.json")
    return result


def save_rf_model(model, feature_scaler, config_dict, output_dir, logger,
                  target_scaler=None, sample_weight_params=None):
    """Save RF model, scalers, config, feature importance and summary"""
    logger.info("\nSAVING MODEL")

    config_dict['data_alignment_version'] = 2
    # Repo-relative so the model still finds its dataset after the repository is moved.
    if config_dict.get('train_path'):
        config_dict['train_path'] = to_repo_relative(config_dict['train_path'])

    # RF model -- joblib (no native .json format like XGBoost)
    joblib.dump(model, os.path.join(output_dir, 'rf_model.pkl'))
    logger.info("  [OK] rf_model.pkl")

    joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    logger.info("  [OK] feature_scaler.pkl")

    if target_scaler is not None:
        joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
        config_dict['target_scaler_params'] = target_scaler.get_params()
        logger.info("  [OK] target_scaler.pkl")

    flat_features = config_dict.get('flat_features', [])
    base_features = config_dict.get('features', [])
    if flat_features:
        save_feature_importance(model, flat_features, output_dir, logger, base_features=base_features)

    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info("  [OK] model_config.json")

    val_metrics   = config_dict['val_metrics']
    train_metrics = config_dict['train_metrics']
    target_scaler_info = (
        'ENABLED (' + target_scaler.method + ')'
        if target_scaler else 'DISABLED'
    )

    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RANDOM FOREST MODEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model ID:   {config_dict['model_id']}\n")
        f.write(f"Data ID:    {config_dict['data_id']}\n")
        f.write(f"Symbol:     {config_dict['symbol'].upper()}\n")
        f.write(f"Horizon:    t+{config_dict['horizon']}\n")
        f.write(f"Features:   {config_dict['n_features']} -> {config_dict['n_flat_features']} flat\n")
        f.write(f"Flattening: {config_dict['flattening_method']}\n")
        f.write(f"Criterion:       {config_dict['params'].get('criterion', 'N/A')}\n")
        f.write(f"Target scaling:  {target_scaler_info}\n")
        oob = config_dict.get('oob_score')
        if oob is not None:
            f.write(f"OOB R2:          {oob:.4f}\n")
        sw = config_dict.get('sample_weight_params')
        if sw and sw.get('mode') is not None:
            f.write(f"Sample weights:  mode={sw['mode']}")
            if sw['mode'] in ('abs', 'abs_clip'):
                f.write(f", alpha={sw['alpha']}")
            if sw['mode'] == 'abs_clip':
                f.write(f", min_w={sw['min_w']}")
            if sw['mode'] == 'percentile':
                f.write(f", pct={sw['percentile']}, low_w={sw['low_w']}")
            f.write("\n")
        else:
            f.write("Sample weights:  DISABLED\n")
        f.write(f"Timestamp:  {config_dict['timestamp']}\n\n")
        f.write("HYPERPARAMETERS:\n")
        for key, value in sorted(config_dict['params'].items()):
            f.write(f"  {key:25s}: {value}\n")
        f.write("\nTRAIN METRICS (log return):\n")
        f.write(f"  MAE:      {train_metrics['mae']:.6f}\n")
        f.write(f"  RMSE:     {train_metrics['rmse']:.6f}\n")
        f.write(f"  R2:       {train_metrics['r2']:.4f}\n")
        f.write(f"  Dir Acc:  {train_metrics['dir_acc']:.2f}%\n\n")
        f.write("VALIDATION METRICS (log return):\n")
        f.write(f"  MAE:      {val_metrics['mae']:.6f}\n")
        f.write(f"  RMSE:     {val_metrics['rmse']:.6f}\n")
        f.write(f"  R2:       {val_metrics['r2']:.4f}\n")
        f.write(f"  Dir Acc:  {val_metrics['dir_acc']:.2f}%\n")
    logger.info("  [OK] summary.txt")


def print_final_summary(model_id, output_dir, train_metrics, val_metrics, logger):
    """Print final training summary"""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model ID:      {model_id}")
    logger.info(f"Output:        {output_dir}")
    logger.info(f"Train MAE:     {train_metrics['mae']:.6f}")
    logger.info(f"Train R2:      {train_metrics['r2']:.4f}")
    logger.info(f"Val MAE:       {val_metrics['mae']:.6f}")
    logger.info(f"Val R2:        {val_metrics['r2']:.4f}")
    logger.info(f"Val Dir Acc:   {val_metrics['dir_acc']:.2f}%")
    logger.info("=" * 80)
