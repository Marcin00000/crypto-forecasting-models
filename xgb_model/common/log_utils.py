"""
Logging utilities for XGBoost pipeline.

All public functions accept an optional `logger` parameter (Proposition 2).
  logger=None  → output goes to print()
  logger=<obj> → output goes to logger.info() (console + file via StreamHandler)

Internal _emit() helper routes to logger.info or print accordingly.
"""


# =============================================================================
# INTERNAL HELPER
# =============================================================================

def _emit(msg, logger=None):
    """Route message to logger.info (with timestamp) or print (plain)."""
    if logger is not None:
        try:
            logger.info(msg)
        except UnicodeEncodeError:
            logger.info(msg.encode('ascii', errors='replace').decode('ascii'))
    else:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('ascii', errors='replace').decode('ascii'))


# =============================================================================
# SHARED — section headers (train + test + tuning)
# =============================================================================

def log_run_header(title, logger=None, width=80, newline=False):
    """Print/log a section separator with title.

    Args:
        title:   Section title string.
        logger:  Logger instance or None (→ print).
        width:   Separator width (default 80).
        newline: Prefix the opening separator with a blank line (default False).
    """
    _emit(("\n" if newline else "") + "=" * width, logger)
    _emit(title, logger)
    _emit("=" * width, logger)


# =============================================================================
# TRAIN — session header and data summary (Proposition 1: via logger only)
# =============================================================================

def log_train_session_header(logger, symbol, horizon, model_id, data_id, output_dir):
    """Log training session header to logger (file + console via StreamHandler)."""
    logger.info("=" * 80)
    logger.info(f"XGBOOST TRAINING - {symbol.upper()} - Horizon t+{horizon}")
    logger.info("=" * 80)
    logger.info(f"Model ID:   {model_id}")
    logger.info(f"Data ID:    {data_id}")
    logger.info(f"Output:     {output_dir}")


def log_data_summary(features, flat_features, flattening_method,
                     lookback, horizon, n_train, n_val, logger=None):
    """Log data loading summary: feature counts, flattening, lookback, train/val sizes."""
    _emit(f"Features:   {len(features)} \u2192 {len(flat_features)} flat", logger)
    _emit(f"Flattening: {flattening_method}", logger)
    _emit(f"Lookback:   {lookback}, Horizon: {horizon}", logger)
    _emit(f"Train:      {n_train} samples, Val: {n_val} samples", logger)


# =============================================================================
# TRAIN — configuration flags (shared print/logger via Proposition 2)
# =============================================================================

def log_config_flags(target_scaler, winsorize_target, winsorize_percentile,
                     sample_weight_mode, reweight_misdir, misdir_approach_a_variant,
                     use_custom_objective, custom_obj_type,
                     custom_obj_sample_weights, custom_obj_misdir_penalty,
                     custom_obj_huber_delta=None, logger=None):
    """Log training configuration flags: target scaler, winsorisation, weights, custom obj.

    custom_obj_huber_delta is optional — included in output only when provided.
    """
    _emit(
        "Target scaler:   "
        + ("ENABLED (" + target_scaler.method + ")" if target_scaler else "DISABLED"),
        logger,
    )
    _emit(
        "Winsorisation:   "
        + ("ENABLED (" + str(winsorize_percentile) + "th pct)" if winsorize_target else "DISABLED"),
        logger,
    )
    _emit(
        "Sample weights:  "
        + ("ENABLED (mode=" + str(sample_weight_mode) + ")" if sample_weight_mode else "DISABLED"),
        logger,
    )
    _emit(
        "Misdir penalty A: "
        + ("ENABLED (variant=" + misdir_approach_a_variant + ")" if reweight_misdir else "DISABLED"),
        logger,
    )
    if use_custom_objective:
        custom_str = (
            f"ENABLED (type={custom_obj_type}, sw={custom_obj_sample_weights}"
            f", misdir={custom_obj_misdir_penalty}"
        )
        if custom_obj_huber_delta is not None:
            custom_str += f", delta={custom_obj_huber_delta}"
        custom_str += ")"
    else:
        custom_str = "DISABLED"
    _emit(f"Custom obj B:    {custom_str}", logger)


def log_signal_threshold(signal_threshold_enabled, resolved_threshold,
                         signal_threshold_mode, use_threshold_penalties,
                         threshold_smooth_transition=False, threshold_smooth_width=2.0,
                         clip_bounds=None, logger=None):
    """Log signal threshold configuration and optional clip bounds."""
    if signal_threshold_enabled and resolved_threshold is not None:
        _emit(
            f"Signal threshold: {resolved_threshold:.6f} (mode={signal_threshold_mode})",
            logger,
        )
        _emit(
            "Threshold penalties: "
            + ("ENABLED" if use_threshold_penalties else "DISABLED (metrics only)"),
            logger,
        )
        if use_threshold_penalties:
            smooth_str = (
                f"ENABLED (width={threshold_smooth_width})"
                if threshold_smooth_transition else "DISABLED"
            )
            _emit(f"  Smooth transition: {smooth_str}", logger)
    else:
        _emit("Signal threshold: DISABLED", logger)
    if clip_bounds:
        _emit(
            f"Clip bounds:     [{clip_bounds['lower']:.6f}, {clip_bounds['upper']:.6f}]",
            logger,
        )


def log_hyperparameters(logger, params, early_stopping_rounds, use_best_iteration):
    """Log XGBoost hyperparameters, early stopping and save-from policy to logger."""
    logger.info("\nHyperparameters:")
    for key, value in sorted(params.items()):
        logger.info(f"  {key:20s}: {value}")
    logger.info(f"  early_stopping:      {early_stopping_rounds}")
    logger.info(
        "Save from:  " + ("best_iteration" if use_best_iteration else "last_iteration")
    )


def _log_thresholded(logger, m):
    """Emit the thresholded metrics block (t_*) when present in a metrics dict.

    Uses _emit() so it works with logger=None (console-only) and with a real logger.
    """
    if 't_mae' not in m:
        return
    t = m['t_threshold']
    _emit(f"--- thresholded (|y| \u2265 {t:.6f}, n={m['t_n_samples']}) ---", logger)
    _emit(f"t_MAE:     {m['t_mae']:.6f}", logger)
    _emit(f"t_RMSE:    {m['t_rmse']:.6f}", logger)
    _emit(f"t_R\u00b2:      {m['t_r2']:.4f}", logger)
    _emit(f"t_PH(\u03b4=1): {m['t_pseudo_huber']:.6f}", logger)
    _emit(f"t_PH(\u03b4=10):{m['t_pseudo_huber_large']:.6f}", logger)
    _emit(f"t_Dir Acc: {m['t_dir_acc']:.2f}%", logger)


def log_train_val_metrics(logger, train_metrics, val_metrics):
    """Log train and validation metrics (MAE, RMSE, R², Dir Acc, t_* if present) to logger."""
    logger.info("\nTRAIN METRICS:")
    logger.info(f"  MAE:      {train_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {train_metrics['rmse']:.6f}")
    logger.info(f"  R²:       {train_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {train_metrics['dir_acc']:.2f}%")
    _log_thresholded(logger, train_metrics)
    logger.info("\nVAL METRICS:")
    logger.info(f"  MAE:      {val_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {val_metrics['rmse']:.6f}")
    logger.info(f"  R²:       {val_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {val_metrics['dir_acc']:.2f}%")
    _log_thresholded(logger, val_metrics)


# =============================================================================
# TEST — console only (logger=None by default)
# =============================================================================

def log_model_info(model_id, config, target_scaler, logger=None):
    """Log loaded model metadata: model_id, data_id, symbol, horizon, features, scaler."""
    _emit(f"Model ID:   {model_id}", logger)
    _emit(f"Data ID:    {config['data_id']}", logger)
    _emit(f"Symbol:     {config['symbol'].upper()}", logger)
    _emit(f"Horizon:    t+{config['horizon']}", logger)
    _emit(f"Features:   {config['n_features']} -> {config['n_flat_features']} flat", logger)
    _emit(f"Flattening: {config['flattening_method']}", logger)
    _emit(
        "Target scaler: "
        + ("ENABLED (" + target_scaler.method + ")" if target_scaler else "DISABLED"),
        logger,
    )


def log_test_run_info(test_id, n_samples, output_dir, logger=None):
    """Log test run identifiers: test_id, number of samples, output directory."""
    _emit(f"\nTest ID:    {test_id}", logger)
    _emit(f"Samples:    {n_samples}", logger)
    _emit(f"Output:     {output_dir}", logger)


def log_test_complete(test_id, model_id, n_samples, metrics, output_dir, logger=None):
    """Log test completion summary with key metrics."""
    _emit("\n" + "=" * 80, logger)
    _emit("TEST COMPLETE", logger)
    _emit("=" * 80, logger)
    _emit(f"Test ID:  {test_id}", logger)
    _emit(f"Model ID: {model_id}", logger)
    _emit(f"Samples:  {n_samples}", logger)
    _emit(f"MAE:      {metrics['mae']:.6f}", logger)
    _emit(f"R\u00b2:       {metrics['r2']:.4f}", logger)
    _emit(f"PH(\u03b4=1):  {metrics['pseudo_huber']:.6f}", logger)
    _emit(f"PH(\u03b4=10): {metrics['pseudo_huber_large']:.6f}", logger)
    _emit(f"Dir Acc:  {metrics['dir_acc']:.2f}%", logger)
    _log_thresholded(logger, metrics)
    _emit(f"Output:   {output_dir}", logger)
    _emit("=" * 80, logger)


# =============================================================================
# TUNING — logger required
# =============================================================================

def log_tuning_header(logger, optuna_id, data_id, output_dir, n_trials,
                      xgb_objective, xgb_config, tune_config):
    """Log tuning session header.

    Args:
        xgb_config:  XGBConfig class (accesses OBJECTIVE, HUBER_SLOPE).
        tune_config: TuningConfig class (accesses OPTIMIZE_TARGET, FEATURE_SET, etc.).
    """
    logger.info("=" * 80)
    logger.info("XGBOOST HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Optuna ID:  {optuna_id}")
    logger.info(f"Data ID:    {data_id}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Trials:     {n_trials}")
    huber_note = (
        f" (huber_slope={xgb_config.HUBER_SLOPE})"
        if xgb_config.OBJECTIVE == 'reg:pseudohubererror_large' else ""
    )
    logger.info(f"Objective:  {xgb_objective}{huber_note}")
    logger.info(
        f"Optuna minimises: {tune_config.OPTIMIZE_TARGET.upper()} loss ({xgb_objective})"
    )
    logger.info(f"Target scaling: {tune_config.SCALE_TARGET} ({tune_config.TARGET_SCALER_TYPE})")
    logger.info(f"Feature set: {tune_config.FEATURE_SET}")
    logger.info(
        "Lookback:   "
        + ("TUNING" if tune_config.TUNE_LOOKBACK else f"FIXED ({tune_config.LOOKBACK})")
    )
    logger.info(
        "Horizon:    "
        + ("TUNING" if tune_config.TUNE_HORIZON else f"FIXED ({tune_config.HORIZON})")
    )
    logger.info(
        "Flattening: "
        + ("TUNING" if not tune_config.FIXED_FLATTENING_METHOD
           else f"FIXED ({tune_config.FIXED_FLATTENING_METHOD})")
    )
    logger.info(
        "Custom obj: "
        + (tune_config.CUSTOM_OBJ_TYPE if tune_config.USE_CUSTOM_OBJECTIVE else "DISABLED")
    )
    logger.info(f"Optimize metric: {tune_config.OPTIMIZE_METRIC}")
    logger.info(
        f"Signal threshold: mode={tune_config.SIGNAL_THRESHOLD_MODE}"
        f", value={tune_config.SIGNAL_THRESHOLD_VALUE}"
    )
    logger.info(
        "Threshold penalties: "
        + ("ENABLED" if tune_config.USE_THRESHOLD_PENALTIES else "DISABLED")
    )


# =============================================================================
# TRAINING_UTILS — moved from training_utils.py
# =============================================================================

def log_train_model_setup(logger, target_scaler, sample_weight, reweight_misdir,
                           misdir_approach_a_variant, misdir_penalty,
                           misdir_min_quantile, custom_objective):
    """Log train_model() setup: target scaler, sample weights, misdir penalty, custom objective."""
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
            f"Misdir penalty A: ENABLED  "
            f"variant={misdir_approach_a_variant}  "
            f"penalty={misdir_penalty:.2f}  "
            f"min_quantile={misdir_min_quantile:.2f}"
        )
    else:
        logger.info("Misdir penalty A: DISABLED")

    if custom_objective is not None:
        logger.info("Custom objective B: ENABLED")
        if reweight_misdir:
            logger.warning(
                "Custom objective B active with reweight_misdir=True -- "
                "misdir penalty A is ignored when custom_objective is used. "
                "Use use_misdir_penalty=True in make_custom_objective instead."
            )
    else:
        logger.info("Custom objective B: DISABLED (standard objective string)")


def log_metrics_comparison(logger, target_scaler, train_metrics, val_metrics):
    """
    Diagnostyczny log por\u00f3wnawczy \u2014 scaled (xgb evals) vs original.

    Pokazuje relacj\u0119 mi\u0119dzy metrykami w skalowanej przestrzeni (logowane
    przez xgb.train evals) a metrykami w oryginalnej przestrzeni log-return\u00f3w
    (obliczanymi post-hoc przez _compute_metrics).
    Przeniesione z training_utils._log_metrics_comparison.
    """
    logger.info("")
    logger.info("METRICS COMPARISON (xgb evals vs original space):")

    if target_scaler is not None:
        try:
            scaler_params = target_scaler.get_params()
            if target_scaler.method == 'robust':
                scale_val = scaler_params.get('scale', None)
                center_val = scaler_params.get('center', None)
                if scale_val is not None:
                    logger.info(
                        f"  Target scaler:     method=robust, center={center_val:.6f}"
                        f", scale(IQR)={scale_val:.6f}"
                    )
                    logger.info(
                        f"  Approx conversion: xgb_rmse \u00d7 {scale_val:.6f} \u2248 original RMSE"
                    )
            elif target_scaler.method == 'standard':
                std_val = scaler_params.get('std', None)
                mean_val = scaler_params.get('mean', None)
                if std_val is not None:
                    logger.info(
                        f"  Target scaler:     method=standard, mean={mean_val:.6f}"
                        f", std={std_val:.6f}"
                    )
                    logger.info(
                        f"  Approx conversion: xgb_rmse \u00d7 {std_val:.6f} \u2248 original RMSE"
                    )
        except Exception:
            logger.info("  Target scaler active but params not available")
    else:
        logger.info("  Target scaler:     DISABLED (xgb evals = original space)")

    logger.info(
        f"  Original space:    train-MAE={train_metrics['mae']:.6f}, "
        f"val-MAE={val_metrics['mae']:.6f}"
    )
    logger.info(
        f"  Original space:    train-RMSE={train_metrics['rmse']:.6f}, "
        f"val-RMSE={val_metrics['rmse']:.6f}"
    )
    logger.info(
        f"  Original space:    train-R\u00b2={train_metrics['r2']:.4f}, "
        f"val-R\u00b2={val_metrics['r2']:.4f}"
    )

    if 't_mae' in train_metrics:
        t = train_metrics['t_threshold']
        logger.info("")
        logger.info(f"  Thresholded (|y| >= {t:.6f}):")
        logger.info(
            f"    train: n={train_metrics['t_n_samples']} ({train_metrics['t_pct_samples']:.1f}%), "
            f"MAE={train_metrics['t_mae']:.6f}, R\u00b2={train_metrics.get('t_r2', 0):.4f}, "
            f"Dir Acc={train_metrics['t_dir_acc']:.2f}%"
        )
        if 't_mae' in val_metrics:
            logger.info(
                f"    val:   n={val_metrics['t_n_samples']} ({val_metrics['t_pct_samples']:.1f}%), "
                f"MAE={val_metrics['t_mae']:.6f}, R\u00b2={val_metrics.get('t_r2', 0):.4f}, "
                f"Dir Acc={val_metrics['t_dir_acc']:.2f}%"
            )
