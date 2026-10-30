"""
Logging utilities for LSTM Log-Return pipeline.

All public functions accept an optional `logger` parameter:
  logger=None  → output goes to print()
  logger=<obj> → output goes to logger.info()
"""


def _emit(msg, logger=None):
    """Route message to logger.info or print."""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# =========================================================================
# HEADERS
# =========================================================================

def log_run_header(title, logger=None, width=80):
    """Print/log a section separator with title."""
    _emit("=" * width, logger)
    _emit(title, logger)
    _emit("=" * width, logger)


def log_train_session_header(logger, symbol, horizon, model_id, data_id,
                             output_dir):
    """Log training session header."""
    logger.info("=" * 80)
    logger.info(f"LSTM LOG-RETURN TRAINING - {symbol.upper()} - Horizon t+{horizon}")
    logger.info("=" * 80)
    logger.info(f"Model ID:   {model_id}")
    logger.info(f"Data ID:    {data_id}")
    logger.info(f"Output:     {output_dir}")


# =========================================================================
# DATA SUMMARY
# =========================================================================

def log_data_summary(features, lookback, horizon, n_train, n_val,
                     val_source='split', logger=None):
    """Log data loading summary."""
    _emit(f"Features:   {len(features)}", logger)
    _emit(f"Lookback:   {lookback}, Horizon: {horizon}", logger)
    _emit(f"Train:      {n_train:,} sequences", logger)
    _emit(f"Val:        {n_val:,} sequences (source: {val_source})", logger)


# =========================================================================
# CONFIGURATION FLAGS
# =========================================================================

def log_config_flags(config, target_scaler, logger=None):
    """Log training configuration flags."""
    ts_str = (f"ENABLED ({target_scaler.method})"
              if target_scaler else "DISABLED")
    _emit(f"Target scaler:    {ts_str}", logger)

    w_str = ("ENABLED" if config.get('winsorize_target') else "DISABLED")
    _emit(f"Winsorisation:    {w_str}", logger)

    feat_r = config.get('feature_round_decimals', 0) or 0
    tgt_r  = config.get('target_round_decimals', 0) or 0
    if feat_r > 0 or tgt_r > 0:
        parts = []
        if feat_r > 0:
            parts.append(f"features={feat_r}d")
        if tgt_r > 0:
            parts.append(f"target={tgt_r}d")
        _emit(f"Pre-train round:  {', '.join(parts)}", logger)
    else:
        _emit("Pre-train round:  off", logger)

    loss_type = config.get('loss_type', 'unknown')
    _emit(f"Loss function:    {loss_type}", logger)

    if loss_type == 'quantile':
        q = config.get('quantile_q', None)
        if q is not None:
            _emit(f"  Quantile q:          {q}", logger)

    if loss_type == 'custom':
        _emit(f"  Sample weights:      {config.get('use_sample_weights', False)}", logger)
        _emit(f"  Misdir penalty:      {config.get('use_misdir_penalty', False)}", logger)
        _emit(f"  Dead-zone:           {config.get('use_dead_zone', False)}", logger)
        use_az = config.get('use_amplified_zone', False)
        az_t = config.get('amplified_zone_threshold', None)
        az_w = config.get('amplified_zone_weight', None)
        az_str = (f"True (threshold={az_t}, weight={az_w})"
                  if use_az and az_t is not None and az_w is not None
                  else str(use_az))
        _emit(f"  Amplified zone:      {az_str}", logger)
        use_corr = config.get('use_correlation_penalty', False)
        corr_w = config.get('correlation_penalty_weight', None)
        corr_str = (f"True (weight={corr_w})" if use_corr and corr_w is not None
                    else str(use_corr))
        _emit(f"  Correlation penalty: {corr_str}", logger)
        use_var = config.get('use_variance_penalty', False)
        var_w = config.get('variance_penalty_weight', None)
        var_str = (f"True (weight={var_w})" if use_var and var_w is not None
                   else str(use_var))
        _emit(f"  Variance penalty:    {var_str}", logger)
        use_dir = config.get('use_directional_loss', False)
        dir_w = config.get('directional_loss_weight', None)
        dir_str = (f"True (weight={dir_w})" if use_dir and dir_w is not None
                   else str(use_dir))
        _emit(f"  Directional loss:    {dir_str}", logger)

    sched = config.get('scheduler_type', 'none')
    _emit(f"Scheduler:        {sched}", logger)


def log_signal_threshold(enabled, threshold_value, mode, logger=None):
    """Log signal threshold configuration."""
    if enabled and threshold_value is not None:
        _emit(f"Signal threshold: {threshold_value:.6f} (mode={mode})", logger)
    else:
        _emit("Signal threshold: DISABLED", logger)


# =========================================================================
# MODEL ARCHITECTURE
# =========================================================================

def log_model_architecture(model, logger=None):
    """Log model architecture summary."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _emit(f"Parameters: {n_params:,}", logger)

    # Extract config from model attributes
    if hasattr(model, 'rnn'):
        rnn = model.rnn
        _emit(f"LSTM:       hidden={rnn.hidden_size}, layers={rnn.num_layers}, "
              f"bidir={rnn.bidirectional}", logger)

    if hasattr(model, 'use_attention'):
        _emit(f"Attention:  {model.use_attention}", logger)


# =========================================================================
# METRICS
# =========================================================================

def _log_thresholded(m, logger=None):
    """Emit thresholded metrics block when present."""
    if 't_mae' not in m:
        return
    t = m['t_threshold']
    _emit(f"  --- thresholded (|y| ≥ {t:.6f}, n={m['t_n_samples']}) ---",
          logger)
    _emit(f"  t_MAE:     {m['t_mae']:.6f}", logger)
    _emit(f"  t_RMSE:    {m['t_rmse']:.6f}", logger)
    _emit(f"  t_R²:      {m['t_r2']:.4f}", logger)
    _emit(f"  t_Dir Acc: {m['t_dir_acc']:.2f}%", logger)


def log_train_val_metrics(logger, train_metrics, val_metrics):
    """Log train and validation metrics."""
    logger.info("\nTRAIN METRICS:")
    logger.info(f"  MAE:      {train_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {train_metrics['rmse']:.6f}")
    logger.info(f"  R²:       {train_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {train_metrics['dir_acc']:.2f}%")
    _log_thresholded(train_metrics, logger)

    logger.info("\nVAL METRICS:")
    logger.info(f"  MAE:      {val_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {val_metrics['rmse']:.6f}")
    logger.info(f"  R²:       {val_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {val_metrics['dir_acc']:.2f}%")
    _log_thresholded(val_metrics, logger)


def log_test_complete(test_id, model_id, n_samples, metrics, output_dir,
                      logger=None):
    """Log test completion summary."""
    _emit("\n" + "=" * 80, logger)
    _emit("TEST COMPLETE", logger)
    _emit("=" * 80, logger)
    _emit(f"Test ID:  {test_id}", logger)
    _emit(f"Model ID: {model_id}", logger)
    _emit(f"Samples:  {n_samples}", logger)
    _emit(f"MAE:      {metrics['mae']:.6f}", logger)
    _emit(f"R²:       {metrics['r2']:.4f}", logger)
    _emit(f"Dir Acc:  {metrics['dir_acc']:.2f}%", logger)
    _log_thresholded(metrics, logger)
    if metrics.get('epochs_trained') is not None:
        best = metrics.get('best_epoch', 'n/a')
        _emit(f"Training ended: epoch {metrics['epochs_trained']}"
              f"  (best checkpoint: epoch {best})", logger)
    _emit(f"Output:   {output_dir}", logger)
    _emit("=" * 80, logger)


def log_final_summary(model_id, output_dir, train_metrics, val_metrics,
                      logger):
    """Print final training summary."""
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
    if 't_mae' in val_metrics:
        logger.info(f"Val t_MAE:     {val_metrics['t_mae']:.6f}")
        logger.info(f"Val t_Dir Acc: {val_metrics['t_dir_acc']:.2f}%")
    logger.info("=" * 80)
