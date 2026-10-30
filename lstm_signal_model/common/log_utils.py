"""
Structured logging utilities for Signal Model training sessions.
"""
from __future__ import annotations


def log_train_session_header(logger, model_id, data_path, resolved):
    """Log session start with key parameters."""
    logger.info("=" * 70)
    logger.info(f"Signal Model Training — {model_id}")
    logger.info("=" * 70)
    logger.info(f"Data path:    {data_path}")
    logger.info(f"Profile:      {resolved.get('training_profile', 'baseline')}")
    logger.info(f"Output mode:  {resolved.get('output_mode', 'dual')}")
    logger.info(f"Lookback:     {resolved['lookback']}")
    logger.info(f"Horizon:      {resolved['horizon']}")
    logger.info(f"Hidden size:  {resolved['hidden_size']}")
    logger.info(f"Loss type:    {resolved['loss_type']}")
    logger.info(f"Threshold:    {resolved['signal_threshold']}")
    logger.info(f"Seed:         {resolved['seed']}")


def log_data_summary(logger, n_train, n_val, n_features, target_stats=None):
    """Log data loading summary."""
    logger.info(f"Train samples: {n_train:,}")
    logger.info(f"Val samples:   {n_val:,}")
    logger.info(f"Features:      {n_features}")
    if target_stats:
        logger.info(
            f"Oracle buy  — mean: {target_stats['buy_mean']:.4f}, "
            f"std: {target_stats['buy_std']:.4f}"
        )
        logger.info(
            f"Oracle sell — mean: {target_stats['sell_mean']:.4f}, "
            f"std: {target_stats['sell_std']:.4f}"
        )


def log_config_flags(logger, resolved):
    """Log active optional flags."""
    flags = []
    if resolved.get('log_signal_diagnostics'):
        flags.append('diagnostics')
    if resolved.get('log_gradient_norms'):
        flags.append('grad_norms')
    if resolved.get('aug_jitter'):
        flags.append('jitter')
    if resolved.get('aug_temporal_dropout'):
        flags.append('temporal_dropout')
    if resolved.get('aug_subsequence'):
        flags.append('subsequence')
    if resolved.get('loss_schedule_ramp_epochs', 0) > 0:
        flags.append(f"scheduled_loss(ramp={resolved['loss_schedule_ramp_epochs']})")
    if resolved.get('best_model_metric') == 'val_composite':
        flags.append('val_composite')

    logger.info(f"Active flags: {', '.join(flags) if flags else '(none)'}")


def log_epoch_summary(logger, epoch, train_loss, val_loss,
                      diagnostics_entry=None, grad_norms=None,
                      loss_components=None):
    """Log per-epoch summary line."""
    parts = [
        f"Epoch {epoch:3d}",
        f"train_loss={train_loss:.6f}",
        f"val_loss={val_loss:.6f}",
    ]
    if diagnostics_entry:
        div = diagnostics_entry.get('signal_diversity', 0)
        status = diagnostics_entry.get('status', '?')
        parts.append(f"div={div:.4f}[{status}]")
    if loss_components:
        comp_str = ",".join([f"{k}={v:.4f}" for k, v in loss_components.items() if k != 'base_loss'])
        if comp_str:
            parts.append(f"loss_breakdown=({comp_str})")
    if grad_norms:
        for group, norm_val in sorted(grad_norms.items()):
            parts.append(f"grad_{group}={norm_val:.4f}")

    logger.info("  ".join(parts))


def log_final_summary(logger, best_epoch, best_val_loss, total_epochs,
                      final_metrics=None):
    """Log training completion summary."""
    logger.info("=" * 70)
    logger.info("Training complete")
    logger.info(f"Best epoch:     {best_epoch}")
    logger.info(f"Best val_loss:  {best_val_loss:.6f}")
    logger.info(f"Total epochs:   {total_epochs}")
    if final_metrics:
        for key, val in final_metrics.items():
            if val is not None:
                logger.info(f"  {key}: {val}")
    logger.info("=" * 70)
