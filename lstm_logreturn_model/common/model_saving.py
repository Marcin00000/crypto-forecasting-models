"""
Model checkpoint + summary persistence for the LSTM Log-Return pipeline.

Split out of `training_utils.py` (CP-085) to keep that module under the
repo's 500-LOC ceiling once the W&B epoch-logging hook was added.
"""
import os
import json

import torch
import joblib


def save_model(model_state, feature_scaler, config_dict, output_dir,
               logger, target_scaler=None, swa_state=None):
    """
    Save trained model and all artifacts.

    Saves:
        - lstm_model.pth       (state_dict)
        - lstm_model_swa.pth   (optional SWA state_dict)
        - feature_scaler.pkl   (joblib)
        - target_scaler.pkl    (joblib, optional)
        - model_config.json    (config dict)
        - summary.txt          (human-readable)

    Note: SWA v1 does not update BatchNorm running stats — models with
    fc_norm=batch may need cautious interpretation of lstm_model_swa.pth.
    """
    logger.info("\nSAVING MODEL")

    pth_path = os.path.join(output_dir, 'lstm_model.pth')
    if os.path.exists(pth_path):
        logger.warning(
            f"Model file already exists at {pth_path} — possible path collision. "
            f"Skipping save to prevent overwrite."
        )
        return

    torch.save(model_state, pth_path)
    logger.info("  ✓ lstm_model.pth")

    if swa_state is not None:
        swa_path = os.path.join(output_dir, 'lstm_model_swa.pth')
        if os.path.exists(swa_path):
            logger.warning(
                f"SWA model file already exists at {swa_path} — skipping save."
            )
        else:
            torch.save(swa_state, swa_path)
            logger.info("  ✓ lstm_model_swa.pth")
            config_dict['has_swa_checkpoint'] = True

    joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    logger.info("  ✓ feature_scaler.pkl")

    if target_scaler is not None:
        joblib.dump(target_scaler, os.path.join(output_dir, 'target_scaler.pkl'))
        config_dict['target_scaler_params'] = target_scaler.get_params()
        logger.info("  ✓ target_scaler.pkl")

    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info("  ✓ model_config.json")

    _write_summary(config_dict, output_dir, target_scaler)
    logger.info("  ✓ summary.txt")


def _write_summary(config_dict, output_dir, target_scaler):
    """Write human-readable training summary."""
    val_m = config_dict.get('val_metrics', {})
    train_m = config_dict.get('train_metrics', {})

    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LSTM LOG-RETURN MODEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model ID:   {config_dict.get('model_id', 'N/A')}\n")
        f.write(f"Data ID:    {config_dict.get('data_id', 'N/A')}\n")
        f.write(f"Symbol:     {config_dict.get('symbol', 'N/A').upper()}\n")
        f.write(f"Horizon:    t+{config_dict.get('horizon', 'N/A')}\n")
        f.write(f"Lookback:   {config_dict.get('lookback', 'N/A')}\n")
        f.write(f"Features:   {config_dict.get('n_features', 'N/A')}\n")
        ts_str = (f"ENABLED ({target_scaler.method})"
                  if target_scaler else "DISABLED")
        f.write(f"Target scl: {ts_str}\n")
        f.write(f"Loss:       {config_dict.get('loss_type', 'N/A')}\n")
        f.write(f"Best epoch: {config_dict.get('best_epoch', 'N/A')}\n")
        f.write(f"Timestamp:  {config_dict.get('timestamp', 'N/A')}\n\n")

        f.write("ARCHITECTURE:\n")
        f.write(f"  Hidden:      {config_dict.get('hidden_size', 'N/A')}\n")
        f.write(f"  Layers:      {config_dict.get('num_layers', 'N/A')}\n")
        f.write(f"  Dropout:     {config_dict.get('dropout', 'N/A')}\n")
        f.write(f"  Bidir:       {config_dict.get('bidirectional', False)}\n")
        f.write(f"  Attention:   {config_dict.get('use_attention', False)}\n")
        f.write(f"  Params:      {config_dict.get('n_params', 0):,}\n\n")

        for label, m in [("TRAIN", train_m), ("VALIDATION", val_m)]:
            if not m:
                continue
            f.write(f"{label} METRICS (log-return):\n")
            f.write(f"  MAE:      {m.get('mae', 0):.6f}\n")
            f.write(f"  RMSE:     {m.get('rmse', 0):.6f}\n")
            f.write(f"  R²:       {m.get('r2', 0):.4f}\n")
            f.write(f"  Dir Acc:  {m.get('dir_acc', 0):.2f}%\n")
            if 't_mae' in m:
                f.write(f"\n  THRESHOLDED (|y| ≥ {m['t_threshold']:.6f},"
                        f" n={m['t_n_samples']}):\n")
                f.write(f"  t_MAE:     {m['t_mae']:.6f}\n")
                f.write(f"  t_RMSE:    {m['t_rmse']:.6f}\n")
                f.write(f"  t_R²:      {m['t_r2']:.4f}\n")
                f.write(f"  t_Dir Acc: {m['t_dir_acc']:.2f}%\n")
            f.write("\n")
