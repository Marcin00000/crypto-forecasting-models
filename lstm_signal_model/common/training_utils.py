"""
Training utilities for Signal Model pipeline.

Contains: training loop, epoch functions, model saving.
Own implementation — no import from lstm_logreturn_model/training_utils
(that module is coupled to LSTMConfig and LR-specific metrics).
"""
import os
import sys
import json
import copy
import logging

import numpy as np
import torch
import joblib
from tqdm import tqdm

from .training_checkpoint import (
    update_swa, resolve_monitor_loss, build_val_signal_ctx,
)

# The progress bar reads the loss from the device only every N batches (host sync is costly).
_PBAR_LOSS_EVERY = 25


# =========================================================================
# LOGGER SETUP
# =========================================================================

def setup_logger(output_dir, name='signal_training'):
    """Create logger with file + console handlers."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        os.path.join(output_dir, 'training.log'), encoding='utf-8')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =========================================================================
# GRADIENT NORMS
# =========================================================================

def _grad_group(param_name):
    """Bucket parameter names into groups for gradient logging."""
    if param_name.startswith('rnn'):
        return 'rnn'
    if param_name.startswith('attention'):
        return 'attention'
    if param_name.startswith('fc') or param_name.startswith('output_linear'):
        return 'fc'
    if param_name.startswith('input_norm'):
        return 'input_norm'
    if param_name.startswith('feature_attention'):
        return 'feature_attention'
    return 'other'


# =========================================================================
# EPOCH FUNCTIONS
# =========================================================================

def train_epoch(model, loader, criterion, optimizer, device,
                grad_clip=1.0, batch_scheduler=None,
                augmenter=None, log_grad_norms=False,
                has_submodel_preds=False):
    """
    Run one training epoch.

    Args:
        has_submodel_preds: if True, last element of each batch tuple
            contains pre-computed submodel predictions to pass to
            model.forward(x, submodel_inputs=...).

    Returns:
        avg_loss (float)
        grad_norms (dict) — only if log_grad_norms=True, else empty dict
    """
    model.train()
    # Per-batch losses stay on the device until the epoch ends: a per-batch .item() would
    # stall the accelerator queue every step. They are summed in batch order afterwards,
    # so avg_loss is identical to accumulating float(loss.item()) step by step.
    batch_losses = []
    grad_sums = {}
    grad_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True, mininterval=0.2, file=sys.stdout)
    for batch in pbar:
        X_batch = batch[0].to(device, non_blocking=True)
        y_batch = batch[1].to(device, non_blocking=True)
        sub_preds = batch[-1].to(device, non_blocking=True) if has_submodel_preds else None

        if augmenter is not None and augmenter.active:
            X_batch = augmenter.apply(X_batch)

        optimizer.zero_grad()
        outputs = model(X_batch, submodel_inputs=sub_preds)
        loss = criterion(outputs, y_batch)
        loss.backward()

        if log_grad_norms:
            # One host transfer per batch instead of one per parameter tensor.
            names, sq_sums = [], []
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                names.append(name)
                sq_sums.append(p.grad.detach().pow(2).sum())
            per_group_sq = {}
            if sq_sums:
                for name, sq in zip(names, torch.stack(sq_sums).tolist()):
                    g = _grad_group(name)
                    per_group_sq[g] = per_group_sq.get(g, 0.0) + float(sq)
            for g, sq in per_group_sq.items():
                grad_sums[g] = grad_sums.get(g, 0.0) + sq ** 0.5
            grad_batches += 1

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        batch_losses.append(loss.detach())
        if len(batch_losses) % _PBAR_LOSS_EVERY == 0:
            pbar.set_postfix({'loss': f'{batch_losses[-1].item():.6f}'}, refresh=False)

    total_loss = 0.0
    if batch_losses:
        for value in torch.stack(batch_losses).tolist():
            total_loss += value

    avg_loss = total_loss / max(len(loader), 1)
    grad_norms = ({k: v / grad_batches for k, v in grad_sums.items()}
                  if grad_batches else {})
    return avg_loss, grad_norms


def val_epoch(model, loader, criterion, device, return_components=False,
              has_submodel_preds=False):
    """
    Run validation epoch.

    Args:
        has_submodel_preds: if True, last element of each batch tuple
            contains pre-computed submodel predictions.

    Returns:
        avg_loss (float)
        all_preds (np.ndarray, shape (n, output_dim))
        all_targets (np.ndarray, shape (n, output_dim))
        [optional] loss_components (dict)
    """
    model.eval()
    total_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].to(device)
            y_batch = batch[1].to(device)
            sub_preds = batch[-1].to(device) if has_submodel_preds else None
            outputs = model(X_batch, submodel_inputs=sub_preds)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(y_batch.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    all_preds = np.concatenate(preds_list, axis=0)
    all_targets = np.concatenate(targets_list, axis=0)
    components = getattr(criterion, '_last_components', {})

    if return_components:
        return avg_loss, all_preds, all_targets, components
    return avg_loss, all_preds, all_targets


# =========================================================================
# MAIN TRAINING LOOP
# =========================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, *, epochs, patience,
                grad_clip=1.0, best_metric='val',
                diversity_weight=0.01,
                min_epochs_before_es=0,
                early_stopping_min_delta=0.0,
                diagnostics=None, logger=None,
                augmenter=None, log_grad_norms=False,
                log_epoch_fn=None,
                scheduler_type='plateau',
                use_swa=False,
                swa_start_epoch=30,
                swa_every=1,
                val_signal_ctx=None,
                val_signal_min_epoch=10,
                has_submodel_preds=False):
    """
    Full training loop with early stopping, SWA and val_signal checkpoint.

    Args:
        best_metric: 'val'           — lowest val_loss
                     'val_composite' — val_loss + diversity_weight*(1-diversity)
                     'val_signal'    — -score(corr_avg, pct_strong, capture_log)
        min_epochs_before_es: minimum epochs to run before early stopping
                              can trigger (0 = no minimum, default)
        early_stopping_min_delta: minimum improvement in score to reset
                                  patience counter (0.0 = any improvement)
        diagnostics: SignalEpochDiagnostics instance (or None)
        log_epoch_fn: callable(epoch, train_loss, val_loss, diag_entry, grad_norms, loss_components)
        use_swa:       enable Stochastic Weight Averaging
        swa_start_epoch: first epoch to include in SWA average (1-based)
        swa_every:     update SWA every N epochs (default 1)
        val_signal_ctx: context dict from build_val_signal_ctx, required when
                        best_metric='val_signal'; contains target_oracle,
                        y_true_lr, horizon, min_signal, start_cash,
                        diversity_weight.
        val_signal_min_epoch: earliest epoch at which val_signal activates.

    Returns:
        dict with best_state_dict, best_epoch, train_losses, val_losses,
        diagnostics_history, swa_state (None when SWA disabled)
    """
    best_for_checkpoint = float('inf')
    best_for_patience = float('inf')
    patience_counter = 0
    best_state = None
    best_epoch = 0
    val_signal_warned = False
    swa_state = None
    n_swa = 0

    train_losses = []
    val_losses = []

    if val_signal_ctx is None and best_metric == 'val_composite':
        # Build a minimal ctx just to carry the diversity_weight through
        # resolve_monitor_loss for val_composite path.
        val_signal_ctx = {'diversity_weight': diversity_weight}

    for epoch in range(1, epochs + 1):
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)

        _batch_sched = scheduler if scheduler_type == 'onecycle' else None
        train_loss, grad_norms = train_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=grad_clip, batch_scheduler=_batch_sched,
            augmenter=augmenter, log_grad_norms=log_grad_norms,
            has_submodel_preds=has_submodel_preds)

        val_loss, val_preds, val_targets, loss_components = val_epoch(
            model, val_loader, criterion, device, return_components=True,
            has_submodel_preds=has_submodel_preds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        diag_entry = None
        if diagnostics is not None:
            diag_entry = diagnostics.update(epoch, val_preds, val_targets)

        if log_epoch_fn:
            import inspect
            sig = inspect.signature(log_epoch_fn)
            if len(sig.parameters) >= 6:
                log_epoch_fn(epoch, train_loss, val_loss, diag_entry, grad_norms, loss_components)
            else:
                log_epoch_fn(epoch, train_loss, val_loss, diag_entry, grad_norms)

        monitor_loss, metric_label, diag_updates, val_signal_warned, domain_switch = (
            resolve_monitor_loss(
                metric_name=best_metric,
                val_loss=val_loss,
                diag_entry=diag_entry,
                val_signal_ctx=val_signal_ctx,
                epoch=epoch - 1,
                val_signal_warned=val_signal_warned,
                val_preds=val_preds,
                val_targets=val_targets,
                val_signal_min_epoch=val_signal_min_epoch,
                logger=logger or _null_logger(),
            ))

        if diag_entry is not None:
            diag_entry.update(diag_updates)

        if domain_switch:
            if logger:
                logger.info(
                    f"  [val_signal] epoch {epoch}: domain switch — "
                    "resetting best/patience to prevent stale comparison "
                    "(val_loss scale vs -score scale)."
                )
            best_for_checkpoint = float('inf')
            best_for_patience = float('inf')
            patience_counter = 0

        # Checkpoint: strict improvement (min_delta does NOT apply here)
        if monitor_loss < best_for_checkpoint:
            best_for_checkpoint = monitor_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            if logger:
                logger.info(
                    f"  >> NEW BEST {metric_label}={monitor_loss:.6f} "
                    f"(epoch {epoch})")

        # Patience: reset only when improvement exceeds min_delta
        if monitor_loss < best_for_patience - early_stopping_min_delta:
            best_for_patience = monitor_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Scheduler step (epoch-level; skip for onecycle — stepped per-batch)
        if scheduler is not None and scheduler_type != 'onecycle':
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # SWA — running average from swa_start_epoch.
        # Must run before the early-stopping break, otherwise the epoch that
        # triggers ES never enters the average.
        if use_swa and epoch >= swa_start_epoch:
            epochs_since = epoch - swa_start_epoch
            if epochs_since % swa_every == 0:
                n_swa += 1
                swa_state, _ = update_swa(swa_state, model, n_swa)

        # Early stopping — only after min_epochs_before_es
        if epoch >= min_epochs_before_es and patience_counter >= patience:
            if logger:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)")
            break

    if swa_state is not None:
        swa_state = update_bn_if_needed(
            model, swa_state, train_loader, device, has_submodel_preds)

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        'best_state_dict': best_state,
        'best_epoch': best_epoch,
        'best_score': best_for_checkpoint,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'diagnostics_history': (
            diagnostics.history if diagnostics else []),
        'swa_state': swa_state,
    }


def update_bn_if_needed(model, swa_state, train_loader, device,
                        has_submodel_preds):
    """Update BatchNorm running statistics for SWA state dict if BatchNorm exists.

    Args:
        has_submodel_preds: if True, last element of each batch tuple contains
            pre-computed submodel predictions. Must be passed explicitly —
            CryptoDataset always yields at least (X, y, idx), so batch length
            alone cannot tell submodel predictions from sample indices.
    """
    if swa_state is None:
        return None
    has_bn = any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                 for m in model.modules())
    if not has_bn:
        return swa_state

    current_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(swa_state)
    model.train()
    with torch.no_grad():
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                sub_inputs = (batch[-1].to(device)
                              if has_submodel_preds else None)
            else:
                x = batch
                sub_inputs = None
            model(x.to(device), submodel_inputs=sub_inputs)
    model.eval()
    updated_swa = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(current_state)
    return updated_swa


def _null_logger():
    """Return a logger that discards all messages (fallback when logger=None)."""
    import logging as _logging
    lg = _logging.getLogger('signal_null')
    if not lg.handlers:
        lg.addHandler(_logging.NullHandler())
    return lg


# =========================================================================
# MODEL SAVING
# =========================================================================

def save_model(model, output_dir, config_dict, feature_scaler=None, swa_state=None):
    """Save model weights, config, optional scaler, and optional SWA state."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(output_dir, 'signal_model.pth'))
    if swa_state is not None:
        torch.save(swa_state,
                   os.path.join(output_dir, 'signal_model_swa.pth'))
        config_dict['has_swa_checkpoint'] = True

    with open(os.path.join(output_dir, 'model_config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    if feature_scaler is not None:
        joblib.dump(feature_scaler,
                    os.path.join(output_dir, 'feature_scaler.pkl'))
