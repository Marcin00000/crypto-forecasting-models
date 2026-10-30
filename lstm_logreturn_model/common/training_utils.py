"""
Training utilities for LSTM Log-Return pipeline.

Contains: training loop, epoch functions, model saving.
Device, Dataset, Schedulers are in torch_utils.py.
"""
import inspect
import os
import sys
import logging

import numpy as np
import torch
from tqdm import tqdm

from lstm_config import LSTMConfig

from .metrics import calculate_metrics
from .target_utils import TargetScaler
from .diagnostics import EpochDiagnostics
from .training_checkpoint import resolve_monitor_loss, update_swa

# The progress bar reads the loss from the device only every N batches (host sync is costly).
_PBAR_LOSS_EVERY = 25


# =========================================================================
# LOGGER SETUP
# =========================================================================

def setup_logger(output_dir, name='lstm_lr_training'):
    """Create logger with file + console handlers."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        os.path.join(output_dir, 'training.log'), encoding='utf-8'
    )
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =========================================================================
# EPOCH FUNCTIONS
# =========================================================================

def _grad_group(param_name):
    """F1.I5: bucket parameter names into 'rnn'/'attention'/'fc'/'other'."""
    if param_name.startswith('rnn'):
        return 'rnn'
    if param_name.startswith('attention'):
        return 'attention'
    if param_name.startswith('fc'):
        return 'fc'
    if param_name.startswith('input_norm'):
        return 'input_norm'
    if param_name.startswith('feature_attention'):
        return 'feature_attention'
    return 'other'


def _unpack_loader_batch(batch, device):
    """Return (X, y, indices, sigma_or_none) from a 3- or 4-tuple batch."""
    if len(batch) == 4:
        X_batch, y_batch, indices, sigma_batch = batch
        return (X_batch.to(device, non_blocking=True),
                y_batch.to(device, non_blocking=True), indices,
                sigma_batch.to(device, non_blocking=True))
    X_batch, y_batch, indices = batch
    return (X_batch.to(device, non_blocking=True),
            y_batch.to(device, non_blocking=True), indices, None)


def _criterion_accepts_sigma(criterion):
    return 'sigma' in inspect.signature(criterion.forward).parameters


def _compute_loss(criterion, outputs, y_batch, *, is_custom_loss=False,
                  indices=None, sigma=None, accepts_sigma=False):
    """Call criterion; forward sigma only when the loss accepts it."""
    if is_custom_loss:
        kwargs = {'indices': indices}
        if sigma is not None and accepts_sigma:
            kwargs['sigma'] = sigma
        return criterion(outputs, y_batch, **kwargs)
    return criterion(outputs, y_batch)


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0,
                is_custom_loss=False, batch_scheduler=None,
                return_components=False, log_grad_norms=False,
                augmenter=None):
    """
    Run one training epoch.

    Args:
        is_custom_loss:    if True, pass indices to criterion
        batch_scheduler:   scheduler that steps per-batch (e.g. OneCycleLR)
        return_components: if True, also return dict of averaged loss
                           components from criterion._last_components.
        log_grad_norms:    if True, return average L2 grad norms per
                           layer-group (rnn / attention / fc / input_norm /
                           other). Adds one extra parameter scan per batch
                           but no extra autograd passes — cheap.

    Returns:
        depending on flags:
            avg_loss
            (avg_loss, components)                       — return_components
            (avg_loss, grad_norms)                       — log_grad_norms only
            (avg_loss, components, grad_norms)           — both flags
    """
    model.train()
    # Per-batch losses stay on the device until the epoch ends: a per-batch .item() would
    # stall the accelerator queue every step. They are summed in batch order afterwards,
    # so avg_loss is identical to accumulating loss.item() step by step.
    batch_losses = []
    comp_sums: dict = {}
    comp_count = 0
    grad_sums: dict = {}
    grad_batches = 0

    accepts_sigma = _criterion_accepts_sigma(criterion) if is_custom_loss else False

    pbar = tqdm(loader, desc="Train", leave=False, dynamic_ncols=True, mininterval=0.2, file=sys.stdout)
    for batch in pbar:
        X_batch, y_batch, indices, sigma_batch = _unpack_loader_batch(
            batch, device)
        if augmenter is not None and augmenter.active:
            X_batch = augmenter.apply(X_batch)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze(-1)

        loss = _compute_loss(
            criterion, outputs, y_batch,
            is_custom_loss=is_custom_loss, indices=indices,
            sigma=sigma_batch, accepts_sigma=accepts_sigma,
        )

        loss.backward()

        # F1.I5: snapshot per-group gradient norms BEFORE clipping so that
        # diagnostics reflect the raw magnitudes the optimiser would see
        # without the clip safety net.
        if log_grad_norms:
            # One host transfer per batch instead of one per parameter tensor.
            names, sq_sums = [], []
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                names.append(name)
                sq_sums.append(p.grad.detach().pow(2).sum())
            per_group_sq: dict = {}
            if sq_sums:
                for name, sq in zip(names, torch.stack(sq_sums).tolist()):
                    g = _grad_group(name)
                    per_group_sq[g] = per_group_sq.get(g, 0.0) + float(sq)
            for g, sq in per_group_sq.items():
                norm = sq ** 0.5
                grad_sums[g] = grad_sums.get(g, 0.0) + norm
            grad_batches += 1

        # F1.I1: grad_clip <= 0 disables clipping. Calling clip_grad_norm_
        # with 0 would scale every gradient to 0 (max_norm/total_norm == 0),
        # so the guard is required, not just an optimisation.
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if batch_scheduler is not None:
            batch_scheduler.step()

        batch_losses.append(loss.detach())
        if len(batch_losses) % _PBAR_LOSS_EVERY == 0:
            pbar.set_postfix({'loss': f'{batch_losses[-1].item():.6f}'}, refresh=False)

        if return_components:
            last = getattr(criterion, '_last_components', None)
            if last:
                for k, v in last.items():
                    comp_sums[k] = comp_sums.get(k, 0.0) + v
                comp_count += 1

    total_loss = 0.0
    if batch_losses:
        for value in torch.stack(batch_losses).tolist():
            total_loss += value
    avg_loss = total_loss / len(loader)
    comps = ({k: v / comp_count for k, v in comp_sums.items()}
             if comp_count else {})
    grad_norms = ({k: v / grad_batches for k, v in grad_sums.items()}
                  if grad_batches else {})

    if return_components and log_grad_norms:
        return avg_loss, comps, grad_norms
    if return_components:
        return avg_loss, comps
    if log_grad_norms:
        return avg_loss, grad_norms
    return avg_loss


def validate_epoch(model, loader, criterion, device, is_custom_loss=False,
                   return_components=False):
    """
    Run one validation epoch.

    Args:
        return_components: if True, also return dict of averaged loss
                           components from criterion._last_components.

    Returns:
        float — average validation loss (when return_components=False)
        (float, dict) tuple when return_components=True
    """
    model.eval()
    total_loss = 0.0
    comp_sums: dict = {}
    comp_count = 0

    accepts_sigma = _criterion_accepts_sigma(criterion) if is_custom_loss else False

    with torch.no_grad():
        for batch in loader:
            X_batch, y_batch, indices, sigma_batch = _unpack_loader_batch(
                batch, device)

            outputs = model(X_batch).squeeze(-1)

            loss = _compute_loss(
                criterion, outputs, y_batch,
                is_custom_loss=is_custom_loss, indices=indices,
                sigma=sigma_batch, accepts_sigma=accepts_sigma,
            )

            total_loss += loss.item()

            if return_components:
                last = getattr(criterion, '_last_components', None)
                if last:
                    for k, v in last.items():
                        comp_sums[k] = comp_sums.get(k, 0.0) + v
                    comp_count += 1

    avg_loss = total_loss / len(loader)
    if not return_components:
        return avg_loss
    comps = ({k: v / comp_count for k, v in comp_sums.items()}
             if comp_count else {})
    return avg_loss, comps


# =========================================================================
# PREDICTION + METRICS
# =========================================================================

def _batch_xy(batch):
    """Extract X and y from a 3- or 4-tuple dataset batch."""
    return batch[0], batch[1]


def predict_all(model, loader, device):
    """Generate predictions for entire dataloader."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            X_batch, _ = _batch_xy(batch)
            X_batch = X_batch.to(device)
            out = model(X_batch).squeeze(-1).cpu().numpy()
            predictions.extend(out.tolist() if out.ndim > 0 else [out.item()])

    return np.array(predictions, dtype=np.float32)


def _apply_precision(arr, decimals):
    """F1.I4: post-hoc decimal rounding (0 = disabled)."""
    if decimals is None or decimals <= 0:
        return arr
    return np.round(arr, decimals=int(decimals))


def compute_metrics(y_true_scaled, y_pred_scaled, target_scaler,
                    threshold=None):
    """
    Compute metrics in original log-return space.

    F1.I4: when LSTMConfig.PREDICTION_PRECISION / TARGET_PRECISION > 0, the
    inverse-transformed arrays are rounded before metric computation. This
    is post-hoc only — gradients are unaffected. Useful for evaluating how
    much decimal noise contributes to MAE / R² instability.
    """
    if target_scaler is not None:
        y_true = target_scaler.inverse_transform(y_true_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
    else:
        y_true = np.asarray(y_true_scaled)
        y_pred = np.asarray(y_pred_scaled)

    y_true = _apply_precision(y_true, getattr(LSTMConfig, 'TARGET_PRECISION', 0))
    y_pred = _apply_precision(y_pred,
                              getattr(LSTMConfig, 'PREDICTION_PRECISION', 0))
    return calculate_metrics(y_true, y_pred, threshold=threshold)


# =========================================================================
# TRAINING LOOP
# =========================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, epochs, early_stopping_patience,
                gradient_clip, logger, target_scaler=None,
                signal_threshold=None, scheduler_type='plateau',
                is_custom_loss=False,
                scheduled_loss=None,
                augmenter=None,
                val_signal_ctx=None,
                min_epochs_before_es=0,
                early_stopping_min_delta=0.0,
                use_swa=False,
                swa_start_epoch=30,
                swa_every=1,
                tracker=None):
    """
    Main training loop with early stopping.

    Args:
        scheduled_loss: optional (base_criterion, full_criterion, warmup_epochs)
                        tuple from FAZA 5. When set, training uses
                        `base_criterion` for the first `warmup_epochs`
                        epochs, then swaps to `full_criterion`. `criterion`
                        is ignored in this mode (caller should pass a
                        sentinel — typically the same as `full_criterion`).
                        Each per-epoch entry of `diagnostics_history` gets a
                        `scheduled_loss_phase` key ('base'/'full') so post-hoc
                        analysis can find the switch boundary.
        val_signal_ctx: optional dict with keys close, lookback, horizon,
                        threshold, window_sigma, min_signal — required when
                        BEST_MODEL_METRIC='val_signal'.
        min_epochs_before_es: block early stopping until this epoch (0=off).
        early_stopping_min_delta: patience resets only when monitor_loss
                        improves by at least this amount; checkpoint still
                        saves on any strict improvement.
        use_swa / swa_start_epoch / swa_every: stochastic weight averaging
                        (default off). Maintains a running mean on CPU from
                        swa_start_epoch, updating every swa_every epochs.
                        BN stats are NOT updated in v1 — see save_model.
        tracker:        optional wandb ExperimentTracker (or NullTracker);
                        logs train/val loss, compression_ratio and grad
                        norms once per epoch. No-op when None.

    Returns:
        best_state_dict, train_losses, val_losses,
        train_metrics, val_metrics, best_epoch,
        diagnostics_history (list of per-epoch dicts from EpochDiagnostics),
        swa_state (dict or None when use_swa=False / no SWA steps taken)
    """
    best_for_checkpoint = float('inf')
    best_for_patience = float('inf')
    patience_counter = 0
    best_state = None
    best_epoch = 0
    train_losses = []
    val_losses = []
    swa_state = None
    n_swa = 0
    val_signal_warned = False

    # F2.0 — per-epoch diagnostic for prediction collapse detection. Runs one
    # extra forward pass on val_loader (cheap relative to training epoch).
    diagnostics = EpochDiagnostics(target_scaler=target_scaler, logger=logger)
    # Cached val ground-truth (scaled); built once on first iter.
    y_val_true_scaled_cached = None

    logger.info(f"\n{'=' * 80}")
    logger.info("TRAINING START")
    logger.info(f"{'=' * 80}")

    # OneCycleLR steps per batch — pass to train_epoch
    is_onecycle = scheduler_type == 'onecycle'

    # Only custom loss populates per-batch components, so we skip the
    # breakdown pass cost for native losses.
    log_grad_norms = bool(getattr(LSTMConfig, 'LOG_GRADIENT_NORMS', False))

    # FAZA 5 — scheduled loss switching state. We resolve `criterion`,
    # `is_custom_loss`, and `track_components` per-epoch so the switch from
    # base → full at the boundary epoch is logged exactly once.
    sched_active = scheduled_loss is not None
    if sched_active:
        sched_base, sched_full, sched_warmup = scheduled_loss
        sched_warmup = int(sched_warmup)
    else:
        sched_base = sched_full = None
        sched_warmup = 0
    prev_phase = None
    # When scheduled-loss is OFF we keep the original behaviour.
    track_components = is_custom_loss

    for epoch in range(epochs):
        if sched_active:
            phase = 'base' if epoch < sched_warmup else 'full'
            active_criterion = sched_base if phase == 'base' else sched_full
            # Re-evaluate is_custom_loss / track_components each phase so
            # the breakdown logger picks up SignalLoss/LSTMCustomLoss only
            # when active. Imported lazily to avoid a hard circular dep.
            from common.losses import LSTMCustomLoss
            from common.signal_loss import SignalLoss
            is_custom_loss = isinstance(
                active_criterion, (LSTMCustomLoss, SignalLoss))
            track_components = is_custom_loss
            if phase != prev_phase:
                logger.info(
                    f"  [scheduled_loss] entering phase={phase!r} at "
                    f"epoch {epoch + 1} (warmup={sched_warmup}). "
                    "val_loss may step-change at the boundary because the "
                    "two losses have different scales."
                )
                prev_phase = phase
            criterion = active_criterion
        # Train
        train_grad_norms = {}
        if track_components and log_grad_norms:
            train_loss, train_comps, train_grad_norms = train_epoch(
                model, train_loader, criterion, optimizer, device,
                gradient_clip, is_custom_loss,
                batch_scheduler=scheduler if is_onecycle else None,
                return_components=True, log_grad_norms=True,
                augmenter=augmenter,
            )
        elif track_components:
            train_loss, train_comps = train_epoch(
                model, train_loader, criterion, optimizer, device,
                gradient_clip, is_custom_loss,
                batch_scheduler=scheduler if is_onecycle else None,
                return_components=True,
                augmenter=augmenter,
            )
        elif log_grad_norms:
            train_loss, train_grad_norms = train_epoch(
                model, train_loader, criterion, optimizer, device,
                gradient_clip, is_custom_loss,
                batch_scheduler=scheduler if is_onecycle else None,
                log_grad_norms=True,
                augmenter=augmenter,
            )
            train_comps = {}
        else:
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device,
                gradient_clip, is_custom_loss,
                batch_scheduler=scheduler if is_onecycle else None,
                augmenter=augmenter,
            )
            train_comps = {}

        # Validate
        if track_components:
            val_loss, val_comps = validate_epoch(
                model, val_loader, criterion, device, is_custom_loss,
                return_components=True,
            )
        else:
            val_loss = validate_epoch(
                model, val_loader, criterion, device, is_custom_loss,
            )
            val_comps = {}

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        # Scheduler step (epoch-level, skip for onecycle)
        if scheduler is not None and not is_onecycle:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        logger.info(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Loss breakdown — show how much each custom-loss component
        # contributes. Only emitted when LSTMCustomLoss is active AND at
        # least one penalty populated `_last_components` with its key.
        if track_components and val_comps:
            parts = []
            base_v = val_comps.get('base')
            if base_v is not None:
                parts.append(f"base={base_v:.4f}")
            if 'corr' in val_comps and 'corr_penalty' in val_comps:
                parts.append(
                    f"corr={val_comps['corr']:+.4f}"
                    f"(pen={val_comps['corr_penalty']:.4f})"
                )
            if 'var_ratio' in val_comps and 'var_penalty' in val_comps:
                parts.append(
                    f"var_ratio={val_comps['var_ratio']:.4f}"
                    f"(pen={val_comps['var_penalty']:.4f})"
                )
            if 'dir_loss' in val_comps:
                parts.append(f"dir_loss={val_comps['dir_loss']:.4f}")
            if 'log_var_penalty' in val_comps:
                parts.append(f"log_var={val_comps['log_var_penalty']:.4f}")
            if 'r2' in val_comps and 'r2_penalty' in val_comps:
                parts.append(
                    f"r2={val_comps['r2']:+.4f}"
                    f"(pen={val_comps['r2_penalty']:.4f})"
                )
            if 'ccc' in val_comps and 'ccc_penalty' in val_comps:
                parts.append(
                    f"ccc={val_comps['ccc']:+.4f}"
                    f"(pen={val_comps['ccc_penalty']:.4f})"
                )
            if parts:
                logger.info(f"  breakdown (val) | {' | '.join(parts)}")

        # F2.0 diagnostics — predict on val set and log pred_std/true_std
        # ratio in raw space. Ground truth cached after first epoch.
        y_val_pred_scaled = predict_all(model, val_loader, device)
        if y_val_true_scaled_cached is None:
            y_val_true_scaled_cached = np.concatenate(
                [y.numpy() for _, y, *_ in val_loader]
            )
        diag_rec = diagnostics.update(
            epoch=epoch + 1,
            y_val_pred_scaled=y_val_pred_scaled,
            y_val_true_scaled=y_val_true_scaled_cached,
        )
        # Enrich diagnostics record with component breakdown for persistence.
        if track_components:
            diag_rec['train_components'] = train_comps
            diag_rec['val_components'] = val_comps

        # FAZA 5 — annotate the diagnostics record with the active phase so
        # the test (and downstream analysis) can correlate metric jumps with
        # the loss switch.
        if sched_active:
            diag_rec['scheduled_loss_phase'] = phase

        # F1.I5 — gradient-norm diagnostics. Logged at INFO and persisted in
        # diagnostics_history so post-hoc analysis can correlate gradient
        # magnitudes with the prediction-collapse signal.
        if log_grad_norms and train_grad_norms:
            diag_rec['grad_norms'] = dict(train_grad_norms)
            gn_str = ' | '.join(
                f"{k}={v:.4f}" for k, v in sorted(train_grad_norms.items()))
            logger.info(f"  grad_norms (train) | {gn_str}")

        if tracker is not None:
            wandb_metrics = {
                'compression_ratio': diag_rec.get('compression_ratio'),
                'lr': current_lr,
            }
            wandb_metrics.update(
                {f'grad_norm_{k}': v for k, v in train_grad_norms.items()})
            tracker.log_epoch(epoch + 1, train_loss, val_loss, wandb_metrics)

        # Early stopping / checkpoint metric.
        metric_name = getattr(LSTMConfig, 'BEST_MODEL_METRIC', 'val')
        monitor_loss, metric_label, sig_diag, val_signal_warned, domain_switch = (
            resolve_monitor_loss(
                metric_name=metric_name,
                train_loss=train_loss,
                val_loss=val_loss,
                diag_rec=diag_rec,
                val_signal_ctx=val_signal_ctx,
                epoch=epoch,
                val_signal_warned=val_signal_warned,
                y_val_pred_scaled=y_val_pred_scaled,
                y_val_true_scaled_cached=y_val_true_scaled_cached,
                target_scaler=target_scaler,
                logger=logger,
            ))
        diag_rec.update(sig_diag)

        if domain_switch:
            logger.info(
                f"  [val_signal] epoka {epoch + 1}: przelaczenie domeny checkpointu "
                "z val_loss (dodatnie) na -score sygnalowy (ujemne, VAL_SIGNAL_MIN_EPOCH) "
                "-- reset best_for_checkpoint/best_for_patience/patience_counter, aby "
                "wartosci sprzed przelaczenia (inna skala/znak) nie byly porownywane "
                "z wynikami po przelaczeniu."
            )
            best_for_checkpoint = float('inf')
            best_for_patience = float('inf')
            patience_counter = 0

        # Checkpoint: strict improvement (min_delta does not apply).
        if monitor_loss < best_for_checkpoint:
            best_for_checkpoint = monitor_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            logger.info(
                f"  >> NEW BEST {metric_label}={monitor_loss:.6f} "
                f"(epoch {best_epoch})"
            )

        # Patience: reset only when improvement exceeds min_delta.
        if monitor_loss < best_for_patience - early_stopping_min_delta:
            best_for_patience = monitor_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if (patience_counter >= early_stopping_patience
                    and epoch + 1 >= min_epochs_before_es):
                logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # SWA — running mean of weights on CPU from swa_start_epoch.
        if use_swa and epoch + 1 >= swa_start_epoch:
            epochs_since_start = epoch + 1 - swa_start_epoch
            if epochs_since_start % swa_every == 0:
                n_swa += 1
                swa_state, _ = update_swa(swa_state, model, n_swa)

    # Load best model for final predictions
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        logger.warning(
            "best_state is None — no epoch improved val_loss. "
            "Using last-epoch weights (may contain NaN)."
        )
    model.to(device)

    # Final metrics in original log-return space.
    # B15 fix: train_loader has shuffle=True — two separate iterations would
    # yield differently ordered samples. Use a dedicated shuffle=False loader
    # so predict_all and y_train_true are correctly paired.
    train_eval_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=LSTMConfig.NUM_WORKERS,
        pin_memory=(device.type == 'cuda' and LSTMConfig.USE_PIN_MEMORY),
    )
    y_train_pred = predict_all(model, train_eval_loader, device)
    y_val_pred = predict_all(model, val_loader, device)

    # Collect true targets from loaders (both shuffle=False → correct order)
    y_train_true = np.concatenate(
        [y.numpy() for _, y, *_ in train_eval_loader])
    y_val_true = np.concatenate([y.numpy() for _, y, *_ in val_loader])

    train_metrics = compute_metrics(
        y_train_true, y_train_pred, target_scaler, threshold=signal_threshold
    )
    val_metrics = compute_metrics(
        y_val_true, y_val_pred, target_scaler, threshold=signal_threshold
    )

    logger.info(
        f"\nBest epoch: {best_epoch}, "
        f"Best monitor loss: {best_for_checkpoint:.6f}"
    )

    # F2.0 — emit diagnostics summary to training.log
    diagnostics.log_summary(best_epoch=best_epoch)

    return (best_state, train_losses, val_losses,
            train_metrics, val_metrics, best_epoch,
            diagnostics.history, swa_state)
