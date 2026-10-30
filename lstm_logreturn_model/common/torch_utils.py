"""
PyTorch utilities for LSTM Log-Return pipeline.

Split from training_utils.py to keep modules under 300 lines.
Contains: device detection, seeding, Dataset, schedulers.
"""
import numpy as np
import random
import torch
from torch.utils.data import Dataset


# =========================================================================
# DEVICE DETECTION
# =========================================================================

def get_device():
    """
    Universal GPU detection (NVIDIA CUDA, AMD ROCm) or CPU fallback.

    PyTorch with ROCm uses the same torch.cuda API — no special code needed.
    Prints diagnostic info: GPU name, VRAM, backend (ROCm / CUDA).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        # Detect ROCm vs CUDA for informational logging
        is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
        backend = f"ROCm {torch.version.hip}" if is_rocm else f"CUDA {torch.version.cuda}"
        try:
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            print(f"Device: {device} ({gpu_name}, {vram_mb} MB VRAM, {backend})")
        except Exception:
            print(f"Device: {device} ({gpu_name}, {backend})")
        return device
    print("Device: cpu")
    return torch.device('cpu')


# =========================================================================
# SEEDING
# =========================================================================

def set_seed(seed=42):
    """Set deterministic seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # F8.3: backend-independent determinism (ROCm + CUDA).
        # warn_only=True logs warnings instead of raising RuntimeError
        # for operations without a deterministic implementation.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions don't support warn_only
            pass


# =========================================================================
# DATASET
# =========================================================================

class CryptoDataset(Dataset):
    """PyTorch Dataset for 3D LSTM sequences with optional index/sigma tracking."""

    def __init__(self, X, y, sigma=None):
        """
        Args:
            X: np.ndarray of shape (n, lookback, n_features)
            y: np.ndarray of shape (n,)
            sigma: optional np.ndarray of shape (n,) — market volatility per sample
        """
        self.X = torch.from_numpy(
            np.asarray(X, dtype=np.float32, order='C'))
        self.y = torch.from_numpy(
            np.asarray(y, dtype=np.float32, order='C'))
        self.sigma = None
        if sigma is not None:
            self.sigma = torch.from_numpy(
                np.asarray(sigma, dtype=np.float32, order='C'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.sigma is not None:
            return self.X[idx], self.y[idx], idx, self.sigma[idx]
        return self.X[idx], self.y[idx], idx


# =========================================================================
# SCHEDULERS
# =========================================================================

def get_scheduler(optimizer, scheduler_type, total_steps=None,
                  lr_warmup_epochs=0, **params):
    """
    Factory for PyTorch learning rate schedulers.

    Args:
        optimizer:      torch.optim.Optimizer
        scheduler_type: 'plateau' | 'cosine' | 'onecycle' | 'step' | 'none'
        total_steps:    int — required for 'onecycle'
        **params:       scheduler-specific parameters

    Returns:
        scheduler instance or None
    """
    if scheduler_type == 'none' or scheduler_type is None:
        return None

    warmup_epochs = int(lr_warmup_epochs or 0)

    if scheduler_type == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=params.get('patience', 5),
            factor=params.get('factor', 0.5),
            mode='min',
        )
        return _wrap_warmup(optimizer, sched, warmup_epochs)

    if scheduler_type == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.get('T_0', 10),
            T_mult=params.get('T_mult', 2),
        )
        return _wrap_warmup(optimizer, sched, warmup_epochs)

    if scheduler_type == 'onecycle':
        if total_steps is None:
            raise ValueError("OneCycleLR requires total_steps parameter")
        # OneCycleLR has its own warmup — ignore LR_WARMUP_EPOCHS.
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.get('max_lr', 0.01),
            total_steps=total_steps,
        )

    if scheduler_type == 'step':
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 30),
            gamma=params.get('gamma', params.get('factor', 0.5)),
        )
        return _wrap_warmup(optimizer, sched, warmup_epochs)

    raise ValueError(f"Unknown scheduler_type: '{scheduler_type}'")


class _WarmupScheduler:
    """
    Linear warmup followed by an arbitrary main scheduler.

    Replaces SequentialLR to avoid its step() incompatibility with
    ReduceLROnPlateau: SequentialLR.step() takes no arguments, but plateau
    requires scheduler.step(metrics). This wrapper forwards the optional
    `metrics` argument to the main scheduler after warmup ends.

    Args:
        warmup:        LinearLR instance covering the warmup phase.
        main:          Main scheduler (e.g. ReduceLROnPlateau, CosineAnnealingWarmRestarts).
        warmup_epochs: Number of warmup epochs; must be >= 1.
    """

    def __init__(self, warmup, main, warmup_epochs: int):
        self._warmup = warmup
        self._main = main
        self._warmup_epochs = int(warmup_epochs)
        self._epoch = 0

    def step(self, metrics=None):
        self._epoch += 1
        if self._epoch <= self._warmup_epochs:
            self._warmup.step()
        else:
            if metrics is not None:
                self._main.step(metrics)
            else:
                self._main.step()

    def get_last_lr(self):
        if self._epoch <= self._warmup_epochs:
            return self._warmup.get_last_lr()
        # ReduceLROnPlateau does not inherit from _LRScheduler and has no
        # get_last_lr(). Fall back to reading LR directly from optimizer param
        # groups, which is always accurate regardless of scheduler type.
        if hasattr(self._main, 'get_last_lr'):
            return self._main.get_last_lr()
        return [pg['lr'] for pg in self._warmup.optimizer.param_groups]


def _wrap_warmup(optimizer, scheduler, warmup_epochs: int):
    """Prepend linear LR warmup before the main epoch-level scheduler."""
    if warmup_epochs is None or int(warmup_epochs) <= 0:
        return scheduler
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=int(warmup_epochs),
    )
    return _WarmupScheduler(warmup, scheduler, int(warmup_epochs))
