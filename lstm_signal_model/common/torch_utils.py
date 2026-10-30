"""
PyTorch utilities for LSTM Signal Model pipeline.

Independent implementation — isolated from lstm_logreturn_model.
Contains: device detection (ROCm/CUDA), seeding, CryptoDataset, schedulers.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================================
# DEVICE DETECTION
# =========================================================================

def get_device():
    """
    Universal GPU detection (NVIDIA CUDA, AMD ROCm) or CPU fallback.

    PyTorch with ROCm uses torch.cuda API seamlessly.
    Logs diagnostic info: GPU name, VRAM, backend (ROCm / CUDA).
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
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
    """Set deterministic seed for reproducibility across PyTorch, NumPy, Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            pass


def seed_worker(worker_id):
    """Worker initialization function for DataLoader determinism."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================================
# DATASET
# =========================================================================

class CryptoDataset(Dataset):
    """
    PyTorch Dataset for 3D LSTM sequences in Signal Model.

    Supports:
        X: (n, lookback, n_features)
        y: (n, 2) for dual sigmoid (p_buy, p_sell) or (n, 1) for single
        sigma: optional (n,) volatility series per sample
        submodel_preds: optional (n, submodel_dim) pre-computed submodel predictions
    """

    def __init__(self, X, y, sigma=None, submodel_preds=None):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32, order='C'))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32, order='C'))
        self.sigma = None
        if sigma is not None:
            self.sigma = torch.from_numpy(np.asarray(sigma, dtype=np.float32, order='C'))
        self.submodel_preds = None
        if submodel_preds is not None:
            self.submodel_preds = torch.from_numpy(
                np.asarray(submodel_preds, dtype=np.float32, order='C'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.sigma is not None:
            base = (self.X[idx], self.y[idx], idx, self.sigma[idx])
        else:
            base = (self.X[idx], self.y[idx], idx)

        if self.submodel_preds is not None:
            return base + (self.submodel_preds[idx],)
        return base


# =========================================================================
# SCHEDULERS
# =========================================================================

class _LinearWarmupWrapper:
    """Wrapper applying linear warmup before main scheduler."""

    def __init__(self, optimizer, base_sched, warmup_epochs):
        self.optimizer = optimizer
        self.base_sched = base_sched
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.target_lrs = [group['lr'] for group in optimizer.param_groups]
        self._current_epoch = 0

    def step(self, metrics=None):
        self._current_epoch += 1
        if self._current_epoch <= self.warmup_epochs:
            frac = self._current_epoch / float(self.warmup_epochs)
            for group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
                group['lr'] = target_lr * frac
        else:
            if metrics is not None and isinstance(self.base_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.base_sched.step(metrics)
            else:
                self.base_sched.step()


def get_scheduler(optimizer, scheduler_type, total_steps=None,
                  lr_warmup_epochs=0, **params):
    """
    Factory for PyTorch learning rate schedulers.

    Args:
        optimizer:      torch.optim.Optimizer
        scheduler_type: 'plateau' | 'cosine' | 'onecycle' | 'step' | 'none'
        total_steps:    int — required for 'onecycle'
        lr_warmup_epochs: int — linear warmup epochs
        **params:       scheduler-specific parameters
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
        return _LinearWarmupWrapper(optimizer, sched, warmup_epochs) if warmup_epochs > 0 else sched

    if scheduler_type == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.get('T_0', 10),
            T_mult=params.get('T_mult', 2),
        )
        return _LinearWarmupWrapper(optimizer, sched, warmup_epochs) if warmup_epochs > 0 else sched

    if scheduler_type == 'onecycle':
        if total_steps is None:
            raise ValueError("OneCycleLR requires total_steps parameter")
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
        return _LinearWarmupWrapper(optimizer, sched, warmup_epochs) if warmup_epochs > 0 else sched

    raise ValueError(f"Unknown scheduler_type: '{scheduler_type}'")
