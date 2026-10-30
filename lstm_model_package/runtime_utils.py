"""
Small self-contained helpers for lstm_model_package.

Kept apart from the package facade so the facade stays well below the repo's
500-LOC ceiling; none of these helpers depend on the training project.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler


class MinRowsInt(int):
    """Integer that also acts as a callable accepting mode ('raw' | 'prepared').

    Older callers read ``min_rows_required`` as a property (int); newer ones call
    ``min_rows_required('prepared')``. One object satisfies both conventions.
    """

    def __new__(cls, val: int, raw_val: int, prep_val: int):
        obj = super().__new__(cls, val)
        obj._raw_val = raw_val
        obj._prep_val = prep_val
        return obj

    def __call__(self, mode: str = 'raw') -> int:
        return self._prep_val if mode in ('prepared', 'prep') else self._raw_val


def apply_precision(arr: np.ndarray, decimals: Optional[int]) -> np.ndarray:
    """Post-hoc decimal rounding (0 or None = disabled)."""
    if decimals is None or decimals <= 0:
        return arr
    return np.round(arr, decimals=int(decimals))


class TargetScaler:
    """Thin wrapper around sklearn scalers for 1-D log-return arrays."""

    def __init__(self, method: str = 'robust'):
        self.method = method
        if method == 'robust':
            self._scaler = RobustScaler()
        elif method == 'standard':
            self._scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler method: '{method}'")

    @classmethod
    def from_params(cls, params: dict) -> 'TargetScaler':
        obj = cls(method=params['method'])
        if params['method'] == 'robust':
            obj._scaler.center_ = np.array([params['center']])
            obj._scaler.scale_ = np.array([params['scale']])
        elif params['method'] == 'standard':
            obj._scaler.mean_ = np.array([params['mean']])
            obj._scaler.scale_ = np.array([params['std']])
            obj._scaler.var_ = np.array([params['std'] ** 2])
            obj._scaler.n_features_in_ = 1
        return obj

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).flatten()
