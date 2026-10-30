"""
Light 1-D target scaler rebuilt from `target_scaler_params` stored in model_config.json.

Rebuilding from plain numbers avoids unpickling rf_model.common.target_utils.TargetScaler,
which would make the package depend on the training code base.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler


class _TargetScaler:
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
    def from_params(cls, params: dict) -> '_TargetScaler':
        """Reconstruct the scaler from target_scaler_params in model_config.json."""
        obj = cls(method=params['method'])
        if params['method'] == 'robust':
            obj._scaler.center_ = np.array([params['center']])
            obj._scaler.scale_ = np.array([params['scale']])
            obj._scaler.n_features_in_ = 1
        elif params['method'] == 'standard':
            obj._scaler.mean_ = np.array([params['mean']])
            obj._scaler.scale_ = np.array([params['std']])
            obj._scaler.var_ = np.array([params['std'] ** 2])
            obj._scaler.n_features_in_ = 1
        return obj

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).flatten()
