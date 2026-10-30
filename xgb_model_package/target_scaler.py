"""
Target scaler reconstruction for xgb_model_package.

The training project stores target scaler parameters as plain JSON in
model_config.json. Rebuilding the sklearn scaler from them avoids unpickling
a class defined in the training project (common.target_utils.TargetScaler),
which would make the package depend on that project being importable.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


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
        """Reconstruct scaler from target_scaler_params stored in model_config.json."""
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
