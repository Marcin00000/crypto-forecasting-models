"""
Target transformation utilities for Random Forest log-return pipeline.

Includes TargetScaler (standard and robust scaling) and target winsorisation.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

__all__ = [
    'TargetScaler',
    'winsorize_targets',
]


class TargetScaler:
    """
    Scaler for log-return targets.
    Scales log-returns to mean=0, std=1 so the model sees meaningful gradients.
    """

    def __init__(self, method: str = 'standard'):
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler method: {method}")

    def fit(self, y: np.ndarray):
        y_arr = np.asarray(y, dtype=np.float64)
        self.scaler.fit(y_arr.reshape(-1, 1))
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        return self.scaler.transform(y_arr.reshape(-1, 1)).flatten().astype(np.float32)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        return self.scaler.fit_transform(y_arr.reshape(-1, 1)).flatten().astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y_scaled, dtype=np.float64)
        return self.scaler.inverse_transform(y_arr.reshape(-1, 1)).flatten().astype(np.float32)

    def get_params(self) -> dict:
        if self.method == 'standard':
            return {
                'method': self.method,
                'mean': float(self.scaler.mean_[0]),
                'std': float(np.sqrt(self.scaler.var_[0])),
            }
        elif self.method == 'robust':
            return {
                'method': self.method,
                'center': float(self.scaler.center_[0]),
                'scale': float(self.scaler.scale_[0]),
            }
        return {'method': self.method}


def winsorize_targets(
    y_train: np.ndarray,
    y_val: np.ndarray,
    percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Clip extreme log-return targets to reduce the influence of flash crashes.

    Bounds are computed ONLY from y_train to prevent data leakage.
    y_val is clipped with the same bounds.
    """
    lower_pct = 100.0 - percentile
    upper_pct = percentile

    lower = float(np.percentile(y_train, lower_pct / 2))
    upper = float(np.percentile(y_train, 50.0 + upper_pct / 2))

    n_clipped_train = int(np.sum((y_train < lower) | (y_train > upper)))
    n_clipped_val = int(np.sum((y_val < lower) | (y_val > upper)))

    y_train_w = np.clip(y_train, lower, upper).astype(np.float32)
    y_val_w = np.clip(y_val, lower, upper).astype(np.float32)

    pct_train = (n_clipped_train / len(y_train) * 100) if len(y_train) > 0 else 0.0
    pct_val = (n_clipped_val / len(y_val) * 100) if len(y_val) > 0 else 0.0

    print(f"Winsorisation ({percentile}th pct): "
          f"bounds=[{lower:.6f}, {upper:.6f}]  "
          f"clipped train={n_clipped_train} ({pct_train:.2f}%)  "
          f"val={n_clipped_val} ({pct_val:.2f}%)")

    clip_bounds = {'lower': lower, 'upper': upper, 'percentile': float(percentile)}
    return y_train_w, y_val_w, clip_bounds
