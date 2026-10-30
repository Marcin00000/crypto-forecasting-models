"""
Target utilities for XGBoost log-return pipeline.

Handles log-return target scaling and winsorisation.
Separated from data_utils.py to keep modules well under 500 lines.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

__all__ = ['TargetScaler', 'winsorize_targets']


class TargetScaler:
    """
    Scaler for log-return targets.
    Scales log-returns to mean=0, std=1 so XGBoost sees meaningful gradients.
    """

    def __init__(self, method='standard'):
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler method: {method}")

    def fit(self, y):
        self.scaler.fit(np.asarray(y).reshape(-1, 1))
        return self

    def transform(self, y):
        return self.scaler.transform(np.asarray(y).reshape(-1, 1)).flatten()

    def fit_transform(self, y):
        return self.scaler.fit_transform(np.asarray(y).reshape(-1, 1)).flatten()

    def inverse_transform(self, y_scaled):
        return self.scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).flatten()

    def get_params(self):
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


def winsorize_targets(y_train, y_val, percentile: float = 99.0):
    """
    Clip extreme log-return targets to reduce the influence of flash crashes
    and pump-and-dump events on model training.

    Bounds are computed ONLY from y_train to prevent data leakage.
    y_val is clipped with the same bounds (no fitting on val).
    """
    lower_pct = 100.0 - percentile
    upper_pct = percentile

    lower = float(np.percentile(y_train, lower_pct / 2.0))
    upper = float(np.percentile(y_train, 50.0 + upper_pct / 2.0))

    n_clipped_train = int(np.sum((y_train < lower) | (y_train > upper)))
    n_clipped_val = int(np.sum((y_val < lower) | (y_val > upper)))

    y_train_w = np.clip(y_train, lower, upper)
    y_val_w = np.clip(y_val, lower, upper)

    pct_train = n_clipped_train / max(len(y_train), 1) * 100.0
    pct_val = n_clipped_val / max(len(y_val), 1) * 100.0

    print(f"Winsorisation ({percentile}th pct): "
          f"bounds=[{lower:.6f}, {upper:.6f}]  "
          f"clipped train={n_clipped_train} ({pct_train:.2f}%)  "
          f"val={n_clipped_val} ({pct_val:.2f}%)")

    clip_bounds = {
        'lower': lower,
        'upper': upper,
        'percentile': percentile,
    }
    return y_train_w, y_val_w, clip_bounds
