"""
Target utilities for LSTM Log-Return pipeline.

Handles log-return target creation, scaling, and winsorisation.
Split from data_utils.py to keep modules under 300 lines.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


# =========================================================================
# TARGET SCALER
# =========================================================================

class TargetScaler:
    """
    Scaler for log-return targets.
    Scales log-returns to mean=0, std=1 so LSTM sees meaningful gradients.
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
        self.scaler.fit(y.reshape(-1, 1))
        return self

    def transform(self, y):
        return self.scaler.transform(y.reshape(-1, 1)).flatten()

    def fit_transform(self, y):
        return self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

    def inverse_transform(self, y_scaled):
        return self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def get_params(self):
        if self.method == 'standard':
            return {
                'method': self.method,
                'mean': float(self.scaler.mean_[0]),
                'std': float(np.sqrt(self.scaler.var_[0]))
            }
        elif self.method == 'robust':
            return {
                'method': self.method,
                'center': float(self.scaler.center_[0]),
                'scale': float(self.scaler.scale_[0])
            }


# =========================================================================
# LOG-RETURN TARGET CREATION
# =========================================================================

def create_log_return_target(df, lookback, horizon):
    """
    Create log-return target aligned with create_sequences output.

    For X[i] = features[i : i+lookback], the last visible price is
    close[i + lookback - 1]. Target y[i] = log(close[i+lookback-1+horizon] /
    close[i+lookback-1]).

    Args:
        df:       DataFrame with 'close' column
        lookback: int — lookback window size
        horizon:  int — prediction horizon

    Returns:
        np.ndarray of shape (n_samples,) with log-return targets
    """
    prices = df['close'].values
    n_samples = len(prices) - lookback - horizon

    if n_samples <= 0:
        raise ValueError(
            f"Insufficient data: need at least {lookback + horizon + 1} samples, "
            f"got {len(prices)}"
        )

    # X[i] window covers rows [i, i+lookback), so the last visible price
    # is close[i + lookback - 1]. Target must use that as 'current'.
    base = lookback - 1
    current = prices[base: base + n_samples]
    future = prices[base + horizon: base + horizon + n_samples]
    y = np.log(future / current).astype(np.float32)

    return y


def load_precomputed_target(data_path, split_name='train'):
    """
    Placeholder for loading pre-computed log-return target from file.

    Will be implemented after data_preparation is updated to save
    log-return targets alongside the prepared data.

    Args:
        data_path:  Path to prepared data directory
        split_name: 'train', 'val', or 'test'

    Returns:
        np.ndarray or None (None = not available, compute on-the-fly)
    """
    # Not available until data_preparation saves log-return targets.
    # Expected file: {data_path}/y_{split_name}_logreturn.npy
    return None


# =========================================================================
# TARGET WINSORISATION
# =========================================================================

def winsorize_targets(y_train, y_val=None, percentile=99.0):
    """
    Clip extreme log-return targets to reduce the influence of flash
    crashes and pump-and-dump events on model training.

    Bounds are computed ONLY from y_train to prevent data leakage.
    y_val is clipped with the same bounds if provided.

    Args:
        y_train:    np.ndarray — training targets (log-returns)
        y_val:      np.ndarray or None — validation targets (log-returns)
        percentile: float in (50, 100) — e.g. 99.0 clips the outer 1%

    Returns:
        y_train_w, y_val_w, clip_bounds
    """
    lower_pct = 100.0 - percentile
    upper_pct = percentile

    lower = float(np.nanpercentile(y_train, lower_pct / 2))
    upper = float(np.nanpercentile(y_train, 50.0 + upper_pct / 2))

    n_clipped_train = int(np.sum((y_train < lower) | (y_train > upper)))
    y_train_w = np.clip(y_train, lower, upper)
    pct_train = n_clipped_train / max(len(y_train), 1) * 100

    if y_val is not None:
        n_clipped_val = int(np.sum((y_val < lower) | (y_val > upper)))
        y_val_w = np.clip(y_val, lower, upper)
        pct_val = n_clipped_val / max(len(y_val), 1) * 100
        val_msg = f"val={n_clipped_val} ({pct_val:.2f}%)"
    else:
        y_val_w = None
        val_msg = "val=None"

    print(f"Winsorisation ({percentile}th pct): "
          f"bounds=[{lower:.6f}, {upper:.6f}]  "
          f"clipped train={n_clipped_train} ({pct_train:.2f}%)  "
          f"{val_msg}")

    clip_bounds = {
        'lower': lower,
        'upper': upper,
        'percentile': percentile,
    }
    return y_train_w, y_val_w, clip_bounds


# =========================================================================
# VOLATILITY NORMALISATION (CP-175)
# =========================================================================

def normalize_target_by_volatility(y_raw: np.ndarray, sigma_aligned: np.ndarray,
                                   eps: float = 1e-6) -> np.ndarray:
    """
    Normalise raw log-returns by market volatility: y_norm = y_raw / max(sigma, eps).
    Preserves dimensionless return units relative to prevailing risk regime.
    """
    y_arr = np.asarray(y_raw, dtype=np.float32)
    sigma_safe = np.maximum(np.asarray(sigma_aligned, dtype=np.float32), eps)
    return y_arr / sigma_safe


def denormalize_target_by_volatility(y_norm: np.ndarray, sigma_aligned: np.ndarray) -> np.ndarray:
    """
    Denormalise predictions back to raw return space: mu = y_norm * sigma.
    """
    y_arr = np.asarray(y_norm, dtype=np.float32)
    sigma = np.asarray(sigma_aligned, dtype=np.float32)
    return y_arr * sigma
