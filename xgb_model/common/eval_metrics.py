"""
Evaluation metrics for XGBoost model validation and early stopping.

Provides custom evaluation metric factories that operate in the raw
log-return space (via target_scaler inverse transform) and support
thresholded metrics (t_rmse, pseudo_huber, corr, t_dir_acc).
All custom evaluation metrics return a loss to be minimized.
"""
from __future__ import annotations

from typing import Callable, Any
import numpy as np


def make_custom_eval_metric(
    metric_name: str | None,
    target_scaler: Any | None = None,
    signal_threshold: float | None = None,
    huber_delta: float = 1.0,
    threshold: float | None = None,
) -> tuple[Callable[[np.ndarray, Any], tuple[str, float]] | None, bool]:
    """
    Factory creating a custom eval metric for xgb.train(custom_metric=...).

    Args:
        metric_name: One of 't_rmse', 'pseudo_huber', 'corr', 't_dir_acc', 'rmse', 'none'
        target_scaler: Scaler instance with inverse_transform() method, or None
        signal_threshold: Threshold in raw log-return space (e.g. 0.001)
        huber_delta: Delta parameter for Pseudo-Huber loss
        threshold: Alias for signal_threshold

    Returns:
        (eval_function, is_higher_better)
        If metric_name in ('rmse', 'none', None), returns (None, False).
    """
    if threshold is not None and signal_threshold is None:
        signal_threshold = threshold

    if metric_name in ('rmse', 'none', None):
        return None, False

    clean_name = str(metric_name).strip().lower()

    def custom_eval_fn(preds: np.ndarray, dmatrix: Any) -> tuple[str, float]:
        labels = dmatrix.get_label()

        # Invert scaling to operate in raw log-return domain
        if target_scaler is not None:
            y_raw = target_scaler.inverse_transform(labels)
            y_pred_raw = target_scaler.inverse_transform(preds)
        else:
            y_raw = labels
            y_pred_raw = preds

        y_raw = np.asarray(y_raw, dtype=np.float64).ravel()
        y_pred_raw = np.asarray(y_pred_raw, dtype=np.float64).ravel()

        if len(y_raw) == 0:
            return clean_name, 0.0

        if clean_name == 't_rmse':
            if signal_threshold is not None and signal_threshold > 0:
                mask = np.abs(y_raw) >= signal_threshold
                if np.sum(mask) == 0:
                    score = float(np.sqrt(np.mean((y_pred_raw - y_raw) ** 2)))
                else:
                    score = float(np.sqrt(np.mean((y_pred_raw[mask] - y_raw[mask]) ** 2)))
            else:
                score = float(np.sqrt(np.mean((y_pred_raw - y_raw) ** 2)))
            return 't_rmse', score

        elif clean_name == 'pseudo_huber':
            residuals = y_pred_raw - y_raw
            r = residuals / max(huber_delta, 1e-8)
            loss = (huber_delta ** 2) * (np.sqrt(1.0 + r ** 2) - 1.0)
            if signal_threshold is not None and signal_threshold > 0:
                mask = np.abs(y_raw) >= signal_threshold
                if np.sum(mask) > 0:
                    loss = loss[mask]
            score = float(np.mean(loss))
            return 'pseudo_huber', score

        elif clean_name == 'corr':
            std_true = float(np.std(y_raw))
            std_pred = float(np.std(y_pred_raw))
            if std_true < 1e-12 or std_pred < 1e-12:
                corr = 0.0
            else:
                c = np.corrcoef(y_raw, y_pred_raw)[0, 1]
                corr = float(c) if not np.isnan(c) else 0.0
            # Loss to minimize: 1.0 - correlation
            score = float(1.0 - corr)
            return 'corr_loss', score

        elif clean_name == 't_dir_acc':
            if signal_threshold is not None and signal_threshold > 0:
                mask = np.abs(y_raw) >= signal_threshold
                if np.sum(mask) == 0:
                    mask = np.ones(len(y_raw), dtype=bool)
            else:
                mask = np.ones(len(y_raw), dtype=bool)

            true_dir = np.sign(y_raw[mask])
            pred_dir = np.sign(y_pred_raw[mask])
            acc = float(np.mean(true_dir == pred_dir))
            # Loss to minimize: 1.0 - accuracy
            score = float(1.0 - acc)
            return 't_dir_err', score

        else:
            # Fallback to standard RMSE
            score = float(np.sqrt(np.mean((y_pred_raw - y_raw) ** 2)))
            return 'rmse', score

    return custom_eval_fn, False
