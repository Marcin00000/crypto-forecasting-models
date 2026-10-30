"""
Permutation importance for the LSTM Log-Return model.

Ported from `lstm_signal_model/common/feature_importance.py` (CP-085, Faza 8).
Answers: which input features actually contribute to model quality? Shuffle
one feature at a time along the feature axis, re-run inference, and measure
the degradation in a chosen metric (default: MAE in scaled log-return space).
"""
from __future__ import annotations

import numpy as np
import torch


def _default_metric_fn(y_pred, y_true):
    """Default degradation metric: MAE (lower = better)."""
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_permutation_importance(model, X_val, y_val, feature_names, *,
                                   metric_fn=None, n_repeats=5,
                                   device='cpu', batch_size=1024,
                                   seed=42):
    """
    Per-feature shuffle -> delta metric. Higher delta = more important feature.

    Args:
        model:         trained LSTMLogReturnModel (eval mode assumed by caller)
        X_val:         (n, lookback, n_features) validation sequences
        y_val:         (n,) targets (scaled log-return space)
        feature_names: list[str], len == n_features
        metric_fn:     callable(y_pred, y_true) -> float (lower = better).
                       Default: mean absolute error.
        n_repeats:     number of shuffles per feature (mean + std reported)
        device:        torch device
        batch_size:    inference batch size
        seed:          RNG seed for shuffling

    Returns:
        dict {feature_name: (mean_delta, std_delta)}
    """
    metric_fn = metric_fn or _default_metric_fn
    rng = np.random.default_rng(seed)
    n_features = X_val.shape[2]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != "
            f"n_features ({n_features})")

    def _predict(X):
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                x_batch = torch.from_numpy(
                    np.asarray(X[start:end], dtype=np.float32)).to(device)
                out = model(x_batch).squeeze(-1).cpu().numpy()
                preds.append(np.atleast_1d(out))
        return np.concatenate(preds, axis=0)

    baseline_pred = _predict(X_val)
    baseline_score = metric_fn(baseline_pred, y_val)

    importances = {}
    for f_idx, name in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_shuffled = X_val.copy()
            perm = rng.permutation(len(X_shuffled))
            X_shuffled[:, :, f_idx] = X_shuffled[perm, :, f_idx]
            shuffled_pred = _predict(X_shuffled)
            shuffled_score = metric_fn(shuffled_pred, y_val)
            deltas.append(shuffled_score - baseline_score)
        importances[name] = (float(np.mean(deltas)), float(np.std(deltas)))

    return importances
