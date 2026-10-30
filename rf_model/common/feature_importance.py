"""
Feature importance extraction and permutation importance for Random Forest.

Implements:
  1. Native MDI (Mean Decrease in Impurity) extraction and aggregation.
  2. Longest-base-match mapping from flattened columns to base features.
  3. Grouped Permutation Importance on OOS test data shuffling all channels
     of each base feature simultaneously (Delta MAE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

__all__ = [
    'map_flat_to_base_feature',
    'extract_mdi_importance',
    'aggregate_mdi_to_base',
    'compute_permutation_importance_grouped',
]


def map_flat_to_base_feature(flat_col: str, base_features: list[str]) -> str:
    """
    Map flattened feature column to its base feature using longest base match.

    Prevents prefix collisions such as 'macd_signal_lag1' matching 'macd' instead
    of 'macd_signal', or 'return_15_mean_6' matching 'return_1'.
    """
    if flat_col in base_features:
        return flat_col

    # Sort candidates by length descending so longest base prefix wins
    sorted_bases = sorted(base_features, key=len, reverse=True)
    for base in sorted_bases:
        if flat_col.startswith(base + '_'):
            return base

    # Fallback to direct substring search if naming pattern differs
    for base in sorted_bases:
        if base in flat_col:
            return base

    return flat_col


def extract_mdi_importance(model, flat_features: list[str]) -> dict[str, float]:
    """Extract raw MDI (feature_importances_) per flattened feature column."""
    if not hasattr(model, 'feature_importances_'):
        return {}
    importances = model.feature_importances_
    return {
        col: float(imp)
        for col, imp in zip(flat_features, importances)
    }


def aggregate_mdi_to_base(
    mdi_flat: dict[str, float],
    base_features: list[str],
) -> dict[str, float]:
    """
    Aggregate flattened MDI importances to base features by summation.
    """
    agg: dict[str, float] = {b: 0.0 for b in base_features}
    unmapped_sum = 0.0

    for flat_col, imp in mdi_flat.items():
        base = map_flat_to_base_feature(flat_col, base_features)
        if base in agg:
            agg[base] += imp
        else:
            unmapped_sum += imp

    if unmapped_sum > 0:
        agg['unmapped'] = unmapped_sum

    # Normalize to sum 1.0
    total = sum(agg.values())
    if total > 1e-12:
        agg = {k: float(v / total) for k, v in agg.items()}

    return agg


def compute_permutation_importance_grouped(
    model,
    X_val,
    y_val,
    flat_features: list[str],
    base_features: list[str],
    *,
    n_repeats: int = 5,
    random_state: int = 42,
    target_scaler=None,
) -> dict[str, dict]:
    """
    Grouped permutation feature importance on OOS validation/test set.

    For each base feature, ALL its flattened sub-columns (lags, windows, signals)
    are shuffled simultaneously with the SAME permutation index vector.
    Computes Delta MAE on raw unscaled log-returns.
    """
    X_arr = np.asarray(X_val, dtype=np.float64).copy()
    y_true_raw = np.asarray(y_val, dtype=np.float64)

    preds_base = model.predict(X_arr)
    if target_scaler is not None:
        preds_base = target_scaler.inverse_transform(preds_base)

    baseline_mae = float(mean_absolute_error(y_true_raw, preds_base))

    # Group flattened column indices by base feature
    groups: dict[str, list[int]] = {b: [] for b in base_features}
    for idx, col in enumerate(flat_features):
        base = map_flat_to_base_feature(col, base_features)
        if base in groups:
            groups[base].append(idx)
        else:
            groups.setdefault(base, []).append(idx)

    rng = np.random.RandomState(random_state)
    n_samples = len(X_arr)
    results: dict[str, dict] = {}

    for base, col_indices in groups.items():
        if not col_indices:
            results[base] = {
                'delta_mae_mean': 0.0,
                'delta_mae_std': 0.0,
                'baseline_mae': baseline_mae,
                'n_columns': 0,
            }
            continue

        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(n_samples)
            X_perm = X_arr.copy()
            # Shuffle all associated columns simultaneously
            for c_idx in col_indices:
                X_perm[:, c_idx] = X_arr[perm, c_idx]

            preds_perm = model.predict(X_perm)
            if target_scaler is not None:
                preds_perm = target_scaler.inverse_transform(preds_perm)

            perm_mae = float(mean_absolute_error(y_true_raw, preds_perm))
            deltas.append(perm_mae - baseline_mae)

        results[base] = {
            'delta_mae_mean': float(np.mean(deltas)),
            'delta_mae_std': float(np.std(deltas)),
            'baseline_mae': baseline_mae,
            'n_columns': len(col_indices),
        }

    return results
