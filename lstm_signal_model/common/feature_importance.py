"""
Permutation importance for the Signal Model.

Answers: which of the ~16-23 input features actually contribute to model
quality? Shuffle one feature at a time along the feature axis, re-run
inference, and measure the degradation in a set of metrics.

Three metrics are reported by default:
    mae           — mean absolute error vs oracle (backwards compatible)
    neg_corr      — -(corr_buy + corr_sell) / 2
    neg_diversity — -signal_diversity(y_pred)
All are "lower = better", so a positive delta after shuffling means the
feature was useful.
"""
from __future__ import annotations

import numpy as np
import torch

from .metrics import signal_correlation, signal_diversity


SUBMODEL_GROUP_KEY = 'submodel_ALL'


def _default_metric_fn(y_pred, y_oracle):
    """Default degradation metric: BCE-like MAE (lower = better)."""
    return float(np.mean(np.abs(y_pred - y_oracle)))


def _neg_corr_metric(y_pred, y_oracle):
    """Negated mean of buy/sell correlation with the oracle (lower = better)."""
    corr = signal_correlation(y_pred, y_oracle)
    return -float((corr['corr_buy'] + corr['corr_sell']) / 2.0)


def _neg_diversity_metric(y_pred, y_oracle):
    """Negated signal diversity — detects shuffles that collapse the output."""
    return -signal_diversity(y_pred)


def default_metric_fns() -> dict:
    """Default metric registry used by compute_permutation_importance()."""
    return {
        'mae': _default_metric_fn,
        'neg_corr': _neg_corr_metric,
        'neg_diversity': _neg_diversity_metric,
    }


def compute_permutation_importance(model, X_val, y_val, feature_names, *,
                                   submodel_preds=None, submodel_names=None,
                                   metric_fns=None, metric_fn=None, n_repeats=5,
                                   device='cpu', batch_size=4096,
                                   seed=42):
    """
    Per-feature shuffle -> delta metric. Higher delta = more important feature.

    Args:
        model:          trained SignalModel (eval mode assumed by caller)
        X_val:          (n, lookback, n_features) validation sequences
        y_val:          (n, 2) oracle targets in [0, 1]
        feature_names:  list[str], len == n_features
        submodel_preds: optional (n, n_submodel_outputs) array of submodel predictions
        submodel_names: optional list[str] of names for submodel outputs
        metric_fns:     dict {metric_name: callable(y_pred, y_oracle) -> float}
                        (lower = better). Default: default_metric_fns().
        metric_fn:      legacy single callable — replaces the 'mae' entry.
        n_repeats:      number of shuffles per feature (mean + std reported)
        device:         torch device
        batch_size:     inference batch size
        seed:           RNG seed for shuffling

    Returns:
        dict with two keys:
          'importances'          {feature_name: (mean_delta_mae, std_delta_mae)}
                                 — unchanged legacy payload
          'importances_by_metric' {metric_name: {feature_name: [mean, std]}}
        When submodel_preds has at least one column, an extra 'submodel_ALL'
        entry permutes every submodel column with the same permutation.
    """
    metric_fns = dict(metric_fns) if metric_fns else default_metric_fns()
    if metric_fn is not None:
        metric_fns['mae'] = metric_fn
    if not metric_fns:
        raise ValueError("metric_fns must contain at least one metric")

    rng = np.random.default_rng(seed)
    n_features = X_val.shape[2]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != "
            f"n_features ({n_features})")

    def _predict(X, sub_preds=None):
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                x_batch = torch.from_numpy(
                    np.asarray(X[start:end], dtype=np.float32)).to(device)
                sub_batch = None
                if sub_preds is not None:
                    sub_batch = torch.from_numpy(
                        np.asarray(sub_preds[start:end], dtype=np.float32)).to(device)
                preds.append(model(x_batch, submodel_inputs=sub_batch).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def _score_all(y_pred):
        return {name: float(fn(y_pred, y_val)) for name, fn in metric_fns.items()}

    baseline_pred = _predict(X_val, submodel_preds)
    baseline_scores = _score_all(baseline_pred)

    by_metric = {name: {} for name in metric_fns}

    def _record(feat_name, shuffled_scores_per_repeat):
        for metric_name in metric_fns:
            deltas = [scores[metric_name] - baseline_scores[metric_name]
                      for scores in shuffled_scores_per_repeat]
            by_metric[metric_name][feat_name] = (
                float(np.mean(deltas)), float(np.std(deltas)))

    for f_idx, name in enumerate(feature_names):
        repeats = []
        for _ in range(n_repeats):
            X_shuffled = X_val.copy()
            perm = rng.permutation(len(X_shuffled))
            X_shuffled[:, :, f_idx] = X_shuffled[perm, :, f_idx]
            repeats.append(_score_all(_predict(X_shuffled, submodel_preds)))
        _record(name, repeats)

    has_submodels = (submodel_preds is not None
                     and getattr(submodel_preds, 'ndim', 0) == 2
                     and submodel_preds.shape[1] > 0)
    if has_submodels:
        n_sub_cols = submodel_preds.shape[1]
        if submodel_names is None or len(submodel_names) != n_sub_cols:
            submodel_names = [f"submodel_{i+1}" for i in range(n_sub_cols)]

        for s_idx, name in enumerate(submodel_names):
            repeats = []
            for _ in range(n_repeats):
                sub_shuffled = submodel_preds.copy()
                perm = rng.permutation(len(sub_shuffled))
                sub_shuffled[:, s_idx] = sub_shuffled[perm, s_idx]
                repeats.append(_score_all(_predict(X_val, sub_shuffled)))
            _record(name, repeats)

        # Group permutation: all submodel channels shuffled with the same
        # permutation, so collinear channels cannot cover for each other.
        repeats = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(submodel_preds))
            sub_shuffled = submodel_preds[perm].copy()
            repeats.append(_score_all(_predict(X_val, sub_shuffled)))
        _record(SUBMODEL_GROUP_KEY, repeats)

    legacy_key = 'mae' if 'mae' in by_metric else next(iter(by_metric))
    return {
        'importances': dict(by_metric[legacy_key]),
        'importances_by_metric': {
            metric: {feat: [vals[0], vals[1]] for feat, vals in feats.items()}
            for metric, feats in by_metric.items()
        },
    }
