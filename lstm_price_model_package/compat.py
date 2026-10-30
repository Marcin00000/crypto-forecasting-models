"""Scaler feature column resolution for lstm_price_model_package."""
from __future__ import annotations

from .scaler_registry import DATASET_SCALER_REGISTRY

__all__ = ['resolve_scaler_columns', 'DATASET_SCALER_REGISTRY']


def resolve_scaler_columns(scaler: object, config: dict) -> list[str]:
    """
    Resolve exact ordered feature columns that feature_scaler was fit on.

    Resolution order:
    1. scaler.feature_names_in_ (if scaler was fit on a DataFrame)
    2. config["scaler_feature_cols"] (if explicitly specified in model_config.json)
    3. DATASET_SCALER_REGISTRY.get(data_id) (offline dataset registry)
    4. config["features"] (model features fallback)
    """
    n_in = getattr(scaler, "n_features_in_", None)

    # 1. Direct DataFrame attribute
    feat_names = getattr(scaler, "feature_names_in_", None)
    if feat_names is not None:
        cols = list(feat_names)
        if n_in is None or len(cols) == n_in:
            return cols

    # 2. Config explicit column list
    cols = config.get("scaler_feature_cols")
    if cols and (n_in is None or len(cols) == n_in):
        return cols

    # 3. Offline registry by data_id
    data_id = config.get("data_id", "")
    if data_id in DATASET_SCALER_REGISTRY:
        cols = DATASET_SCALER_REGISTRY[data_id]
        if n_in is None or len(cols) == n_in:
            return cols

    # 4. Fallback to model features
    feats = config.get("features", [])
    if feats and (n_in is None or len(feats) == n_in):
        return feats

    if n_in is not None:
        raise ValueError(
            f"Unable to resolve scaler feature columns for data_id='{data_id}'. "
            f"Scaler expects {n_in} features, but candidate resolution failed."
        )
    return feats
