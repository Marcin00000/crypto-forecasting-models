"""Compatibility shims and scaler feature resolution for lstm_model_package."""
from __future__ import annotations

import contextlib
import sys
import types

from .scaler_registry import DATASET_SCALER_REGISTRY

__all__ = [
    'LRChannelCalibrator',
    'temporary_calibrator_shim',
    'resolve_scaler_columns',
    'DATASET_SCALER_REGISTRY',
]


class LRChannelCalibrator:
    """Channel calibrator stub for unpickling calibrator.pkl outside training repo."""

    def __init__(self, model: object = None, method: str = "isotonic"):
        self.model = model
        self.method = method

    def predict(self, x):
        import numpy as np
        x = np.asarray(x, dtype=np.float64)
        if self.method == "isotonic":
            return self.model.predict(x)
        # Platt / logistic probability in [0, 1] scaled to [0, 100]
        probs = self.model.predict_proba(x.reshape(-1, 1))[:, 1]
        return probs * 100.0


@contextlib.contextmanager
def temporary_calibrator_shim():
    """
    Temporarily register common.calibrator_fit stub in sys.modules during unpickling.
    Restores the original state of sys.modules on exit to avoid namespace pollution.
    """
    created_common = False
    created_calibrator_fit = False
    orig_common = sys.modules.get("common")
    orig_calibrator_fit = sys.modules.get("common.calibrator_fit")

    try:
        if "common" not in sys.modules:
            sys.modules["common"] = types.ModuleType("common")
            created_common = True
        if "common.calibrator_fit" not in sys.modules:
            stub = types.ModuleType("common.calibrator_fit")
            stub.LRChannelCalibrator = LRChannelCalibrator
            sys.modules["common.calibrator_fit"] = stub
            created_calibrator_fit = True
        elif not hasattr(sys.modules["common.calibrator_fit"], "LRChannelCalibrator"):
            sys.modules["common.calibrator_fit"].LRChannelCalibrator = LRChannelCalibrator
        yield
    finally:
        if created_common:
            sys.modules.pop("common", None)
        elif orig_common is not None:
            sys.modules["common"] = orig_common

        if created_calibrator_fit:
            sys.modules.pop("common.calibrator_fit", None)
        elif orig_calibrator_fit is not None:
            sys.modules["common.calibrator_fit"] = orig_calibrator_fit


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
