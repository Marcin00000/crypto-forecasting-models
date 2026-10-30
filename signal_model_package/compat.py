"""
Compatibility shim and scaler resolution for standalone signal_model_package.
Provides dynamic unpickler shims for legacy/external environments and resolves
feature column ordering across diverse datasets.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np

from .scaler_registry import DATASET_SCALER_REGISTRY


# =========================================================================
# CALIBRATOR STUB FOR UNPICKLING
# =========================================================================

class _DualSignalCalibratorStub:
    """Stub class matching common.calibrator_fit.DualSignalCalibrator."""
    def __init__(self, *args, **kwargs):
        self.method = kwargs.get('method', 'isotonic')
        self.calibrator_buy = None
        self.calibrator_sell = None
        self.is_fitted = False

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.asarray(y_pred, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        # IsotonicRegression.predict returns the calibrated probability.
        # LogisticRegression.predict returns a class label; probabilities
        # come from predict_proba, matching DualSignalCalibrator.transform.
        if self.method == 'isotonic':
            cal_buy = self.calibrator_buy.predict(y_pred[:, 0])
            cal_sell = self.calibrator_sell.predict(y_pred[:, 1])
        else:
            cal_buy = self.calibrator_buy.predict_proba(y_pred[:, 0].reshape(-1, 1))[:, 1]
            cal_sell = self.calibrator_sell.predict_proba(y_pred[:, 1].reshape(-1, 1))[:, 1]
        return np.column_stack([cal_buy, cal_sell])


@contextmanager
def temporary_calibrator_shim():
    """
    Context manager injecting dynamic stubs for calibrators into sys.modules.
    Allows unpickling calibrator.pkl outside of the training repository.
    """
    original_common = sys.modules.get('common')
    original_cal_fit = sys.modules.get('common.calibrator_fit')

    try:
        if 'common' not in sys.modules:
            common_mod = types.ModuleType('common')
            sys.modules['common'] = common_mod
        else:
            common_mod = sys.modules['common']

        cal_mod = types.ModuleType('common.calibrator_fit')
        cal_mod.DualSignalCalibrator = _DualSignalCalibratorStub
        sys.modules['common.calibrator_fit'] = cal_mod
        setattr(common_mod, 'calibrator_fit', cal_mod)

        yield
    finally:
        if original_cal_fit is not None:
            sys.modules['common.calibrator_fit'] = original_cal_fit
        else:
            sys.modules.pop('common.calibrator_fit', None)

        if original_common is not None:
            sys.modules['common'] = original_common
        else:
            sys.modules.pop('common', None)


# =========================================================================
# SCALER COLUMN RESOLUTION
# =========================================================================

def resolve_scaler_columns(scaler: Any, config: dict) -> list[str]:
    """
    Determine the exact list and order of feature columns expected by feature_scaler.pkl.

    Hierarchy:
    1. scaler.feature_names_in_ (if preserved in scikit-learn scaler)
    2. config['scaler_feature_cols'] (explicit list saved in model_config.json)
    3. DATASET_SCALER_REGISTRY[data_id] (exact match from metadata.json)
    4. config['features'] (fallback to model's input feature subset)
    """
    if scaler is None:
        return []

    # 1. Direct sklearn attribute
    if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
        cols = list(scaler.feature_names_in_)
        if hasattr(scaler, 'n_features_in_') and len(cols) == scaler.n_features_in_:
            return cols

    # 2. Explicit config field
    if 'scaler_feature_cols' in config and config['scaler_feature_cols']:
        cols = list(config['scaler_feature_cols'])
        if hasattr(scaler, 'n_features_in_') and len(cols) == scaler.n_features_in_:
            return cols

    # 3. Static registry from dataset metadata.json
    data_id = str(config.get('data_id', '')).strip().upper()
    if data_id in DATASET_SCALER_REGISTRY:
        cols = list(DATASET_SCALER_REGISTRY[data_id])
        if hasattr(scaler, 'n_features_in_'):
            if len(cols) == scaler.n_features_in_:
                return cols

    # 4. Fallback to model features. A length mismatch here would make transform()
    # fail later (or, worse, mis-assign columns), so fail now with the cause.
    features = list(config.get('features', []))
    n_in = getattr(scaler, 'n_features_in_', None)
    if n_in is not None and len(features) != n_in:
        raise ValueError(
            f"Unable to resolve scaler feature columns for data_id='{config.get('data_id', '')}'. "
            f"Scaler expects {n_in} features, but no candidate list matched "
            f"(model features: {len(features)})."
        )
    return features
