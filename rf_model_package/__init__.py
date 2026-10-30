"""
Standalone Random Forest log-return inference package.

Self-contained: loads rf_model.pkl + model_config.json and emits probabilistic BUY/SELL
signals (same math as xgb_model_package). No imports from the training project.
"""
from .rf_model_package import RFModelPackage, SignalConfig

__all__ = ['RFModelPackage', 'SignalConfig']
