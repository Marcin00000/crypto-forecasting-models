"""
Standalone XGBoost log-return inference package.

Self-contained: rebuilds a trained XGBoost model from model_config.json +
xgb_model.json and emits probabilistic BUY/SELL signals. No imports from the
training project.
"""
from .xgb_model_package import XGBModelPackage, SignalConfig

__all__ = ['XGBModelPackage', 'SignalConfig']
