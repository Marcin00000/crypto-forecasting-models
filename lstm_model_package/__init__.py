"""
Standalone LSTM log-return inference package.

Self-contained: rebuilds a trained LSTM from model_config.json + lstm_model.pth
and emits probabilistic BUY/SELL signals (same math as xgb_model_package).
No imports from the training project.
"""
from .lstm_model_package import LSTMModelPackage, SignalConfig

__all__ = ['LSTMModelPackage', 'SignalConfig']
