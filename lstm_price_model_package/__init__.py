"""
Standalone price-LSTM inference package.

Self-contained: rebuilds a trained price LSTM from model_config.json + best_model.pth,
derives the implied log-return mu = log(P_hat) - log(P_t) and emits probabilistic
BUY/SELL signals (Gaussian approach). No imports from the training project.
"""
from .lstm_price_model_package import LSTMPriceModelPackage, SignalConfig

__all__ = ['LSTMPriceModelPackage', 'SignalConfig']
