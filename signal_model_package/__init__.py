"""
Standalone LSTM Signal Inference Package.

Self-contained: rebuilds a trained LSTM Signal Model from model_config.json + signal_model.pth
and emits direct probabilistic BUY/SELL signals ([prob_buy, prob_sell] in [0, 1] and [0, 100]).
No imports from the training repository.
"""
from .signal_model_package import SignalModelPackage, SignalInferenceConfig

__all__ = ['SignalModelPackage', 'SignalInferenceConfig']
