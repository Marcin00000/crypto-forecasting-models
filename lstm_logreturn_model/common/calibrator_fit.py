"""Post-hoc calibration on validation signals (opt-in)."""
from __future__ import annotations

import os
import joblib
import numpy as np

from .simulation import (
    align_sigma, compute_sigma_series, compute_signals, fill_sigma_nan,
)
from .training_utils import predict_all


class LRChannelCalibrator:
    """Channel calibrator implementing predict(signal) returning 0-100 range."""

    def __init__(self, model, method: str):
        self.model = model
        self.method = method

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.method == 'isotonic':
            return self.model.predict(x)
        # Platt / logistic: returns probability in [0, 1] scaled to [0, 100]
        probs = self.model.predict_proba(x.reshape(-1, 1))[:, 1]
        return probs * 100.0


class LRSignalCalibrator:
    """
    Calibrator for LR signals (0-100 scale).
    Supports 'isotonic' and 'platt'/'logistic' regression.
    Produces dictionary compatible with calibrator.pkl consumers.
    """

    def __init__(self, method: str = 'isotonic', threshold: float = 0.001,
                 window_sigma: int = 100):
        if method not in ('isotonic', 'platt', 'logistic'):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = 'logistic' if method == 'platt' else method
        self.threshold = float(threshold)
        self.window_sigma = int(window_sigma)
        self.calibrator_buy = None
        self.calibrator_sell = None
        self.is_fitted = False

    def fit(self, signal_buy, signal_sell, y_true):
        signal_buy = np.asarray(signal_buy, dtype=np.float64)
        signal_sell = np.asarray(signal_sell, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        buy_target = (y_true > self.threshold).astype(int)
        sell_target = (y_true < -self.threshold).astype(int)

        if self.method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            if np.any(signal_buy > 0):
                reg_b = IsotonicRegression(out_of_bounds='clip')
                reg_b.fit(signal_buy, buy_target)
                self.calibrator_buy = reg_b
            if np.any(signal_sell > 0):
                reg_s = IsotonicRegression(out_of_bounds='clip')
                reg_s.fit(signal_sell, sell_target)
                self.calibrator_sell = reg_s
        else:
            from sklearn.linear_model import LogisticRegression
            if np.any(signal_buy > 0) and len(np.unique(buy_target)) > 1:
                clf_b = LogisticRegression(C=1.0, solver='lbfgs')
                clf_b.fit(signal_buy.reshape(-1, 1), buy_target)
                self.calibrator_buy = LRChannelCalibrator(clf_b, 'logistic')
            if np.any(signal_sell > 0) and len(np.unique(sell_target)) > 1:
                clf_s = LogisticRegression(C=1.0, solver='lbfgs')
                clf_s.fit(signal_sell.reshape(-1, 1), sell_target)
                self.calibrator_sell = LRChannelCalibrator(clf_s, 'logistic')

        self.is_fitted = (self.calibrator_buy is not None or self.calibrator_sell is not None)
        return self

    def to_dict(self) -> dict:
        return {
            'threshold': float(self.threshold),
            'window_sigma': int(self.window_sigma),
            'method': self.method,
            'buy': self.calibrator_buy,
            'sell': self.calibrator_sell,
        }


# DualSignalCalibrator alias for cross-module naming parity
DualSignalCalibrator = LRSignalCalibrator


def fit_and_save_calibrator(model, val_loader, device, target_scaler,
                            y_val_raw, val_close, lookback, horizon,
                            signal_threshold, cfg, output_dir, logger,
                            method: str = 'isotonic'):
    """Fit calibrators on val signals and save calibrator.pkl."""
    model.eval()
    y_pred_scaled = predict_all(model, val_loader, device)
    if target_scaler is not None:
        y_pred_raw = target_scaler.inverse_transform(y_pred_scaled)
    else:
        y_pred_raw = y_pred_scaled

    n = min(len(y_val_raw), len(y_pred_raw))
    y_true = np.asarray(y_val_raw[:n], dtype=np.float64)
    y_pred = np.asarray(y_pred_raw[:n], dtype=np.float64)
    threshold = (signal_threshold
                 if signal_threshold is not None
                 else cfg.SIGNAL_THRESHOLD_VALUE)

    sigma_series = compute_sigma_series(val_close, cfg.SIGNAL_WINDOW_SIGMA)
    sigma = align_sigma(sigma_series, lookback, n)
    sigma = fill_sigma_nan(sigma)

    signal_buy, signal_sell = compute_signals(y_pred, sigma, threshold)

    calibrator = LRSignalCalibrator(
        method=method, threshold=threshold, window_sigma=cfg.SIGNAL_WINDOW_SIGMA)
    calibrator.fit(signal_buy, signal_sell, y_true)
    cal_dict = calibrator.to_dict()

    out_path = os.path.join(output_dir, 'calibrator.pkl')
    joblib.dump(cal_dict, out_path)
    logger.info(f"  ✓ calibrator.pkl (method={method}, buy={cal_dict['buy'] is not None}, "
                f"sell={cal_dict['sell'] is not None})")
    return calibrator
