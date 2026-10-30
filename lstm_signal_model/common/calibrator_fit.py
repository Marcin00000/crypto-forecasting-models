"""
Probability calibrator for Dual Sigmoid predictions (p_buy, p_sell).

Independent implementation for lstm_signal_model.
Supports Platt Scaling (Logistic Regression) and Isotonic Regression
per output channel independently.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class DualSignalCalibrator:
    """
    Independent calibrator for dual output probabilities (p_buy, p_sell).

    Aligns raw model output probabilities [0, 1] to true empirical
    win probabilities.
    """

    def __init__(self, method: str = 'isotonic'):
        if method not in ('isotonic', 'platt', 'logistic'):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = 'logistic' if method == 'platt' else method
        self.calibrator_buy = None
        self.calibrator_sell = None
        self.is_fitted = False

    def fit(self, y_pred, y_oracle):
        """
        Fit calibrators for buy and sell channels.

        Args:
            y_pred:   (n, 2) raw probabilities
            y_oracle: (n, 2) binary or soft oracle targets in [0, 1]
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_oracle = np.asarray(y_oracle, dtype=np.float64)

        # Convert targets to binary labels >= 0.5 for fitting
        b_buy = (y_oracle[:, 0] >= 0.5).astype(int)
        b_sell = (y_oracle[:, 1] >= 0.5).astype(int)

        if self.method == 'isotonic':
            self.calibrator_buy = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            self.calibrator_sell = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
            self.calibrator_buy.fit(y_pred[:, 0], b_buy)
            self.calibrator_sell.fit(y_pred[:, 1], b_sell)
        else:  # 'logistic' / 'platt'
            self.calibrator_buy = LogisticRegression(C=1.0, solver='lbfgs')
            self.calibrator_sell = LogisticRegression(C=1.0, solver='lbfgs')
            self.calibrator_buy.fit(y_pred[:, 0].reshape(-1, 1), b_buy)
            self.calibrator_sell.fit(y_pred[:, 1].reshape(-1, 1), b_sell)

        self.is_fitted = True
        return self

    def transform(self, y_pred):
        """
        Transform raw probabilities to calibrated probabilities.

        Args:
            y_pred: (n, 2)
        Returns:
            calibrated_pred: (n, 2)
        """
        if not self.is_fitted:
            return np.asarray(y_pred, dtype=np.float64)

        y_pred = np.asarray(y_pred, dtype=np.float64)
        cal_buy = np.zeros(len(y_pred), dtype=np.float64)
        cal_sell = np.zeros(len(y_pred), dtype=np.float64)

        if self.method == 'isotonic':
            cal_buy = self.calibrator_buy.predict(y_pred[:, 0])
            cal_sell = self.calibrator_sell.predict(y_pred[:, 1])
        else:
            cal_buy = self.calibrator_buy.predict_proba(y_pred[:, 0].reshape(-1, 1))[:, 1]
            cal_sell = self.calibrator_sell.predict_proba(y_pred[:, 1].reshape(-1, 1))[:, 1]

        return np.column_stack([cal_buy, cal_sell])
