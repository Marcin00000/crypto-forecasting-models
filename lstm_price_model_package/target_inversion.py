"""
Inversion of the scaled price-model output into implied log-returns and dollar prices.

Mirrors ``lstm_model/common/target_utils.py`` (compute_implied_log_returns and
convert_predictions_to_dollars) so that package output equals stored test results.
"""
from __future__ import annotations

import numpy as np


def _inverse_price(y_pred_scaled: np.ndarray, price_scaler) -> np.ndarray:
    """Undo the target scaling; datasets without a price scaler ('none') are used as-is."""
    y_2d = np.asarray(y_pred_scaled, dtype=np.float64).reshape(-1, 1)
    if price_scaler is not None:
        return price_scaler.inverse_transform(y_2d).flatten()
    return y_2d.flatten()


def implied_log_returns(y_pred_scaled: np.ndarray, current_prices: np.ndarray,
                        price_scaler, use_log_prices: bool = True) -> np.ndarray:
    """
    mu = log(P_hat) - log(P_t).

    With ``use_log_prices`` the (inverse-scaled) output already is log(P_hat); otherwise
    it is the price itself and mu = log(P_hat / P_t).
    """
    y_inv = _inverse_price(y_pred_scaled, price_scaler)
    cur_safe = np.maximum(np.asarray(current_prices, dtype=np.float64).flatten(), 1e-12)
    if use_log_prices:
        mu = y_inv - np.log(cur_safe)
    else:
        mu = np.log(np.maximum(y_inv, 1e-12) / cur_safe)
    return mu.astype(np.float64)


def predicted_dollars(y_pred_scaled: np.ndarray, price_scaler, use_log_prices: bool = True) -> np.ndarray:
    """Predicted price P_hat in quote currency."""
    y_inv = _inverse_price(y_pred_scaled, price_scaler)
    return (np.exp(y_inv) if use_log_prices else y_inv).astype(np.float64)
