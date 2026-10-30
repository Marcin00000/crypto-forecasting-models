"""
Target transformation and price-to-return inversion utilities for LSTM price model.

Handles scaling, logarithmic inversion, dollar conversions, and target returns.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    'compute_implied_log_returns',
    'compute_actual_log_returns',
    'convert_predictions_to_dollars',
    'winsorize_targets',
]


def compute_implied_log_returns(
    y_pred_scaled: np.ndarray,
    current_prices: np.ndarray,
    price_scaler,
    use_log_prices: bool = True,
) -> np.ndarray:
    """
    Compute implied log-return mu from model output and current entry price.

    When use_log_prices=True:
      1. inverse_transform(y_pred_scaled) -> log(P_hat)
      2. mu = log(P_hat) - log(current_prices)
    When use_log_prices=False:
      1. inverse_transform(y_pred_scaled) -> P_hat
      2. mu = log(P_hat / current_prices)
    """
    y_pred_2d = np.asarray(y_pred_scaled, dtype=np.float64).reshape(-1, 1)
    if price_scaler is not None:
        y_inv = price_scaler.inverse_transform(y_pred_2d).flatten()
    else:
        y_inv = y_pred_2d.flatten()

    cur = np.asarray(current_prices, dtype=np.float64).flatten()
    cur_safe = np.maximum(cur, 1e-12)

    if use_log_prices:
        log_pred = y_inv
        log_cur = np.log(cur_safe)
        mu = log_pred - log_cur
    else:
        pred_safe = np.maximum(y_inv, 1e-12)
        mu = np.log(pred_safe / cur_safe)

    return mu.astype(np.float64)


def compute_actual_log_returns(
    future_prices: np.ndarray,
    current_prices: np.ndarray,
) -> np.ndarray:
    """
    Compute ground-truth log-return r_true = log(P_{t+H} / P_t).
    Identical to create_log_return_target in the LR pipeline.
    """
    fut = np.asarray(future_prices, dtype=np.float64).flatten()
    cur = np.asarray(current_prices, dtype=np.float64).flatten()
    fut_safe = np.maximum(fut, 1e-12)
    cur_safe = np.maximum(cur, 1e-12)
    r_true = np.log(fut_safe / cur_safe)
    return r_true.astype(np.float64)


def convert_predictions_to_dollars(
    y_pred_scaled: np.ndarray,
    price_scaler,
    use_log_prices: bool = True,
) -> np.ndarray:
    """
    Convert model output to raw price space in dollars ($) for metrics and plotting.

    When use_log_prices=True:
      log(P_hat) = inverse_transform(y_pred_scaled)
      P_hat = exp(log(P_hat))
    When use_log_prices=False:
      P_hat = inverse_transform(y_pred_scaled)
    """
    y_pred_2d = np.asarray(y_pred_scaled, dtype=np.float64).reshape(-1, 1)
    if price_scaler is not None:
        y_inv = price_scaler.inverse_transform(y_pred_2d).flatten()
    else:
        y_inv = y_pred_2d.flatten()

    if use_log_prices:
        dollars = np.exp(y_inv)
    else:
        dollars = y_inv

    return dollars.astype(np.float64)


def winsorize_targets(
    y: np.ndarray,
    lower_p: float = 0.5,
    upper_p: float = 99.5,
) -> np.ndarray:
    """
    Optional winsorization of target values for outlier damping.
    Default pipeline standard leaves this disabled (winsorize_target=False).
    """
    arr = np.asarray(y, dtype=np.float64).copy()
    low = np.percentile(arr, lower_p)
    high = np.percentile(arr, upper_p)
    return np.clip(arr, low, high)
