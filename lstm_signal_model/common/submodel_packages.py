"""
Adapter between SubmodelManager and the standalone inference packages of the
XGBoost, Random Forest and Price-LSTM families.

Each non-LR submodel emits one channel of shape (N, 1) for the Signal fusion head:
  - xgb / rf: mu (log-return predicted by the model),
  - price:    P_hat (predicted price in quote currency, the quantity the price
              model actually predicts; LR models are the dedicated source of mu).

The packages are imported lazily so that LR-only runs do not need xgboost.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# Detection order matters: the first marker file found in the directory wins.
_KIND_MARKERS = (
    ('xgb', 'xgb_model.json'),
    ('rf', 'rf_model.pkl'),
    ('price', 'best_model.pth'),
    ('price', 'lstm_price_model.pth'),
    ('lr', 'lstm_model.pth'),
)
PACKAGE_KINDS = ('xgb', 'rf', 'price')
_CHUNK = 50_000  # bounds flattening memory (hybrid on 39 features is ~600 columns)


def detect_submodel_kind(model_dir: str) -> str:
    """Return 'lr' | 'xgb' | 'rf' | 'price' based on the weight files in model_dir."""
    for kind, marker in _KIND_MARKERS:
        if os.path.exists(os.path.join(str(model_dir), marker)):
            return kind
    return 'lr'  # backward compatible: the LR loader reports missing weights itself


def load_package(kind: str, model_dir: str, device):
    """Load the sibling inference package for kind; returns (package, model_config dict)."""
    model_dir = str(model_dir)
    # window_sigma only drives the volatility estimate, which is not used here; the
    # smallest value keeps the packages' minimum-row validation out of the way.
    if kind == 'xgb':
        from xgb_model_package import XGBModelPackage, SignalConfig
        pkg = XGBModelPackage.from_directory(model_dir, signal_config=SignalConfig(window_sigma=1))
    elif kind == 'rf':
        from rf_model_package import RFModelPackage, SignalConfig
        pkg = RFModelPackage.from_directory(model_dir, signal_config=SignalConfig(window_sigma=1))
    elif kind == 'price':
        from lstm_price_model_package import LSTMPriceModelPackage, SignalConfig
        pkg = LSTMPriceModelPackage.from_directory(
            model_dir, signal_config=SignalConfig(window_sigma=1), device=str(device))
    else:
        raise ValueError(f"load_package: unsupported submodel kind '{kind}'")
    with open(os.path.join(model_dir, 'model_config.json'), 'r', encoding='utf-8') as f:
        config = json.load(f)
    return pkg, config


def submodel_lookback(config: dict) -> int:
    """Window length the submodel was trained with."""
    return int(config.get('lookback', 1))


def predict_package_output(kind: str, pkg, X: np.ndarray, features: list[str],
                           sub_lookback: int = 1, chunk: int = _CHUNK) -> np.ndarray:
    """
    One channel (N, 1) float32 for the Signal samples X of shape (N, L, F).

    Sample i is the window ending at candle t = i + L - 1 and may only use rows <= t.
    Price: the window itself is the model input -> P_hat for t + horizon.
    XGB / RF: features are shifted by one candle inside the package (row r sees rows
    <= r-1), so the decision row is t + 1; a placeholder row is appended whose own
    feature values are never read. Chunks carry their own L-1 candles of history,
    so the result does not depend on the chunk size.
    """
    n_total, window, n_feat = X.shape
    if n_feat != len(features):
        raise ValueError(f"X has {n_feat} features, submodel expects {len(features)}")
    if window < sub_lookback:
        raise ValueError(
            f"Signal lookback ({window}) is shorter than the {kind} submodel lookback "
            f"({sub_lookback}); the first samples would lack history.")
    column = 'predicted_price' if kind == 'price' else 'mu'
    parts = []
    for start in range(0, n_total, chunk):
        xb = X[start:start + chunk]
        if len(xb) > 1 and not np.array_equal(xb[1:, :-1], xb[:-1, 1:]):
            raise ValueError("Signal windows are not consecutive candles (stride 1 required)")
        series = np.concatenate([xb[0, :-1], xb[:, -1]]).astype(np.float64)
        if kind != 'price':
            series = np.vstack([series, series[-1:]])
        if not np.isfinite(series).all():
            raise ValueError(f"{kind} submodel input contains NaN/inf values")
        df = pd.DataFrame(series, columns=features)
        # close is only required by the packages' input validation; neither mu of the
        # tabular models nor predicted_price depends on it.
        df['close'] = 1.0
        out = pkg.predict_batch(df, data_mode='prepared', drop_warmup=False,
                                return_df=False)[column]
        if len(out) < len(xb):
            raise ValueError(
                f"{kind} submodel returned {len(out)} rows for {len(xb)} samples: some rows "
                f"have undefined features (not enough history in the Signal window for the "
                f"submodel's flattening, e.g. 'hybrid' diff_5 needs 6 candles).")
        parts.append(np.asarray(out)[-len(xb):])
    return np.concatenate(parts).reshape(-1, 1).astype(np.float32)
