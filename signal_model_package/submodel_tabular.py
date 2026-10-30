"""
Embedded inference for XGBoost and Random Forest submodels of the Signal fusion model.

The package must stay copyable without sibling packages, so the flattening of
xgb_model_package / rf_model_package is duplicated here; tests compare this copy with
both siblings on synthetic data. Every flattening method shifts features by one candle:
row r only sees rows <= r-1.

Alignment with Signal samples: sample i ends at candle t = i + window - 1 and is decided
on row t + 1. The series therefore gets one placeholder row at the end whose own feature
values are never read (shift by one candle), so no information after t is used.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_CHUNK = 50_000


# ---------------------------------------------------------------------------
# Flattening (copy of xgb_model_package/flattening.py)
# ---------------------------------------------------------------------------

def _create_aggregated_features(df, features, lookback):
    agg_list, names = [], []
    for feat in features:
        rolled = df[feat].shift(1).rolling(window=lookback, min_periods=1)
        agg_list += [rolled.mean(), rolled.std(), rolled.min(), rolled.max(), df[feat].shift(1)]
        names += [
            f'{feat}_mean_{lookback}', f'{feat}_std_{lookback}',
            f'{feat}_min_{lookback}', f'{feat}_max_{lookback}', f'{feat}_last',
        ]
    flat = pd.concat(agg_list, axis=1)
    flat.columns = names
    return flat, names


def _create_lag_features(df, features, lookback):
    lag_list, names = [], []
    for feat in features:
        for lag in range(1, lookback + 1):
            lag_list.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
    flat = pd.concat(lag_list, axis=1)
    flat.columns = names
    return flat, names


def _create_hybrid_features(df, features, lookback, lag_points=None, windows=None):
    lag_points = [l for l in (lag_points or [1, 2, 3, 5, 10, 24]) if l <= lookback]
    windows = [w for w in (windows or [6, 12, 24, 48]) if w <= lookback]
    parts, names = [], []
    for feat in features:
        for lag in lag_points:
            parts.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
        for w in windows:
            rolled = df[feat].shift(1).rolling(window=w, min_periods=1)
            parts += [rolled.mean(), rolled.std()]
            names += [f'{feat}_mean_{w}', f'{feat}_std_{w}']
        parts.append(df[feat].shift(1).diff(1))
        names.append(f'{feat}_diff_1')
        if lookback >= 5:
            parts.append(df[feat].shift(1).diff(5))
            names.append(f'{feat}_diff_5')
    flat = pd.concat(parts, axis=1)
    flat.columns = names
    return flat, names


def _create_signal_features(df, features, lookback,
                            short_windows=None, long_windows=None, diff_lags=None):
    short_windows = [w for w in (short_windows or [3, 6, 12]) if w <= lookback]
    long_windows = [w for w in (long_windows or [24, 48]) if w <= lookback]
    diff_lags = [l for l in (diff_lags or [1, 3, 5, 12]) if l <= lookback]
    parts, names = [], []
    for feat in features:
        col = df[feat].shift(1)
        short_means = {sw: col.rolling(window=sw, min_periods=1).mean() for sw in short_windows}
        long_means = {lw: col.rolling(window=lw, min_periods=1).mean() for lw in long_windows}
        for sw, sm in short_means.items():
            for lw, lm in long_means.items():
                parts.append(sm - lm)
                names.append(f'{feat}_trend_{sw}_{lw}')
        parts.append(col - col.rolling(window=lookback, min_periods=1).mean())
        names.append(f'{feat}_last_vs_mean')
        parts.append(col.rolling(window=lookback, min_periods=1).apply(
            lambda x: float(np.sum(x > 0)) / max(len(x), 1), raw=True))
        names.append(f'{feat}_sign_frac')
        diffs = {}
        for lag in diff_lags:
            diffs[lag] = col.diff(lag)
            parts.append(diffs[lag])
            names.append(f'{feat}_diff_{lag}')
        if 1 in diffs and 5 in diffs:
            parts.append(diffs[1] - diffs[5])
            names.append(f'{feat}_accel_1_5')
    flat = pd.concat(parts, axis=1)
    flat.columns = names
    return flat, names


def _create_none_features(df, features):
    return pd.DataFrame({feat: df[feat].shift(1) for feat in features}), list(features)


def _flatten_features(df, features, lookback, method):
    """Dispatch to the flattening method stored in model_config.json."""
    if method == 'agg':
        return _create_aggregated_features(df, features, lookback)
    if method == 'lag':
        return _create_lag_features(df, features, lookback)
    if method == 'hybrid':
        return _create_hybrid_features(df, features, lookback)
    if method == 'signal':
        return _create_signal_features(df, features, lookback)
    if method == 'none':
        return _create_none_features(df, features)
    raise ValueError(f"Unknown flattening method: '{method}'. "
                     f"Expected 'agg', 'lag', 'hybrid', 'signal', or 'none'.")


# ---------------------------------------------------------------------------
# Submodel
# ---------------------------------------------------------------------------

class _TabularSubmodel:
    """XGBoost booster or Random Forest producing mu (log-return) for Signal samples."""

    def __init__(self, kind: str, model_dir: str, cfg: dict, target_scaler=None):
        self.kind = kind
        self.features = list(cfg['features'])
        self.lookback = int(cfg['lookback'])
        self.method = cfg.get('flattening_method', 'hybrid')
        self.target_scaler = target_scaler
        self.iter_range = None
        if kind == 'xgb':
            import xgboost as xgb
            self._xgb = xgb
            self.model = xgb.Booster()
            self.model.load_model(os.path.join(model_dir, 'xgb_model.json'))
            # saved_iteration = number of trees used in training-time evaluation,
            # clamped to the trees stored in the file.
            saved = self.model.attr('saved_iteration')
            n_rounds = self.model.num_boosted_rounds()
            n = min(int(saved), n_rounds) if saved is not None else n_rounds
            self.iter_range = (0, max(1, n))
        elif kind == 'rf':
            import joblib
            self.model = joblib.load(os.path.join(model_dir, 'rf_model.pkl'))
            self.model.verbose = 0  # training stores verbose=1 (progress on every predict)
        else:
            raise ValueError(f"_TabularSubmodel supports 'xgb' and 'rf', got '{kind}'")

    def _predict_flat(self, X: np.ndarray) -> np.ndarray:
        if self.kind == 'xgb':
            return self.model.predict(self._xgb.DMatrix(X), iteration_range=self.iter_range)
        return self.model.predict(X)

    def predict(self, series: np.ndarray, n_samples: int, window: int,
                chunk: int = _CHUNK) -> np.ndarray:
        """
        mu for each of n_samples windows; series has n_samples + window - 1 rows
        (columns ordered as self.features). Returns shape (n_samples,).
        """
        if window < self.lookback:
            raise ValueError(
                f"Signal lookback ({window}) is shorter than the {self.kind} submodel "
                f"lookback ({self.lookback}); the first samples would lack history.")
        series = np.asarray(series, dtype=np.float64)
        if not np.isfinite(series).all():
            raise ValueError(f"{self.kind} submodel input contains NaN/inf values")
        out = []
        for s in range(0, n_samples, chunk):
            e = min(s + chunk, n_samples)
            block = series[s:e + window - 1]
            block = np.vstack([block, block[-1:]])  # placeholder decision row (t + 1)
            flat, _ = _flatten_features(
                pd.DataFrame(block, columns=self.features), self.features,
                self.lookback, self.method)
            X = flat.iloc[window:window + (e - s)].to_numpy(dtype=np.float32)
            if np.isnan(X).any():
                raise ValueError(
                    f"{self.kind} submodel feature matrix contains NaN values (not enough "
                    f"history in the Signal window for the submodel's flattening, e.g. "
                    f"'hybrid' diff_5 needs 6 candles).")
            y = self._predict_flat(X)
            if self.target_scaler is not None:
                y = self.target_scaler.inverse_transform(y)
            out.append(np.asarray(y, dtype=np.float64))
        return np.concatenate(out)
