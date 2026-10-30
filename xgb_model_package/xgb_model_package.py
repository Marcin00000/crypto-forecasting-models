"""
XGBModelPackage -- standalone inference package for XGBoost log-return models.

Produces probabilistic BUY / SELL signals from a trained XGBoost model.

Signal model (Approach A -- two independent thresholds, normal distribution):
    mu    = model prediction (log-return, after inverse_transform)
    sigma = rolling std of log-returns from close prices (market volatility)
    T     = threshold (minimum economically meaningful move)

    signal_buy  = P(r >  T) = (1 - Phi((T  - mu) / sigma)) * 100   [0-100]
    signal_sell = P(r < -T) = Phi((-T - mu) / sigma) * 100          [0-100]

Row alignment (identical to xgb_model training and xgb_test.py):
    Features are shifted by one candle, the first `lookback` rows of the
    flattened matrix are always dropped (also for flattening 'none'), and sample
    i is decided on row `lookback + i` of the input frame. timestamp, close and
    sigma are read from that very row; the original row index is preserved so
    that a NaN prefix (feature warm-up) never shifts mu against sigma.

Inputs accepted:
    'raw'      -- raw OHLCV from Binance (open_time, open, high, low, close, volume)
    'prepared' -- prepared CSV from data_preparation (features already scaled,
                  close is the raw price)
    'auto'     -- auto-detect based on column presence (default)

Usage:
    from xgb_model_package import XGBModelPackage, SignalConfig

    pkg = XGBModelPackage.from_directory('xgb_model/xgb_models/X239_eth_xgb_model_E109_0610')
    df_signals = pkg.predict_batch(df)
    # columns: timestamp, close, mu, sigma, signal_buy, signal_sell

    signal = pkg.predict_live(df_buffer)   # dict with the same keys, latest candle
    n = pkg.min_rows_required              # int; pkg.min_rows_required('prepared') for prepared mode
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .feature_engineering import (
    prepare_features, get_required_warmup, uses_legacy_vol_percentile,
)
from .flattening import _flatten_features
from .signals import compute_sigma_series, fill_sigma_nan, compute_signals
from .target_scaler import _TargetScaler

_DEFAULT_THRESHOLD = 0.001
# Repo root when the package sits at the top level of the project (used to
# resolve repo-relative train_path values stored in model_config.json).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# CONFIG & SMALL HELPERS
# ===========================================================================

@dataclasses.dataclass
class SignalConfig:
    """
    Parameters for the probabilistic signal generator.

    threshold:    Minimum log-return to trigger a signal; should cover fees and
                  spread. Default 0.001 (about 0.1%).
    window_sigma: Rolling window (in candles) for the volatility estimate.
                  Default 100 is the window used by the simulation.
    """
    threshold: float = _DEFAULT_THRESHOLD
    window_sigma: int = 100


class _MinRowsInt(int):
    """Integer that also acts as a callable accepting mode ('raw' | 'prepared')."""

    def __new__(cls, val: int, raw_val: int, prep_val: int):
        obj = super().__new__(cls, val)
        obj._raw_val = raw_val
        obj._prep_val = prep_val
        return obj

    def __call__(self, mode: str = 'raw') -> int:
        return self._prep_val if mode in ('prepared', 'prep') else self._raw_val


def _default_threshold(config: dict) -> float:
    """Training-time threshold: resolved value, then fixed value, then 0.001."""
    for key in ('signal_threshold_resolved', 'signal_threshold_value'):
        value = config.get(key)
        if value is not None:
            return float(value)
    return _DEFAULT_THRESHOLD


# ===========================================================================
# VALIDATION
# ===========================================================================

_RAW_REQUIRED = {'open_time', 'open', 'high', 'low', 'close', 'volume'}
_PREP_REQUIRED = {'close'}


def _detect_mode(df: pd.DataFrame, config: dict) -> str:
    """Return 'prepared' if feature columns already present, else 'raw'."""
    return 'prepared' if config['features'][0] in df.columns else 'raw'


def _validate(df: pd.DataFrame, mode: str, config: dict, window_sigma: int) -> None:
    """Raise ValueError with a descriptive message if df is not suitable."""
    lookback = int(config['lookback'])
    features = config['features']

    if mode == 'raw':
        missing = _RAW_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"[raw mode] Missing required columns: {sorted(missing)}\n"
                f"Expected: {sorted(_RAW_REQUIRED)}"
            )
        warmup = get_required_warmup(features)
        min_rows = lookback + window_sigma + warmup
        if len(df) < min_rows:
            raise ValueError(
                f"[raw mode] Insufficient rows: got {len(df)}, need >= {min_rows} "
                f"(lookback={lookback} + window_sigma={window_sigma} + warmup={warmup})"
            )
        nan_cols = [c for c in ('open', 'high', 'low', 'close', 'volume') if df[c].isna().any()]
        if nan_cols:
            raise ValueError(f"Columns {nan_cols} contain NaN values.")
    elif mode == 'prepared':
        missing_base = _PREP_REQUIRED - set(df.columns)
        if missing_base:
            raise ValueError(f"[prepared mode] Missing base columns: {sorted(missing_base)}")
        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            extra = f' ... (+{len(missing_feats) - 5} more)' if len(missing_feats) > 5 else ''
            raise ValueError(
                f"[prepared mode] Missing feature columns: {missing_feats[:5]}{extra}\n"
                f"Model expects {len(features)} features from config."
            )
        min_rows = lookback + window_sigma
        if len(df) < min_rows:
            raise ValueError(
                f"[prepared mode] Insufficient rows: got {len(df)}, need >= {min_rows} "
                f"(lookback={lookback} + window_sigma={window_sigma})"
            )
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'raw', 'prepared', or 'auto'.")

    if df['close'].isna().any():
        raise ValueError("Column 'close' contains NaN values.")
    if (df['close'] <= 0).any():
        raise ValueError("Column 'close' contains zero or negative values.")


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class XGBModelPackage:
    """
    Standalone inference + signal package for a trained XGBoost log-return model.

    Parameters
    ----------
    model_dir : str
        Directory produced by xgb_train.py. Must contain xgb_model.json,
        feature_scaler.pkl and model_config.json (target_scaler.pkl is optional:
        the scaler is rebuilt from model_config.json when possible).
    signal_config : SignalConfig, optional
        Signal parameters. When omitted the threshold is taken from the model
        config (signal_threshold_resolved -> signal_threshold_value -> 0.001).
    config : SignalConfig, optional
        Alias of signal_config (same ergonomics as the other model packages).
    """

    def __init__(self, model_dir: str,
                 signal_config: Optional[SignalConfig] = None,
                 config: Optional[SignalConfig] = None):
        self._model_dir = model_dir
        self._config = self._load_config()
        effective_cfg = signal_config if signal_config is not None else config
        if effective_cfg is None:
            effective_cfg = SignalConfig(threshold=_default_threshold(self._config))
        self._sig_cfg = effective_cfg

        self._model = self._load_model()
        self._feature_scaler = self._load_feature_scaler()
        self._target_scaler = self._load_target_scaler()
        self._iter_range = self._resolve_iter_range()
        self._scaler_feature_cols = self._resolve_scaler_feature_cols()

    @classmethod
    def from_directory(cls, model_dir: str,
                       signal_config: Optional[SignalConfig] = None,
                       config: Optional[SignalConfig] = None,
                       **kwargs) -> 'XGBModelPackage':
        """Factory constructor matching the other model packages."""
        return cls(model_dir, signal_config=signal_config, config=config, **kwargs)

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        path = os.path.join(self._model_dir, 'model_config.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"model_config.json not found in: {self._model_dir}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_model(self) -> xgb.Booster:
        path = os.path.join(self._model_dir, 'xgb_model.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"xgb_model.json not found in: {self._model_dir}")
        model = xgb.Booster()
        model.load_model(path)
        return model

    def _load_feature_scaler(self):
        # Prepared data uses 'scaler_features.pkl'; training saves 'feature_scaler.pkl'
        for name in ('feature_scaler.pkl', 'scaler_features.pkl'):
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                return joblib.load(path)
        raise FileNotFoundError(
            f"Feature scaler not found in {self._model_dir} "
            f"(tried feature_scaler.pkl and scaler_features.pkl)"
        )

    def _load_target_scaler(self) -> Optional['_TargetScaler']:
        """
        Load the target scaler.

        target_scaler_params from model_config.json is preferred: it needs no
        pickle and no class from the training project. target_scaler.pkl is only
        a fallback for models saved without params; unpickling it requires
        common.target_utils.TargetScaler to be importable.
        """
        if not self._config.get('scale_target', False):
            return None

        params = self._config.get('target_scaler_params')
        if params:
            return _TargetScaler.from_params(params)

        pkl_path = os.path.join(self._model_dir, 'target_scaler.pkl')
        if os.path.exists(pkl_path):
            try:
                raw = joblib.load(pkl_path)
            except (ModuleNotFoundError, ImportError) as exc:
                raise RuntimeError(
                    f"Cannot load target_scaler.pkl ({exc}) and "
                    f"'target_scaler_params' is missing from model_config.json. "
                    f"The pickle needs common.target_utils.TargetScaler on the import path; "
                    f"re-save the model with xgb_train.py to embed scaler params."
                ) from exc
            if isinstance(raw, _TargetScaler):
                return raw
            wrapper = _TargetScaler.__new__(_TargetScaler)
            wrapper.method = self._config.get('target_scaler_type', 'robust')
            wrapper._scaler = raw
            return wrapper

        return None

    def _resolve_iter_range(self) -> tuple:
        """
        Iteration range [start, end) for Booster.predict (end is exclusive).

        xgb_train.py stores `saved_iteration` as the NUMBER of trees to use
        (best_iteration + 1, or total rounds), and xgb_test.py predicts with
        (0, saved_iteration). The count is clamped to the trees physically in
        the file so a stale attribute can never raise an out-of-range error.
        """
        actual_rounds = self._model.num_boosted_rounds()
        saved = self._model.attr('saved_iteration')
        n = min(int(saved), actual_rounds) if saved is not None else actual_rounds
        return (0, max(1, n))

    def _resolve_scaler_feature_cols(self) -> Optional[list]:
        """
        Ordered column list the feature scaler was fitted on (needed in raw
        mode to build the full matrix before scaler.transform()).

        Resolution order:
          1. model_config 'scaler_feature_cols' (self-contained models)
          2. prepared-data metadata.json located through train_path (relative to
             the repo root, the model directory, or absolute for older models)
          3. model_config 'features' when the count matches the scaler
          4. None (raw mode then raises)
        """
        n = self._feature_scaler.n_features_in_

        cols = self._config.get('scaler_feature_cols')
        if cols and len(cols) == n:
            return cols

        train_path = self._config.get('train_path', '')
        if train_path:
            candidates = [
                train_path,
                os.path.join(_REPO_ROOT, train_path),
                os.path.join(self._model_dir, '..', '..', train_path),
                os.path.join(self._model_dir, '..', train_path),
            ]
            for base in candidates:
                meta_path = os.path.normpath(os.path.join(base, 'metadata.json'))
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            cols = json.load(f).get('features', [])
                    except (OSError, ValueError):
                        continue
                    if len(cols) == n:
                        return cols

        model_feats = self._config.get('features', [])
        return model_feats if len(model_feats) == n else None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @property
    def min_rows_required(self) -> _MinRowsInt:
        """
        Minimum input rows for predict_live. The int value is the raw-mode
        requirement; call it with 'prepared' for prepared input.

        raw:      lookback + window_sigma + feature warm-up
        prepared: lookback + window_sigma (the first `lookback` rows are always
                  dropped, also for flattening 'none')
        """
        lookback = int(self._config['lookback'])
        window = int(self._sig_cfg.window_sigma)
        raw_val = lookback + window + get_required_warmup(self._config['features'])
        return _MinRowsInt(raw_val, raw_val, lookback + window)

    def get_model_info(self) -> dict:
        """Return key model metadata."""
        c = self._config
        return {
            'model_id':          c.get('model_id', 'unknown'),
            'data_id':           c.get('data_id', 'unknown'),
            'symbol':            c.get('symbol', 'unknown'),
            'lookback':          c.get('lookback'),
            'horizon':           c.get('horizon'),
            'n_features':        c.get('n_features'),
            'n_flat_features':   c.get('n_flat_features'),
            'flattening_method': c.get('flattening_method'),
            'feature_set':       c.get('feature_set'),
            'scale_target':      c.get('scale_target'),
            'target_scaler_type': c.get('target_scaler_type'),
            'train_metrics':     c.get('train_metrics'),
            'val_metrics':       c.get('val_metrics'),
            'timestamp':         c.get('timestamp'),
            'signal_threshold':  self._sig_cfg.threshold,
            'signal_window_sigma': self._sig_cfg.window_sigma,
            'iter_range':        self._iter_range,
        }

    def predict_batch(self, df: pd.DataFrame,
                      data_mode: str = 'auto',
                      input_type: Optional[str] = None,
                      return_df: bool = True,
                      drop_warmup: bool = True) -> Union[pd.DataFrame, dict]:
        """
        Signals for every valid sample of df.

        Parameters
        ----------
        df : pd.DataFrame
            Raw OHLCV or prepared data (see module docstring). The time column
            may be named open_time, timestamp or time; rows are sorted ascending.
        data_mode : str
            'raw' | 'prepared' | 'auto' (default).
        input_type : str, optional
            Alias of data_mode, takes precedence when given.
        return_df : bool
            False returns a dict of numpy arrays instead of a DataFrame.
        drop_warmup : bool
            True (default) drops samples whose sigma is NaN (first window_sigma
            candles). False keeps them with sigma = NaN and signals computed from
            the median sigma, which reproduces the step grid of the simulation.

        Returns
        -------
        timestamp, close, mu, sigma, signal_buy, signal_sell
        (timestamp, close and sigma come from the decision row lookback + i).
        """
        df = self._normalise_input(df)
        mode = input_type if input_type is not None else data_mode
        if mode == 'auto':
            mode = _detect_mode(df, self._config)
        _validate(df, mode, self._config, self._sig_cfg.window_sigma)

        df_work = self._prepare_df(df, mode)
        mu, row_idx = self._predict_mu(df_work)

        sigma_series = compute_sigma_series(df_work['close'], self._sig_cfg.window_sigma)
        sigma = sigma_series[row_idx]
        timestamps = (df_work['open_time'].iloc[row_idx].to_numpy()
                      if 'open_time' in df_work.columns else row_idx)
        close = df_work['close'].iloc[row_idx].to_numpy(dtype=np.float64)

        signal_buy, signal_sell = compute_signals(
            mu, fill_sigma_nan(sigma), self._sig_cfg.threshold)

        keep = np.isfinite(sigma) if drop_warmup else np.ones(len(mu), dtype=bool)
        out = {
            'timestamp':   timestamps[keep],
            'close':       close[keep],
            'mu':          mu[keep],
            'sigma':       sigma[keep],
            'signal_buy':  signal_buy[keep],
            'signal_sell': signal_sell[keep],
        }
        return pd.DataFrame(out) if return_df else out

    def predict_live(self, df: pd.DataFrame,
                     data_mode: str = 'auto',
                     input_type: Optional[str] = None) -> dict:
        """
        Single signal for the latest candle (live / streaming mode).

        The caller maintains a rolling buffer of at least min_rows_required rows.

        Returns
        -------
        dict {timestamp, close, mu, sigma, signal_buy, signal_sell}
        """
        mode = input_type if input_type is not None else data_mode
        result = self.predict_batch(df, data_mode=mode, return_df=False)
        if len(result['mu']) == 0:
            raise RuntimeError(
                "predict_live: no valid signals produced. "
                f"Provide at least {self.min_rows_required(mode)} rows."
            )
        return {
            'timestamp':   result['timestamp'][-1],
            'close':       float(result['close'][-1]),
            'mu':          float(result['mu'][-1]),
            'sigma':       float(result['sigma'][-1]),
            'signal_buy':  float(result['signal_buy'][-1]),
            'signal_sell': float(result['signal_sell'][-1]),
        }

    # ------------------------------------------------------------------
    # INTERNAL PIPELINE
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_input(df: pd.DataFrame) -> pd.DataFrame:
        """Copy, map time-column aliases, sort ascending and cast OHLCV to float."""
        df = df.copy().reset_index(drop=True)
        if 'open_time' not in df.columns:
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'open_time'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'open_time'})
        if 'open_time' in df.columns and not df['open_time'].is_monotonic_increasing:
            df = df.sort_values('open_time').reset_index(drop=True)
        for col in ('open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume'):
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _prepare_df(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Return a frame with the same rows as df: scaled feature columns and raw
        close. Raw mode runs feature engineering, so the first rows hold NaN
        features (warm-up); they are kept so that row positions stay intact.
        """
        if mode != 'raw':
            return df

        features = self._config['features']
        df_feat = prepare_features(
            df, features=features,
            legacy_vol_percentile=uses_legacy_vol_percentile(self._config.get('data_id')))
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

        # A model feature that feature engineering cannot build (e.g. buy_pressure
        # without taker volume) must fail loudly instead of being imputed.
        missing = [f for f in features if f not in df_feat.columns]
        if missing:
            raise ValueError(
                f"[raw mode] Model features could not be computed from the input: {missing}. "
                f"Provide the required source columns (e.g. taker_buy_base_asset_volume "
                f"for buy_pressure)."
            )

        scaler_cols = self._scaler_feature_cols
        if scaler_cols is None:
            raise ValueError(
                "[raw mode] Cannot resolve the column order of the feature scaler "
                "(no scaler_feature_cols in model_config.json and no matching metadata.json)."
            )
        # The scaler was fitted on every dataset feature, but the model only uses a
        # subset; columns that are not model features are scaled as zeros (they are
        # never read) because scaling is column-independent.
        for col in scaler_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0.0
        df_feat[scaler_cols] = self._feature_scaler.transform(df_feat[scaler_cols])
        return df_feat

    def _predict_mu(self, df_work: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Flatten -> XGBoost predict -> inverse target scaling.

        Always drops the first `lookback` rows (matches training for every
        flattening method) without resetting the index, so row_idx maps each
        prediction back to the row of df_work it was decided on. NaN is only
        allowed as a leading prefix (feature warm-up); a gap inside the series
        would silently pair predictions with the wrong rows and raises instead.

        Returns (mu, row_idx).
        """
        features = self._config['features']
        lookback = int(self._config['lookback'])
        method = self._config['flattening_method']

        X_flat, _ = _flatten_features(df_work, features, lookback, method)
        X_flat = X_flat.iloc[lookback:]

        valid = X_flat.notna().all(axis=1).to_numpy()
        if not valid.any():
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
        first_valid = int(np.argmax(valid))
        if not valid[first_valid:].all():
            raise ValueError(
                "NaN values inside the feature matrix after the warm-up prefix; "
                "predictions cannot be aligned with rows. Check the input for gaps."
            )

        row_idx = X_flat.index.to_numpy()[valid]
        X_np = X_flat.to_numpy(dtype=np.float32)[valid]
        y_scaled = self._model.predict(xgb.DMatrix(X_np), iteration_range=self._iter_range)

        mu = (self._target_scaler.inverse_transform(y_scaled)
              if self._target_scaler is not None else y_scaled)
        return np.asarray(mu, dtype=np.float64), row_idx
