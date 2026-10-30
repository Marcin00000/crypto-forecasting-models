"""
RFModelPackage -- standalone inference package for Random Forest crypto models.

Produces probabilistic BUY / SELL signals from a trained Random Forest log-return model
(Gaussian "Approach A", identical to the XGB and LR packages):

    mu    = model prediction (log-return, after inverse_transform)
    sigma = rolling std of 1-candle log-returns computed from close prices
    T     = threshold (minimum economically meaningful move)

    signal_buy  = P(r >  T) = (1 - Phi((T  - mu) / sigma)) * 100   [0-100]
    signal_sell = P(r < -T) = Phi((-T - mu) / sigma) * 100          [0-100]

Inputs accepted:
    'raw'      -- raw OHLCV from the exchange (open_time, open, high, low, close, volume;
                  taker_buy_base_asset_volume when the model uses buy_pressure)
    'prepared' -- prepared CSV from data_preparation (features already scaled, raw close)
    'auto'     -- detect from the presence of the first model feature (default)

Alignment contract (mirrors rf_test.py and rf_model/common/simulation_profile.py):
    The first `lookback` rows are always dropped, also for flattening 'none' (training
    does iloc[lookback:]). Original row labels are preserved, so sample i belongs to
    candle `lookback + first_valid_idx + i`; timestamp, close and sigma are read from
    that very candle. NaN feature rows can only form a prefix (indicator warm-up).

Differences vs XGBModelPackage: loads rf_model.pkl (joblib), plain model.predict(X)
without DMatrix / iteration_range.

Usage:
    from rf_model_package import RFModelPackage, SignalConfig

    pkg = RFModelPackage.from_directory('rf_model/rf_models/R054_eth_rf_model_E109_0610')
    df_signals = pkg.predict_batch(df)   # timestamp, close, mu, sigma, signal_buy, signal_sell
    signal = pkg.predict_live(df_buffer) # same fields for the latest candle
    n = pkg.min_rows_required            # int; also callable: n('raw') / n('prepared')
"""
from __future__ import annotations

import dataclasses
import json
import os
import warnings
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd

from .feature_engineering import (
    prepare_features, get_required_warmup, uses_legacy_vol_percentile,
)
from .flattening import _flatten_features
from .signals import compute_sigma_series, fill_sigma_nan, compute_signals
from .target_scaler import _TargetScaler


# ===========================================================================
# CONFIG / HELPERS
# ===========================================================================

class _MinRowsInt(int):
    """Integer that also acts as a callable accepting the mode ('raw' | 'prepared')."""

    def __new__(cls, val: int, raw_val: int, prep_val: int):
        obj = super().__new__(cls, val)
        obj._raw_val = raw_val
        obj._prep_val = prep_val
        return obj

    def __call__(self, mode: str = 'raw') -> int:
        return self._prep_val if mode in ('prepared', 'prep') else self._raw_val


@dataclasses.dataclass
class SignalConfig:
    """
    Parameters for the probabilistic signal generator.

    threshold:    Minimum log-return that triggers a signal; should cover fees and spread.
                  When no SignalConfig is passed the package uses the value the model was
                  tested with (signal_threshold -> dir_acc_threshold -> 0.001).
    window_sigma: Rolling window (candles) for market volatility. Default 100.
    """
    threshold:    float = 0.001
    window_sigma: int   = 100


_RAW_REQUIRED = {'open_time', 'open', 'high', 'low', 'close', 'volume'}
_PREP_REQUIRED = {'close'}
_NUMERIC_COLS = ('open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume')


def _detect_mode(df: pd.DataFrame, config: dict) -> str:
    """'prepared' if feature columns already present, else 'raw'."""
    return 'prepared' if config['features'][0] in df.columns else 'raw'


def _validate(df: pd.DataFrame, mode: str, config: dict, window_sigma: int) -> None:
    """Raise ValueError with a descriptive message if df is not usable (cheap checks only)."""
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
        nan_cols = [c for c in ('open', 'high', 'low', 'close', 'volume')
                    if df[c].isna().any()]
        if nan_cols:
            raise ValueError(f"Columns {nan_cols} contain NaN values.")
        if (df['volume'] < 0).any():
            raise ValueError("Column 'volume' contains negative values.")
    elif mode == 'prepared':
        missing_base = _PREP_REQUIRED - set(df.columns)
        if missing_base:
            raise ValueError(f"[prepared mode] Missing base columns: {sorted(missing_base)}")
        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            extra = (f' ... (+{len(missing_feats) - 5} more)'
                     if len(missing_feats) > 5 else '')
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
        if df['close'].isna().any():
            raise ValueError("Column 'close' contains NaN values.")
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'raw', 'prepared', or 'auto'.")

    if (df['close'] <= 0).any():
        raise ValueError("Column 'close' contains zero or negative values.")


def _normalise_input(df: pd.DataFrame) -> pd.DataFrame:
    """Copy, unify the time column name, sort chronologically, coerce numerics to float."""
    df = df.copy().reset_index(drop=True)
    if 'open_time' not in df.columns:
        for alias in ('timestamp', 'time'):
            if alias in df.columns:
                df = df.rename(columns={alias: 'open_time'})
                break
    if 'open_time' in df.columns and not df['open_time'].is_monotonic_increasing:
        df = df.sort_values('open_time').reset_index(drop=True)
    for col in _NUMERIC_COLS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class RFModelPackage:
    """
    Standalone inference + signal package for a trained Random Forest log-return model.

    Parameters
    ----------
    model_dir : str
        Directory produced by rf_train.py. Must contain rf_model.pkl, feature_scaler.pkl
        (or scaler_features.pkl) and model_config.json.
    signal_config : SignalConfig, optional
        Signal parameters. When omitted the threshold comes from model_config.json
        (signal_threshold -> dir_acc_threshold -> 0.001), as in rf_test.py.
    config : SignalConfig, optional
        Alias of `signal_config` (same ergonomics as the other model packages).
    n_jobs : int, optional
        Overrides the loaded estimator's n_jobs (None = keep the stored value). Trading
        loops predict a handful of rows per call, where thread start-up costs more than
        the trees, so they should pass 1.
    """

    def __init__(self, model_dir: str,
                 signal_config: Optional[SignalConfig] = None,
                 config: Optional[SignalConfig] = None,
                 n_jobs: Optional[int] = None):
        self._model_dir = model_dir
        self._config = self._load_config()

        effective_cfg = signal_config if signal_config is not None else config
        if effective_cfg is None:
            threshold = self._config.get(
                'signal_threshold', self._config.get('dir_acc_threshold', 0.001)
            )
            effective_cfg = SignalConfig(threshold=float(threshold))
        self._sig_cfg = effective_cfg

        self._model = self._load_model()
        # Training stores verbose=1, which prints joblib progress on every predict call.
        self._model.verbose = 0
        if n_jobs is not None:
            self._model.n_jobs = int(n_jobs)
        self._feature_scaler = self._load_feature_scaler()
        self._target_scaler = self._load_target_scaler()
        self._scaler_feature_cols = self._resolve_scaler_feature_cols()

    @classmethod
    def from_directory(cls, model_dir: str,
                       signal_config: Optional[SignalConfig] = None,
                       config: Optional[SignalConfig] = None,
                       **kwargs) -> 'RFModelPackage':
        """Factory matching the other model packages (extra kwargs: n_jobs)."""
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

    def _load_model(self):
        path = os.path.join(self._model_dir, 'rf_model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"rf_model.pkl not found in: {self._model_dir}")
        return joblib.load(path)

    def _load_feature_scaler(self):
        for name in ('feature_scaler.pkl', 'scaler_features.pkl'):
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                return joblib.load(path)
        raise FileNotFoundError(
            f"Feature scaler not found in {self._model_dir} "
            f"(tried feature_scaler.pkl and scaler_features.pkl)"
        )

    def _load_target_scaler(self) -> Optional[_TargetScaler]:
        """
        Priority: target_scaler_params from model_config.json (no pickle, no module
        dependencies), then target_scaler.pkl for models saved without params.
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
                    f"'target_scaler_params' is missing from model_config.json.\n"
                    f"The pickle references common.target_utils.TargetScaler; run from the "
                    f"rf_model directory or re-save the model with rf_train.py."
                ) from exc
            if isinstance(raw, _TargetScaler):
                return raw
            wrapper = _TargetScaler.__new__(_TargetScaler)
            wrapper.method = self._config.get('target_scaler_type', 'robust')
            wrapper._scaler = raw
            return wrapper
        return None

    def _resolve_scaler_feature_cols(self) -> Optional[list]:
        """
        Ordered list of columns the feature scaler was fitted on (raw mode must pass the
        full matrix in that order).

        Resolution order: scaler_feature_cols from model_config.json; metadata.json of the
        prepared dataset referenced by train_path (repo-relative or legacy absolute);
        model_config['features'] when the count matches; otherwise None.
        """
        n = self._feature_scaler.n_features_in_

        cols = self._config.get('scaler_feature_cols')
        if cols and len(cols) == n:
            return cols

        train_path = self._config.get('train_path', '')
        if train_path:
            # model_dir = <repo>/rf_model/rf_models/<run>, so '../../..' is the repo root.
            candidates = [
                train_path,
                os.path.join(self._model_dir, '..', '..', '..', train_path),
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
        if len(model_feats) == n:
            return model_feats
        return None

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @property
    def min_rows_required(self) -> _MinRowsInt:
        """
        Minimum rows for predict_live. Behaves as an int (the raw-mode value, safe for
        both modes) and as a callable: min_rows_required('prepared') = lookback +
        window_sigma, min_rows_required('raw') adds the feature warm-up. Flattening
        'none' still needs `lookback` rows because they are always dropped.
        """
        lookback = int(self._config['lookback'])
        window_sigma = int(self._sig_cfg.window_sigma)
        warmup = get_required_warmup(self._config['features'])
        raw_val = lookback + window_sigma + warmup
        return _MinRowsInt(raw_val, raw_val, lookback + window_sigma)

    def get_model_info(self) -> dict:
        """Return key model metadata."""
        c = self._config
        return {
            'model_id':            c.get('model_id',          'unknown'),
            'data_id':             c.get('data_id',           'unknown'),
            'symbol':              c.get('symbol',            'unknown'),
            'lookback':            c.get('lookback'),
            'horizon':             c.get('horizon'),
            'n_features':          c.get('n_features'),
            'n_flat_features':     c.get('n_flat_features'),
            'features':            list(c.get('features', [])),
            'flattening_method':   c.get('flattening_method'),
            'feature_set':         c.get('feature_set'),
            'criterion':           c.get('params', {}).get('criterion', 'N/A'),
            'n_estimators':        c.get('params', {}).get('n_estimators'),
            'n_jobs':              getattr(self._model, 'n_jobs', None),
            'scale_target':        c.get('scale_target'),
            'target_scaler_type':  c.get('target_scaler_type'),
            'oob_score':           c.get('oob_score'),
            'train_metrics':       c.get('train_metrics'),
            'val_metrics':         c.get('val_metrics'),
            'timestamp':           c.get('timestamp'),
            'dir_acc_threshold':   c.get('dir_acc_threshold'),
            'signal_threshold':    self._sig_cfg.threshold,
            'signal_window_sigma': self._sig_cfg.window_sigma,
        }

    def predict_batch(self, df: pd.DataFrame,
                      data_mode: str = 'auto',
                      input_type: Optional[str] = None,
                      return_df: bool = True,
                      drop_warmup: bool = True) -> Union[pd.DataFrame, dict]:
        """
        Signals for every candle that has a complete feature row.

        Parameters
        ----------
        df : pd.DataFrame
            Input data (see module docstring). Time column aliases `timestamp` / `time`
            are accepted; rows are sorted chronologically.
        data_mode : 'raw' | 'prepared' | 'auto'
        input_type : optional alias of `data_mode` (takes precedence when given).
        return_df : False returns a dict of numpy arrays (faster for live loops).
        drop_warmup : True drops rows whose sigma is still NaN (rolling warm-up).
            False keeps them with sigma = NaN and signals computed from the median
            sigma; this reproduces the full step grid of the Simulation v2 evaluation.

        Returns
        -------
        columns: timestamp, close, mu, sigma, signal_buy, signal_sell
            Every field is taken from the decision candle `lookback + i` of the working
            frame, i.e. the candle whose close the stored test results treat as entry.
        """
        effective_mode = input_type if input_type is not None else data_mode
        df = _normalise_input(df)
        mode = (_detect_mode(df, self._config)
                if effective_mode == 'auto' else effective_mode)
        _validate(df, mode, self._config, self._sig_cfg.window_sigma)

        df_work = self._prepare_df(df, mode)
        mu, row_idx = self._predict_mu(df_work)

        sigma_series = compute_sigma_series(df_work['close'], self._sig_cfg.window_sigma)
        sigma = sigma_series[row_idx]
        close = df_work['close'].to_numpy(dtype=np.float64)[row_idx]
        timestamps = (df_work['open_time'].to_numpy()[row_idx]
                      if 'open_time' in df_work.columns else row_idx)

        signal_buy, signal_sell = compute_signals(
            mu, fill_sigma_nan(sigma), self._sig_cfg.threshold
        )
        keep = np.isfinite(sigma) if drop_warmup else np.ones(len(mu), dtype=bool)
        out = {
            'timestamp': timestamps[keep],
            'close': close[keep],
            'mu': mu[keep],
            'sigma': sigma[keep],
            'signal_buy': signal_buy[keep],
            'signal_sell': signal_sell[keep],
        }
        return pd.DataFrame(out) if return_df else out

    def predict_live(self, df: pd.DataFrame,
                     data_mode: str = 'auto',
                     input_type: Optional[str] = None) -> dict:
        """
        Single signal for the latest candle of a rolling buffer holding at least
        `min_rows_required` rows.

        Returns dict: {timestamp, close, mu, sigma, signal_buy, signal_sell}
        """
        effective_mode = input_type if input_type is not None else data_mode
        result = self.predict_batch(df, data_mode=effective_mode, return_df=False)
        if len(result['timestamp']) == 0:
            req = self.min_rows_required(
                'raw' if effective_mode == 'auto' else effective_mode
            )
            raise RuntimeError(
                f"predict_live: no valid signals produced. Provide at least {req} rows."
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

    def _prepare_df(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Frame with scaled feature columns and raw close. The index is NOT reset or
        filtered, so row labels stay equal to candle positions of the input.
        """
        if mode != 'raw':
            return df   # prepared: features already scaled, close is the raw price

        features = self._config['features']
        df_feat = prepare_features(
            df, features=features,
            legacy_vol_percentile=uses_legacy_vol_percentile(self._config.get('data_id')))
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

        missing = [f for f in features if f not in df_feat.columns]
        if missing:
            hint = (" (buy_pressure needs 'taker_buy_base_asset_volume' in the input)"
                    if 'buy_pressure' in missing else '')
            raise ValueError(
                f"[raw mode] Model features could not be generated: {missing}{hint}. "
                f"Imputing them would silently corrupt predictions."
            )

        # feature_scaler was fitted on the full dataset feature list; it needs every
        # column in training order. Only columns that are not model inputs may be
        # zero-filled -- their transformed values never reach the model.
        scaler_cols = self._scaler_feature_cols
        if scaler_cols is not None:
            for col in scaler_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            df_feat[scaler_cols] = self._feature_scaler.transform(df_feat[scaler_cols])
        else:
            warnings.warn(
                "Could not resolve scaler column order from model_config.json or the "
                "dataset metadata. Scaling only the model features -- predictions may "
                "be inaccurate.",
                RuntimeWarning,
            )
            df_feat[features] = self._feature_scaler.transform(df_feat[features])
        return df_feat

    def _predict_mu(self, df_work: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Flatten -> drop first `lookback` rows (always, as in training) -> predict.

        Returns (mu, row_idx); row_idx holds the df_work row label of every predicted
        sample, which is the decision candle used for timestamp, close and sigma.
        """
        features = self._config['features']
        lookback = int(self._config['lookback'])
        method = self._config.get('flattening_method', 'hybrid')

        X_flat, _ = _flatten_features(df_work, features, lookback, method)
        X_flat = X_flat.iloc[lookback:]
        valid = X_flat.notna().all(axis=1).to_numpy()
        row_idx = X_flat.index.to_numpy()[valid]
        if len(row_idx) == 0:
            return np.empty(0, dtype=np.float64), row_idx

        y_scaled = self._model.predict(X_flat.to_numpy(dtype=np.float32)[valid])
        if self._target_scaler is not None:
            mu = self._target_scaler.inverse_transform(y_scaled)
        else:
            mu = y_scaled
        return np.asarray(mu, dtype=np.float64), row_idx
