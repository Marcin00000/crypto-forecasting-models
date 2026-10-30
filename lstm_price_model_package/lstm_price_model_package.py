"""
LSTMPriceModelPackage - standalone inference package for price-prediction LSTM models.

Reproduces lstm_model/lstm_test_price.py: the network predicts a scaled (log-)price at
t + horizon, which is inverted to P_hat and converted to the implied log-return
mu = log(P_hat) - log(P_t); signals are Gaussian probabilities from mu and the rolling
volatility sigma (see signals.py).
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
import torch

from .compat import resolve_scaler_columns
from .feature_engineering import prepare_features, uses_legacy_vol_percentile
from .lstm_arch import build_model_from_config
from .signals import (
    align_sigma,
    compute_signals,
    compute_sigma_series,
    fill_sigma_nan,
    get_required_warmup,
)
from .target_inversion import implied_log_returns, predicted_dollars

DEFAULT_SIGNAL_THRESHOLD = 0.001
_RAW_REQUIRED = {'open_time', 'open', 'high', 'low', 'close', 'volume'}
_OHLCV_NUMERIC = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']


class _MinRowsInt(int):
    """Integer that is also callable with a mode ('raw' | 'prepared')."""

    def __new__(cls, val: int, raw_val: int, prep_val: int):
        obj = super().__new__(cls, val)
        obj._raw_val = raw_val
        obj._prep_val = prep_val
        return obj

    def __call__(self, mode: str = 'raw') -> int:
        return self._prep_val if mode in ('prepared', 'prep') else self._raw_val


@dataclass
class SignalConfig:
    """
    threshold: minimum log-return move that counts as a trade signal (default 0.001).
    window_sigma: rolling window of the 1-step log-return volatility (default 100).
    """
    threshold: float = DEFAULT_SIGNAL_THRESHOLD
    window_sigma: int = 100


class LSTMPriceModelPackage:
    """
    Inference + signal package for a trained price LSTM.

    model_dir must contain model_config.json (or .pkl), best_model.pth (or the legacy
    lstm_price_model.pth) and feature_scaler.pkl / price_scaler.pkl when the dataset was
    scaled.
    """

    def __init__(self, model_dir: str,
                 signal_config: Optional[SignalConfig] = None,
                 config: Optional[SignalConfig] = None,
                 device: str = 'cpu'):
        self._model_dir = str(model_dir)
        self._device = torch.device(device)
        self._config = self._load_config()
        self._features = list(self._config['features'])
        self._lookback = int(self._config['lookback'])
        self._use_log_prices = bool(self._config.get('use_log_prices', True))

        sig_cfg = signal_config if signal_config is not None else config
        if sig_cfg is None:
            sig_cfg = SignalConfig(threshold=float(
                self._config.get('signal_threshold', DEFAULT_SIGNAL_THRESHOLD)))
        self._sig_cfg = sig_cfg

        self._feature_scaler = self._load_pickle(('feature_scaler.pkl', 'scaler_features.pkl'))
        self._price_scaler = self._load_pickle(('price_scaler.pkl', 'scaler_price.pkl'))
        self._scaler_feature_cols = (
            resolve_scaler_columns(self._feature_scaler, self._config)
            if self._feature_scaler is not None else [])
        self._checkpoint_name, self._model = self._load_model()
        self._warned_missing_taker = False

    @classmethod
    def from_directory(cls, model_dir: str,
                       signal_config: Optional[SignalConfig] = None,
                       config: Optional[SignalConfig] = None,
                       device: str = 'cpu') -> 'LSTMPriceModelPackage':
        """Factory constructor matching the other model packages."""
        return cls(model_dir=model_dir, signal_config=signal_config, config=config, device=device)

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        json_path = os.path.join(self._model_dir, 'model_config.json')
        pkl_path = os.path.join(self._model_dir, 'model_config.pkl')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        if os.path.exists(pkl_path):
            return joblib.load(pkl_path)
        raise FileNotFoundError(f"model_config.json / model_config.pkl not found in: {self._model_dir}")

    def _load_pickle(self, names: tuple[str, ...]):
        for name in names:
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                return joblib.load(path)
        return None

    def _load_model(self):
        for name in ('best_model.pth', 'lstm_price_model.pth'):
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                break
        else:
            raise FileNotFoundError(
                f"Model weights (best_model.pth / lstm_price_model.pth) not found in: {self._model_dir}")
        model = build_model_from_config(self._config)
        state = torch.load(path, map_location=self._device, weights_only=True)
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        return name, model

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @property
    def min_rows_required(self) -> _MinRowsInt:
        """Minimum input rows; callable as min_rows_required('raw' | 'prepared')."""
        window = int(self._sig_cfg.window_sigma)
        raw_val = self._lookback + window + get_required_warmup(self._features)
        return _MinRowsInt(raw_val, raw_val, self._lookback + window)

    def get_model_info(self) -> dict:
        c = self._config
        return {
            'model_id': c.get('model_id', 'unknown'),
            'data_id': c.get('data_id', 'unknown'),
            'symbol': c.get('symbol', 'unknown'),
            'lookback': self._lookback,
            'horizon': c.get('horizon'),
            'n_features': len(self._features),
            'features': list(self._features),
            'feature_set': c.get('feature_set'),
            'hidden_size': c.get('hidden_size'),
            'num_layers': c.get('num_layers'),
            'use_log_prices': self._use_log_prices,
            'checkpoint': self._checkpoint_name,
            'has_feature_scaler': self._feature_scaler is not None,
            'has_price_scaler': self._price_scaler is not None,
            'device': str(self._device),
            'signal_threshold': self._sig_cfg.threshold,
            'signal_window_sigma': self._sig_cfg.window_sigma,
        }

    def predict_batch(self, df: pd.DataFrame,
                      data_mode: str = 'auto',
                      input_type: Optional[str] = None,
                      batch_size: int = 4096,
                      dataset_id: Optional[str] = None,
                      return_df: bool = True,
                      drop_warmup: bool = True) -> Union[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Predict mu and signals for every complete lookback window.

        Output row i belongs to decision candle i + lookback - 1 (the last candle of the
        window) and has columns timestamp, close, mu, predicted_price, sigma, signal_buy,
        signal_sell. The horizon is not trimmed: the last `horizon` rows have no realised
        target yet, which is exactly the live-trading situation.

        drop_warmup=True removes rows whose sigma is still undefined (first window_sigma
        candles); False keeps them with sigma=NaN and signals computed from a median-filled
        sigma, which is what a step-for-step comparison with stored test results needs.
        """
        effective_mode = input_type if input_type is not None else data_mode
        df = self._normalise_input(df)
        mode = self._detect_mode(df) if effective_mode == 'auto' else effective_mode
        self._validate(df, mode, drop_warmup)
        self._warn_cross_dataset(df, mode, dataset_id)
        self._warn_missing_taker(df, mode)

        df_work = self._prepare_df(df, mode)
        sigma_full = compute_sigma_series(df['close'].to_numpy(dtype=np.float64), self._sig_cfg.window_sigma)

        n = len(df_work) - self._lookback + 1
        if n <= 0:
            return self._empty_result(return_df)

        y_scaled = self._predict_scaled(df_work, batch_size)
        pos = np.arange(n) + (self._lookback - 1)
        close_row = df_work['close'].to_numpy(dtype=np.float64)[pos]
        # The training/test pipeline stores prices as float32 (create_sequences); mirror that
        # rounding so mu equals the stored predicted_log instead of differing by ~1e-7.
        mu = implied_log_returns(y_scaled, close_row.astype(np.float32).astype(np.float64),
                                 self._price_scaler, self._use_log_prices)
        price_hat = predicted_dollars(y_scaled, self._price_scaler, self._use_log_prices)

        sigma = align_sigma(sigma_full[df_work.index.to_numpy()], self._lookback, n)
        sig_buy, sig_sell = compute_signals(mu, fill_sigma_nan(sigma), self._sig_cfg.threshold)
        timestamps = (df_work['open_time'].to_numpy()[pos] if 'open_time' in df_work.columns
                      else np.arange(n))

        out = {
            'timestamp': timestamps,
            'close': close_row,
            'mu': mu,
            'predicted_price': price_hat,
            'sigma': sigma,
            'signal_buy': sig_buy,
            'signal_sell': sig_sell,
        }
        if drop_warmup:
            keep = np.isfinite(sigma)
            out = {k: v[keep] for k, v in out.items()}
        return pd.DataFrame(out) if return_df else out

    def predict_live(self, df: pd.DataFrame,
                     data_mode: str = 'auto',
                     input_type: Optional[str] = None,
                     dataset_id: Optional[str] = None) -> dict:
        """Single signal for the latest closed candle."""
        effective_mode = input_type if input_type is not None else data_mode
        result = self.predict_batch(df, data_mode=effective_mode, dataset_id=dataset_id, return_df=False)
        if len(result['timestamp']) == 0:
            raise RuntimeError(
                f"predict_live: no valid signals produced. Ensure at least "
                f"{self.min_rows_required(effective_mode if effective_mode != 'auto' else 'raw')} rows.")
        return {
            'timestamp': result['timestamp'][-1],
            'close': float(result['close'][-1]),
            'mu': float(result['mu'][-1]),
            'predicted_price': float(result['predicted_price'][-1]),
            'sigma': float(result['sigma'][-1]),
            'signal_buy': float(result['signal_buy'][-1]),
            'signal_sell': float(result['signal_sell'][-1]),
        }

    # ------------------------------------------------------------------
    # INTERNAL PIPELINE
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_input(df: pd.DataFrame) -> pd.DataFrame:
        """Copy, map time aliases to open_time, sort chronologically and coerce OHLCV to float."""
        df = df.copy().reset_index(drop=True)
        if 'open_time' not in df.columns:
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'open_time'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'open_time'})
        if 'open_time' in df.columns and not df['open_time'].is_monotonic_increasing:
            df = df.sort_values('open_time').reset_index(drop=True)
        for col in _OHLCV_NUMERIC:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _detect_mode(self, df: pd.DataFrame) -> str:
        return 'prepared' if self._features[0] in df.columns else 'raw'

    def _validate(self, df: pd.DataFrame, mode: str, drop_warmup: bool) -> None:
        window = int(self._sig_cfg.window_sigma)
        if mode == 'raw':
            missing = _RAW_REQUIRED - set(df.columns)
            if missing:
                raise ValueError(f"[raw mode] Missing required columns: {sorted(missing)}")
            need = self.min_rows_required('raw')
            if len(df) < need:
                raise ValueError(f"[raw mode] Insufficient rows: got {len(df)}, need >= {need} "
                                 f"(lookback + window_sigma + feature warm-up)")
            nan_cols = [c for c in ('open', 'high', 'low', 'close', 'volume') if df[c].isna().any()]
            if nan_cols:
                raise ValueError(f"Columns {nan_cols} contain NaN values.")
            if (df['volume'] < 0).any():
                raise ValueError("Column 'volume' contains negative values.")
        elif mode == 'prepared':
            if 'close' not in df.columns:
                raise ValueError("[prepared mode] Missing base columns: ['close'] (raw, unscaled close is required)")
            missing_feats = [f for f in self._features if f not in df.columns]
            if missing_feats:
                extra = f' ... (+{len(missing_feats) - 5} more)' if len(missing_feats) > 5 else ''
                raise ValueError(f"[prepared mode] Missing feature columns: {missing_feats[:5]}{extra}")
            need = self._lookback + (window if drop_warmup else 0)
            if len(df) < need:
                raise ValueError(f"[prepared mode] Insufficient rows: got {len(df)}, need >= {need}")
            if df['close'].isna().any():
                raise ValueError("Column 'close' contains NaN values.")
        else:
            raise ValueError(f"Unknown mode: '{mode}'. Use 'raw', 'prepared', or 'auto'.")
        if (df['close'] <= 0).any():
            raise ValueError("Column 'close' contains zero or negative values.")

    def _warn_cross_dataset(self, df: pd.DataFrame, mode: str, dataset_id: Optional[str]) -> None:
        if mode != 'prepared':
            return
        given = dataset_id or (df['data_id'].iloc[0] if 'data_id' in df.columns else None)
        expected = self._config.get('data_id')
        if given and expected and str(given).upper() != str(expected).upper():
            warnings.warn(
                f"Cross-Dataset Warning: model was trained on {expected}, but input data is marked as {given}. "
                "Different datasets use different normalisation statistics.", UserWarning, stacklevel=3)

    def _warn_missing_taker(self, df: pd.DataFrame, mode: str) -> None:
        if (mode == 'raw' and 'buy_pressure' in self._features
                and 'taker_buy_base_asset_volume' not in df.columns and not self._warned_missing_taker):
            warnings.warn(
                "Model requires 'buy_pressure', but 'taker_buy_base_asset_volume' is missing from input. "
                "Imputing neutral buy_pressure=0.5.", UserWarning, stacklevel=3)
            self._warned_missing_taker = True

    def _prepare_df(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Return the frame the windows are cut from; its index keeps the original row positions.

        raw: feature engineering -> scaler transform -> drop rows with undefined features.
        prepared: features were scaled when the dataset was built (scale_csv=true) and must
        not be scaled again. Only datasets built with scale_csv=false carry raw feature
        values, which the test pipeline scales with the saved feature scaler.
        """
        if mode == 'prepared':
            df_work = df.copy()
            if self._config.get('scale_csv', True) is False and self._feature_scaler is not None:
                cols = self._scaler_feature_cols
                df_work[cols] = self._feature_scaler.transform(df_work[cols])
            return df_work

        df_feat = prepare_features(
            df.copy(), features=self._features,
            legacy_vol_percentile=uses_legacy_vol_percentile(self._config.get('data_id')))
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
        if 'buy_pressure' in self._features and 'buy_pressure' not in df_feat.columns:
            df_feat['buy_pressure'] = 0.5

        missing = [f for f in self._features if f not in df_feat.columns]
        if missing:
            raise ValueError(f"Feature engineering did not produce model features: {missing}")

        if self._feature_scaler is None and self._config.get('scale_csv', False) is True:
            # Training consumed scaled CSV features (scale_csv=True); a model saved without its
            # scaler would silently receive unscaled live features. Datasets without feature
            # scaling (scale_csv=False) legitimately have no scaler.
            raise ValueError(
                "Raw mode requires the feature scaler used in training "
                "(feature_scaler.pkl or scaler_features.pkl), which is missing from the model "
                "directory. Copy scaler_features.pkl of the training dataset there, or use "
                "mode='prepared' with already scaled features.")
        if self._feature_scaler is not None:
            # Scaler columns that are not model features (e.g. 39-column scaler, 6-feature model)
            # are never read by the network; scalers act per column, so 0.0 is a harmless filler.
            for col in self._scaler_feature_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
            df_feat[self._scaler_feature_cols] = self._feature_scaler.transform(
                df_feat[self._scaler_feature_cols])
        return df_feat.dropna(subset=self._features)

    def _predict_scaled(self, df_work: pd.DataFrame, batch_size: int) -> np.ndarray:
        data = np.ascontiguousarray(df_work[self._features].to_numpy(dtype=np.float32))
        # Zero-copy sliding windows: (n_windows, lookback, n_features)
        windows = torch.from_numpy(data).unfold(0, self._lookback, 1).transpose(1, 2)
        preds = []
        with torch.inference_mode():
            for start in range(0, windows.shape[0], batch_size):
                x_batch = windows[start:start + batch_size].to(self._device, non_blocking=True)
                preds.append(np.atleast_1d(self._model(x_batch).squeeze(-1).cpu().numpy()))
        return np.concatenate(preds)

    @staticmethod
    def _empty_result(return_df: bool):
        cols = ['timestamp', 'close', 'mu', 'predicted_price', 'sigma', 'signal_buy', 'signal_sell']
        return pd.DataFrame({c: [] for c in cols}) if return_df else {c: np.empty(0) for c in cols}
