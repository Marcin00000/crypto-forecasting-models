"""
LSTMModelPackage — standalone high-performance inference package for LSTM log-return models.
Optimised for high-throughput backtesting and live trading signal generation.
"""
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from .lstm_arch import build_model_from_config
from .feature_engineering import prepare_features, uses_legacy_vol_percentile
from .compat import temporary_calibrator_shim, resolve_scaler_columns
from .runtime_utils import MinRowsInt, TargetScaler, apply_precision
from .signals import (
    compute_sigma_series,
    align_sigma,
    fill_sigma_nan,
    denormalize_mu_vol_norm,
    compute_signals,
    get_required_warmup,
    valid_sigma_mask,
)

# ---------------------------------------------------------------------------
# HELPERS & CONFIG
# ---------------------------------------------------------------------------

_MinRowsInt = MinRowsInt
_apply_precision = apply_precision
_TargetScaler = TargetScaler
_VALID_CHECKPOINTS = ('best', 'last', 'swa')


@dataclass
class SignalConfig:
    """
    Configuration parameters for probabilistic signal generation.
    threshold: Minimum log-return move to trigger signal. Default 0.001.
    window_sigma: Rolling window for volatility estimation. Default 100.
    """
    threshold: float = 0.001
    window_sigma: int = 100


# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

_RAW_REQUIRED = {'open_time', 'open', 'high', 'low', 'close', 'volume'}
_PREP_REQUIRED = {'close'}


def _detect_mode(df: pd.DataFrame, config: dict) -> str:
    first_feat = config['features'][0]
    return 'prepared' if first_feat in df.columns else 'raw'


def _validate(df: pd.DataFrame, mode: str, config: dict, window_sigma: int) -> None:
    lookback = config['lookback']
    features = config['features']

    if mode == 'raw':
        missing = _RAW_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"[raw mode] Missing required columns: {sorted(missing)}")
        warmup = get_required_warmup(features)
        min_rows = lookback + window_sigma + warmup
        if len(df) < min_rows:
            raise ValueError(
                f"[raw mode] Insufficient rows: got {len(df)}, need >= {min_rows} "
                f"(lookback={lookback} + window_sigma={window_sigma} + warmup={warmup})"
            )
        core_cols = ['open', 'high', 'low', 'close', 'volume']
        nan_cols = [c for c in core_cols if c in df.columns and df[c].isna().any()]
        if nan_cols:
            raise ValueError(f"Columns {nan_cols} contain NaN values.")
        if (df['close'] <= 0).any():
            raise ValueError("Column 'close' contains zero or negative values.")
        if 'volume' in df.columns and (df['volume'] < 0).any():
            raise ValueError("Column 'volume' contains negative values.")
    elif mode == 'prepared':
        missing_base = _PREP_REQUIRED - set(df.columns)
        if missing_base:
            raise ValueError(f"[prepared mode] Missing base columns: {sorted(missing_base)}")
        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            preview = missing_feats[:5]
            extra = f' ... (+{len(missing_feats) - 5} more)' if len(missing_feats) > 5 else ''
            raise ValueError(f"[prepared mode] Missing feature columns: {preview}{extra}")
        min_rows = lookback + window_sigma
        if len(df) < min_rows:
            raise ValueError(
                f"[prepared mode] Insufficient rows: got {len(df)}, need >= {min_rows}"
            )
        if df['close'].isna().any():
            raise ValueError("Column 'close' contains NaN values.")
        if (df['close'] <= 0).any():
            raise ValueError("Column 'close' contains zero or negative values.")
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'raw', 'prepared', or 'auto'.")


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class LSTMModelPackage:
    """
    Standalone inference + signal package for trained LSTM log-return models.

    Parameters
    ----------
    model_dir : str
        Directory produced by lstm_train.py. Must contain:
            lstm_model.pth, feature_scaler.pkl, model_config.json
    signal_config : SignalConfig, optional
    device : str, optional ('cpu' default, or 'cuda')
    checkpoint : 'best' (default, lstm_model.pth) | 'swa' | 'last'
        'last' is a deprecated alias of 'best': training stores one non-SWA
        checkpoint (the best-validation weights), there is no separate last-epoch file.
    apply_calibration : bool
        Opt-in. lstm_test.py and the simulation never apply calibrator.pkl to the
        primary metrics, so the default keeps package signals identical to them.
    """

    def __init__(self, model_dir: str,
                 signal_config: Optional[SignalConfig] = None,
                 config: Optional[SignalConfig] = None,
                 device: str = 'cpu',
                 checkpoint: str = 'best',
                 apply_calibration: bool = False):
        self._model_dir = model_dir
        self._device = torch.device(device)
        self._checkpoint = self._normalise_checkpoint(checkpoint)
        self._apply_calibration_enabled = bool(apply_calibration)

        self._config = self._load_config()
        effective_cfg = signal_config if signal_config is not None else config
        if effective_cfg is None:
            effective_cfg = SignalConfig(
                threshold=float(self._config.get('signal_threshold', 0.001)),
            )
        self._sig_cfg = effective_cfg
        self._target_scaler = self._load_target_scaler()
        self._feature_scaler = self._load_feature_scaler()
        self._scaler_feature_cols = resolve_scaler_columns(self._feature_scaler, self._config)
        self._model = self._load_model()
        self._calibrator = self._load_calibrator()
        if self._apply_calibration_enabled and self._calibrator is None:
            warnings.warn(
                f"apply_calibration=True but no usable calibrator.pkl in {model_dir}; "
                "signals stay uncalibrated.", UserWarning, stacklevel=2)
        self._warned_missing_taker: bool = False

    @classmethod
    def from_directory(cls, model_dir: str,
                       signal_config: Optional[SignalConfig] = None,
                       config: Optional[SignalConfig] = None,
                       device: str = 'cpu',
                       checkpoint: str = 'best',
                       apply_calibration: bool = False) -> 'LSTMModelPackage':
        """Alternative factory constructor matching SignalModelPackage ergonomics."""
        return cls(model_dir=model_dir, signal_config=signal_config, config=config,
                   device=device, checkpoint=checkpoint,
                   apply_calibration=apply_calibration)

    @staticmethod
    def _normalise_checkpoint(checkpoint: str) -> str:
        if checkpoint not in _VALID_CHECKPOINTS:
            raise ValueError(
                f"Unknown checkpoint '{checkpoint}'. Use one of {list(_VALID_CHECKPOINTS)}.")
        if checkpoint == 'last':
            warnings.warn(
                "checkpoint='last' is deprecated: there is no separate last-epoch file, "
                "it loads lstm_model.pth exactly like 'best'.",
                DeprecationWarning, stacklevel=3)
            return 'best'
        return checkpoint

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
        if self._checkpoint == 'swa':
            path = os.path.join(self._model_dir, 'lstm_model_swa.pth')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"SWA checkpoint requested ('swa'), but lstm_model_swa.pth not found in: {self._model_dir}"
                )
        else:
            path = os.path.join(self._model_dir, 'lstm_model.pth')
            if not os.path.exists(path):
                raise FileNotFoundError(f"lstm_model.pth not found in: {self._model_dir}")
        input_size = self._config.get('n_features') or len(self._config['features'])
        model = build_model_from_config(self._config, input_size)
        state = torch.load(path, map_location=self._device, weights_only=True)
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        return model

    def _load_feature_scaler(self):
        for name in ('feature_scaler.pkl', 'scaler_features.pkl'):
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                import joblib
                return joblib.load(path)
        return None

    def _load_target_scaler(self) -> Optional['_TargetScaler']:
        if not self._config.get('scale_target', False):
            return None
        params = self._config.get('target_scaler_params')
        if params:
            return _TargetScaler.from_params(params)
        pkl_path = os.path.join(self._model_dir, 'target_scaler.pkl')
        if os.path.exists(pkl_path):
            import joblib
            raw = joblib.load(pkl_path)
            if isinstance(raw, _TargetScaler):
                return raw
            wrapper = _TargetScaler.__new__(_TargetScaler)
            wrapper.method = self._config.get('target_scaler_type', 'robust')
            wrapper._scaler = raw
            return wrapper
        return None

    def _load_calibrator(self) -> Optional[dict]:
        path = os.path.join(self._model_dir, 'calibrator.pkl')
        if not os.path.exists(path):
            return None
        try:
            with temporary_calibrator_shim():
                import joblib
                cal = joblib.load(path)
                return cal if isinstance(cal, dict) else None
        except Exception as e:
            warnings.warn(f"Failed to load calibrator from {path}: {e}", UserWarning)
            return None

    def _apply_calibration(self, signal_buy: np.ndarray, signal_sell: np.ndarray):
        if self._calibrator is None:
            return signal_buy, signal_sell
        buy_model = self._calibrator.get('buy')
        sell_model = self._calibrator.get('sell')
        if buy_model is not None:
            signal_buy = np.clip(buy_model.predict(signal_buy), 0.0, 100.0)
        if sell_model is not None:
            signal_sell = np.clip(sell_model.predict(signal_sell), 0.0, 100.0)
        return signal_buy, signal_sell

    # Backward compatibility alias
    _apply_signal_calibration = _apply_calibration

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @property
    def min_rows_required(self) -> _MinRowsInt:
        lookback = int(self._config['lookback'])
        window_sigma = int(self._sig_cfg.window_sigma)
        warmup = get_required_warmup(self._config['features'])
        raw_val = lookback + window_sigma + warmup
        prep_val = lookback + window_sigma
        return _MinRowsInt(raw_val, raw_val, prep_val)

    def get_model_info(self) -> dict:
        c = self._config
        return {
            'model_id': c.get('model_id', 'unknown'),
            'data_id': c.get('data_id', 'unknown'),
            'symbol': c.get('symbol', 'unknown'),
            'lookback': c.get('lookback'),
            'horizon': c.get('horizon'),
            'n_features': c.get('n_features'),
            'features': list(c.get('features', [])),
            'feature_set': c.get('feature_set'),
            'rnn_type': c.get('rnn_type', 'lstm'),
            'hidden_size': c.get('hidden_size'),
            'num_layers': c.get('num_layers'),
            'use_attention': c.get('use_attention'),
            'use_rnn_norm': c.get('use_rnn_norm', False),
            'pooling_mode': c.get('pooling_mode', 'last'),
            'target_mode': c.get('target_mode', 'raw'),
            'checkpoint': self._checkpoint,
            'has_swa': os.path.isfile(os.path.join(self._model_dir, 'lstm_model_swa.pth')),
            'is_fusion': False,
            'is_standalone': True,
            'device': str(self._device),
            'signal_threshold': self._sig_cfg.threshold,
            'signal_window_sigma': self._sig_cfg.window_sigma,
            'calibrated': self._calibrator is not None,
            'calibrator_available': self._calibrator is not None,
            'calibration_applied': (self._calibrator is not None
                                    and self._apply_calibration_enabled),
        }

    def predict_batch(self, df: pd.DataFrame,
                      data_mode: str = 'auto',
                      input_type: Optional[str] = None,
                      batch_size: int = 4096,
                      dataset_id: Optional[str] = None,
                      return_df: bool = True,
                      drop_warmup: bool = True) -> Union[pd.DataFrame, dict[str, np.ndarray]]:
        """
        Compute probabilistic signals for all valid samples.
        High-throughput zero-copy sequence unfolding.

        drop_warmup: True drops windows whose rolling sigma is not yet defined (live
            use). False keeps every window with sigma left as NaN while signals use the
            median-filled sigma, which reproduces the step grid of lstm_test.py and
            the simulation v2 (needed for parity checks).
        """
        effective_mode = input_type if input_type is not None else data_mode
        df = df.copy().reset_index(drop=True)
        # Transparent time column aliases (e.g. CCXT or exchange APIs using timestamp/time)
        if 'open_time' not in df.columns:
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'open_time'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'open_time'})

        # Sort chronologically if open_time is present
        if 'open_time' in df.columns and not df['open_time'].is_monotonic_increasing:
            df = df.sort_values('open_time').reset_index(drop=True)

        # Cast raw OHLCV columns to float if stringified from JSON/CSV
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        mode = _detect_mode(df, self._config) if effective_mode == 'auto' else effective_mode
        _validate(df, mode, self._config, self._sig_cfg.window_sigma)

        # Cross-dataset warning in prepared mode
        if mode == 'prepared':
            data_id_in = dataset_id or getattr(df, 'data_id', None) or (df['data_id'].iloc[0] if 'data_id' in df.columns else None)
            expected_id = self._config.get('data_id')
            if data_id_in and expected_id and str(data_id_in).upper() != str(expected_id).upper():
                warnings.warn(
                    f"Cross-Dataset Warning: Model was trained on {expected_id}, but input data is marked as {data_id_in}. "
                    "Different datasets use different normalisation statistics.",
                    UserWarning, stacklevel=2
                )

        if mode == 'raw' and 'buy_pressure' in self._config['features'] and 'taker_buy_base_asset_volume' not in df.columns:
            if not self._warned_missing_taker:
                warnings.warn(
                    "Model requires 'buy_pressure', but 'taker_buy_base_asset_volume' is missing from input. "
                    "Imputing neutral buy_pressure=0.5. For optimal accuracy, provide Binance data with taker volume.",
                    UserWarning, stacklevel=2
                )
                self._warned_missing_taker = True

        df_work = self._prepare_df(df, mode)
        mu = self._predict_mu(df_work, batch_size=batch_size)

        lookback = self._config['lookback']
        window = self._sig_cfg.window_sigma
        sigma_series = compute_sigma_series(df_work['close'], window)
        sigma_aligned = align_sigma(sigma_series, lookback, len(mu))
        sigma_safe = fill_sigma_nan(sigma_aligned)

        idx = np.clip(np.arange(len(mu)) + (lookback - 1), 0, len(df_work) - 1)
        timestamps = (df_work['open_time'].iloc[idx].to_numpy()
                      if 'open_time' in df_work.columns
                      else np.arange(len(mu)))
        close_aligned = df_work['close'].iloc[idx].to_numpy(dtype=np.float64)

        signal_buy, signal_sell = compute_signals(mu, sigma_safe, self._sig_cfg.threshold)
        if self._apply_calibration_enabled:
            signal_buy, signal_sell = self._apply_calibration(signal_buy, signal_sell)

        if drop_warmup:
            valid_mask = valid_sigma_mask(
                df_work['close'], sigma_aligned, lookback,
                vol_norm=self._config.get('target_mode') == 'vol_norm')
        else:
            valid_mask = np.ones(len(mu), dtype=bool)
        out_dict = {
            'timestamp': timestamps[valid_mask],
            'close': close_aligned[valid_mask],
            'mu': mu[valid_mask],
            'sigma': sigma_aligned[valid_mask],
            'signal_buy': signal_buy[valid_mask],
            'signal_sell': signal_sell[valid_mask],
        }
        return pd.DataFrame(out_dict) if return_df else out_dict

    def predict_live(self, df: pd.DataFrame,
                     data_mode: str = 'auto',
                     input_type: Optional[str] = None,
                     dataset_id: Optional[str] = None) -> dict:
        """Single probabilistic signal for the latest closed candle."""
        effective_mode = input_type if input_type is not None else data_mode
        result = self.predict_batch(df, data_mode=effective_mode, dataset_id=dataset_id, return_df=False)
        if len(result['timestamp']) == 0:
            req_rows = self.min_rows_required(effective_mode)
            raise RuntimeError(
                f"predict_live: no valid signals produced. Ensure at least {req_rows} rows."
            )
        return {
            'timestamp': result['timestamp'][-1],
            'close': float(result['close'][-1]),
            'mu': float(result['mu'][-1]),
            'sigma': float(result['sigma'][-1]),
            'signal_buy': float(result['signal_buy'][-1]),
            'signal_sell': float(result['signal_sell'][-1]),
        }

    # ------------------------------------------------------------------
    # INTERNAL PIPELINE
    # ------------------------------------------------------------------

    def _prepare_df(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        features = self._config['features']
        if mode == 'raw':
            df_feat = prepare_features(
                df.copy(), features=features,
                legacy_vol_percentile=uses_legacy_vol_percentile(self._config.get('data_id')))
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            
            if 'buy_pressure' in features and 'buy_pressure' not in df_feat.columns:
                df_feat['buy_pressure'] = 0.5

            # Fill missing scaler columns with 0.0 before transform
            scaler_cols = self._scaler_feature_cols
            if self._feature_scaler is None:
                # Training consumed scaled CSV features; unscaled live features would silently
                # shift every network input, so a model saved without its scaler cannot serve raw data.
                raise ValueError(
                    "Raw mode requires the feature scaler used in training "
                    "(feature_scaler.pkl or scaler_features.pkl), which is missing from the model "
                    "directory. Copy scaler_features.pkl of the training dataset there, or use "
                    "mode='prepared' with already scaled features.")
            if scaler_cols:
                for col in scaler_cols:
                    if col not in df_feat.columns:
                        df_feat[col] = 0.0
                df_feat[scaler_cols] = self._feature_scaler.transform(df_feat[scaler_cols])

            available = [f for f in features if f in df_feat.columns]
            return df_feat.dropna(subset=available).reset_index(drop=True)

        return df.copy().reset_index(drop=True)

    def _predict_mu(self, df_work: pd.DataFrame, batch_size: int = 4096) -> np.ndarray:
        features = self._config['features']
        lookback = self._config['lookback']
        data = df_work[features].to_numpy(dtype=np.float32)

        feat_round = int(self._config.get('feature_round_decimals', 0) or 0)
        if feat_round > 0:
            data = np.round(data, decimals=feat_round)

        if len(data) < lookback:
            return np.empty(0, dtype=np.float64)

        # Zero-copy tensor unfolding (1ms for 200k samples)
        tensor_data = torch.from_numpy(np.ascontiguousarray(data))
        windows = tensor_data.unfold(0, lookback, 1).transpose(1, 2)
        n_samples = windows.shape[0]

        preds = []
        with torch.inference_mode():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = windows[start:end].to(self._device, non_blocking=True)
                out = self._model(x_batch).squeeze(-1).cpu().numpy()
                preds.append(np.atleast_1d(out))

        y_scaled = np.concatenate(preds)

        if self._target_scaler is not None:
            mu = self._target_scaler.inverse_transform(y_scaled).astype(np.float64)
        else:
            mu = y_scaled.astype(np.float64)

        # Denormalise vol_norm predictions (y_norm * sigma_100)
        if self._config.get('target_mode') == 'vol_norm':
            close_prices = df_work['close'].to_numpy(dtype=np.float64)
            mu = denormalize_mu_vol_norm(mu, close_prices, lookback)

        pred_prec = int(self._config.get('prediction_precision', 0) or 0)
        return _apply_precision(mu, pred_prec)
