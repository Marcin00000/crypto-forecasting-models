"""
SignalModelPackage — standalone high-performance inference package for LSTM Signal models.
Optimised for batched backtesting (zero-copy window unfolding) and live bot signal generation.
"""
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .signal_arch import build_signal_model_from_config
from .feature_engineering import (
    prepare_features, get_required_warmup, uses_legacy_vol_percentile,
)
from .compat import temporary_calibrator_shim, resolve_scaler_columns
from .runtime_utils import MinRowsInt
from .submodels import StandaloneSubmodelRunner


# ===========================================================================
# CONFIG DATACLASS
# ===========================================================================

@dataclass
class SignalInferenceConfig:
    """
    Execution configuration for standalone Signal model inference.

    checkpoint:    'best' (signal_model.pth) or 'swa' (signal_model_swa.pth).
    device:        'cpu' (default) or 'cuda'.
    submodel_dirs: optional list of directory paths to LR / XGB / RF / Price submodels
                   (for submodel_fusion).
    """
    checkpoint: str = 'best'
    device: str = 'cpu'
    submodel_dirs: Optional[list[str]] = None


# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

_RAW_REQUIRED = {'open_time', 'open', 'high', 'low', 'close', 'volume'}
_PREP_REQUIRED = {'close'}


def _detect_mode(df: pd.DataFrame, config: dict) -> str:
    first_feat = config['features'][0]
    return 'prepared' if first_feat in df.columns else 'raw'


def _validate(df: pd.DataFrame, mode: str, config: dict) -> None:
    lookback = config['lookback']
    features = config['features']

    if mode == 'raw':
        missing = _RAW_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"[raw mode] Missing required columns: {sorted(missing)}")
        warmup = get_required_warmup(features)
        min_rows = lookback + warmup
        if len(df) < min_rows:
            raise ValueError(
                f"[raw mode] Insufficient rows: got {len(df)}, need >= {min_rows} "
                f"(lookback={lookback} + required_warmup={warmup})"
            )
    elif mode == 'prepared':
        missing_base = _PREP_REQUIRED - set(df.columns)
        if missing_base:
            raise ValueError(f"[prepared mode] Missing base columns: {sorted(missing_base)}")
        missing_feats = [f for f in features if f not in df.columns]
        if missing_feats:
            preview = missing_feats[:5]
            extra = f' ... (+{len(missing_feats) - 5} more)' if len(missing_feats) > 5 else ''
            raise ValueError(f"[prepared mode] Missing feature columns: {preview}{extra}")
        if len(df) < lookback:
            raise ValueError(
                f"[prepared mode] Insufficient rows: got {len(df)}, need >= {lookback}"
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

class SignalModelPackage:
    """
    Standalone inference + signal generation package for trained LSTM Signal models.

    Parameters
    ----------
    model_dir : str
        Directory produced by signal_train.py. Must contain:
            model_config.json, signal_model.pth (and optionally feature_scaler.pkl, calibrator.pkl)
    config : SignalInferenceConfig, optional
    device : str, optional ('cpu' default, or 'cuda')
    submodel_dirs : list of str, optional (paths to LR submodels for submodel_fusion)
    apply_calibration : bool
        Opt-in. signal_test.py and the simulation never apply calibrator.pkl to the
        primary metrics, so the default keeps package probabilities identical to them.
    """

    @classmethod
    def from_directory(cls, model_dir: str,
                       config: Optional[SignalInferenceConfig] = None,
                       device: Optional[str] = None,
                       submodel_dirs: Optional[list[str]] = None,
                       apply_calibration: bool = False) -> SignalModelPackage:
        return cls(model_dir=model_dir, config=config, device=device,
                   submodel_dirs=submodel_dirs, apply_calibration=apply_calibration)

    def __init__(self, model_dir: str,
                 config: Optional[SignalInferenceConfig] = None,
                 device: Optional[str] = None,
                 submodel_dirs: Optional[list[str]] = None,
                 apply_calibration: bool = False):
        self._model_dir = str(model_dir)
        self._cfg = config or SignalInferenceConfig()
        self._apply_calibration_enabled = bool(apply_calibration)

        # Precedence: Constructor argument > Config field > Default
        chosen_device = device if device is not None else self._cfg.device
        self._device = torch.device(chosen_device or 'cpu')

        chosen_submodels = submodel_dirs if submodel_dirs is not None else self._cfg.submodel_dirs
        self._submodel_dirs = chosen_submodels

        self._config = self._load_config()
        self._checkpoint_mode = self._cfg.checkpoint or 'best'
        self._feature_scaler = self._load_feature_scaler()
        self._scaler_feature_cols = resolve_scaler_columns(self._feature_scaler, self._config)
        self._model = self._load_model()
        self._calibrator = self._load_calibrator()
        if self._apply_calibration_enabled and self._calibrator is None:
            warnings.warn(
                f"apply_calibration=True but no usable calibrator.pkl in {self._model_dir}; "
                "probabilities stay uncalibrated.", UserWarning, stacklevel=2)
        self._submodel_runner = self._init_submodels()
        self._warned_missing_taker: bool = False

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        path = os.path.join(self._model_dir, 'model_config.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"model_config.json not found in: {self._model_dir}")
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # predict_batch reads both output columns ([:, 0] buy, [:, 1] sell); a
        # single-output model would fail later with an unreadable index error.
        output_mode = str(cfg.get('output_mode', 'dual')).lower()
        if output_mode != 'dual':
            raise ValueError(
                f"Model {cfg.get('model_id', self._model_dir)} has output_mode="
                f"'{output_mode}'; signal_model_package supports only 'dual' "
                "(buy + sell probabilities)."
            )
        return cfg

    def _load_model(self):
        if self._checkpoint_mode == 'swa':
            path = os.path.join(self._model_dir, 'signal_model_swa.pth')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"SWA weights checkpoint not found in {self._model_dir}: signal_model_swa.pth"
                )
        else:
            path = os.path.join(self._model_dir, 'signal_model.pth')
            if not os.path.exists(path):
                raise FileNotFoundError(f"signal_model.pth not found in: {self._model_dir}")

        input_size = self._config.get('n_features') or len(self._config['features'])
        state = torch.load(path, map_location=self._device, weights_only=True)
        has_attention = any('attention' in k for k in state.keys())
        use_content_heads = any('content_heads' in k for k in state.keys()) if has_attention else True
        model = build_signal_model_from_config(self._config, input_size, use_content_heads=use_content_heads)
        model.load_state_dict(state, strict=True)
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

    def _load_calibrator(self):
        path = os.path.join(self._model_dir, 'calibrator.pkl')
        if not os.path.exists(path):
            return None
        try:
            with temporary_calibrator_shim():
                import joblib
                return joblib.load(path)
        except Exception as e:
            warnings.warn(f"Failed to load calibrator from {path}: {e}", UserWarning)
            return None

    def _init_submodels(self) -> Optional[StandaloneSubmodelRunner]:
        arch = str(self._config.get('arch_type', 'legacy')).lower()
        sub_size = int(self._config.get('submodel_size', 0) or 0)
        if arch == 'submodel_fusion' and sub_size > 0:
            if not self._submodel_dirs:
                req_dirs = self._config.get('submodel_dirs', [])
                raise ValueError(
                    f"Model {self._config.get('model_id')} has arch_type='submodel_fusion' and requires "
                    f"{sub_size} submodel(s) ({req_dirs}). "
                    f"Provide submodel_dirs=['path/to/LR...', 'path/to/X...', ...] (LR / XGB / RF / "
                    f"Price model directories) in SignalModelPackage constructor, "
                    f"or use a standalone model (e.g. SIG1152, SIG1126, SIG1100, SIG992)."
                )
            return StandaloneSubmodelRunner(
                self._submodel_dirs, self._device,
                max_missing_ratio=float(self._config.get('submodel_max_missing_ratio', 0.30)))
        return None

    def _apply_calibration(self, prob_buy: np.ndarray, prob_sell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._calibrator is None:
            return prob_buy, prob_sell
        stacked = np.column_stack([prob_buy, prob_sell])
        calibrated = self._calibrator.transform(stacked)
        return calibrated[:, 0], calibrated[:, 1]

    # ------------------------------------------------------------------
    # PUBLIC PROPERTIES & METADATA
    # ------------------------------------------------------------------

    @property
    def lookback(self) -> int:
        return int(self._config['lookback'])

    @property
    def min_rows_required(self) -> MinRowsInt:
        """int (raw-mode minimum) that is also callable: ``min_rows_required('prepared')``."""
        lookback = int(self._config['lookback'])
        # Features used only by submodels need their own indicator warm-up in raw mode.
        features = list(self._config['features']) + self._submodel_features()
        raw_val = lookback + get_required_warmup(features)
        return MinRowsInt(raw_val, raw_val, lookback)

    def _submodel_features(self) -> list[str]:
        runner = getattr(self, '_submodel_runner', None)
        return runner.all_features() if runner is not None else []

    def get_model_info(self) -> dict:
        c = self._config
        is_fus = c.get('arch_type') == 'submodel_fusion' and c.get('submodel_size', 0) > 0
        return {
            'model_id': c.get('model_id', 'unknown'),
            'data_id': c.get('data_id', 'unknown'),
            'symbol': c.get('symbol', 'unknown'),
            'lookback': c.get('lookback'),
            'horizon': c.get('horizon', 1),
            'n_features': c.get('n_features'),
            'feature_set': c.get('feature_set'),
            'arch_type': c.get('arch_type', 'legacy'),
            'rnn_type': c.get('rnn_type', 'lstm'),
            'hidden_size': c.get('hidden_size'),
            'num_layers': c.get('num_layers'),
            'use_attention': c.get('use_attention'),
            'use_conv1d_front': c.get('use_conv1d_front', False),
            'checkpoint': self._checkpoint_mode,
            'calibrated': self._calibrator is not None,
            'calibrator_available': self._calibrator is not None,
            'calibration_applied': (self._calibrator is not None
                                    and self._apply_calibration_enabled),
            'submodel_size': c.get('submodel_size', 0),
            'submodel_kinds': ([s.get('kind', 'lr') for s in self._submodel_runner.submodels]
                               if self._submodel_runner is not None else []),
            'is_fusion': is_fus,
            'is_standalone': not is_fus,
            'device': str(self._device),
            'features': list(c.get('features', [])),
            'has_swa': os.path.isfile(os.path.join(self._model_dir, 'signal_model_swa.pth')),
        }

    # ------------------------------------------------------------------
    # INFERENCE PIPELINE
    # ------------------------------------------------------------------

    def predict_batch(self, df: pd.DataFrame,
                      data_mode: str = 'auto',
                      input_type: Optional[str] = None,
                      batch_size: int = 4096,
                      dataset_id: Optional[str] = None,
                      return_df: bool = True) -> pd.DataFrame:
        """
        Compute probabilistic signals [prob_buy, prob_sell] and strengths [0, 100].
        High-throughput zero-copy sequence unfolding.
        """
        effective_mode = input_type if input_type is not None else data_mode
        df = df.copy().reset_index(drop=True)

        # Transparent time column aliases
        if 'open_time' not in df.columns:
            if 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'open_time'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'open_time'})

        # Sort chronologically if open_time is present
        if 'open_time' in df.columns and not df['open_time'].is_monotonic_increasing:
            df = df.sort_values('open_time').reset_index(drop=True)

        # Cast raw OHLCV columns to float if stringified
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        mode = _detect_mode(df, self._config) if effective_mode == 'auto' else effective_mode
        _validate(df, mode, self._config)

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
                    "Model requires 'buy_pressure', but 'taker_buy_base_asset_volume' is missing. "
                    "Imputing neutral buy_pressure=0.5.",
                    UserWarning, stacklevel=2
                )
                self._warned_missing_taker = True

        df_work = self._prepare_df(df, mode)
        features = self._config['features']
        lookback = int(self._config['lookback'])

        data = df_work[features].to_numpy(dtype=np.float32)
        if len(data) < lookback:
            return pd.DataFrame(columns=['timestamp', 'close', 'prob_buy', 'prob_sell', 'signal_buy', 'signal_sell'])

        # Zero-copy sequence unfolding (live-aligned: len - lookback + 1 windows)
        tensor_data = torch.from_numpy(np.ascontiguousarray(data))
        windows = tensor_data.unfold(0, lookback, 1).transpose(1, 2)
        n_samples = windows.shape[0]

        # Precompute submodels if fusion model
        submodel_inputs = None
        if self._submodel_runner is not None:
            df_dict = {
                f: df_work[f].to_numpy(dtype=np.float32)
                for f in df_work.columns
                if f not in ('open_time', 'timestamp', 'time', 'datetime', 'data_id')
                and pd.api.types.is_numeric_dtype(df_work[f])
            }
            submodel_inputs = self._submodel_runner.run_submodels(
                df_dict, n_samples=n_samples, primary_lookback=lookback, batch_size=batch_size
            )

        preds = []
        with torch.inference_mode():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_b = windows[start:end].to(self._device)
                sub_b = torch.from_numpy(submodel_inputs[start:end]).to(self._device) if submodel_inputs is not None else None
                out = self._model(x_b, submodel_inputs=sub_b).cpu().numpy()
                preds.append(np.atleast_2d(out))

        y_pred = np.vstack(preds)
        prob_buy, prob_sell = y_pred[:, 0], y_pred[:, 1]
        if self._apply_calibration_enabled:
            prob_buy, prob_sell = self._apply_calibration(prob_buy, prob_sell)

        idx = np.clip(np.arange(n_samples) + (lookback - 1), 0, len(df_work) - 1)
        timestamps = (df_work['open_time'].iloc[idx].to_numpy()
                      if 'open_time' in df_work.columns
                      else np.arange(n_samples))
        close_aligned = df_work['close'].iloc[idx].to_numpy(dtype=np.float64)

        signal_buy = np.clip(prob_buy * 100.0, 0.0, 100.0)
        signal_sell = np.clip(prob_sell * 100.0, 0.0, 100.0)

        out_dict = {
            'timestamp': timestamps,
            'close': close_aligned,
            'prob_buy': prob_buy,
            'prob_sell': prob_sell,
            'signal_buy': signal_buy,
            'signal_sell': signal_sell,
        }
        return pd.DataFrame(out_dict) if return_df else out_dict

    def predict_live(self, df: pd.DataFrame,
                     data_mode: str = 'auto',
                     input_type: Optional[str] = None,
                     dataset_id: Optional[str] = None) -> dict:
        """Compute probabilistic signal for the latest closed candle."""
        effective_mode = input_type if input_type is not None else data_mode
        result_df = self.predict_batch(df, data_mode=effective_mode, dataset_id=dataset_id, return_df=True)
        if result_df.empty:
            req_rows = self.min_rows_required('prepared' if effective_mode == 'prepared' else 'raw')
            raise RuntimeError(
                f"predict_live: no valid signals produced. Ensure at least {req_rows} rows."
            )
        row = result_df.iloc[-1]
        return {
            'timestamp': row['timestamp'],
            'close': float(row['close']),
            'prob_buy': float(row['prob_buy']),
            'prob_sell': float(row['prob_sell']),
            'signal_buy': float(row['signal_buy']),
            'signal_sell': float(row['signal_sell']),
        }

    # ------------------------------------------------------------------
    # INTERNAL PREPARATION
    # ------------------------------------------------------------------

    def _prepare_df(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        features = self._config['features']
        if mode == 'raw':
            scaler_cols = self._scaler_feature_cols or []
            sub_features = self._submodel_features()
            needed = list(set(features) | set(scaler_cols) | set(sub_features))
            df_feat = prepare_features(
                df.copy(), features=needed,
                legacy_vol_percentile=uses_legacy_vol_percentile(self._config.get('data_id')))
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

            if 'buy_pressure' in features and 'buy_pressure' not in df_feat.columns:
                df_feat['buy_pressure'] = 0.5

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

            # Rows where a submodel-only feature is still warming up must be dropped too.
            available = [f for f in dict.fromkeys(list(features) + sub_features)
                         if f in df_feat.columns]
            return df_feat.dropna(subset=available).reset_index(drop=True)

        return df.copy().reset_index(drop=True)
