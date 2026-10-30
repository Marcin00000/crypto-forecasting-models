"""
Submodel Manager for Dynamic (Mode 2) submodel integration.

Loads trained submodels (LSTM Log-Return, XGBoost, Random Forest, Price LSTM)
and generates one prediction channel per submodel.  Predictions are pre-computed
once and cached as numpy arrays so training loop has zero additional overhead
from submodels.

Public API:
    SubmodelManager(submodel_dirs, device)
        .precompute_predictions(X_3d, signal_features) -> np.ndarray
        .diagnose(preds, split_name, logger) -> list[dict]
        .total_output_dim() -> int
    apply_submodel_mode(preds, mode, seed, rng_offset) -> np.ndarray
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import joblib

from .submodel_packages import (
    detect_submodel_kind, load_package, predict_package_output, submodel_lookback,
)

SUBMODEL_MODES = ('real', 'shuffle', 'noise', 'zero')

# Degeneracy guards used by SubmodelManager.diagnose()
DEGENERATE_STD = 1e-6
LOW_STD_RATIO = 0.30


# =========================================================================
# SUBMODEL ENTRY (single loaded model)
# =========================================================================

@dataclass
class SubmodelEntry:
    """Descriptor for a single loaded submodel."""
    model_dir: str
    model: Any  # LR: torch module; xgb / rf / price: standalone inference package
    config: dict
    feature_list: list[str]
    target_scaler: Any  # sklearn scaler or None
    output_dim: int = 1
    zero_filled_features: list[str] = field(default_factory=list)
    kind: str = 'lr'  # 'lr' | 'xgb' | 'rf' | 'price'


# =========================================================================
# FEATURE MAPPING
# =========================================================================

def _map_features(
    signal_features: list[str],
    lr_features: list[str],
    X_signal: np.ndarray,
    max_missing_ratio: float = 0.30,
) -> tuple[np.ndarray, list[str]]:
    """
    Map Signal Model features to LR model features.

    For each LR feature:
      - if present in signal_features → copy column
      - if missing → fill with zeros (warning emitted, raises ValueError if > max_missing_ratio)

    Args:
        signal_features:   column names in X_signal
        lr_features:       column names expected by LR model
        X_signal:          (N, lookback, n_signal_features)
        max_missing_ratio: max allowable fraction of missing features (default 0.30)

    Returns:
        (X_lr, missing) — X_lr of shape (N, lookback, n_lr_features) and the list
        of LR feature names that had to be zero-filled.
    """
    sig_idx = {name: i for i, name in enumerate(signal_features)}
    n_samples, lookback, _ = X_signal.shape
    n_lr = len(lr_features)
    X_lr = np.zeros((n_samples, lookback, n_lr), dtype=np.float32)
    missing = []

    for j, feat in enumerate(lr_features):
        if feat in sig_idx:
            X_lr[:, :, j] = X_signal[:, :, sig_idx[feat]]
        else:
            missing.append(feat)

    if missing:
        missing_ratio = len(missing) / max(n_lr, 1)
        if missing_ratio > max_missing_ratio:
            raise ValueError(
                f"SubmodelManager: Too many missing LR features in Signal data "
                f"({len(missing)}/{n_lr} missing, {missing_ratio:.1%} > max allowed {max_missing_ratio:.0%}). "
                f"Missing: {missing}"
            )
        warnings.warn(
            f"SubmodelManager: {len(missing)} LR features missing in Signal "
            f"data (filled with zeros): {missing}",
            stacklevel=3,
        )
    return X_lr, missing


# =========================================================================
# LR MODEL LOADER
# =========================================================================

def _load_lr_model(model_dir: str, device: torch.device) -> SubmodelEntry:
    """
    Load a trained LSTM Log-Return model from disk.

    Reads model_config.json, reconstructs LSTMLogReturnModel, loads weights
    from lstm_model.pth, and optionally loads target_scaler.pkl.
    """
    model_dir = str(model_dir)
    config_path = os.path.join(model_dir, 'model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"model_config.json not found in {model_dir}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Import LR model class (both modules live in the same repo)
    from lstm_logreturn_model.common.model import LSTMLogReturnModel

    # Reconstruct model architecture from saved config
    # model_config.json uses lowercase keys matching constructor params
    n_features = config.get('n_features', config.get('input_size', 23))
    fc_layers = config.get('fc_layers', [32])
    if isinstance(fc_layers, str):
        fc_layers = [int(x) for x in fc_layers.split(',')]

    model = LSTMLogReturnModel(
        input_size=n_features,
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.2),
        bidirectional=config.get('bidirectional', False),
        use_attention=config.get('use_attention', False),
        fc_layers=fc_layers,
        activation=config.get('activation', 'relu'),
        fc_norm=config.get('fc_norm', 'layer'),
        rnn_type=config.get('rnn_type', 'lstm'),
        use_input_norm=config.get('use_input_norm', False),
        attention_heads=config.get('attention_heads', 1),
        lstm_dropout=config.get('lstm_dropout'),
        fc_dropout=config.get('fc_dropout'),
        arch_type=config.get('arch_type', 'legacy'),
        use_feature_attention=config.get('use_feature_attention', False),
        use_rnn_norm=config.get('use_rnn_norm', False),
        pooling_mode=config.get('pooling_mode', 'last'),
        use_conv1d_front=config.get('use_conv1d_front', False),
        conv1d_kernel=config.get('conv1d_kernel', 5),
        use_fc_residual=config.get('use_fc_residual', False),
        use_learnable_output_scale=config.get(
            'use_learnable_output_scale', False),
    )

    # Load weights
    weights_path = os.path.join(model_dir, 'lstm_model.pth')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"lstm_model.pth not found in {model_dir}")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    # count_parameters() only counts trainable tensors, so read it before freezing
    n_params = model.count_parameters()
    model.requires_grad_(False)

    # Load target scaler (for inverse transform)
    target_scaler = None
    scaler_path = os.path.join(model_dir, 'target_scaler.pkl')
    if os.path.exists(scaler_path):
        target_scaler = joblib.load(scaler_path)

    # Feature list from config
    feature_list = config.get('features', [])
    if not feature_list:
        feature_list = config.get('feature_names', [])

    model_id = os.path.basename(model_dir).split('_')[0]
    print(f"  SubmodelManager: loaded {model_id} "
          f"({n_params:,} params, "
          f"{len(feature_list)} features)")

    return SubmodelEntry(
        model_dir=model_dir,
        model=model,
        config=config,
        feature_list=feature_list,
        target_scaler=target_scaler,
        output_dim=1,
    )


def _load_submodel(model_dir: str, device: torch.device) -> SubmodelEntry:
    """Load a submodel of any supported family (LR, XGB, RF, Price LSTM)."""
    kind = detect_submodel_kind(model_dir)
    if kind == 'lr':
        return _load_lr_model(model_dir, device)

    pkg, config = load_package(kind, model_dir, device)
    feature_list = list(config.get('features', []))
    model_id = os.path.basename(str(model_dir)).split('_')[0]
    print(f"  SubmodelManager: loaded {model_id} ({kind}, {len(feature_list)} features)")
    return SubmodelEntry(
        model_dir=str(model_dir), model=pkg, config=config,
        feature_list=feature_list, target_scaler=None, output_dim=1, kind=kind,
    )


# =========================================================================
# SUBMODEL MANAGER
# =========================================================================

class SubmodelManager:
    """
    Manager for submodel integration (Offline Precompute Mode).

    Loads one or more trained LSTM Log-Return models and provides
    batched GPU inference for pre-computing predictions before training/test.
    Predictions are generated in no_grad mode and passed into SignalModel.

    Usage:
        sm = SubmodelManager(['lstm_lr_models/LR1107_...'], device)
        preds = sm.precompute_predictions(X_train_3d, signal_features)
        # preds.shape == (N, total_output_dim)
    """

    INFERENCE_BATCH_SIZE = 4096

    def __init__(
        self,
        submodel_dirs: list[str],
        device: torch.device,
        expected_symbol: str | None = None,
        expected_data_id: str | None = None,
        max_missing_ratio: float | None = None,
        expected_scaler: str | None = None,
        strict: bool = True,
    ):
        """
        Args:
            expected_scaler: feature scaler name ('robust'/'minmax'/...) of the
                dataset the submodels will be fed with. Compared against the
                scaler of the dataset each submodel was trained on.
            strict: raise ValueError instead of warning on symbol / data_id /
                scaler mismatch, and default max_missing_ratio to 0.0.
            max_missing_ratio: None → 0.0 when strict, 0.30 otherwise.
        """
        if not submodel_dirs:
            raise ValueError("submodel_dirs must be a non-empty list")
        self.device = device
        self.strict = bool(strict)
        if max_missing_ratio is None:
            max_missing_ratio = 0.0 if self.strict else 0.30
        self.max_missing_ratio = float(max_missing_ratio)
        self.expected_scaler = expected_scaler
        self.entries: list[SubmodelEntry] = []
        for d in submodel_dirs:
            resolved = _resolve_model_dir(d)
            entry = _load_submodel(resolved, device)
            self.entries.append(entry)
            self._check_entry_consistency(
                entry, resolved, expected_symbol, expected_data_id)

        self.channel_names = self._build_channel_names()
        print(f"SubmodelManager: {len(self.entries)} submodel(s) loaded, "
              f"total output dim = {self.total_output_dim()}, "
              f"strict={self.strict}, max_missing_ratio={self.max_missing_ratio}")

    # -- consistency guards -------------------------------------------------

    def _fail(self, message: str):
        """Raise in strict mode, warn otherwise."""
        if self.strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=3)

    def _check_entry_consistency(self, entry, resolved_dir, expected_symbol,
                                 expected_data_id):
        """Symbol / data_id / feature-scaler compatibility between submodel and data."""
        model_id = os.path.basename(resolved_dir).split('_')[0]
        sm_symbol = entry.config.get('symbol')
        sm_data_id = entry.config.get('data_id')

        # Scaler first: it is the most informative failure — a submodel fed data
        # scaled differently than it was trained on produces worthless output.
        if self.expected_scaler:
            sm_scaler = _read_dataset_scaler(entry.config.get('train_path'), sm_data_id)
            entry.config['_train_data_scaler'] = sm_scaler
            if sm_scaler is None:
                warnings.warn(
                    f"SubmodelManager: cannot read the feature scaler of the dataset "
                    f"submodel {model_id} was trained on (train_path="
                    f"{entry.config.get('train_path')!r}) - scaler compatibility unverified.",
                    stacklevel=3,
                )
            elif sm_scaler != self.expected_scaler:
                self._fail(
                    f"SubmodelManager: submodel {model_id} was trained on "
                    f"{sm_data_id or 'its own dataset'} (scaler={sm_scaler}), but you are "
                    f"feeding it {expected_data_id or 'this dataset'} "
                    f"(scaler={self.expected_scaler}) - its predictions will be worthless; "
                    f"use --allow_submodel_data_mismatch 1 if this is intentional."
                )

        if expected_symbol and sm_symbol and expected_symbol.upper() != sm_symbol.upper():
            self._fail(
                f"SubmodelManager: Symbol mismatch for submodel {model_id}! "
                f"Submodel symbol='{sm_symbol}', expected='{expected_symbol}'. "
                f"Use --allow_submodel_data_mismatch 1 if this is intentional."
            )
        if expected_data_id and sm_data_id and expected_data_id != sm_data_id:
            self._fail(
                f"SubmodelManager: Data ID mismatch for submodel {model_id}! "
                f"Submodel data_id='{sm_data_id}', expected='{expected_data_id}'. "
                f"Use --allow_submodel_data_mismatch 1 if this is intentional."
            )

    def _build_channel_names(self) -> list[str]:
        names = []
        for entry in self.entries:
            model_id = os.path.basename(entry.model_dir.rstrip(os.sep)).split('_')[0]
            if entry.output_dim == 1:
                names.append(model_id)
            else:
                names.extend(f"{model_id}_{k}" for k in range(entry.output_dim))
        return names

    def _reference_std(self, channel_idx: int) -> float | None:
        """Prediction std recorded in the submodel's own training config, if any."""
        cursor = 0
        for entry in self.entries:
            if channel_idx < cursor + entry.output_dim:
                for key in ('val_metrics', 'train_metrics'):
                    metrics = entry.config.get(key) or {}
                    val = metrics.get('pred_std')
                    if isinstance(val, (int, float)) and val > 0:
                        return float(val)
                return None
            cursor += entry.output_dim
        return None

    def total_output_dim(self) -> int:
        """Sum of output dimensions across all submodels."""
        return sum(e.output_dim for e in self.entries)

    def zero_filled_features(self) -> dict[str, list[str]]:
        """Map model_id → LR features that had to be zero-filled during mapping."""
        return {
            os.path.basename(e.model_dir.rstrip(os.sep)).split('_')[0]:
                list(e.zero_filled_features)
            for e in self.entries
        }

    def diagnose(self, preds, split_name, logger=None, strict=None) -> list[dict]:
        """
        Log per-channel statistics of precomputed submodel predictions.

        Raises ValueError (strict mode) when a channel is fully degenerate
        (std < 1e-6) and warns when its std collapsed below 30% of the value
        recorded in the submodel's own training config.

        Args:
            preds:      (N, total_output_dim) array from precompute_predictions()
            split_name: 'train' | 'val' | 'test' — used in log messages only
            logger:     optional logging.Logger; falls back to print()
            strict:     override the manager-level strict flag

        Returns:
            list of per-channel stat dicts.
        """
        strict = self.strict if strict is None else bool(strict)
        arr = np.asarray(preds, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        emit = logger.info if logger is not None else print

        stats = []
        for j in range(arr.shape[1]):
            col = arr[:, j]
            name = (self.channel_names[j] if j < len(self.channel_names)
                    else f"channel_{j}")
            entry_stats = {
                'channel': name,
                'split': split_name,
                'mean': float(np.mean(col)),
                'std': float(np.std(col)),
                'min': float(np.min(col)),
                'max': float(np.max(col)),
                'frac_unique': float(len(np.unique(col)) / max(len(col), 1)),
            }
            stats.append(entry_stats)
            emit(
                f"  submodel[{split_name}] {name}: mean={entry_stats['mean']:+.6f} "
                f"std={entry_stats['std']:.6f} min={entry_stats['min']:+.6f} "
                f"max={entry_stats['max']:+.6f} "
                f"frac_unique={entry_stats['frac_unique']:.4f}"
            )

            if entry_stats['std'] < DEGENERATE_STD:
                msg = (
                    f"SubmodelManager.diagnose: submodel channel '{name}' is degenerate "
                    f"on split '{split_name}' (std={entry_stats['std']:.2e} < "
                    f"{DEGENERATE_STD:.0e}) - it emits a constant value and carries no "
                    f"information. Most likely the data scaler does not match the one "
                    f"the submodel was trained on. Use --allow_submodel_data_mismatch 1 "
                    f"to continue anyway."
                )
                if strict:
                    raise ValueError(msg)
                logging.getLogger('submodel_manager').warning(msg)
                continue

            ref_std = self._reference_std(j)
            if ref_std and entry_stats['std'] < LOW_STD_RATIO * ref_std:
                logging.getLogger('submodel_manager').warning(
                    f"SubmodelManager.diagnose: submodel channel '{name}' on split "
                    f"'{split_name}' has std={entry_stats['std']:.6f}, i.e. "
                    f"{entry_stats['std'] / ref_std:.1%} of the std recorded during its "
                    f"own training ({ref_std:.6f}) - output is strongly compressed."
                )
        return stats

    def precompute_predictions(
        self,
        X_signal: np.ndarray,
        signal_features: list[str],
    ) -> np.ndarray:
        """
        Generate predictions for all submodels via batched GPU inference.

        Args:
            X_signal:        (N, lookback, n_signal_features)
            signal_features: column names matching X_signal axis 2

        Returns:
            np.ndarray of shape (N, total_output_dim) — concatenated
            predictions from all submodels
        """
        n_samples = X_signal.shape[0]
        all_preds = []

        for entry in self.entries:
            preds = self._predict_single(entry, X_signal, signal_features)
            all_preds.append(preds)

        return np.concatenate(all_preds, axis=1)

    def _predict_single(
        self,
        entry: SubmodelEntry,
        X_signal: np.ndarray,
        signal_features: list[str],
    ) -> np.ndarray:
        """
        Generate predictions for a single submodel.

        Returns np.ndarray of shape (N, output_dim).
        """
        # Map features (trees and the price net must not receive zero-filled columns)
        X_lr, missing = _map_features(
            signal_features, entry.feature_list, X_signal,
            max_missing_ratio=self.max_missing_ratio if entry.kind == 'lr' else 0.0,
        )
        entry.zero_filled_features = missing
        if entry.kind != 'lr':
            return predict_package_output(
                entry.kind, entry.model, X_lr, entry.feature_list,
                sub_lookback=submodel_lookback(entry.config))
        n_samples = X_lr.shape[0]
        preds_list = []

        # Batched inference on GPU
        with torch.no_grad():
            for start in range(0, n_samples, self.INFERENCE_BATCH_SIZE):
                end = min(start + self.INFERENCE_BATCH_SIZE, n_samples)
                batch = torch.from_numpy(X_lr[start:end]).to(self.device)
                out = entry.model(batch)  # (B, 1)
                preds_list.append(out.cpu().numpy())

        preds = np.concatenate(preds_list, axis=0)  # (N, 1)

        # Inverse transform if scaler exists
        if entry.target_scaler is not None:
            try:
                preds = entry.target_scaler.inverse_transform(preds)
            except Exception:
                # Some scalers expect 1D input
                preds_flat = preds.ravel()
                preds_flat = entry.target_scaler.inverse_transform(
                    preds_flat.reshape(-1, 1)).ravel()
                preds = preds_flat.reshape(-1, 1)

        return preds.astype(np.float32)


# =========================================================================
# ABLATION MODES
# =========================================================================

def apply_submodel_mode(preds, mode, seed, rng_offset=0) -> np.ndarray:
    """
    Ablation control for submodel predictions.

    Modes:
        'real'    — unchanged (default)
        'shuffle' — row permutation: marginal distribution preserved, the
                    row-wise link to the target is destroyed
        'noise'   — resample each column from N(mean, std) of that column
        'zero'    — all zeros

    Deterministic for a given (seed, rng_offset); use a different rng_offset
    per split so train and val are not permuted identically.
    """
    if mode is None or mode == 'real':
        return preds
    if mode not in SUBMODEL_MODES:
        raise ValueError(
            f"Unknown submodel_mode '{mode}'. Available: {list(SUBMODEL_MODES)}")
    if preds is None:
        return preds

    arr = np.asarray(preds, dtype=np.float32)
    if mode == 'zero':
        return np.zeros_like(arr)

    rng = np.random.default_rng(int(seed) + int(rng_offset))
    if mode == 'shuffle':
        perm = rng.permutation(arr.shape[0])
        return arr[perm].copy()

    # 'noise'
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return rng.normal(loc=mean, scale=std, size=arr.shape).astype(np.float32)


# =========================================================================
# PATH RESOLVER
# =========================================================================

def _read_dataset_scaler(train_path: str | None,
                         data_id: str | None = None) -> str | None:
    """
    Read metadata['scalers']['features'] of the dataset a submodel was trained on.

    Falls back to searching data_preparation/prepared_data by data_id when the
    absolute train_path stored in the submodel config no longer exists.
    """
    candidates = []
    if train_path:
        # New configs store train_path relative to the repo root.
        from .path_utils import resolve_repo_path
        candidates.append(Path(resolve_repo_path(train_path)))
    if data_id:
        prepared = Path(__file__).resolve().parents[2] / 'data_preparation' / 'prepared_data'
        if prepared.exists():
            candidates.extend(
                d for d in prepared.iterdir()
                if d.is_dir() and d.name.startswith(f"{data_id}_")
            )

    for candidate in candidates:
        meta_path = candidate / 'metadata.json'
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        scaler = (meta.get('scalers') or {}).get('features')
        if scaler:
            return str(scaler)
    return None



def _resolve_model_dir(path_or_id: str) -> str:
    """
    Resolve a model directory from a path or LR model ID.

    Accepts:
        - Full/relative path to model directory
        - Model ID like 'LR1107' (searches lstm_logreturn_model/lstm_lr_models and lstm_lr_models)

    Returns absolute path string.
    """
    # Direct path
    if os.path.isdir(path_or_id):
        return os.path.abspath(path_or_id)

    model_id = path_or_id.strip()
    repo_root = Path(__file__).resolve().parents[2]

    # Possible candidate root directories (LR, XGB, RF, Price LSTM models)
    search_dirs = [
        repo_root / 'lstm_logreturn_model' / 'lstm_lr_models',
        repo_root / 'lstm_lr_models',
        repo_root / 'xgb_model' / 'xgb_models',
        repo_root / 'rf_model' / 'rf_models',
        repo_root / 'lstm_model' / 'lstm_models',
    ]

    for lr_models_dir in search_dirs:
        if lr_models_dir.exists():
            candidates = [
                d for d in lr_models_dir.iterdir()
                if d.is_dir() and d.name.startswith(f"{model_id}_")
            ]
            if len(candidates) == 1:
                return str(candidates[0])
            if len(candidates) > 1:
                best = max(candidates, key=lambda d: d.stat().st_mtime)
                return str(best)

    # Try relative to repo root
    candidate = repo_root / path_or_id
    if candidate.is_dir():
        return str(candidate)

    raise FileNotFoundError(
        f"Could not resolve submodel directory: '{path_or_id}'. "
        f"Provide a full path or model ID (e.g. 'LR1107', 'X240', 'R057', 'L012')."
    )


def union_submodel_features(submodel_dirs: list[str]) -> list[str]:
    """Ordered union of config['features'] over all submodels (no duplicates)."""
    merged: dict[str, None] = {}
    for d in submodel_dirs:
        with open(os.path.join(_resolve_model_dir(d), 'model_config.json'),
                  'r', encoding='utf-8') as f:
            cfg = json.load(f)
        for name in cfg.get('features') or cfg.get('feature_names') or []:
            merged.setdefault(name, None)
    return list(merged)


def resolve_submodel_feature_set(submodel_dirs, symbol='ETH', explicit_set=None):
    """Forwarder to common.data_utils.resolve_submodel_feature_set."""
    from .data_utils import resolve_submodel_feature_set as _resolver
    return _resolver(submodel_dirs, symbol, explicit_set)
