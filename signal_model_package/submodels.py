"""
Standalone submodel runner for Signal models with submodel_fusion architecture.
Contains an embedded, self-contained inference implementation of LSTMLogReturnModel,
supporting both Legacy linear head and DualHead architectures without external dependencies.
XGBoost / Random Forest (submodel_tabular.py) and Price-LSTM (submodel_price.py) submodels
are dispatched by the weight files found in the submodel directory.
"""
from __future__ import annotations

import os
import glob
import json
import warnings
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, RobustScaler

from .signal_arch import (
    FeatureAttention,
    FCLayerBlock,
    TemporalAttention,
    build_norm_layer,
    get_activation,
)
from .submodel_price import _PriceSubmodel
from .submodel_tabular import _TabularSubmodel


# =========================================================================
# TARGET SCALER WRAPPER FOR LR SUBMODELS
# =========================================================================

class _LRTargetScaler:
    """Wrapper around sklearn scalers for 1-D log-return outputs."""
    def __init__(self, method: str = 'robust'):
        self.method = method
        self._scaler = RobustScaler() if method == 'robust' else StandardScaler()

    @classmethod
    def from_params(cls, params: dict) -> '_LRTargetScaler':
        obj = cls(method=params.get('method', 'robust'))
        if params['method'] == 'robust':
            obj._scaler.center_ = np.array([params['center']])
            obj._scaler.scale_ = np.array([params['scale']])
        elif params['method'] == 'standard':
            obj._scaler.mean_ = np.array([params['mean']])
            obj._scaler.scale_ = np.array([params['std']])
            obj._scaler.var_ = np.array([params['std'] ** 2])
            obj._scaler.n_features_in_ = 1
        return obj

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).flatten()


# =========================================================================
# SLIM LR MODEL (EMBEDDED ARCHITECTURE)
# =========================================================================

class _LRDualHeadOutput(nn.Module):
    """Direction/amplitude output head for Log-Return models."""
    def __init__(self, input_size: int):
        super().__init__()
        self.direction = nn.Linear(input_size, 1)
        self.amplitude = nn.Linear(input_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direction = torch.tanh(self.direction(x))
        amplitude = F.softplus(self.amplitude(x))
        return direction * amplitude


class _SlimLRModel(nn.Module):
    """
    Self-contained pure-inference LSTM/GRU Log-Return architecture.
    Supports the full checklist of submodel flags:
    bidirectional, use_rnn_norm, fc_norm, use_fc_residual, dual_head, output_scale.
    """
    def __init__(self, config: dict, input_size: int, use_content_heads: bool = True):
        super().__init__()
        c = config
        self.arch_type = str(c.get('arch_type', 'legacy')).lower()
        hidden_size = int(c.get('hidden_size', 128))
        num_layers = int(c.get('num_layers', 2))
        dropout = float(c.get('dropout', 0.2))
        self.bidirectional = bool(c.get('bidirectional', False))
        rnn_type = str(c.get('rnn_type', 'lstm')).lower()
        self.use_input_norm = bool(c.get('use_input_norm', False))
        self.use_rnn_norm = bool(c.get('use_rnn_norm', False))
        self.use_fc_residual = bool(c.get('use_fc_residual', False))
        fc_layers = c.get('fc_layers')
        if isinstance(fc_layers, str):
            fc_layers = [int(x) for x in fc_layers.split(',')]
        activation = str(c.get('activation', 'relu'))
        fc_norm = str(c.get('fc_norm', 'layer'))
        use_attention = bool(c.get('use_attention', False))
        attention_heads = int(c.get('attention_heads', 1))
        self.pooling_mode = str(c.get('pooling_mode', 'last'))
        use_learnable_output_scale = bool(c.get('use_learnable_output_scale', False))
        use_conv1d_front = bool(c.get('use_conv1d_front', False))
        conv1d_kernel = int(c.get('conv1d_kernel', 5))
        use_feature_attention = bool(c.get('use_feature_attention', False))

        self.input_norm = nn.LayerNorm(input_size) if self.use_input_norm else None

        # Same module path as the training FeatureAttention (`gate.0`), so
        # checkpoints that enable per-feature gating load under strict=True.
        if use_feature_attention:
            self.feature_attention = FeatureAttention(input_size)
        else:
            self.feature_attention = None

        if use_conv1d_front:
            self.conv1d_front = nn.Conv1d(input_size, input_size, kernel_size=conv1d_kernel, padding=conv1d_kernel // 2)
            self.conv1d_activation = get_activation(activation)
        else:
            self.conv1d_front = None
            self.conv1d_activation = None

        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0,
                           bidirectional=self.bidirectional)

        rnn_output_size = hidden_size * (2 if self.bidirectional else 1)
        self.rnn_norm = nn.LayerNorm(rnn_output_size) if self.use_rnn_norm else None

        if use_attention:
            self.attention = TemporalAttention(
                rnn_output_size, n_heads=attention_heads,
                use_content_heads=use_content_heads)
        else:
            self.attention = None

        fc_sizes = fc_layers if fc_layers else [rnn_output_size // 2]
        prev_size = rnn_output_size

        if self.use_fc_residual:
            self.fc_blocks = nn.ModuleList([
                FCLayerBlock(prev_size if i == 0 else fc_sizes[i - 1], fc_size, fc_norm, activation, 0.0)
                for i, fc_size in enumerate(fc_sizes)
            ])
            prev_size = fc_sizes[-1] if fc_sizes else rnn_output_size
            self.fc = None
            if self.arch_type == 'legacy':
                self.fc_out = nn.Linear(prev_size, 1)
                self.output_head = None
            else:
                self.fc_out = None
                self.output_head = _LRDualHeadOutput(prev_size)
        else:
            self.fc_blocks = None
            self.fc_out = None
            layers = []
            for fc_size in fc_sizes:
                layers.append(nn.Linear(prev_size, fc_size))
                norm = build_norm_layer(fc_norm, fc_size)
                if norm is not None:
                    layers.append(norm)
                layers.append(get_activation(activation))
                layers.append(nn.Dropout(0.0))
                prev_size = fc_size
            if self.arch_type == 'legacy':
                layers.append(nn.Linear(prev_size, 1))
                self.fc = nn.Sequential(*layers)
                self.output_head = None
            else:
                self.fc = nn.Sequential(*layers)
                self.output_head = _LRDualHeadOutput(prev_size)

        if use_learnable_output_scale:
            self.output_scale = nn.Parameter(torch.ones(1))
            self.output_bias = nn.Parameter(torch.zeros(1))
        else:
            self.output_scale = None
            self.output_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.feature_attention is not None:
            x = self.feature_attention(x)
        if self.conv1d_front is not None:
            x = self.conv1d_front(x.transpose(1, 2)).transpose(1, 2)
            x = self.conv1d_activation(x)

        rnn_out, _ = self.rnn(x)
        if self.rnn_norm is not None:
            rnn_out = self.rnn_norm(rnn_out)

        if self.attention is not None:
            context = self.attention(rnn_out)
        elif self.pooling_mode == 'mean':
            context = rnn_out.mean(dim=1)
        elif self.pooling_mode == 'max':
            context = rnn_out.max(dim=1)[0]
        else:
            context = rnn_out[:, -1, :]

        if self.fc_blocks is not None:
            h = context
            for b in self.fc_blocks:
                h = b(h)
            out = self.output_head(h) if self.output_head is not None else self.fc_out(h)
        else:
            h = self.fc(context)
            out = self.output_head(h) if self.output_head is not None else h

        if self.output_scale is not None:
            out = out * self.output_scale + self.output_bias
        return out


# =========================================================================
# STANDALONE SUBMODEL RUNNER
# =========================================================================

def _resolve_submodel_path(path_or_id: str) -> str:
    """Resolve submodel directory from direct path or model ID."""
    if os.path.isdir(path_or_id):
        return path_or_id
    search_dirs = [
        os.path.join("lstm_logreturn_model", "lstm_lr_models"),
        os.path.join("models", "logreturns"),
        os.path.join("models", "lr"),
        os.path.join("xgb_model", "xgb_models"),
        os.path.join("rf_model", "rf_models"),
        os.path.join("lstm_model", "lstm_models"),
        os.path.join("models", "xgb"),
        os.path.join("models", "rf"),
        os.path.join("models", "price"),
    ]
    for base in search_dirs:
        if os.path.isdir(base):
            matches = glob.glob(os.path.join(base, f"*{path_or_id}*"))
            if matches:
                return matches[0]
    return path_or_id


# Weight file that identifies each submodel family; first match wins.
_KIND_MARKERS = (
    ('xgb', 'xgb_model.json'),
    ('rf', 'rf_model.pkl'),
    ('price', 'best_model.pth'),
    ('price', 'lstm_price_model.pth'),
    ('lr', 'lstm_model.pth'),
)


def _detect_kind(model_dir: str) -> str:
    """'xgb' | 'rf' | 'price' | 'lr' from the weight files in model_dir (default 'lr')."""
    for kind, marker in _KIND_MARKERS:
        if os.path.exists(os.path.join(model_dir, marker)):
            return kind
    return 'lr'


def _load_target_scaler(cfg: dict, model_dir: str):
    """Target scaler of a submodel: params in model_config.json, else target_scaler.pkl."""
    if not cfg.get('scale_target', False):
        return None
    params = cfg.get('target_scaler_params')
    if params:
        return _LRTargetScaler.from_params(params)
    t_path = os.path.join(model_dir, 'target_scaler.pkl')
    if os.path.exists(t_path):
        import joblib
        raw = joblib.load(t_path)
        scaler = raw if isinstance(raw, _LRTargetScaler) else _LRTargetScaler(method='robust')
        if hasattr(raw, 'center_'):
            scaler._scaler = raw
        return scaler
    return None


def _infer_use_content_heads(state: dict) -> bool:
    """Whether a checkpoint's temporal attention carries per-head content projections.

    Checkpoints that pre-date content heads have only `attention.heads.*` and
    `attention.out_proj`; building the new layout would fail under strict loading.
    """
    has_temporal = any(k.startswith('attention.') for k in state)
    return any('content_heads' in k for k in state) if has_temporal else True


class StandaloneSubmodelRunner:
    """Manages loading and batched inference for one or more LR submodels.

    max_missing_ratio: largest fraction of a submodel's features that may be absent
        from the Signal input (zero-filled with a warning). Above it a ValueError is
        raised, matching the training-side SubmodelManager and the Signal config key
        ``submodel_max_missing_ratio``.
    """

    def __init__(self, submodel_dirs: list[str], device: torch.device,
                 max_missing_ratio: float = 0.30):
        if not submodel_dirs:
            raise ValueError("submodel_dirs must be a non-empty list of paths.")
        self.device = device
        self.max_missing_ratio = float(max_missing_ratio)
        self.submodels = []

        for raw_dir in submodel_dirs:
            m_dir = _resolve_submodel_path(raw_dir)
            cfg_path = os.path.join(m_dir, 'model_config.json')
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Submodel config not found in: {m_dir} (from '{raw_dir}')")
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            kind = _detect_kind(m_dir)
            target_scaler = _load_target_scaler(cfg, m_dir)
            if kind == 'lr':
                w_path = os.path.join(m_dir, 'lstm_model.pth')
                if not os.path.exists(w_path):
                    raise FileNotFoundError(f"Submodel weights not found in: {m_dir} (from '{raw_dir}')")
                input_size = cfg.get('n_features') or len(cfg.get('features', []))
                state = torch.load(w_path, map_location=device, weights_only=True)
                model = _SlimLRModel(cfg, input_size,
                                     use_content_heads=_infer_use_content_heads(state))
                model.load_state_dict(state, strict=True)
                model.to(device)
                model.eval()
            elif kind == 'price':
                model = _PriceSubmodel(m_dir, cfg, device)
            else:
                model = _TabularSubmodel(kind, m_dir, cfg, target_scaler)
                target_scaler = None  # applied inside the tabular submodel

            self.submodels.append({
                'model': model,
                'config': cfg,
                'features': cfg.get('features', []),
                'target_scaler': target_scaler,
                'lookback': cfg.get('lookback', 48),
                'kind': kind,
            })

    def all_features(self) -> list[str]:
        """Ordered union of the feature names used by all submodels."""
        return list(dict.fromkeys(f for sub in self.submodels for f in sub['features']))

    def run_submodels(self, df_features: dict[str, np.ndarray], n_samples: int,
                      primary_lookback: int, batch_size: int = 4096) -> np.ndarray:
        """
        Generate aligned submodel predictions for all primary samples.

        Each window uses ``primary_lookback`` (the Signal sequence), not the
        lookback stored in the LR checkpoint. The fusion head was trained on
        predictions from that Signal window: SubmodelManager copies LR features
        out of ``X_signal`` and does not rebuild a separate LR-length sequence.
        df_features: dict mapping feature_name -> 1D numpy array of values.
        """
        preds_all = []
        window = int(primary_lookback)

        for sub in self.submodels:
            model = sub['model']
            lr_feats = sub['features']
            kind = sub.get('kind', 'lr')  # tests build runners without 'kind'

            # Extract and align feature matrix for this LR submodel
            sub_matrix = np.zeros((n_samples + window - 1, len(lr_feats)), dtype=np.float32)
            missing = []
            for j, f_name in enumerate(lr_feats):
                if f_name in df_features:
                    sub_matrix[:, j] = df_features[f_name]
                else:
                    missing.append(f_name)
            if missing:
                ratio = len(missing) / max(len(lr_feats), 1)
                # Trees and the price net must not receive zero-filled columns.
                if ratio > (self.max_missing_ratio if kind == 'lr' else 0.0):
                    raise ValueError(
                        f"Submodel input is missing {len(missing)}/{len(lr_feats)} features "
                        f"({ratio:.1%} > allowed {self.max_missing_ratio:.0%}): {missing}. "
                        "Feed a dataset that contains the submodel features."
                    )
                warnings.warn(f"Submodel missing features {missing} — filled with 0.0", UserWarning)

            if kind != 'lr':
                # sub_matrix already is the candle series (n_samples + window - 1 rows)
                preds_all.append(
                    model.predict(sub_matrix, n_samples, window).reshape(-1, 1))
                continue

            tensor_sub = torch.from_numpy(np.ascontiguousarray(sub_matrix))
            windows_sub = tensor_sub.unfold(0, window, 1).transpose(1, 2)
            if windows_sub.shape[0] != n_samples:
                raise ValueError(
                    f"Submodel window count {windows_sub.shape[0]} != signal samples "
                    f"{n_samples} (signal lookback={window})."
                )

            sub_preds = []
            with torch.inference_mode():
                for start in range(0, n_samples, batch_size):
                    end = min(start + batch_size, n_samples)
                    x_b = windows_sub[start:end].to(self.device)
                    y_b = model(x_b).squeeze(-1).cpu().numpy()
                    sub_preds.append(np.atleast_1d(y_b))

            sub_y = np.concatenate(sub_preds)
            if sub['target_scaler'] is not None:
                sub_y = sub['target_scaler'].inverse_transform(sub_y)
            preds_all.append(sub_y.reshape(-1, 1))

        return np.hstack(preds_all).astype(np.float32)
