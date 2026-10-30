"""
Signal model definition for direct buy/sell signal prediction.

Configurable architecture: hidden size, layers, dropout, bidirectional,
attention, FC layers, activation. Output: dual sigmoid (p_buy, p_sell)
or single sigmoid (placeholder).

Architecture components are self-contained — no import from
lstm_logreturn_model to ensure isolation from LR-specific changes.

F6 flags (Faza C — parity with LR):
    USE_CONV1D_FRONT      1D conv before RNN as local feature extractor
    POOLING_MODE          'last' | 'mean' | 'max'  (when attention disabled)
    USE_FC_RESIDUAL       residual skip in FC head (active only when in==out)
    USE_RNN_NORM          LayerNorm on RNN output before attention/pooling
    USE_LEARNABLE_OUTPUT_SCALE  per-head learnable scale+bias before sigmoid
    FREEZE_RNN            freeze RNN weights (fine-tuning FC / sub-models)
"""
import torch
import torch.nn as nn


# =========================================================================
# HELPERS
# =========================================================================

def get_activation(name):
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.01),
        'silu': nn.SiLU(),
        'swish': nn.SiLU(),
        'mish': nn.Mish(),
    }
    if name not in activations:
        raise ValueError(
            f"Unknown activation: '{name}'. "
            f"Available: {list(activations.keys())}"
        )
    return activations[name]


def build_norm_layer(norm_type, size):
    if norm_type == 'batch':
        return nn.BatchNorm1d(size)
    if norm_type == 'layer':
        return nn.LayerNorm(size)
    if norm_type == 'none':
        return None
    raise ValueError(
        f"Unknown fc_norm: '{norm_type}'. Available: ['batch', 'layer', 'none']"
    )


# =========================================================================
# FC LAYER BLOCK (C3 — residual skip)
# =========================================================================

class FCLayerBlock(nn.Module):
    """Linear → norm → activation → dropout, with optional residual add.

    The residual skip is active only when ``in_size == out_size`` — this
    avoids the dimension-mismatch bug present in early LR implementations
    (CP-067 backlog).  When sizes differ the block behaves identically to
    a plain sequential layer.
    """

    def __init__(self, in_size, out_size, fc_norm, activation, dropout):
        super().__init__()
        self.use_residual = (in_size == out_size)
        self.linear = nn.Linear(in_size, out_size)
        self.norm = build_norm_layer(fc_norm, out_size)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + x
        return out


# =========================================================================
# FEATURE ATTENTION
# =========================================================================

class FeatureAttention(nn.Module):
    """Squeeze-and-excitation gate for per-feature reweighting before RNN."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )
        self.last_weights = None

    def forward(self, x):
        avg = x.mean(dim=1)
        weights = self.gate(avg)
        self.last_weights = weights.detach().mean(dim=0)
        return x * weights.unsqueeze(1)


# =========================================================================
# TEMPORAL ATTENTION
# =========================================================================

class TemporalAttention(nn.Module):
    """Multi-head temporal attention over the RNN output sequence."""

    def __init__(self, input_size, n_heads=1):
        super().__init__()
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if input_size % n_heads != 0:
            raise ValueError(
                f"input_size ({input_size}) must be divisible by "
                f"n_heads ({n_heads})"
            )
        self.n_heads = int(n_heads)
        self.input_size = int(input_size)
        self.head_dim = input_size // n_heads

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, self.head_dim),
                nn.Tanh(),
                nn.Linear(self.head_dim, 1),
            )
            for _ in range(self.n_heads)
        ])
        self.content_heads = nn.ModuleList([
            nn.Linear(input_size, self.head_dim)
            for _ in range(self.n_heads)
        ])
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, input_size)
        self.last_weights = None

    def forward(self, rnn_output):
        contexts = []
        all_weights = []
        for i, head in enumerate(self.heads):
            scores = head(rnn_output)
            weights = torch.softmax(scores, dim=1)
            head_proj = self.content_heads[i](rnn_output)
            ctx = (head_proj * weights).sum(dim=1)
            contexts.append(ctx)
            all_weights.append(weights.detach())
        self.last_weights = (
            torch.stack(all_weights, dim=0).mean(dim=(0, 1)).squeeze(-1))
        concat = torch.cat(contexts, dim=-1)
        return self.out_proj(concat)


# =========================================================================
# SIGNAL MODEL
# =========================================================================

class SignalModel(nn.Module):
    """
    LSTM/GRU + attention + FC → dual sigmoid for direct signal prediction.

    Forward pass (all steps optional/conditional):

        x (B,T,F)
          → [input_norm]
          → [feature_attention]
          → [conv1d_front]        C1: local feature extractor
          → RNN (LSTM/GRU)
          → [rnn_norm]            C4: LayerNorm on RNN output
          → [attention | pooling] C2: 'last'/'mean'/'max' when no attention
          → FC (plain or C3 residual blocks)
          → output_linear
          → [learnable_scale]     C5: per-head scale+bias
          → sigmoid

    F6 Args (Faza C):
        use_conv1d_front:         bool  — 1D conv before RNN
        conv1d_kernel:            int   — kernel width for conv1d (default 5)
        pooling_mode:             str   — 'last'|'mean'|'max' (attention off)
        use_fc_residual:          bool  — residual in FC blocks
        use_rnn_norm:             bool  — LayerNorm after RNN
        use_learnable_output_scale: bool — learnable scale+bias on output
    """

    def __init__(self, input_size, hidden_size=32, num_layers=1,
                 dropout=0.35, bidirectional=False, use_attention=True,
                 fc_layers=None, activation='gelu', fc_norm='layer',
                 rnn_type='lstm', use_input_norm=True,
                 attention_heads=1,
                 lstm_dropout=None, fc_dropout=None,
                 use_feature_attention=False,
                 output_mode='dual', arch_type='legacy',
                 submodel_size=0,
                 use_submodel_norm=True,
                 submodel_norm_type='auto',
                 # F6 flags
                 use_conv1d_front=False, conv1d_kernel=5,
                 pooling_mode='last',
                 use_fc_residual=False,
                 use_rnn_norm=False,
                 use_learnable_output_scale=False):
        super().__init__()
        if output_mode not in ('dual', 'single'):
            raise ValueError(
                f"Unknown output_mode: '{output_mode}'. "
                "Available: ['dual', 'single']"
            )
        if pooling_mode not in ('last', 'mean', 'max'):
            raise ValueError(
                f"Unknown pooling_mode: '{pooling_mode}'. "
                "Available: ['last', 'mean', 'max']"
            )
        if arch_type not in ('legacy', 'dual_head', 'submodel_fusion'):
            raise ValueError(
                f"Unknown arch_type: '{arch_type}'. "
                "Available: ['legacy', 'dual_head', 'submodel_fusion']"
            )
        self.arch_type = arch_type
        self.output_mode = output_mode
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_input_norm = bool(use_input_norm)
        self.attention_heads = int(attention_heads)
        self.pooling_mode = pooling_mode
        self.use_fc_residual = bool(use_fc_residual)
        self.use_rnn_norm = bool(use_rnn_norm)
        self.use_learnable_output_scale = bool(use_learnable_output_scale)
        self.submodel_size = int(submodel_size)
        self.use_submodel_norm = bool(use_submodel_norm)
        self.submodel_norm_type = str(submodel_norm_type or 'auto').lower()

        self.lstm_dropout = (
            float(lstm_dropout) if lstm_dropout is not None else float(dropout))
        self.fc_dropout = (
            float(fc_dropout) if fc_dropout is not None else float(dropout))

        # Input processing
        self.input_norm = nn.LayerNorm(input_size) if self.use_input_norm else None
        self.feature_attention = (
            FeatureAttention(input_size) if use_feature_attention else None)

        # C1: 1D conv front-end (local temporal feature extractor)
        if bool(use_conv1d_front):
            kernel = int(conv1d_kernel)
            if kernel < 1:
                raise ValueError(f"conv1d_kernel must be >= 1, got {kernel}")
            self.conv1d_front = nn.Conv1d(
                input_size, input_size,
                kernel_size=kernel, padding=kernel // 2,
            )
            self.conv1d_act = get_activation(activation)
        else:
            self.conv1d_front = None
            self.conv1d_act = None

        # RNN
        rnn_dropout = self.lstm_dropout if num_layers > 1 else 0
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown rnn_type: '{rnn_type}'")

        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # C4: LayerNorm on RNN output
        self.rnn_norm = nn.LayerNorm(rnn_output_size) if self.use_rnn_norm else None

        # Attention / pooling
        if use_attention:
            self.attention = TemporalAttention(
                rnn_output_size, n_heads=self.attention_heads)
        else:
            self.attention = None

        # FC head (plain sequential or C3 residual blocks)
        fc_sizes = fc_layers if fc_layers else [hidden_size // 2]
        n_out = 2 if output_mode == 'dual' else 1

        if self.use_fc_residual:
            self.fc_blocks = nn.ModuleList()
            prev_size = rnn_output_size
            for fc_size in fc_sizes:
                self.fc_blocks.append(
                    FCLayerBlock(prev_size, fc_size, fc_norm,
                                 activation, self.fc_dropout))
                prev_size = fc_size
            self.fc = None
            self.output_linear = nn.Linear(prev_size, n_out)
        else:
            self.fc_blocks = None
            layers = []
            prev_size = rnn_output_size
            for fc_size in fc_sizes:
                layers.append(nn.Linear(prev_size, fc_size))
                norm = build_norm_layer(fc_norm, fc_size)
                if norm is not None:
                    layers.append(norm)
                layers.append(get_activation(activation))
                layers.append(nn.Dropout(self.fc_dropout))
                prev_size = fc_size
            self.fc = nn.Sequential(*layers)
            self.output_linear = nn.Linear(prev_size, n_out)

        if self.arch_type == 'submodel_fusion' and self.submodel_size > 0:
            from .model_variants import SubmodelFusionHead
            self.fusion_head = SubmodelFusionHead(
                primary_size=prev_size,
                submodel_size=self.submodel_size,
                out_features=n_out,
                use_submodel_norm=self.use_submodel_norm,
                submodel_norm_type=self.submodel_norm_type,
            )
        else:
            self.fusion_head = None

        if self.arch_type == 'dual_head':
            if self.output_mode != 'dual':
                raise ValueError("arch_type='dual_head' requires output_mode='dual'")
            from .model_variants import DualHeadOutput
            self.dual_head = DualHeadOutput(prev_size)
        else:
            self.dual_head = None

        # C5: learnable per-output scale+bias (applied before sigmoid)
        # Shape (n_out,) so each channel (buy/sell) has independent scaling.
        if self.use_learnable_output_scale:
            self.output_scale = nn.Parameter(torch.ones(n_out))
            self.output_bias = nn.Parameter(torch.zeros(n_out))
        else:
            self.output_scale = None
            self.output_bias = None

    def forward(self, x, submodel_inputs=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            submodel_inputs: (batch, submodel_size) optional submodel predictions/embeddings
        Returns:
            (batch, 2) for dual mode, (batch, 1) for single — values in [0,1]
        """
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.feature_attention is not None:
            x = self.feature_attention(x)

        # C1: conv1d — operates on (B, F, T), transpose required
        if self.conv1d_front is not None:
            x = self.conv1d_front(x.transpose(1, 2)).transpose(1, 2)
            x = self.conv1d_act(x)

        rnn_out, _ = self.rnn(x)

        # C4: LayerNorm on RNN sequence output
        if self.rnn_norm is not None:
            rnn_out = self.rnn_norm(rnn_out)

        # Temporal pooling / attention
        if self.attention is not None:
            context = self.attention(rnn_out)
        elif self.pooling_mode == 'mean':
            context = rnn_out.mean(dim=1)
        elif self.pooling_mode == 'max':
            context = rnn_out.max(dim=1).values
        else:
            context = rnn_out[:, -1, :]

        # FC head
        if self.use_fc_residual:
            h = context
            for block in self.fc_blocks:
                h = block(h)
        else:
            h = self.fc(context)

        if self.arch_type == 'submodel_fusion':
            if submodel_inputs is not None:
                if self.fusion_head is None:
                    raise ValueError(
                        "arch_type='submodel_fusion' but fusion_head is None. "
                        "Set submodel_size > 0 when constructing SignalModel."
                    )
                logits = self.fusion_head(h, submodel_inputs)
            else:
                import warnings
                warnings.warn(
                    "arch_type='submodel_fusion' but submodel_inputs is None — "
                    "falling back to output_linear which may be untrained.",
                    stacklevel=2,
                )
                logits = self.output_linear(h)
        elif self.arch_type == 'dual_head':
            logits = self.dual_head(h)
        else:
            logits = self.output_linear(h)

        # C5: learnable output scale+bias before sigmoid
        if self.output_scale is not None:
            logits = logits * self.output_scale + self.output_bias

        return torch.sigmoid(logits)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# WEIGHT INITIALISATION
# =========================================================================

def _init_weights(model, mode, activation='gelu'):
    """Apply weight initialisation to nn.Linear layers."""
    if mode == 'default':
        return
    gain_nonlin = 'leaky_relu' if activation == 'leaky_relu' else 'relu'

    linear_modules = []
    if model.fc is not None:
        linear_modules.extend(
            m for m in model.fc.modules() if isinstance(m, nn.Linear))
    if model.fc_blocks is not None:
        for block in model.fc_blocks:
            linear_modules.append(block.linear)
    linear_modules.append(model.output_linear)
    if getattr(model, 'fusion_head', None) is not None:
        linear_modules.append(model.fusion_head.projection)
        linear_modules.append(model.fusion_head.output_linear)
    if getattr(model, 'dual_head', None) is not None:
        linear_modules.append(model.dual_head.direction)
        linear_modules.append(model.dual_head.strength)

    for module in linear_modules:
        if mode == 'kaiming':
            nn.init.kaiming_normal_(module.weight, nonlinearity=gain_nonlin)
        elif mode == 'xavier':
            nn.init.xavier_normal_(module.weight)
        else:
            raise ValueError(
                f"Unknown WEIGHT_INIT mode: {mode!r}. "
                f"Expected default/kaiming/xavier."
            )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =========================================================================
# FACTORY
# =========================================================================

def build_signal_model(resolved, input_size, device):
    """Build SignalModel from resolved config dict and move to device."""
    model = SignalModel(
        input_size=input_size,
        hidden_size=resolved['hidden_size'],
        num_layers=resolved['num_layers'],
        dropout=resolved['dropout'],
        bidirectional=resolved.get('bidirectional', False),
        use_attention=resolved['use_attention'],
        fc_layers=resolved['fc_layers'],
        activation=resolved['activation'],
        fc_norm=resolved['fc_norm'],
        rnn_type=resolved.get('rnn_type', 'lstm'),
        use_input_norm=resolved.get('use_input_norm', True),
        attention_heads=resolved.get('attention_heads', 1),
        lstm_dropout=resolved.get('lstm_dropout'),
        fc_dropout=resolved.get('fc_dropout'),
        use_feature_attention=resolved.get('use_feature_attention', False),
        output_mode=resolved.get('output_mode', 'dual'),
        arch_type=resolved.get('arch_type', 'legacy'),
        submodel_size=resolved.get('submodel_size', 0),
        use_submodel_norm=resolved.get('use_submodel_norm', True),
        submodel_norm_type=resolved.get('submodel_norm_type', 'auto'),
        # F6
        use_conv1d_front=resolved.get('use_conv1d_front', False),
        conv1d_kernel=resolved.get('conv1d_kernel', 5),
        pooling_mode=resolved.get('pooling_mode', 'last'),
        use_fc_residual=resolved.get('use_fc_residual', False),
        use_rnn_norm=resolved.get('use_rnn_norm', False),
        use_learnable_output_scale=resolved.get(
            'use_learnable_output_scale', False),
    )
    init_mode = resolved.get('weight_init', 'default')
    if init_mode != 'default':
        _init_weights(model, init_mode, activation=resolved['activation'])

    # C6: freeze RNN weights (for sub-model fine-tuning)
    if resolved.get('freeze_rnn', False):
        for param in model.rnn.parameters():
            param.requires_grad = False

    return model.to(device)
