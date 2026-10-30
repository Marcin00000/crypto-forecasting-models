"""
LSTM model definition for Log-Return prediction.

Configurable architecture: hidden size, layers, dropout, bidirectional,
attention, FC layers, activation. GRU placeholder for future exploration.
"""
import torch
import torch.nn as nn

from .model_variants import DualHeadOutput, FCLayerBlock, build_norm_layer, get_activation

class FeatureAttention(nn.Module):
    """Squeeze-and-excitation gate for per-feature reweighting before RNN."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid(),
        )
        # CP-085 (Faza 8): mean per-feature gate weight from the last
        # forward pass, for W&B logging in lstm_test.py.
        self.last_weights = None

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        avg = x.mean(dim=1)
        weights = self.gate(avg)
        self.last_weights = weights.detach().mean(dim=0)
        return x * weights.unsqueeze(1)


# =========================================================================
# ATTENTION MODULE
# =========================================================================

class TemporalAttention(nn.Module):
    """
    Multi-head temporal attention over the RNN output sequence.

    Each head learns its own softmax over the time axis on a low-dimensional
    projection of the RNN output (`head_dim = input_size // n_heads`). The
    per-head context vectors are concatenated and projected back to
    `input_size`. With `n_heads=1` this is functionally equivalent to the
    original single-head attention (modulo the output projection, which
    replaces the implicit identity).

    Args:
        input_size: int — dimensionality of the RNN output per timestep.
                    Must be divisible by `n_heads`.
        n_heads:    int — number of attention heads. Default 1.
    """

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
        # Projection of (B, n_heads * head_dim) → (B, input_size). For
        # n_heads=1, head_dim == input_size, so this collapses to a plain
        # square Linear, which preserves expressivity without bias.
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, input_size)
        # CP-085 (Faza 8): mean per-timestep attention weight (averaged over
        # batch and heads) from the last forward pass, for W&B logging.
        self.last_weights = None

    def forward(self, rnn_output):
        """
        Args:
            rnn_output: (batch, seq_len, input_size)

        Returns:
            context: (batch, input_size) — weighted sum over timesteps,
                     then concatenated across heads and re-projected.
        """
        contexts = []
        head_weights = []
        for i, head in enumerate(self.heads):
            scores = head(rnn_output)                             # (B, T, 1)
            weights = torch.softmax(scores, dim=1)                # (B, T, 1)
            head_proj = self.content_heads[i](rnn_output)         # (B, T, head_dim)
            ctx = (head_proj * weights).sum(dim=1)                # (B, head_dim)
            contexts.append(ctx)
            head_weights.append(weights.detach())
        concat = torch.cat(contexts, dim=-1)                      # (B, H*head_dim)
        self.last_weights = (
            torch.stack(head_weights, dim=0).mean(dim=(0, 1)).squeeze(-1))
        return self.out_proj(concat)                              # (B, input_size)


# =========================================================================
# LSTM MODEL
# =========================================================================

class LSTMLogReturnModel(nn.Module):
    """
    Configurable LSTM for log-return prediction.

    Architecture:
        LSTM → [Attention | LastTimestep] → FC layers → Linear(1)

    Args:
        input_size:    int — number of input features per timestep
        hidden_size:   int — LSTM hidden state size (default 128)
        num_layers:    int — number of stacked LSTM layers (default 2)
        dropout:       float — dropout rate (default 0.2)
        bidirectional: bool — bidirectional LSTM (default False)
        use_attention: bool — attention vs last-timestep (default False)
        fc_layers:     list[int] — FC hidden sizes after LSTM (default [64])
        activation:    str — 'relu' | 'gelu' | 'leaky_relu' (default 'relu')
        fc_norm:       str — 'batch' | 'layer' | 'none' (default 'layer').
                       Normalization inside FC head. 'batch' is legacy and
                       can suppress amplitude gradient on flattened outputs.
        rnn_type:      str — 'lstm' | 'gru' (gru is placeholder)
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.2, bidirectional=False, use_attention=False,
                 fc_layers=None, activation='relu', fc_norm='layer',
                 rnn_type='lstm', use_input_norm=False,
                 attention_heads=1,
                 lstm_dropout=None, fc_dropout=None,
                 arch_type='legacy',
                 use_feature_attention=False,
                 use_rnn_norm=False,
                 pooling_mode='last',
                 use_conv1d_front=False, conv1d_kernel=5,
                 use_fc_residual=False,
                 use_learnable_output_scale=False):
        super().__init__()

        if arch_type not in ('legacy', 'dual_head'):
            raise ValueError(
                f"Unknown arch_type: '{arch_type}'. "
                "Available: ['legacy', 'dual_head']"
            )
        self.arch_type = arch_type
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_input_norm = bool(use_input_norm)
        self.attention_heads = int(attention_heads)

        # F4.M4: split dropout. The two dropout rates govern different parts of
        # the model — `lstm_dropout` is forwarded to nn.LSTM/GRU between stacked
        # layers (only active when num_layers > 1), `fc_dropout` is applied
        # inside the FC head. Both fall back to the unified `dropout` argument
        # when None so legacy callers keep their existing behaviour.
        self.lstm_dropout = (
            float(lstm_dropout) if lstm_dropout is not None else float(dropout))
        self.fc_dropout = (
            float(fc_dropout) if fc_dropout is not None else float(dropout))

        # F1.I2: input LayerNorm over the feature dimension. Applied before
        # the RNN so each (batch, t, feature) entry is normalised independently
        # across features — recommended when raw features have heterogeneous
        # scales. Disabled by default to keep baseline runs unchanged.
        if self.use_input_norm:
            self.input_norm = nn.LayerNorm(input_size)
        else:
            self.input_norm = None

        self.use_feature_attention = bool(use_feature_attention)
        if self.use_feature_attention:
            self.feature_attention = FeatureAttention(input_size)
        else:
            self.feature_attention = None

        self.use_rnn_norm = bool(use_rnn_norm)
        self.pooling_mode = pooling_mode
        if pooling_mode not in ('last', 'mean', 'max'):
            raise ValueError(
                f"Unknown pooling_mode: '{pooling_mode}'. "
                "Available: ['last', 'mean', 'max']"
            )
        self.use_fc_residual = bool(use_fc_residual)
        self.use_learnable_output_scale = bool(use_learnable_output_scale)

        if bool(use_conv1d_front):
            kernel = int(conv1d_kernel)
            if kernel < 1:
                raise ValueError(f"conv1d_kernel must be >= 1, got {kernel}")
            self.conv1d_front = nn.Conv1d(
                input_size, input_size, kernel_size=kernel,
                padding=kernel // 2,
            )
            self.conv1d_activation = get_activation(activation)
        else:
            self.conv1d_front = None
            self.conv1d_activation = None

        # RNN layer
        rnn_dropout = self.lstm_dropout if num_layers > 1 else 0
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == 'gru':
            # F4.M6: GRU implementation. PyTorch's GRU shares interface and
            # dropout semantics with nn.LSTM, so the rest of the model
            # (attention, FC head) does not need to branch on rnn_type.
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unknown rnn_type: '{rnn_type}'")

        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        if self.use_rnn_norm:
            self.rnn_norm = nn.LayerNorm(rnn_output_size)
        else:
            self.rnn_norm = None

        # Attention (optional). F4.M3: multi-head when attention_heads > 1.
        if use_attention:
            self.attention = TemporalAttention(
                rnn_output_size, n_heads=self.attention_heads)
        else:
            self.attention = None

        # FC head (configurable)
        self.fc_norm = fc_norm
        fc_sizes = fc_layers if fc_layers else [rnn_output_size // 2]
        prev_size = rnn_output_size
        if self.use_fc_residual:
            self.fc_blocks = nn.ModuleList([
                FCLayerBlock(
                    prev_size if i == 0 else fc_sizes[i - 1],
                    fc_size, fc_norm, activation, self.fc_dropout,
                )
                for i, fc_size in enumerate(fc_sizes)
            ])
            prev_size = fc_sizes[-1] if fc_sizes else rnn_output_size
            self.fc = None
            if self.arch_type == 'legacy':
                self.fc_out = nn.Linear(prev_size, 1)
                self.output_head = None
            else:
                self.fc_out = None
                self.output_head = DualHeadOutput(prev_size)
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
                layers.append(nn.Dropout(self.fc_dropout))
                prev_size = fc_size
            if self.arch_type == 'legacy':
                layers.append(nn.Linear(prev_size, 1))
                self.fc = nn.Sequential(*layers)
                self.output_head = None
            else:
                self.fc = nn.Sequential(*layers)
                self.output_head = DualHeadOutput(prev_size)

        if self.use_learnable_output_scale:
            self.output_scale = nn.Parameter(torch.ones(1))
            self.output_bias = nn.Parameter(torch.zeros(1))
        else:
            self.output_scale = None
            self.output_bias = None

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            (batch, 1) — predicted log-return
        """
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.feature_attention is not None:
            x = self.feature_attention(x)
        if self.conv1d_front is not None:
            x = self.conv1d_front(x.transpose(1, 2)).transpose(1, 2)
            x = self.conv1d_activation(x)
        rnn_out, _ = self.rnn(x)
        # rnn_out: (batch, seq_len, hidden * dirs)

        if self.rnn_norm is not None:
            rnn_out = self.rnn_norm(rnn_out)

        if self.attention is not None:
            context = self.attention(rnn_out)
        elif self.pooling_mode == 'last':
            context = rnn_out[:, -1, :]
        elif self.pooling_mode == 'mean':
            context = rnn_out.mean(dim=1)
        else:
            context = rnn_out.max(dim=1).values

        if self.use_fc_residual:
            h = context
            for block in self.fc_blocks:
                h = block(h)
            if self.output_head is not None:
                out = self.output_head(h)
            else:
                out = self.fc_out(h)
        else:
            features = self.fc(context)
            if self.output_head is not None:
                out = self.output_head(features)
            else:
                out = features

        if self.output_scale is not None:
            out = out * self.output_scale + self.output_bias
        return out

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =========================================================================
# WEIGHT INITIALISATION (F1.I3)
# =========================================================================

def _init_weights(model, mode, activation='relu'):
    """
    Apply weight initialisation to `nn.Linear` layers.

    PyTorch's defaults already use Kaiming for Linear, but the magnitude is
    `kaiming_uniform_(a=sqrt(5))`, which is conservative and was originally
    chosen for backward compatibility with Lua-Torch. Switching to the
    explicit Kaiming normal (`fan_in`, gain matched to the activation) often
    helps regression heads escape the early flatness phase, especially when
    combined with input LayerNorm (F1.I2).

    LSTM/GRU layers are left untouched — PyTorch's RNN init is already
    well-tuned and changing it risks instability.
    """
    if mode == 'default':
        return
    if activation == 'leaky_relu':
        gain_nonlin = 'leaky_relu'
    elif activation in ('relu', 'gelu'):
        gain_nonlin = 'relu'
    else:
        gain_nonlin = 'relu'

    linear_modules = []
    if model.fc is not None:
        linear_modules.extend(
            m for m in model.fc.modules() if isinstance(m, nn.Linear))
    if model.fc_blocks is not None:
        for block in model.fc_blocks:
            linear_modules.append(block.linear)
    if model.fc_out is not None:
        linear_modules.append(model.fc_out)

    for module in linear_modules:
        if mode == 'kaiming':
            nn.init.kaiming_normal_(
                module.weight, nonlinearity=gain_nonlin)
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
# FACTORY FUNCTION
# =========================================================================

def build_model(config, input_size, device):
    """
    Build LSTM model from LSTMConfig and move to device.

    Args:
        config:     LSTMConfig class (or object with matching attributes)
        input_size: int — number of features per timestep
        device:     torch.device

    Returns:
        LSTMLogReturnModel on the specified device
    """
    model = LSTMLogReturnModel(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        use_attention=config.USE_ATTENTION,
        fc_layers=config.FC_LAYERS,
        activation=config.ACTIVATION,
        fc_norm=getattr(config, 'FC_NORM', 'layer'),
        rnn_type=config.RNN_TYPE,
        use_input_norm=getattr(config, 'USE_INPUT_NORM', False),
        attention_heads=getattr(config, 'ATTENTION_HEADS', 1),
        lstm_dropout=getattr(config, 'LSTM_DROPOUT', None),
        fc_dropout=getattr(config, 'FC_DROPOUT', None),
        arch_type=getattr(config, 'ARCH_TYPE', 'legacy'),
        use_feature_attention=getattr(config, 'USE_FEATURE_ATTENTION', False),
        use_rnn_norm=getattr(config, 'USE_RNN_NORM', False),
        pooling_mode=getattr(config, 'POOLING_MODE', 'last'),
        use_conv1d_front=getattr(config, 'USE_CONV1D_FRONT', False),
        conv1d_kernel=getattr(config, 'CONV1D_KERNEL', 5),
        use_fc_residual=getattr(config, 'USE_FC_RESIDUAL', False),
        use_learnable_output_scale=getattr(
            config, 'USE_LEARNABLE_OUTPUT_SCALE', False),
    )
    init_mode = getattr(config, 'WEIGHT_INIT', 'default')
    if init_mode != 'default':
        if model.output_head is None:
            _init_weights(model, init_mode, activation=config.ACTIVATION)
        seed = int(getattr(config, 'SEED', 42))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if getattr(config, 'FREEZE_RNN', False):
        for param in model.rnn.parameters():
            param.requires_grad = False
    return model.to(device)
