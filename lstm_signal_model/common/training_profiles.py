"""Named training profiles for reproducible signal model recipes."""
from __future__ import annotations

PROFILE_CHOICES = (
    "baseline",
    "aggressive_signal",
    "conservative_signal",
    "custom_sim",
    "rank_focused",
    "conv1d_gru_silu",
    "conv1d_gru_mish",
)

SIGNAL_PROFILES = {
    "baseline": {
        "description": "Default SignalConfig values — no override.",
        "resolved": {},
    },
    "aggressive_signal": {
        "description": "Lower threshold, more signals, higher recall.",
        "resolved": {
            "signal_threshold": 0.0005,
            "learning_rate": 5e-4,
        },
    },
    "conservative_signal": {
        "description": "Higher threshold, fewer signals, higher precision.",
        "resolved": {
            "signal_threshold": 0.002,
            "early_stopping_patience": 35,
        },
    },
    "custom_sim": {
        "description": (
            "Simulation-tuned: canonical oracle, higher diversity weight, "
            "longer training — optimised for sim_capture_log."
        ),
        "resolved": {
            "best_model_metric": "val_signal",
            "loss_type": "bce_diversity",
            "loss_diversity_weight": 0.05,
            "loss_diversity_target": 0.30,
            "early_stopping_patience": 40,
            "lr_warmup_epochs": 5,
            "scheduler_type": "cosine",
        },
    },
    "rank_focused": {
        "description": (
            "Rank-optimised: SWA + larger model + diversity penalty. "
            "Best-guess recipe inspired by LR high-rank runs."
        ),
        "resolved": {
            "best_model_metric": "val_signal",
            "loss_type": "bce_diversity",
            "loss_diversity_weight": 0.10,
            "loss_diversity_target": 0.30,
            "hidden_size": 128,
            "num_layers": 2,
            "use_attention": True,
            "attention_heads": 2,
            "use_swa": True,
            "swa_start_epoch": 20,
            "swa_every": 2,
            "early_stopping_patience": 50,
            "lr_warmup_epochs": 5,
            "scheduler_type": "cosine",
            "aug_jitter": True,
            "aug_jitter_std": 0.01,
        },
    },
    "conv1d_gru_silu": {
        "description": (
            "Conv1D front-end + GRU + SiLU activation recipe "
            "(F6 architecture baseline)."
        ),
        "resolved": {
            "use_conv1d_front": True,
            "rnn_type": "gru",
            "activation": "silu",
            "use_learnable_output_scale": True,
            "use_rnn_norm": True,
            "use_fc_residual": True,
            "bidirectional": True,
            "weight_init": "kaiming",
            "conv1d_kernel": 3,
        },
    },
    "conv1d_gru_mish": {
        "description": (
            "Conv1D front-end + GRU + Mish activation recipe "
            "(F6 architecture variant)."
        ),
        "resolved": {
            "use_conv1d_front": True,
            "rnn_type": "gru",
            "activation": "mish",
            "use_learnable_output_scale": True,
            "use_rnn_norm": True,
            "use_fc_residual": True,
            "bidirectional": True,
            "weight_init": "kaiming",
            "conv1d_kernel": 3,
        },
    },
}


def normalize_profile_name(name):
    """Validate and return canonical profile name."""
    if name not in PROFILE_CHOICES:
        raise ValueError(
            f"Unknown training profile: '{name}'. "
            f"Available: {list(PROFILE_CHOICES)}"
        )
    return name
