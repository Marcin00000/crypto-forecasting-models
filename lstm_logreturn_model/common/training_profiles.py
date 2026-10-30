"""Named training profiles for reproducible rank/sim LSTM recipes."""
from __future__ import annotations

PROFILE_CHOICES = ("baseline", "rank_focused", "sim_focused", "custom_sim")


TRAINING_PROFILES = {
    "baseline": {
        "description": "Do not override LSTMConfig or CLI values.",
        "resolved": {},
        "loss": {},
    },
    "rank_focused": {
        "description": "LR594-style recipe: maximize rank_score tier.",
        "resolved": {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.22201981500408824,
            "bidirectional": True,
            "use_attention": False,
            "attention_heads": 1,
            "fc_layers": [64],
            "activation": "relu",
            "fc_norm": "none",
            "use_input_norm": False,
            "lstm_dropout": 0.1,
            "fc_dropout": 0.4,
            "lr": 0.00021700190985768128,
            "weight_decay": 6.881340829229464e-06,
            "scheduler_type": "cosine",
            "loss_type": "signal",
        },
        "loss": {
            "signal_use_r2": False,
            "signal_use_ccc": False,
            "signal_use_log_var": False,
            "signal_use_std_match": True,
            "signal_std_match_weight": 0.05,
            "signal_use_dir_threshold": False,
        },
    },
    "sim_focused": {
        "description": "LR342-style recipe: maximize sim_$ with rank guardrail.",
        "resolved": {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.051,
            "bidirectional": True,
            "use_attention": False,
            "attention_heads": 1,
            "fc_layers": [32],
            "activation": "relu",
            "fc_norm": "batch",
            "use_input_norm": True,
            "lstm_dropout": 0.1,
            "fc_dropout": 0.4,
            "lr": 0.000619,
            "weight_decay": 1.244e-05,
            "scheduler_type": "cosine",
            "best_model_metric": "val",
            "loss_type": "signal",
        },
        "loss": {
            "signal_use_r2": False,
            "signal_use_ccc": False,
            "signal_use_log_var": False,
            "signal_use_std_match": True,
            "signal_std_match_weight": 0.05,
            "signal_use_dir_threshold": False,
        },
    },
    "custom_sim": {
        "description": "LR843 DNA: sim_focused arch + custom corr/var + amplified 3.5 + warmup.",
        "resolved": {
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.051,
            "bidirectional": True,
            "use_attention": False,
            "attention_heads": 1,
            "fc_layers": [32],
            "activation": "relu",
            "fc_norm": "batch",
            "use_input_norm": True,
            "lstm_dropout": 0.1,
            "fc_dropout": 0.4,
            "lr": 0.000619,
            "weight_decay": 1.244e-05,
            "scheduler_type": "cosine",
            "best_model_metric": "val",
            "loss_type": "custom",
            "lr_warmup_epochs": 5,
        },
        "loss": {
            "use_correlation_penalty": True,
            "correlation_penalty_weight": 0.1,
            "use_variance_penalty": True,
            "variance_penalty_weight": 0.1,
            "use_amplified_zone": True,
            "amplified_zone_weight": 3.5,
            "signal_use_std_match": True,
            "signal_std_match_weight": 0.05,
        },
    },
}


def normalize_profile_name(name: str | None) -> str:
    """Normalize user-facing hyphen spelling to config-friendly underscores."""
    if not name:
        return "baseline"
    normalized = name.replace("-", "_")
    if normalized not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training_profile: {name!r}. "
            f"Available: {', '.join(PROFILE_CHOICES)}"
        )
    return normalized


def apply_training_profile(resolved: dict, profile_name: str,
                           explicit_keys: set[str]) -> tuple[dict, dict]:
    """Apply profile defaults without overriding explicitly supplied kwargs."""
    profile_name = normalize_profile_name(profile_name)
    profile = TRAINING_PROFILES[profile_name]
    updated = dict(resolved)
    loss_overrides = {}

    for key, value in profile["resolved"].items():
        if key not in explicit_keys:
            updated[key] = value

    for key, value in profile["loss"].items():
        if key not in explicit_keys:
            loss_overrides[key] = value

    updated["training_profile"] = profile_name
    return updated, loss_overrides
