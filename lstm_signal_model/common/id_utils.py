"""
ID management utilities for LSTM Signal pipeline.

Independent implementation for lstm_signal_model.
Prefixes:
    SIG   — model ID   (SIG001, SIG002, ...)
    SIGO  — Optuna ID  (SIGO001, ...)
    SIGT  — test ID    (SIGT001, ...)
    SIGB  — data ID    (SIGB001, ...)
"""
import os
import re
from typing import Tuple


def get_next_id(base_dir, prefix, suffix_pattern='_'):
    """
    Get next available ID with given prefix.

    Scans base_dir for directories starting with prefix and containing
    suffix_pattern, extracts numeric part, and returns prefix + next num.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return f"{prefix}001"

    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith(prefix) and suffix_pattern in d
    ]

    if not existing:
        return f"{prefix}001"

    numbers = []
    for d in existing:
        m = re.search(r'\d+', d[len(prefix):])
        if m:
            numbers.append(int(m.group()))

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_num:03d}"


def get_next_model_id(base_dir=None):
    """Get next Signal model ID (SIG001, SIG002, ...)."""
    if base_dir is None:
        from signal_config import SignalConfig
        base_dir = SignalConfig.MODELS_DIR
    return get_next_id(base_dir, 'SIG', '_')


def get_next_test_id(base_dir=None):
    """Get next Signal test ID (SIGT001, SIGT002, ...)."""
    if base_dir is None:
        from signal_config import SignalConfig
        base_dir = SignalConfig.DATA_DIR
    return get_next_id(base_dir, 'SIGT', '_')


def get_next_optuna_id(base_dir=None):
    """Get next Signal Optuna ID (SIGO001, SIGO002, ...)."""
    if base_dir is None:
        from signal_config import SignalConfig
        base_dir = SignalConfig.DATA_DIR
    return get_next_id(base_dir, 'SIGO', '_')


def reserve_next_model_dir(run_name_template: str, base_dir=None) -> Tuple[str, str]:
    """
    Atomically reserve model ID and create output directory (TOCTOU protection).

    Args:
        run_name_template: Template string with {model_id} placeholder
        base_dir: Models directory
    """
    if base_dir is None:
        from signal_config import SignalConfig
        base_dir = SignalConfig.MODELS_DIR
    for _ in range(200):
        model_id = get_next_model_id(base_dir)
        candidate_dir = os.path.join(
            base_dir, run_name_template.format(model_id=model_id)
        )
        try:
            os.makedirs(candidate_dir, exist_ok=False)
            return model_id, candidate_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not reserve model ID in {base_dir} after 200 attempts.")


def reserve_next_test_dir(run_name_template: str, base_dir=None) -> Tuple[str, str]:
    """Atomically reserve test ID and create directory."""
    if base_dir is None:
        from signal_config import SignalConfig
        base_dir = SignalConfig.DATA_DIR
    for _ in range(200):
        test_id = get_next_test_id(base_dir)
        candidate_dir = os.path.join(
            base_dir, run_name_template.format(test_id=test_id)
        )
        try:
            os.makedirs(candidate_dir, exist_ok=False)
            return test_id, candidate_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not reserve test ID in {base_dir} after 200 attempts.")


def reserve_next_optuna_dir(run_name_template: str, base_dir=None) -> Tuple[str, str]:
    """Atomically reserve Optuna ID and create directory."""
    if base_dir is None:
        from signal_config import SignalConfig
        base_dir = SignalConfig.DATA_DIR
    for _ in range(200):
        optuna_id = get_next_optuna_id(base_dir)
        candidate_dir = os.path.join(
            base_dir, run_name_template.format(optuna_id=optuna_id)
        )
        try:
            os.makedirs(candidate_dir, exist_ok=False)
            return optuna_id, candidate_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not reserve optuna ID in {base_dir} after 200 attempts.")


def extract_data_id(data_path: str) -> str:
    """Extract data batch ID (e.g., E104, B104) from data directory path."""
    name = os.path.basename(os.path.normpath(data_path))
    m = re.match(r'([A-Z0-9]+)_prep_', name)
    if m:
        return m.group(1)
    return name.split('_')[0]


def detect_symbol(path_or_name: str) -> str:
    """Detect symbol (ETH, BTC, etc.) from path or folder name."""
    name = os.path.basename(os.path.normpath(path_or_name)).lower()
    if 'btc' in name:
        return 'btc'
    if 'eth' in name:
        return 'eth'
    return 'unknown'


def extract_signal_model_id(model_dir: str) -> str:
    """Extract SIG### model ID from directory name."""
    name = os.path.basename(os.path.normpath(model_dir))
    m = re.search(r'SIG\d+', name)
    return m.group(0) if m else name
