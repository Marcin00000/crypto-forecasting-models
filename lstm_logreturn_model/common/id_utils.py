"""
ID management utilities for LSTM Log-Return pipeline.

Prefixes:
    LR   — model ID   (LR001, LR002, ...)
    LRO  — Optuna ID  (LRO001, ...)
    LRT  — test ID    (LRT001, ...)
    LRB  — data ID    (LRB001, ...)
"""
import os
import re
import json
from typing import Tuple


# =========================================================================
# GENERIC ID GENERATION
# =========================================================================

def get_next_id(base_dir, prefix, suffix_pattern):
    """
    Get next available ID with given prefix.

    Scans base_dir for directories starting with prefix and containing
    suffix_pattern, extracts the numeric part, and returns prefix + next num.

    Args:
        base_dir:       Directory to scan
        prefix:         ID prefix (e.g. 'LR', 'LRO', 'LRT')
        suffix_pattern: Pattern to match in directory names (e.g. '_')

    Returns:
        str: Next ID (e.g. 'LR001', 'LRO002')
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


# =========================================================================
# TYPED ID GENERATORS
# =========================================================================

def get_next_model_id(base_dir=None):
    """Get next LSTM model ID (LR001, LR002, ...)."""
    if base_dir is None:
        from lstm_config import LSTMConfig
        base_dir = LSTMConfig.MODELS_DIR
    return get_next_id(base_dir, 'LR', '_')


def reserve_next_model_dir(run_name_template: str,
                            base_dir=None) -> Tuple[str, str]:
    """
    Atomowo rezerwuje ID modelu i tworzy katalog wyjściowy.

    Rozwiązuje TOCTOU między get_next_model_id() a os.makedirs:
    używa os.makedirs(exist_ok=False) jako atomowej operacji rezerwacji.
    Przy kolizji (FileExistsError) inkrementuje i próbuje ponownie.

    Args:
        run_name_template: Szablon nazwy katalogu z placeholderem {model_id},
                           np. "{model_id}_eth_model_E104_0207"
        base_dir:          Katalog modeli (domyślnie LSTMConfig.MODELS_DIR)

    Returns:
        Tuple[str, str]: (model_id, output_dir) — zarezerwowane ID i ścieżka
                         do istniejącego katalogu modelu.

    Raises:
        RuntimeError: Jeśli nie uda się zarezerwować ID po 200 próbach.
    """
    if base_dir is None:
        from lstm_config import LSTMConfig
        base_dir = LSTMConfig.MODELS_DIR
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
    raise RuntimeError(
        f"Nie udało się zarezerwować ID modelu po 200 próbach "
        f"(base_dir={base_dir!r})."
    )


def get_next_optuna_id(base_dir=None):
    """Get next Optuna tuning ID (LRO001, LRO002, ...)."""
    if base_dir is None:
        from lstm_config import LSTMConfig
        base_dir = LSTMConfig.DATA_DIR
    return get_next_id(base_dir, 'LRO', '_optuna_')


def get_next_test_id(base_dir=None):
    """Get next test ID (LRT001, LRT002, ...)."""
    if base_dir is None:
        from lstm_config import LSTMConfig
        base_dir = LSTMConfig.DATA_DIR
    return get_next_id(base_dir, 'LRT', '_test_results_')


# =========================================================================
# ID EXTRACTION
# =========================================================================

def extract_model_id(model_dir):
    """
    Extract model ID from model directory path.

    Args:
        model_dir: Path to model directory (e.g. 'lstm_lr_models/LR001_eth_...')

    Returns:
        str: Model ID (e.g. 'LR001') or 'UNKNOWN'
    """
    basename = os.path.basename(model_dir)
    if '_' in basename and basename.startswith('LR'):
        parts = basename.split('_')
        candidate = parts[0]
        # Must be LR + 3 digits (e.g. LR001)
        if len(candidate) >= 5 and candidate[2:5].isdigit():
            return candidate
    return 'UNKNOWN'


def extract_data_id_from_path(data_path):
    """
    Extract data ID from path or metadata.

    Checks metadata.json first (most reliable), then falls back to
    parsing the directory name for LRB/XB prefix patterns.

    Args:
        data_path: Path to data (directory or file)

    Returns:
        str: Data ID or 'UNKNOWN'
    """
    if os.path.isdir(data_path):
        metadata_path = os.path.join(data_path, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('data_id', 'UNKNOWN')
            except (json.JSONDecodeError, IOError):
                pass

        # Fallback: parse directory name (<ID>_prep_..., e.g. E114_prep_eth_...;
        # same pattern as the Signal pipeline so both families agree on data_id)
        basename = os.path.basename(os.path.normpath(data_path))
        m = re.match(r'([A-Z0-9]+)_prep_', basename)
        if m:
            return m.group(1)
        if '_' in basename:
            parts = basename.split('_')
            prefix = parts[0]
            # Match LRB001 or XB001 patterns
            for known_prefix in ('LRB', 'XB'):
                if prefix.startswith(known_prefix):
                    suffix = prefix[len(known_prefix):]
                    if suffix.isdigit():
                        return prefix
    elif os.path.isfile(data_path):
        basename = os.path.basename(data_path)
        if '_' in basename:
            parts = basename.split('_')
            prefix = parts[0]
            for known_prefix in ('LRB', 'XB'):
                if prefix.startswith(known_prefix):
                    suffix = prefix[len(known_prefix):]
                    if suffix.isdigit():
                        return prefix

    return 'UNKNOWN'


def extract_data_id(metadata_or_path):
    """
    Extract data ID from metadata dict or path.

    Args:
        metadata_or_path: Either metadata dict or path string

    Returns:
        str: Data ID or 'UNKNOWN'
    """
    if isinstance(metadata_or_path, dict):
        return metadata_or_path.get('data_id', 'UNKNOWN')
    elif isinstance(metadata_or_path, str):
        return extract_data_id_from_path(metadata_or_path)
    return 'UNKNOWN'


# =========================================================================
# SYMBOL DETECTION
# =========================================================================

_SYMBOL_MAP = {
    'btc': 'btc', 'bitcoin': 'btc',
    'eth': 'eth', 'ethereum': 'eth',
    'bnb': 'bnb', 'binance': 'bnb',
    'sol': 'sol', 'solana': 'sol',
    'ada': 'ada', 'cardano': 'ada',
    'xrp': 'xrp', 'ripple': 'xrp',
    'doge': 'doge', 'dogecoin': 'doge',
    'dot': 'dot', 'polkadot': 'dot',
    'matic': 'matic', 'polygon': 'matic',
    'avax': 'avax', 'avalanche': 'avax',
}


def detect_symbol(filename):
    """
    Detect cryptocurrency symbol from filename.

    Args:
        filename: File or directory name

    Returns:
        str: Symbol (e.g. 'btc', 'eth') or 'crypto'
    """
    filename_lower = filename.lower()
    for key, symbol in _SYMBOL_MAP.items():
        if key in filename_lower:
            return symbol
    return 'crypto'
