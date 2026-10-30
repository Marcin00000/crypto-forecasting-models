"""
ID management utilities for XGBoost pipeline
"""
import os
import re
import json

# Prepared-data ID in a directory/file name: legacy 'XB001' and universal 'E109' / 'B104'.
_DATA_ID_PATTERN = re.compile(r'^[A-Z]{1,2}\d{3,}$')

# <repo>/xgb_model/common/id_utils.py -> repository root
_XGB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(_XGB_ROOT)


def to_repo_relative(path):
    """
    Path as stored in model_config.json: relative to the repository root with '/'
    separators, so a model keeps working when the repository is moved or cloned.
    Stays absolute when no relative form exists (different drive).
    """
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, REPO_ROOT).replace(os.sep, '/')
    except ValueError:
        return abs_path


def resolve_repo_path(path):
    """
    Inverse of to_repo_relative. Absolute values (older models) are returned unchanged;
    relative ones are tried against the repository root, then against xgb_model/ (the
    working directory of early training runs, e.g. '../data_preparation/...').
    """
    if not path or os.path.isabs(path):
        return path
    for base in (REPO_ROOT, _XGB_ROOT):
        candidate = os.path.normpath(os.path.join(base, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def get_next_id(base_dir, prefix, suffix_pattern):
    """
    Get next available ID with given prefix

    Args:
        base_dir: Directory to scan
        prefix: ID prefix (e.g., 'X', 'XB', 'XO')
        suffix_pattern: Pattern to match in directory names

    Returns:
        str: Next ID (e.g., 'X001', 'XB002')
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        return f"{prefix}001"

    existing = [d for d in os.listdir(base_dir)
                if d.startswith(prefix) and suffix_pattern in d]

    if not existing:
        return f"{prefix}001"

    numbers = []
    for d in existing:
        # \d+ (not a fixed 3 digits) keeps counting correctly past X999
        match = re.match(r'\d+', d[len(prefix):])
        if match:
            numbers.append(int(match.group()))

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_num:03d}"


def get_next_model_id(base_dir=None):
    """
    Get next available XGBoost model ID

    Args:
        base_dir: Models directory (default: XGBConfig.MODELS_DIR, absolute, so the
                  ID does not depend on the current working directory)

    Returns:
        str: Model ID (e.g., 'X001', 'X002')
    """
    if base_dir is None:
        from xgb_config import XGBConfig
        base_dir = XGBConfig.MODELS_DIR
    return get_next_id(base_dir, 'X', '_')


def get_next_optuna_id(base_dir=None):
    """
    Get next available Optuna tuning ID

    Args:
        base_dir: Data directory (default: XGBConfig.LOGS_DIR)

    Returns:
        str: Optuna ID (e.g., 'XO001', 'XO002')
    """
    if base_dir is None:
        from xgb_config import XGBConfig
        base_dir = XGBConfig.LOGS_DIR
    return get_next_id(base_dir, 'XO', '_optuna_')


def get_next_test_id(base_dir=None):
    """
    Get next available test ID

    Args:
        base_dir: Data directory (default: XGBConfig.LOGS_DIR)

    Returns:
        str: Test ID (e.g., 'XT001', 'XT002')
    """
    if base_dir is None:
        from xgb_config import XGBConfig
        base_dir = XGBConfig.LOGS_DIR
    return get_next_id(base_dir, 'XT', '_test_results_')


def get_next_data_id(base_dir='lstm_prepared_data'):
    """
    Get next available prepared data ID

    Args:
        base_dir: Prepared data directory

    Returns:
        str: Data ID (e.g., 'XB001', 'XB002')
    """
    return get_next_id(base_dir, 'XB', '_prep_')


def extract_model_id(model_dir):
    """
    Extract model ID from model directory path

    Args:
        model_dir: Path to model directory

    Returns:
        str: Model ID or 'UNKNOWN'
    """
    basename = os.path.basename(model_dir)
    if '_' in basename and basename.startswith('X'):
        parts = basename.split('_')
        if parts[0] and len(parts[0]) >= 4:  # X001 or longer
            return parts[0]
    return 'UNKNOWN'


def extract_data_id_from_path(data_path):
    """
    Extract data ID from path or metadata

    Args:
        data_path: Path to data (directory or file)

    Returns:
        str: Data ID or 'UNKNOWN'
    """
    # If it's a directory with metadata
    if os.path.isdir(data_path):
        metadata_path = os.path.join(data_path, 'metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('data_id', 'UNKNOWN')
            except (json.JSONDecodeError, OSError):
                pass

        # Try to extract from directory name
        basename = os.path.basename(data_path)
        if '_' in basename:
            candidate = basename.split('_')[0]
            if _DATA_ID_PATTERN.match(candidate):
                return candidate

    # If it's a file
    elif os.path.isfile(data_path):
        basename = os.path.basename(data_path)
        if '_' in basename:
            candidate = basename.split('_')[0]
            if _DATA_ID_PATTERN.match(candidate):
                return candidate

    return 'UNKNOWN'


def extract_data_id(metadata_or_path):
    """
    Extract data ID from metadata dict or path

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


def detect_symbol(filename):
    """
    Detect cryptocurrency symbol from filename

    Args:
        filename: File or directory name

    Returns:
        str: Symbol (e.g., 'btc', 'eth') or 'crypto'
    """
    filename_lower = filename.lower()

    # Common crypto symbols
    symbols = {
        'btc': 'btc',
        'bitcoin': 'btc',
        'eth': 'eth',
        'ethereum': 'eth',
        'bnb': 'bnb',
        'binance': 'bnb',
        'sol': 'sol',
        'solana': 'sol',
        'ada': 'ada',
        'cardano': 'ada',
        'xrp': 'xrp',
        'ripple': 'xrp',
        'doge': 'doge',
        'dogecoin': 'doge',
        'dot': 'dot',
        'polkadot': 'dot',
        'matic': 'matic',
        'polygon': 'matic',
        'avax': 'avax',
        'avalanche': 'avax',
    }

    for key, symbol in symbols.items():
        if key in filename_lower:
            return symbol

    return 'crypto'
