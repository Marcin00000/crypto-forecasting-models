"""
ID management utilities for LSTM price prediction pipeline.

Handles sequential ID allocation (L### for models, LT### for tests), regex-based
extraction of identifiers, and data ID resolution.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Absolute anchors: ID allocation and dataset lookup must not depend on the caller's
# working directory (batch runner, tests and scripts start from different folders).
LSTM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LSTM_ROOT.parent
DEFAULT_MODELS_DIR = LSTM_ROOT / 'lstm_models'
DEFAULT_DATA_DIR = LSTM_ROOT / 'lstm_data'
PREPARED_DATA_DIR = REPO_ROOT / 'data_preparation' / 'prepared_data'
# Legacy per-family copy of prepared datasets; searched last (canonical copies win).
FAMILY_PREPARED_DIR = LSTM_ROOT / 'lstm_prepared_data'


def get_next_id(base_dir: str | Path, prefix: str, suffix_pattern: str) -> str:
    """Get next available sequential ID with given prefix."""
    if not os.path.exists(base_dir):
        return f"{prefix}001"

    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith(prefix) and suffix_pattern in d
    ]
    if not existing:
        return f"{prefix}001"

    numbers = []
    p_len = len(prefix)
    for d in existing:
        try:
            m = re.match(rf"^{re.escape(prefix)}(\d{{3,4}})", d)
            if m:
                numbers.append(int(m.group(1)))
            else:
                num = int(d[p_len:p_len + 3])
                numbers.append(num)
        except Exception:
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}{next_num:03d}"


def get_next_model_id(base_dir: str | Path = DEFAULT_MODELS_DIR) -> str:
    """Get next available model ID (e.g. L001, L002)."""
    return get_next_id(base_dir, 'L', '_')


def get_next_optuna_id(base_dir: str | Path = DEFAULT_MODELS_DIR) -> str:
    """Get next available Optuna tuning ID (e.g. O001)."""
    return get_next_id(base_dir, 'O', '_optuna_tuning_')


def get_next_test_id(base_dir: str | Path = DEFAULT_DATA_DIR) -> str:
    """Get next available test ID using the standard LT prefix (e.g. LT001)."""
    return get_next_id(base_dir, 'LT', '_test_results_')


def extract_lt_num(test_dir_or_id: str) -> int:
    """Extract numeric suffix from test ID (e.g., LT005 -> 5, T005 -> 5)."""
    basename = os.path.basename(str(test_dir_or_id))
    m = re.search(r"^(?:LT|T)(\d{3,4})", basename)
    if m:
        return int(m.group(1))
    return -1


def extract_test_id(test_dir: str) -> str:
    """Extract test ID (LT### or legacy T###) from directory path or string."""
    basename = os.path.basename(str(test_dir))
    m = re.search(r"^(LT\d{3,4}|T\d{3,4})", basename)
    if m:
        return m.group(1)
    return 'UNKNOWN'


def extract_model_id(model_dir_or_test_dir: str) -> str:
    """
    Extract model ID (L###) from a model directory or a test results directory.
    Priority:
      1. If test directory like LT001_test_results_L005_..., extract L005.
      2. If model directory like L005_lstm_..., extract L005.
    """
    basename = os.path.basename(str(model_dir_or_test_dir))
    if '_test_results_' in basename:
        after = basename.split('_test_results_', 1)[1]
        m = re.match(r"^(L\d{3,4})", after)
        if m:
            return m.group(1)
    m = re.match(r"^(L\d{3,4})", basename)
    if m:
        return m.group(1)
    return 'UNKNOWN'


_DATA_ID_COMPONENT = re.compile(r"^([A-Z]\d{3,4})(?:_|$)")


def extract_data_id_from_path(data_path: str) -> str:
    """
    Extract data ID from a prepared data directory, CSV file or metadata.json path.

    The dataset directory name (``E114_prep_...``) is the authoritative source, so path
    components are inspected from the innermost outwards before the loose regex fallback.
    """
    norm_str = str(data_path).replace("\\", "/")
    for part in reversed([p for p in norm_str.split("/") if p]):
        m = _DATA_ID_COMPONENT.match(part)
        if m:
            return m.group(1).upper()
    m = re.search(r"(?:^|[/_])([A-Z]\d{3})(?:$|[/_])", norm_str) or re.search(r"([A-Z]\d{3})", norm_str)
    if m:
        return m.group(1).upper()
    return 'UNKNOWN'


def extract_data_id(metadata_or_path) -> str:
    """Extract data ID from metadata dictionary or file path."""
    if isinstance(metadata_or_path, dict):
        return metadata_or_path.get('data_id', 'UNKNOWN')
    elif isinstance(metadata_or_path, (str, Path)):
        return extract_data_id_from_path(str(metadata_or_path))
    return 'UNKNOWN'


def is_known_data_id(data_id) -> bool:
    """False for empty values and the 'UNKNOWN' sentinel returned by the extractors."""
    return data_id not in (None, '', 'UNKNOWN')


def resolve_data_id(path, metadata: dict | None = None, config_data_id: str | None = None) -> str:
    """
    Resolve the dataset ID with the priority: directory name -> metadata -> config.

    The directory prefix wins over ``metadata['data_id']`` because some archived datasets
    carry a stale ID in their metadata (E101 stores 'E001').
    """
    for candidate in (
        extract_data_id(str(path)) if path else None,
        extract_data_id(metadata) if metadata else None,
        config_data_id,
    ):
        if is_known_data_id(candidate):
            return str(candidate)
    return 'UNKNOWN'


def to_repo_relative(path: str | Path) -> str:
    """
    Return ``path`` relative to the repository root with '/' separators.

    Paths outside the repository (e.g. temporary datasets) stay absolute, because a
    '..' chain would not be portable between machines either.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a stored path: absolute paths are kept, relative ones are anchored at the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_dataset_path(data_id_or_path: str, search_roots=None) -> Path:
    """
    Resolve a dataset reference to an existing directory.

    Order: existing directory (as given) -> repo-relative directory -> directory in the
    canonical ``data_preparation/prepared_data`` whose name equals or starts with the
    reference (``E109`` matches ``E109_prep_eth_universal_max_1m``) -> legacy family copy.
    """
    raw = Path(data_id_or_path)
    if raw.is_dir():
        return raw.resolve()
    anchored = REPO_ROOT / raw
    if not raw.is_absolute() and anchored.is_dir():
        return anchored.resolve()

    prefix = str(data_id_or_path)
    if '_' not in prefix:
        prefix += '_'  # 'E1' must not match 'E109_...'
    roots = [PREPARED_DATA_DIR, FAMILY_PREPARED_DIR] if search_roots is None else list(search_roots)
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        candidates = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix))
        if candidates:
            return candidates[0].resolve()

    raise FileNotFoundError(f"Could not resolve dataset path for: {data_id_or_path}")
