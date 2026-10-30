"""
Narzędzia zarządzania identyfikatorami i katalogami testów dla modułu Foundation Models.

Wzorowane na standardzie id_utils.py z modułów LR i Signal:
- Prefiks testowy: FMT### (Foundation Model Test)
- Format katalogu: FMT###_test_results_<MODEL>_<DATASET>_<DATE>
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Tuple, Optional


def get_next_fmt_id(base_dir: Optional[str] = None) -> str:
    """
    Zwraca kolejny dostępny identyfikator testu FMT###.

    Skanuje base_dir (domyślnie FoundationConfig.DATA_DIR) w poszukiwaniu folderów z prefiksem FMT.
    """
    if base_dir is None:
        from foundation_config import FoundationConfig
        base_dir = FoundationConfig.DATA_DIR

    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith("FMT") and os.path.isdir(os.path.join(base_dir, d))
    ]

    numbers = []
    for d in existing:
        m = re.search(r"FMT(\d+)", d)
        if m:
            numbers.append(int(m.group(1)))

    next_num = max(numbers) + 1 if numbers else 1
    return f"FMT{next_num:03d}"


def create_test_output_dir(
    base_dir: Optional[str] = None,
    model_name: str = "FoundationModel",
    data_id: str = "E109",
    run_label: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Tworzy ustandaryzowany katalog na wyniki testu w stylu LR/Signal.

    Format: {base_dir}/FMT{num:03d}_test_results_{label}_{data_id}_{MMDD}
    Returns:
        (test_id, output_dir_path)
    """
    if base_dir is None:
        from foundation_config import FoundationConfig
        base_dir = FoundationConfig.DATA_DIR
    os.makedirs(base_dir, exist_ok=True)
    clean_model = run_label or os.path.basename(model_name.rstrip("/\\"))
    # Oczyszczenie nazwy ze znaków niedozwolonych w ścieżkach
    clean_model = re.sub(r"[^\w\-.]", "_", clean_model)
    date_str = datetime.now().strftime("%d%m")

    for _ in range(200):
        test_id = get_next_fmt_id(base_dir)
        dir_name = f"{test_id}_test_results_{clean_model}_{data_id}_{date_str}"
        target_dir = os.path.join(base_dir, dir_name)
        try:
            os.makedirs(target_dir, exist_ok=False)
            return test_id, target_dir
        except FileExistsError:
            continue

    # Fallback jeśli po 200 próbach wystąpił konflikt
    time_str = datetime.now().strftime("%H%M%S")
    dir_name = f"{test_id}_{time_str}_test_results_{clean_model}_{data_id}_{date_str}"
    target_dir = os.path.join(base_dir, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    return test_id, target_dir
