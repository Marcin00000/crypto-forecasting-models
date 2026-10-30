"""
Common utilities for LSTM Log-Return model
"""

from .id_utils import (
    get_next_model_id,
    reserve_next_model_dir,
    get_next_optuna_id,
    get_next_test_id,
    extract_model_id,
    extract_data_id,
    extract_data_id_from_path,
    detect_symbol,
)

from .target_utils import (
    TargetScaler,
    create_log_return_target,
    winsorize_targets,
)

from .metrics import (
    calculate_metrics,
    print_metrics_summary,
    interpret_metrics,
)

from .torch_utils import (
    get_device,
    set_seed,
    CryptoDataset,
    get_scheduler,
)

__all__ = [
    # ID
    'get_next_model_id', 'reserve_next_model_dir',
    'get_next_optuna_id', 'get_next_test_id',
    'extract_model_id', 'extract_data_id', 'extract_data_id_from_path',
    'detect_symbol',
    # Target
    'TargetScaler', 'create_log_return_target', 'winsorize_targets',
    # Metrics
    'calculate_metrics', 'print_metrics_summary', 'interpret_metrics',
    # Torch
    'get_device', 'set_seed', 'CryptoDataset', 'get_scheduler',
]
