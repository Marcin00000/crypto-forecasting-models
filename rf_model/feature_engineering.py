"""
Feature engineering bridge -- delegates to the universal module at the repository root.

Uses importlib.util to load the root feature_engineering.py directly by file
path, avoiding sys.path collisions where this bridge file (also named
feature_engineering.py) would shadow itself during import resolution.
"""
import os
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))   # rf_model/
_ROOT = os.path.dirname(_HERE)                        # repository root
_UNIVERSAL_FE = os.path.join(_ROOT, 'feature_engineering.py')

_spec = importlib.util.spec_from_file_location(
    'feature_engineering_universal', _UNIVERSAL_FE
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

prepare_features    = _mod.prepare_features
get_feature_columns = _mod.get_feature_columns
get_feature_sets    = _mod.get_feature_sets
select_features     = _mod.select_features
get_binance_columns = _mod.get_binance_columns

__all__ = [
    'prepare_features',
    'get_feature_columns',
    'get_feature_sets',
    'select_features',
    'get_binance_columns',
]
