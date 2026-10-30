"""
Small self-contained helpers for signal_model_package.

Deliberately duplicated from lstm_model_package so each package stays copyable on
its own; tests/test_signal_model_package.py checks the two copies stay equivalent.
"""
from __future__ import annotations


class MinRowsInt(int):
    """Integer that also acts as a callable accepting mode ('raw' | 'prepared').

    Older callers read ``min_rows_required`` as a property (int); newer ones call
    ``min_rows_required('prepared')``. One object satisfies both conventions.
    """

    def __new__(cls, val: int, raw_val: int, prep_val: int):
        obj = super().__new__(cls, val)
        obj._raw_val = raw_val
        obj._prep_val = prep_val
        return obj

    def __call__(self, mode: str = 'raw') -> int:
        return self._prep_val if mode in ('prepared', 'prep') else self._raw_val
