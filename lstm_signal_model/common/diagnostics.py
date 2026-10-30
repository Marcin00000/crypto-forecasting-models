"""
Training-time diagnostics for Signal Model.

Monitors per-epoch signal quality to detect collapse toward neutral (0.5).
Analogous to lstm_logreturn_model/common/diagnostics.py but adapted for
[0, 1] signal outputs instead of log-return regression.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any

import numpy as np


def _classify_diversity(diversity: float) -> str:
    """
    Classify signal diversity into a qualitative tag.

    diversity = mean(|pred - 0.5|) — 0.0 means all predictions are 0.5 (flat),
    0.5 means all predictions are at extremes (0 or 1).
    """
    if not np.isfinite(diversity):
        return 'invalid'
    if diversity < 0.02:
        return 'FLAT'
    if diversity < 0.05:
        return 'low'
    if diversity < 0.15:
        return 'moderate'
    return 'healthy'


class SignalEpochDiagnostics:
    """
    Per-epoch signal collapse monitoring.

    Tracks:
      - pred_std / oracle_std per channel (buy, sell) = compression ratio
      - signal_diversity = mean(|pred - 0.5|) — anti-collapse metric
      - status tag: 'FLAT' | 'low' | 'moderate' | 'healthy'

    Enabled by SignalConfig.LOG_SIGNAL_DIAGNOSTICS=True.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.history: List[Dict[str, Any]] = []

    def update(self, epoch: int, y_pred: np.ndarray,
               y_oracle: np.ndarray) -> Dict[str, Any]:
        """
        Compute + log diagnostics for a single epoch.

        Args:
            epoch:    1-based epoch index
            y_pred:   model predictions, shape (n, 2) in [0, 1]
            y_oracle: oracle targets, shape (n, 2) in [0, 1]

        Returns:
            dict with diagnostic metrics for this epoch
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_oracle = np.asarray(y_oracle, dtype=np.float64)

        pred_buy_std = float(np.std(y_pred[:, 0]))
        pred_sell_std = float(np.std(y_pred[:, 1]))
        oracle_buy_std = float(np.std(y_oracle[:, 0]))
        oracle_sell_std = float(np.std(y_oracle[:, 1]))

        cr_buy = pred_buy_std / max(oracle_buy_std, 1e-12)
        cr_sell = pred_sell_std / max(oracle_sell_std, 1e-12)
        cr_mean = (cr_buy + cr_sell) / 2.0

        diversity = float(np.mean(np.abs(y_pred - 0.5)))

        status = _classify_diversity(diversity)

        entry = {
            'epoch': epoch,
            'pred_buy_std': pred_buy_std,
            'pred_sell_std': pred_sell_std,
            'oracle_buy_std': oracle_buy_std,
            'oracle_sell_std': oracle_sell_std,
            'compression_ratio_buy': cr_buy,
            'compression_ratio_sell': cr_sell,
            'compression_ratio_mean': cr_mean,
            'signal_diversity': diversity,
            'status': status,
        }
        self.history.append(entry)

        if self.logger:
            self.logger.info(
                f"  Diagnostics ep{epoch}: diversity={diversity:.4f} "
                f"CR_buy={cr_buy:.3f} CR_sell={cr_sell:.3f} [{status}]"
            )

        return entry

    @property
    def last_status(self) -> str:
        return self.history[-1]['status'] if self.history else 'unknown'

    @property
    def last_diversity(self) -> float:
        return self.history[-1]['signal_diversity'] if self.history else 0.0
