"""
Training-time diagnostics for LSTM Log-Return pipeline.

Primary focus (F2.0): per-epoch prediction collapse monitoring.
Tracks prediction std vs ground-truth std in raw (log-return) space and
surfaces a compression ratio so collapse-toward-zero is visible in the
training log without waiting for the post-training summary.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any

import numpy as np


EARLY_BEST_THRESHOLD = 15
EARLY_BEST_WARNING = (
    f"best_epoch < {EARLY_BEST_THRESHOLD} — possible lucky early checkpoint"
)


def _classify(ratio: float) -> str:
    """
    Classify compression ratio pred_std/true_std into a qualitative tag.

    Thresholds are heuristic, informed by LR015–LR034 where flattened
    models clustered at <0.01 and borderline models at 0.02–0.10.
    """
    if not np.isfinite(ratio):
        return 'invalid'
    if ratio < 0.05:
        return 'FLAT'
    if ratio < 0.15:
        return 'low'
    if ratio < 0.50:
        return 'moderate'
    return 'healthy'


class EpochDiagnostics:
    """
    Record per-epoch prediction variance diagnostics.

    Metrics are computed in raw (un-scaled) log-return space so magnitudes
    are comparable across runs with different target scalers / winsorization.

    Stored fields per epoch (see `update()`):
      - epoch (1-based)
      - val_pred_std, val_true_std
      - compression_ratio = val_pred_std / val_true_std
      - val_pred_mean, val_true_mean
      - status tag ('FLAT' | 'low' | 'moderate' | 'healthy' | 'invalid')
    """

    def __init__(self, target_scaler=None, logger=None):
        self.target_scaler = target_scaler
        self.logger = logger
        self.history: List[Dict[str, Any]] = []

    def _inverse(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return np.asarray(y_scaled, dtype=np.float64)
        return np.asarray(
            self.target_scaler.inverse_transform(y_scaled), dtype=np.float64
        )

    def update(self,
               epoch: int,
               y_val_pred_scaled: np.ndarray,
               y_val_true_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Compute + log diagnostics for a single epoch.

        Args:
            epoch:              1-based epoch index.
            y_val_pred_scaled:  model predictions on val set (scaled space).
            y_val_true_scaled:  ground-truth y on val set (scaled space).

        Returns:
            dict of diagnostics recorded for this epoch.
        """
        y_pred = self._inverse(np.asarray(y_val_pred_scaled).reshape(-1))
        y_true = self._inverse(np.asarray(y_val_true_scaled).reshape(-1))

        pred_std = float(np.std(y_pred)) if y_pred.size else 0.0
        true_std = float(np.std(y_true)) if y_true.size else 0.0
        pred_mean = float(np.mean(y_pred)) if y_pred.size else 0.0
        true_mean = float(np.mean(y_true)) if y_true.size else 0.0

        ratio = pred_std / true_std if true_std > 0 else float('nan')
        status = _classify(ratio)

        record = {
            'epoch': int(epoch),
            'val_pred_std': pred_std,
            'val_true_std': true_std,
            'val_pred_mean': pred_mean,
            'val_true_mean': true_mean,
            'compression_ratio': ratio,
            'status': status,
        }
        self.history.append(record)

        if self.logger is not None:
            self.logger.info(
                f"  diag   | pred_std={pred_std:.3e} "
                f"true_std={true_std:.3e} "
                f"ratio={ratio:.4f} [{status}]"
            )

        return record

    def summary(self) -> Dict[str, Any]:
        """Return diagnostics for best-epoch-candidate interpretation."""
        if not self.history:
            return {}
        last = self.history[-1]
        ratios = [
            h['compression_ratio']
            for h in self.history
            if np.isfinite(h['compression_ratio'])
        ]
        if ratios:
            best_ratio = max(ratios)
            best_idx = int(np.argmax(ratios))
            best_rec = self.history[best_idx]
        else:
            best_ratio = float('nan')
            best_rec = last
        return {
            'last_epoch': last,
            'best_ratio_epoch': best_rec,
            'best_ratio': best_ratio,
            'n_flat_epochs': sum(1 for h in self.history if h['status'] == 'FLAT'),
            'n_epochs': len(self.history),
        }

    def log_summary(self, logger=None, best_epoch=None) -> None:
        """Emit a compact end-of-training diagnostic summary."""
        target = logger or self.logger
        if target is None:
            return
        s = self.summary()
        if not s:
            return
        if best_epoch is not None and int(best_epoch) < EARLY_BEST_THRESHOLD:
            target.warning(f"  {EARLY_BEST_WARNING} (best_epoch={best_epoch})")
        target.info("\nDIAGNOSTICS SUMMARY (F2.0)")
        target.info(
            f"  epochs: {s['n_epochs']}, FLAT epochs: {s['n_flat_epochs']}"
        )
        best = s['best_ratio_epoch']
        target.info(
            f"  best  ratio: {s['best_ratio']:.4f} @ epoch {best['epoch']} "
            f"(pred_std={best['val_pred_std']:.3e})"
        )
        last = s['last_epoch']
        target.info(
            f"  final ratio: {last['compression_ratio']:.4f} @ epoch "
            f"{last['epoch']} [{last['status']}] "
            f"(pred_std={last['val_pred_std']:.3e})"
        )
