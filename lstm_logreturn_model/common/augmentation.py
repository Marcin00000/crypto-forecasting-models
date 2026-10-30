"""
Per-batch sequence augmentation for LSTM training (F7.B).

Applied only during training, after tensors are moved to device.
All transforms are GPU-friendly and no-op when disabled in config.
"""
from __future__ import annotations

import torch


class SequenceAugmenter:
    """Configurable augmentation pipeline for (B, T, F) sequence batches."""

    def __init__(self, *,
                 jitter: bool = False,
                 jitter_std: float = 0.01,
                 temporal_dropout: bool = False,
                 temporal_dropout_rate: float = 0.05,
                 subsequence: bool = False,
                 subsequence_offset: int = 2):
        self.jitter = bool(jitter)
        self.jitter_std = float(jitter_std)
        self.temporal_dropout = bool(temporal_dropout)
        self.temporal_dropout_rate = float(temporal_dropout_rate)
        self.subsequence = bool(subsequence)
        self.subsequence_offset = int(subsequence_offset)

    @property
    def active(self) -> bool:
        return (self.jitter or self.temporal_dropout or self.subsequence)

    @classmethod
    def from_config(cls, config) -> 'SequenceAugmenter':
        """Build from LSTMConfig class or compatible object."""
        return cls(
            jitter=getattr(config, 'AUG_JITTER', False),
            jitter_std=getattr(config, 'AUG_JITTER_STD', 0.01),
            temporal_dropout=getattr(config, 'AUG_TEMPORAL_DROPOUT', False),
            temporal_dropout_rate=getattr(
                config, 'AUG_TEMPORAL_DROPOUT_RATE', 0.05),
            subsequence=getattr(config, 'AUG_SUBSEQUENCE', False),
            subsequence_offset=getattr(config, 'AUG_SUBSEQUENCE_OFFSET', 2),
        )

    @classmethod
    def from_trial_params(cls, params: dict) -> 'SequenceAugmenter':
        """Build from Optuna ``trial.params`` dict (0/1 bool flags)."""
        def _on(key: str) -> bool:
            return bool(int(params.get(key, 0)))

        return cls(
            jitter=_on('aug_jitter'),
            jitter_std=float(params.get('aug_jitter_std', 0.01)),
            temporal_dropout=_on('aug_temporal_dropout'),
            temporal_dropout_rate=float(
                params.get('aug_temporal_dropout_rate', 0.05)),
            subsequence=_on('aug_subsequence'),
            subsequence_offset=int(params.get('aug_subsequence_offset', 2)),
        )

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enabled transforms to input batch x: (B, T, F)."""
        if not self.active:
            return x
        if self.subsequence and self.subsequence_offset > 0:
            x = self._subsequence_shift(x)
        if self.jitter and self.jitter_std > 0:
            x = self._jitter(x)
        if self.temporal_dropout and self.temporal_dropout_rate > 0:
            x = self._temporal_dropout(x)
        return x

    def _jitter(self, x: torch.Tensor) -> torch.Tensor:
        feat_std = x.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
        noise = torch.randn_like(x) * feat_std * self.jitter_std
        return x + noise

    def _temporal_dropout(self, x: torch.Tensor) -> torch.Tensor:
        keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device)
        keep = (keep > self.temporal_dropout_rate).to(x.dtype)
        return x * keep

    def _subsequence_shift(self, x: torch.Tensor) -> torch.Tensor:
        offset = int(torch.randint(
            -self.subsequence_offset,
            self.subsequence_offset + 1,
            (1,),
            device=x.device,
        ).item())
        if offset == 0:
            return x
        if offset > 0:
            pad = torch.zeros_like(x[:, :offset, :])
            return torch.cat([pad, x[:, :-offset, :]], dim=1)
        else:
            n = -offset
            pad = torch.zeros_like(x[:, :n, :])
            return torch.cat([x[:, n:, :], pad], dim=1)
