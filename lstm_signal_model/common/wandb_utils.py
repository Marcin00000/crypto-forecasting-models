"""
Opt-in Weights & Biases experiment tracking for the Signal Model pipeline.

Graceful degradation: when wandb is not installed, or --use_wandb=0,
ExperimentTracker.create() returns a NullTracker whose methods are silent
no-ops. Callers never need to branch on availability.
"""
from __future__ import annotations

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

PROJECT_NAME = 'lstm-signal-model'


class NullTracker:
    """No-op tracker used when W&B is disabled or unavailable."""

    def log_epoch(self, epoch, train_loss, val_loss, metrics=None):
        pass

    def log_signal_metrics(self, pct_strong, corr_buy, corr_sell,
                           diversity, sim=None):
        pass

    def log_feature_attention_weights(self, weights, feature_names):
        pass

    def log_temporal_attention_weights(self, weights):
        pass

    def log_permutation_importance(self, importances: dict):
        pass

    def log_summary(self, final_metrics: dict):
        pass

    def finish(self):
        pass


class ExperimentTracker:
    """
    Thin W&B wrapper. Construct via ``ExperimentTracker.create()`` — that
    factory returns a ``NullTracker`` automatically when tracking should
    be disabled, so call sites never need an availability check.
    """

    def __init__(self, project: str, config: dict, run_name: str):
        self._run = _wandb.init(
            project=project, config=config, name=run_name, reinit=True)

    @classmethod
    def create(cls, enabled: bool, config: dict, run_name: str,
              project: str = PROJECT_NAME):
        """Factory: returns ExperimentTracker if enabled+available, else NullTracker."""
        if not enabled or not _WANDB_AVAILABLE:
            return NullTracker()
        try:
            return cls(project=project, config=config, run_name=run_name)
        except Exception as exc:  # network / auth failures should not crash training
            print(f"[wandb] init failed ({exc}); falling back to NullTracker")
            return NullTracker()

    def log_epoch(self, epoch, train_loss, val_loss, metrics=None):
        payload = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        if metrics:
            payload.update(metrics)
        _wandb.log(payload, step=epoch)

    def log_signal_metrics(self, pct_strong, corr_buy, corr_sell,
                           diversity, sim=None):
        payload = {
            'pct_strong': pct_strong,
            'corr_buy': corr_buy,
            'corr_sell': corr_sell,
            'diversity': diversity,
        }
        if sim:
            payload.update({f'sim_{k}': v for k, v in sim.items()
                           if not k.startswith('_')})
        _wandb.log(payload)

    def log_feature_attention_weights(self, weights, feature_names):
        """weights: 1D array-like of per-feature mean attention (len == n_features)."""
        try:
            data = [[name, float(w)] for name, w in zip(feature_names, weights)]
            table = _wandb.Table(data=data, columns=['feature', 'weight'])
            _wandb.log({'feature_attention': _wandb.plot.bar(
                table, 'feature', 'weight', title='Feature Attention Weights')})
        except Exception as exc:
            print(f"[wandb] log_feature_attention_weights failed: {exc}")

    def log_temporal_attention_weights(self, weights):
        """weights: 1D array-like of per-timestep mean attention (len == seq_len)."""
        try:
            data = [[t, float(w)] for t, w in enumerate(weights)]
            table = _wandb.Table(data=data, columns=['timestep', 'weight'])
            _wandb.log({'temporal_attention': _wandb.plot.line(
                table, 'timestep', 'weight',
                title='Temporal Attention Weights')})
        except Exception as exc:
            print(f"[wandb] log_temporal_attention_weights failed: {exc}")

    def log_permutation_importance(self, importances: dict):
        """importances: {feature_name: (mean_delta, std)}"""
        try:
            data = [[name, float(mean), float(std)]
                   for name, (mean, std) in importances.items()]
            table = _wandb.Table(
                data=data, columns=['feature', 'mean_delta', 'std'])
            _wandb.log({'permutation_importance': _wandb.plot.bar(
                table, 'feature', 'mean_delta',
                title='Permutation Importance')})
        except Exception as exc:
            print(f"[wandb] log_permutation_importance failed: {exc}")

    def log_summary(self, final_metrics: dict):
        for k, v in final_metrics.items():
            if not k.startswith('_'):
                self._run.summary[k] = v

    def finish(self):
        _wandb.finish()
