"""
Opt-in Weights & Biases experiment tracking for the LSTM Log-Return pipeline.

Ported from `lstm_signal_model/common/wandb_utils.py` (CP-085, Faza 8 of the
signal-model parity plan). Graceful degradation: when wandb is not
installed, or --use_wandb=0, `ExperimentTracker.create()` returns a
`NullTracker` whose methods are silent no-ops. Callers never need to branch
on availability.
"""
from __future__ import annotations

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

PROJECT_NAME = 'lstm-logreturn-model'


class NullTracker:
    """No-op tracker used when W&B is disabled or unavailable."""

    def log_epoch(self, epoch, train_loss, val_loss, metrics=None):
        pass

    def log_test_metrics(self, metrics: dict):
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

    def log_test_metrics(self, metrics: dict):
        """metrics: flat dict — mae/rmse/r2/t_dir_acc/rank_score/sim_* etc."""
        payload = {k: v for k, v in metrics.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
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
