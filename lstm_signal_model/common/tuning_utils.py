"""
Optuna tuning results saving for Signal Model pipeline.

Split from signal_hyperparameter_tuning.py to keep the main file slim.
"""
import os
import json
from datetime import datetime

import pandas as pd
import joblib


def _build_signal_config_snapshot():
    """Capture current SignalConfig signal/loss flags as a dict for metadata."""
    from signal_config import SignalConfig
    keys = [
        'LOSS_TYPE', 'LOSS_SCHEDULE_RAMP_EPOCHS',
        'SIGNAL_THRESHOLD', 'WINDOW_SIGMA', 'MIN_SIGNAL', 'SIM_START_CASH',
        'OUTPUT_MODE', 'USE_FEATURE_ATTENTION',
        'AUG_JITTER', 'AUG_JITTER_STD',
        'AUG_TEMPORAL_DROPOUT', 'AUG_TEMPORAL_DROPOUT_RATE',
        'AUG_SUBSEQUENCE', 'AUG_SUBSEQUENCE_OFFSET',
        'BEST_MODEL_METRIC', 'BEST_MODEL_DIVERSITY_WEIGHT',
    ]
    return {k: getattr(SignalConfig, k, None) for k in keys}


# Legacy filenames for ranks 1..3. Ranks 4+ fall through to topNN pattern
# so new result files can be added without breaking loaders that look for
# `best_params.json` / `second_best_params.json` / `third_best_params.json`.
_LEGACY_LABELS = ['best', 'second_best', 'third_best']


def _rank_label(rank_zero_idx):
    """Return file label for rank (0=best, 1=second_best, ...)."""
    if rank_zero_idx < len(_LEGACY_LABELS):
        return _LEGACY_LABELS[rank_zero_idx]
    return f"top{rank_zero_idx + 1:02d}"


def save_params(params_dict, metadata, filepath):
    """Save params with metadata to JSON."""
    output = dict(params_dict)
    output['_metadata'] = metadata
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def save_top_n_params(study, output_dir, metadata_base, direction='minimize',
                      n_best=3, verbose=True):
    """
    Save the top-N trials' params to separate JSON files.

    Used both at the end of tuning (via ``save_tuning_results``) and as an
    Optuna callback after every new best trial (incremental save).
    Returns list of written file paths.
    """
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    if not completed:
        return []
    reverse = (direction == 'maximize')
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=reverse)

    written = []
    for i in range(min(n_best, len(sorted_trials))):
        trial = sorted_trials[i]
        meta = dict(metadata_base)
        meta['best_value'] = trial.value
        meta['trial_number'] = trial.number
        meta['rank'] = i + 1
        meta['n_completed'] = len(completed)
        meta['n_total'] = len(study.trials)
        meta['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(output_dir, f'{_rank_label(i)}_params.json')
        save_params(trial.params, meta, path)
        written.append(path)
        if verbose:
            print(f"  {os.path.basename(path)} (trial {trial.number}, "
                  f"value={trial.value:.6f})")
    return written


def make_incremental_save_callback(output_dir, metadata_base,
                                   direction='minimize', n_best=3):
    """
    Build an Optuna callback that persists top-N params after every new best.

    Provides crash-safe checkpoints so an interrupted tuning run always
    leaves a valid ``best_params.json`` on disk.
    """
    last_best_value = {'v': None}

    def _callback(study, trial):
        completed = [t for t in study.trials if t.state.name == 'COMPLETE']
        if not completed:
            return
        best_val = study.best_value
        prev = last_best_value['v']
        if prev is None:
            changed = True
        elif direction == 'maximize':
            changed = best_val > prev
        else:
            changed = best_val < prev
        if not changed:
            return
        last_best_value['v'] = best_val
        save_top_n_params(study, output_dir, metadata_base,
                          direction=direction, n_best=n_best, verbose=False)

    return _callback


def save_tuning_results(study, tuning_cfg, optuna_id, data_id, output_dir,
                        direction='minimize'):
    """
    Save top-N params, trials CSV, study pickle, summary.

    Args:
        study:      optuna.Study
        tuning_cfg: SignalTuningConfig
        optuna_id:  str (e.g. 'SIGO001')
        data_id:    str (e.g. 'E106')
        output_dir: str — output directory
        direction:  'minimize' | 'maximize' — sort order for top trials
    """
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']

    metadata_base = {
        'optuna_id': optuna_id,
        'data_id': data_id,
        'feature_set': tuning_cfg.FEATURE_SET,
        'direction': direction,
        'score_type': tuning_cfg.OPTUNA_SCORE_TYPE,
        'loss_config': _build_signal_config_snapshot(),
        'training_profile': getattr(tuning_cfg, 'TRAINING_PROFILE', 'baseline'),
        'baseline_params': getattr(tuning_cfg, 'BASELINE_PARAMS_PATH', None),
    }

    n_best = getattr(tuning_cfg, 'N_BEST_RESULTS', 3)
    save_top_n_params(study, output_dir, metadata_base,
                      direction=direction, n_best=n_best, verbose=True)

    trials_data = []
    for t in completed:
        row = {'trial': t.number, 'value': t.value}
        row.update(t.params)
        trials_data.append(row)

    pd.DataFrame(trials_data).to_csv(
        os.path.join(output_dir, 'trials.csv'), index=False)
    print("  trials.csv")

    joblib.dump(study, os.path.join(output_dir, 'study.pkl'))
    print("  study.pkl")

    best = study.best_trial
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SIGNAL MODEL OPTUNA TUNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Optuna ID:  {optuna_id}\n")
        f.write(f"Data ID:    {data_id}\n")
        f.write(f"Completed:  {len(completed)} / {len(study.trials)} trials\n")
        f.write(f"Best trial: {best.number}\n")
        f.write(f"Best value: {best.value:.6f}\n")
        f.write(f"Score:      {metadata_base.get('score_type', 'val_loss')}\n")
        f.write(f"Direction:  {direction}\n")
        f.write(f"N_BEST:     {n_best}\n\n")
        f.write("BEST PARAMETERS:\n")
        for k, v in sorted(best.params.items()):
            f.write(f"  {k}: {v}\n")
    print("  summary.txt")
