"""
Optuna tuning results saving for LSTM Log-Return pipeline.

Split from lstm_hyperparameter_tuning.py to keep main file under 500 lines.
"""
import os
import json
from datetime import datetime

import pandas as pd
import joblib


def _build_loss_config_snapshot():
    """Capture current LSTMConfig custom loss flags as a dict for metadata."""
    from lstm_config import LSTMConfig
    keys = [
        'USE_DEAD_ZONE', 'DEAD_ZONE_WEIGHT', 'SIGNAL_THRESHOLD_VALUE',
        'USE_AMPLIFIED_ZONE', 'AMPLIFIED_ZONE_THRESHOLD', 'AMPLIFIED_ZONE_WEIGHT',
        'USE_CORRELATION_PENALTY', 'CORRELATION_PENALTY_WEIGHT',
        'USE_VARIANCE_PENALTY', 'VARIANCE_PENALTY_WEIGHT',
        'USE_DIRECTIONAL_LOSS', 'DIRECTIONAL_LOSS_WEIGHT',
        'USE_MISDIR_PENALTY', 'MISDIR_PENALTY', 'MISDIR_MIN_QUANTILE',
        'USE_SAMPLE_WEIGHTS', 'SAMPLE_WEIGHT_MODE',
        'SAMPLE_WEIGHT_PERCENTILE', 'SAMPLE_WEIGHT_LOW_W',
        'HUBER_DELTA', 'QUANTILE_LOSS_Q',
        # F2.V1: anti-collapse penalties added in losses.py
        'USE_LOG_VARIANCE_PENALTY', 'LOG_VARIANCE_PENALTY_WEIGHT',
        'USE_R2_PENALTY', 'R2_PENALTY_WEIGHT',
        'USE_CCC_PENALTY', 'CCC_PENALTY_WEIGHT',
        # FAZA 3.V2: SignalLoss v2 component flags
        'SIGNAL_USE_R2', 'SIGNAL_R2_WEIGHT',
        'SIGNAL_USE_CCC', 'SIGNAL_CCC_WEIGHT',
        'SIGNAL_USE_LOG_VAR', 'SIGNAL_LOG_VAR_WEIGHT',
        # Anti-collapse std-match and directional threshold (2026-06-08)
        'SIGNAL_USE_STD_MATCH', 'SIGNAL_STD_MATCH_WEIGHT',
        'SIGNAL_USE_DIR_THRESHOLD', 'SIGNAL_DIR_THRESHOLD_WEIGHT',
        'SIGNAL_EMA_PENALTIES', 'SIGNAL_EMA_ALPHA',
        'SIGNAL_LABEL_SMOOTHING', 'SIGNAL_LABEL_SMOOTHING_ALPHA',
        'USE_FEATURE_ATTENTION',
        'AUG_JITTER', 'AUG_JITTER_STD',
        'AUG_TEMPORAL_DROPOUT', 'AUG_TEMPORAL_DROPOUT_RATE',
        'AUG_SUBSEQUENCE', 'AUG_SUBSEQUENCE_OFFSET',
        'LR_WARMUP_EPOCHS',
        # v3 (CP-046+): utility σ proxy, soft threshold, CR ceiling, direction BCE
        'SIGNAL_USE_UTILITY', 'SIGNAL_UTILITY_WEIGHT', 'SIGNAL_SOFT_THRESHOLD',
        'USE_CR_CEILING', 'CR_CEILING', 'CR_CEILING_WEIGHT',
        'SIGNAL_USE_DIRECTION_BCE', 'SIGNAL_DIRECTION_BCE_WEIGHT',
    ]
    return {k: getattr(LSTMConfig, k, None) for k in keys}


# Legacy filenames for ranks 1..3. Ranks 4+ fall through to topNN pattern
# so new result files can be added without breaking loaders that look for
# `best_params.json` / `second_best_params.json` / `third_best_params.json`.
_LEGACY_LABELS = ['best', 'second_best', 'third_best']


def _rank_label(rank_zero_idx):
    """Return file label for rank (0=best, 1=second_best, ...)."""
    if rank_zero_idx < len(_LEGACY_LABELS):
        return _LEGACY_LABELS[rank_zero_idx]
    # rank_zero_idx=3 → top04, 4 → top05, ...
    return f"top{rank_zero_idx + 1:02d}"


def save_params(params_dict, metadata, filepath):
    """Save params with metadata to JSON."""
    output = dict(params_dict)
    output['_metadata'] = metadata
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def save_top_n_params(study, output_dir, metadata_base, direction='minimize',
                      n_best=3, verbose=True):
    """
    Save the top-N trials' params to separate JSON files.

    Used both at the end of tuning (via ``save_tuning_results``) and as an
    Optuna callback after every new best trial (F2.5.3 incremental save).
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
            print(f"  ✓ {os.path.basename(path)} (trial {trial.number}, "
                  f"value={trial.value:.6f})")
    return written


def make_incremental_save_callback(output_dir, metadata_base,
                                   direction='minimize', n_best=3):
    """
    Build an Optuna callback that persists top-N params after every new best.

    F2.5.3 — provides crash-safe checkpoints so an interrupted tuning run
    always leaves a valid ``best_params.json`` on disk. Called per-trial,
    but writes only when the NEW best improves over the previous one.
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
        # Silent save — per-trial prints would spam Optuna's progress bar.
        save_top_n_params(study, output_dir, metadata_base,
                          direction=direction, n_best=n_best, verbose=False)

    return _callback


def save_tuning_results(study, tuning_cfg, optuna_id, data_id, output_dir,
                        direction='minimize'):
    """
    Save top-N params (F2.5.4), trials CSV, study pickle, summary.

    Args:
        study:      optuna.Study
        tuning_cfg: TuningConfig
        optuna_id:  str (e.g. 'LRO001')
        data_id:    str (e.g. 'XB001')
        output_dir: str — output directory
        direction:  'minimize' | 'maximize' — sort order for top trials
    """
    from lstm_config import LSTMConfig
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']

    metadata_base = {
        'optuna_id': optuna_id,
        'data_id': data_id,
        'feature_set': tuning_cfg.FEATURE_SET,
        'scale_target': tuning_cfg.SCALE_TARGET,
        'target_scaler_type': tuning_cfg.TARGET_SCALER_TYPE,
        'direction': direction,
        'score_type': LSTMConfig.OPTUNA_SCORE_TYPE,
        'loss_type': LSTMConfig.LOSS_TYPE,
        'loss_config': _build_loss_config_snapshot(),
        'training_profile': getattr(tuning_cfg, 'TRAINING_PROFILE', 'baseline'),
        'baseline_params': getattr(tuning_cfg, 'BASELINE_PARAMS_PATH', None),
    }

    n_best = getattr(tuning_cfg, 'N_BEST_RESULTS', 3)
    save_top_n_params(study, output_dir, metadata_base,
                      direction=direction, n_best=n_best, verbose=True)

    # Trials CSV
    trials_data = []
    for t in completed:
        row = {'trial': t.number, 'value': t.value}
        row.update(t.params)
        trials_data.append(row)

    pd.DataFrame(trials_data).to_csv(
        os.path.join(output_dir, 'trials.csv'), index=False)
    print("  ✓ trials.csv")

    # Study pickle
    joblib.dump(study, os.path.join(output_dir, 'study.pkl'))
    print("  ✓ study.pkl")

    # Summary
    best = study.best_trial
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("LSTM LOG-RETURN OPTUNA TUNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Optuna ID:  {optuna_id}\n")
        f.write(f"Data ID:    {data_id}\n")
        f.write(f"Completed:  {len(completed)} / {len(study.trials)} trials\n")
        f.write(f"Best trial: {best.number}\n")
        f.write(f"Best value: {best.value:.6f}\n")
        f.write(f"Loss type:  {LSTMConfig.LOSS_TYPE}\n")
        f.write(f"Score:      {metadata_base.get('score_type', 'val_loss')}\n")
        f.write(f"Direction:  {direction}\n")
        f.write(f"N_BEST:     {n_best}\n\n")
        f.write("BEST PARAMETERS:\n")
        for k, v in sorted(best.params.items()):
            f.write(f"  {k}: {v}\n")
    print("  ✓ summary.txt")
