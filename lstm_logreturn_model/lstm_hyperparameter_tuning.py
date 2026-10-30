"""
LSTM Log-Return Prediction â€” Hyperparameter Tuning (Optuna)

Top-level orchestration: parses CLI arguments, builds a TuningConfig,
constructs the study and runs it with a callback set.

Search space and objective live in dedicated modules:
    common/tuning_search_space.py â€” `suggest_hyperparameters`, `_should_tune`
    common/tuning_objective.py    â€” `create_objective`, `compute_composite_score`
"""
import os
import argparse
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from lstm_config import LSTMConfig
from common.data_utils import load_data
from common.torch_utils import set_seed, get_device
from common.tuning_objective import create_objective
from common.visualization import plot_optuna_results
from common.tuning_utils import (
    save_tuning_results, make_incremental_save_callback,
    _build_loss_config_snapshot,
)
from common.tuning_baseline import build_tuning_baseline
from common.id_utils import get_next_optuna_id, extract_data_id, detect_symbol
from common.training_profiles import PROFILE_CHOICES




# =========================================================================
# TUNING CONFIG
# =========================================================================

class TuningConfig:
    """Configuration specific to Optuna tuning runs."""

    def __init__(self):
        self.TRAIN_PATH = None
        self.FEATURE_SET = LSTMConfig.DEFAULT_FEATURE_SET
        self.CUSTOM_FEATURES = None
        self.PREFERRED_FORMAT = LSTMConfig.PREFERRED_DATA_FORMAT

        self.N_TRIALS = 100
        self.TIMEOUT = None
        self.SEED = LSTMConfig.SEED

        self.TUNING_EPOCHS = 100
        self.TUNING_PATIENCE = 10

        self.SCALE_TARGET = LSTMConfig.SCALE_TARGET
        self.TARGET_SCALER_TYPE = LSTMConfig.TARGET_SCALER_TYPE
        self.WINSORIZE_TARGET = LSTMConfig.WINSORIZE_TARGET
        self.WINSORIZE_PERCENTILE = LSTMConfig.WINSORIZE_PERCENTILE
        self.SIGNAL_THRESHOLD_ENABLED = LSTMConfig.SIGNAL_THRESHOLD_ENABLED
        self.SIGNAL_THRESHOLD_MODE = LSTMConfig.SIGNAL_THRESHOLD_MODE
        self.SIGNAL_THRESHOLD_VALUE = LSTMConfig.SIGNAL_THRESHOLD_VALUE
        self.SIGNAL_THRESHOLD_PERCENTILE = LSTMConfig.SIGNAL_THRESHOLD_PERCENTILE

        self.VAL_SOURCE = LSTMConfig.VAL_SOURCE
        self.SCORE_SOURCE = LSTMConfig.OPTUNA_SCORE_SOURCE

        self.LOOKBACK = None
        self.HORIZON = None
        self.GRADIENT_CLIP = LSTMConfig.GRADIENT_CLIP

        self.N_BEST_RESULTS = getattr(LSTMConfig, 'OPTUNA_N_BEST_RESULTS', 3)

        self.TUNE_PARAMS = dict(getattr(LSTMConfig, 'TUNE_PARAMS', {}))

        self.BASELINE_PARAMS: dict = {}
        self.TRAINING_PROFILE = 'baseline'
        self.BASELINE_PARAMS_PATH = None


# =========================================================================
# HYPERPARAMETER SUGGESTIONS â€” moved to common/tuning_search_space.py
# =========================================================================
# Search space, freeze logic and per-loss conditional dimensions live in
# `common/tuning_search_space.suggest_hyperparameters`. The trial objective
# (training loop, data cache, score computation) lives in
# `common/tuning_objective.create_objective`. This file keeps only the
# `TuningConfig` aggregator, `run_tuning` orchestration and the CLI parser.

# =========================================================================
# MAIN
# =========================================================================

def run_tuning(data_path, n_trials=None, timeout=None, feature_set=None,
               lookback=None, horizon=None, optimize_target=None,
               val_source=None, preferred_format=None, seed=None,
               n_best_results=None, tune_params=None,
               training_profile=None, baseline_params=None):
    """
    Run Optuna hyperparameter tuning.

    Args:
        data_path:       Path to prepared data or raw CSV (required)
        n_trials:        Number of trials (default from TuningConfig)
        optimize_target: Loss type to optimize (e.g. 'custom', 'huber').
                         Overrides LSTMConfig.LOSS_TYPE for this run.
                         Score is always val_loss (minimize).
    """
    cfg = TuningConfig()
    cfg.TRAIN_PATH = data_path
    if n_trials is not None:
        cfg.N_TRIALS = n_trials
    if timeout is not None:
        cfg.TIMEOUT = timeout
    if feature_set is not None:
        cfg.FEATURE_SET = feature_set
    if lookback is not None:
        cfg.LOOKBACK = lookback
    if horizon is not None:
        cfg.HORIZON = horizon
    if optimize_target is not None:
        LSTMConfig.LOSS_TYPE = optimize_target
        LSTMConfig.validate()
    if val_source is not None:
        cfg.VAL_SOURCE = val_source
    if preferred_format is not None:
        cfg.PREFERRED_FORMAT = preferred_format
    if seed is not None:
        cfg.SEED = seed
    if n_best_results is not None:
        cfg.N_BEST_RESULTS = n_best_results
    if tune_params is not None:
        # Overlay on top of existing TUNE_PARAMS (CLI override > config).
        merged = dict(cfg.TUNE_PARAMS)
        merged.update(tune_params)
        cfg.TUNE_PARAMS = merged

    if training_profile is not None:
        cfg.TRAINING_PROFILE = training_profile
    if baseline_params is not None:
        cfg.BASELINE_PARAMS_PATH = baseline_params
    cfg.BASELINE_PARAMS = build_tuning_baseline(
        training_profile=cfg.TRAINING_PROFILE,
        baseline_params_path=cfg.BASELINE_PARAMS_PATH,
    )

    set_seed(cfg.SEED)
    device = get_device()
    LSTMConfig.ensure_dirs()

    # IDs
    optuna_id = get_next_optuna_id()
    symbol = detect_symbol(os.path.basename(data_path))
    data_id = 'UNKNOWN'

    # Quick data load to get data_id
    try:
        _, _, meta = load_data(
            data_path, preferred_format=cfg.PREFERRED_FORMAT,
            lookback=cfg.LOOKBACK or LSTMConfig.DEFAULT_LOOKBACK,
            horizon=cfg.HORIZON or LSTMConfig.DEFAULT_HORIZON,
            feature_set=cfg.FEATURE_SET,
            scale_target=cfg.SCALE_TARGET,
            target_scaler_type=cfg.TARGET_SCALER_TYPE,
            val_source=cfg.VAL_SOURCE,
        )
        data_id = extract_data_id(meta)
    except Exception:
        pass

    # Output directory
    date_str = datetime.now().strftime('%d%m')
    run_name = f"{optuna_id}_optuna_{symbol}_{data_id}_{date_str}"
    output_dir = os.path.join(LSTMConfig.DATA_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Direction is always minimize; score source from config
    direction = 'minimize'
    score_label = 'train_loss' if cfg.SCORE_SOURCE == 'train' else 'val_loss'

    print("=" * 80)
    print(f"LSTM LOG-RETURN OPTUNA TUNING â€” {symbol.upper()}")
    print("=" * 80)
    print(f"Optuna ID:  {optuna_id}")
    print(f"Data ID:    {data_id}")
    print(f"Data:       {data_path}")
    print(f"N trials:   {cfg.N_TRIALS}")
    print(f"Timeout:    {cfg.TIMEOUT}s" if cfg.TIMEOUT else "Timeout:    None")
    print(f"Loss type:  {LSTMConfig.LOSS_TYPE}")
    print(f"Score:      {score_label} (minimize)")
    print(f"Epochs:     {cfg.TUNING_EPOCHS} (patience={cfg.TUNING_PATIENCE})")
    print(f"Device:     {device}")
    print(f"Output:     {output_dir}")
    print(f"N_BEST:     {cfg.N_BEST_RESULTS}")
    frozen = sorted(k for k, v in cfg.TUNE_PARAMS.items() if v is False)
    if frozen:
        print(f"Frozen:     {', '.join(frozen)}")
    if cfg.TRAINING_PROFILE and cfg.TRAINING_PROFILE != 'baseline':
        print(f"Profile:    {cfg.TRAINING_PROFILE}")
    if cfg.BASELINE_PARAMS_PATH:
        print(f"Baseline:   {cfg.BASELINE_PARAMS_PATH}")
    if cfg.BASELINE_PARAMS:
        print(f"Baseline keys: {len(cfg.BASELINE_PARAMS)}")
    print("=" * 80)

    sampler = TPESampler(seed=cfg.SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        study_name=run_name,
    )

    # Suppress Optuna trial logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # F2.5.3 â€” incremental checkpoint: persist top-N params after every new
    # best trial so crashed/interrupted runs still leave a valid
    # best_params.json. Metadata written here matches the final snapshot.
    inc_metadata = {
        'optuna_id': optuna_id,
        'data_id': data_id,
        'feature_set': cfg.FEATURE_SET,
        'scale_target': cfg.SCALE_TARGET,
        'target_scaler_type': cfg.TARGET_SCALER_TYPE,
        'score_type': score_label,
        'loss_type': LSTMConfig.LOSS_TYPE,
        'loss_config': _build_loss_config_snapshot(),
        'direction': direction,
        'incremental': True,
        'training_profile': cfg.TRAINING_PROFILE,
        'baseline_params': cfg.BASELINE_PARAMS_PATH,
    }
    incremental_cb = make_incremental_save_callback(
        output_dir, inc_metadata,
        direction=direction,
        n_best=cfg.N_BEST_RESULTS,
    )

    # Run
    objective = create_objective(cfg, device)
    study.optimize(
        objective,
        n_trials=cfg.N_TRIALS,
        timeout=cfg.TIMEOUT,
        show_progress_bar=True,
        callbacks=[incremental_cb],
    )

    # Results
    print("\n" + "=" * 80)
    print("TUNING COMPLETE")
    print("=" * 80)

    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    pruned = [t for t in study.trials if t.state.name == 'PRUNED']
    print(f"Completed:  {len(completed)}")
    print(f"Pruned:     {len(pruned)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.6f}")
    print(f"\nBest parameters:")
    for k, v in sorted(study.best_trial.params.items()):
        print(f"  {k}: {v}")

    # Save
    print("\nSaving results...")
    save_tuning_results(study, cfg, optuna_id, data_id, output_dir,
                        direction=direction)
    plot_optuna_results(study, output_dir)

    print(f"\n{'='*80}")
    print(f"Output: {output_dir}")
    print(f"Use best params: python lstm_train.py --data_path {data_path} "
          f"--load_optuna_params {os.path.join(output_dir, 'best_params.json')}")
    print(f"{'='*80}")

    return study


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSTM log-return hyperparameter tuning (Optuna)"
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to prepared data directory or raw CSV")
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None,
                        help="Timeout in seconds")
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--lookback', type=int, default=None,
                        help="Fixed lookback (skip tuning)")
    parser.add_argument('--horizon', type=int, default=None,
                        help="Fixed horizon (skip tuning)")
    parser.add_argument('--optimize_target', type=str, default=None,
                        choices=['mse', 'mae', 'huber', 'smooth_l1',
                                 'pseudo_huber', 'quantile', 'custom',
                                 'signal'],
                        help=("Loss type / optimization target. Overrides "
                              "LSTMConfig.LOSS_TYPE for this run. All trials "
                              "train on the chosen loss, score = val_loss. "
                              "Default: LSTMConfig.LOSS_TYPE (custom). "
                              "Example: --optimize_target custom"))
    parser.add_argument('--val_source', type=str, default=None,
                        choices=['split', 'test'],
                        help=("Validation source for Optuna trials. "
                              "'split' (default via LSTMConfig.VAL_SOURCE) "
                              "= 80/20 chronological split of train.csv; "
                              "avoids leaking test.csv into tuning. 'test' "
                              "= test.csv as val (faster but biased)."))
    parser.add_argument('--preferred_format', type=str, default=None,
                        choices=['csv', 'npy'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--optuna_score_type', type=str, default=None,
                        choices=['val_loss', 'composite', 'signal'],
                        help=("F2.V1-5: 'val_loss' (default) or 'composite' "
                              "= val_loss + collapse_penalty + r2_penalty. "
                              "Composite actively penalises trials with low "
                              "compression ratio or negative RÂ˛. "
                              "'signal' = symmetric |1-CR| penalty (penalises "
                              "both collapse and explosion)."))
    parser.add_argument('--n_best_results', type=int, default=None,
                        help=("F2.5.4: how many top-N params JSON files to "
                              "save. Default = LSTMConfig.OPTUNA_N_BEST_RESULTS "
                              "(3). Ranks 1..3 use legacy names "
                              "(best/second_best/third_best), ranks 4+ use "
                              "topNN_params.json."))
    parser.add_argument('--freeze', type=str, default=None,
                        help=("F2.5.5: comma-separated parameter names to "
                              "FREEZE (NOT tune). Each frozen param uses the "
                              "training profile / baseline JSON / LSTMConfig "
                              "default instead of being sampled. "
                              "E.g. --freeze fc_layers,activation "
                              "pins FC head/activation to baseline values. "
                              "Special key 'fc_layers' freezes the entire FC "
                              "head group (n_fc_layers + fc_size_*)."))
    parser.add_argument('--training_profile', type=str, default=None,
                        choices=list(PROFILE_CHOICES),
                        help=("Named recipe for frozen-param defaults "
                              "(rank_focused / sim_focused / custom_sim)."))
    parser.add_argument('--baseline_params', type=str, default=None,
                        help=("Optional Optuna JSON to seed BASELINE_PARAMS "
                              "(e.g. LRO019 best_params.json)."))

    args = parser.parse_args()

    tune_params_override = None
    if args.freeze:
        frozen = [s.strip() for s in args.freeze.split(',') if s.strip()]
        tune_params_override = {name: False for name in frozen}

    # F2.V1-5: composite score is global (affects ALL trials), so we mutate
    # the class attribute up front. Validation happens via argparse choices.
    if args.optuna_score_type is not None:
        LSTMConfig.OPTUNA_SCORE_TYPE = args.optuna_score_type

    run_tuning(
        data_path=args.data_path,
        n_trials=args.n_trials,
        timeout=args.timeout,
        feature_set=args.feature_set,
        lookback=args.lookback,
        horizon=args.horizon,
        optimize_target=args.optimize_target,
        val_source=args.val_source,
        preferred_format=args.preferred_format,
        seed=args.seed,
        n_best_results=args.n_best_results,
        tune_params=tune_params_override,
        training_profile=args.training_profile,
        baseline_params=args.baseline_params,
    )
