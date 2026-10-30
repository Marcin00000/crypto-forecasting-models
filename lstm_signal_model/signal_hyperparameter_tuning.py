"""
LSTM Signal Model â€” Hyperparameter Tuning (Optuna)

Top-level orchestration: parses CLI arguments, builds a SignalTuningConfig,
constructs the study and runs it with a callback set.

Search space and objective live in dedicated modules:
    common/tuning_search_space.py â€” `suggest_hyperparameters`, `_should_tune`
    common/tuning_objective.py    â€” `create_objective`, `compute_composite_score`
"""
import os
import sys
import argparse
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from signal_config import SignalConfig
from common.data_utils import load_signal_data
from common.tuning_objective import create_objective
from common.tuning_utils import (
    save_tuning_results, make_incremental_save_callback,
    _build_signal_config_snapshot,
)
from common.tuning_baseline import build_tuning_baseline
from common.training_profiles import PROFILE_CHOICES

from common.torch_utils import set_seed, get_device
from common.id_utils import (
    get_next_id, extract_data_id, detect_symbol,
)


# =========================================================================
# TUNING CONFIG
# =========================================================================

class SignalTuningConfig:
    """Configuration specific to Optuna tuning runs for the signal model."""

    def __init__(self):
        self.TRAIN_PATH = None
        self.FEATURE_SET = SignalConfig.DEFAULT_FEATURE_SET

        self.N_TRIALS = 100
        self.TIMEOUT = None
        self.SEED = SignalConfig.SEED

        self.TUNING_EPOCHS = 100
        self.TUNING_PATIENCE = 10

        self.VAL_SOURCE = SignalConfig.VAL_SOURCE
        self.OPTUNA_SCORE_TYPE = 'val_loss'
        self.MIN_SIGNAL = SignalConfig.MIN_SIGNAL

        self.LOOKBACK = None
        self.HORIZON = None
        self.GRADIENT_CLIP = SignalConfig.GRADIENT_CLIP

        self.N_BEST_RESULTS = 3

        self.TUNE_PARAMS: dict = {}
        self.BASELINE_PARAMS: dict = {}
        self.TRAINING_PROFILE = 'baseline'
        self.BASELINE_PARAMS_PATH = None
        self.FREEZE_HYPERPARAMS = False
        self.FEATURE_SET_CANDIDATES = None
        self.SUBMODEL_NORM_TYPE = 'auto'
        self.SUBMODEL_FEATURE_SET = None


# =========================================================================
# MAIN
# =========================================================================

def run_tuning(data_path, n_trials=None, timeout=None, feature_set=None,
               lookback=None, horizon=None, val_source=None, seed=None,
               optuna_score_type=None, n_best_results=None, freeze=None,
               training_profile=None, baseline_params=None,
               submodel_dirs=None, tune_feature_set=None,
               feature_set_candidates=None, freeze_hyperparams=None,
               submodel_norm_type=None, submodel_feature_set=None,
               tuning_epochs=None, tuning_patience=None):
    """
    Run Optuna hyperparameter tuning for the signal model.

    Args:
        data_path: Path to prepared data directory (required)
        n_trials:  Number of trials (default from SignalTuningConfig)
    """
    cfg = SignalTuningConfig()
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
    if val_source is not None:
        cfg.VAL_SOURCE = val_source
    if seed is not None:
        cfg.SEED = seed
    if optuna_score_type is not None:
        cfg.OPTUNA_SCORE_TYPE = optuna_score_type
    if n_best_results is not None:
        cfg.N_BEST_RESULTS = n_best_results
    if freeze:
        frozen = [s.strip() for s in freeze.split(',') if s.strip()]
        cfg.TUNE_PARAMS = {name: False for name in frozen}

    if tune_feature_set is not None:
        cfg.TUNE_PARAMS['feature_set'] = bool(tune_feature_set)
    if feature_set_candidates is not None:
        cfg.FEATURE_SET_CANDIDATES = feature_set_candidates
    if freeze_hyperparams is not None:
        cfg.FREEZE_HYPERPARAMS = bool(freeze_hyperparams)
    if submodel_norm_type is not None:
        cfg.SUBMODEL_NORM_TYPE = submodel_norm_type
    if submodel_feature_set is not None:
        cfg.SUBMODEL_FEATURE_SET = submodel_feature_set
    if tuning_epochs is not None:
        cfg.TUNING_EPOCHS = tuning_epochs
    if tuning_patience is not None:
        cfg.TUNING_PATIENCE = tuning_patience

    if training_profile is not None:
        cfg.TRAINING_PROFILE = training_profile
    if baseline_params is not None:
        cfg.BASELINE_PARAMS_PATH = baseline_params
    cfg.BASELINE_PARAMS = build_tuning_baseline(
        training_profile=cfg.TRAINING_PROFILE,
        baseline_params_path=cfg.BASELINE_PARAMS_PATH,
    )

    # Submodel integration
    if submodel_dirs:
        cfg.SUBMODEL_DIRS = submodel_dirs
        cfg.HAS_SUBMODELS = True

    set_seed(cfg.SEED)
    device = get_device()
    SignalConfig.ensure_dirs()

    optuna_id = get_next_id(SignalConfig.DATA_DIR, 'SIGO', '_optuna_')
    symbol = detect_symbol(os.path.basename(data_path))
    data_id = 'UNKNOWN'

    try:
        data = load_signal_data(
            data_path,
            feature_set=cfg.FEATURE_SET,
            lookback=cfg.LOOKBACK or SignalConfig.DEFAULT_LOOKBACK,
            horizon=cfg.HORIZON or SignalConfig.DEFAULT_HORIZON,
            val_source=cfg.VAL_SOURCE,
        )
        data_id = extract_data_id(data['metadata'])
    except Exception:
        pass

    date_str = datetime.now().strftime('%d%m')
    run_name = f"{optuna_id}_optuna_{symbol}_{data_id}_{date_str}"
    output_dir = os.path.join(SignalConfig.DATA_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    direction = 'minimize'
    print("=" * 80)
    print(f"SIGNAL MODEL OPTUNA TUNING â€” {symbol.upper()}")
    print("=" * 80)
    print(f"Optuna ID:  {optuna_id}")
    print(f"Data ID:    {data_id}")
    print(f"Data:       {data_path}")
    print(f"N trials:   {cfg.N_TRIALS}")
    print(f"Timeout:    {cfg.TIMEOUT}s" if cfg.TIMEOUT else "Timeout:    None")
    print(f"Score:      {cfg.OPTUNA_SCORE_TYPE} (minimize)")
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

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    inc_metadata = {
        'optuna_id': optuna_id,
        'data_id': data_id,
        'feature_set': cfg.FEATURE_SET,
        'score_type': cfg.OPTUNA_SCORE_TYPE,
        'loss_config': _build_signal_config_snapshot(),
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

    objective = create_objective(cfg, device)
    study.optimize(
        objective,
        n_trials=cfg.N_TRIALS,
        timeout=cfg.TIMEOUT,
        show_progress_bar=True,
        callbacks=[incremental_cb],
    )

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

    print("\nSaving results...")
    save_tuning_results(study, cfg, optuna_id, data_id, output_dir,
                        direction=direction)

    print(f"\n{'='*80}")
    print(f"Output: {output_dir}")
    print(f"Use best params: python signal_train.py --data_path {data_path} "
          f"--load_optuna_params {os.path.join(output_dir, 'best_params.json')}")
    print(f"{'='*80}")

    return study


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal Model hyperparameter tuning (Optuna)"
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help="Path to prepared data directory")
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--timeout', type=int, default=None,
                        help="Timeout in seconds")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--lookback', type=int, default=None,
                        help="Fixed lookback (skip tuning)")
    parser.add_argument('--horizon', type=int, default=None,
                        help="Fixed horizon (skip tuning)")
    parser.add_argument('--val_source', type=str, default=None,
                        choices=['split', 'test'])
    parser.add_argument('--optuna_score_type', type=str, default=None,
                        choices=['val_loss', 'composite', 'signal_rank'],
                        help=("'val_loss' (default), 'composite' = "
                              "val_loss + diversity collapse penalty, "
                              "'signal_rank' = -compute_signal_rank_score "
                              "(directly optimises rank score)."))
    parser.add_argument('--n_best_results', type=int, default=None,
                        help="How many top-N params JSON files to save.")
    parser.add_argument('--freeze', type=str, default=None,
                        help=("Comma-separated parameter names to FREEZE "
                              "(NOT tune). Each frozen param uses the "
                              "training profile / baseline JSON / "
                              "SignalConfig default instead of being "
                              "sampled. E.g. --freeze fc_layers,activation"))
    parser.add_argument('--training_profile', type=str, default=None,
                        choices=list(PROFILE_CHOICES),
                        help="Named recipe for frozen-param defaults.")
    parser.add_argument('--baseline_params', type=str, default=None,
                        help="Optional Optuna JSON to seed BASELINE_PARAMS.")
    parser.add_argument('--submodel_dirs', type=str, nargs='+', default=None,
                        help='LR, XGB, RF or Price-LSTM model directories (or IDs) for '
                             'submodel fusion (e.g. LR1107 X240 R057 L012)')
    parser.add_argument('--tune_feature_set', type=int, default=None, choices=[0, 1],
                        help="1 = sample feature_set from candidates in Optuna search space")
    parser.add_argument('--feature_set_candidates', type=str, nargs='+', default=None,
                        help="List of feature set names to sample during tuning")
    parser.add_argument('--freeze_hyperparams', type=int, default=None, choices=[0, 1],
                        help="1 = Mode A (feature set selection only, freeze all architecture)")
    parser.add_argument('--submodel_norm_type', type=str, default=None,
                        choices=['auto', 'batch', 'zscore', 'layer', 'none'],
                        help="Normalization adapter for submodels")
    parser.add_argument('--submodel_feature_set', type=str, default=None,
                        help="Feature set for submodel disjoint feeding")
    parser.add_argument('--tuning_epochs', type=int, default=None,
                        help="Max epochs per trial (default 50)")
    parser.add_argument('--tuning_patience', type=int, default=None,
                        help="Early stopping patience per trial (default 10)")

    args = parser.parse_args()

    run_tuning(
        data_path=args.data_path,
        n_trials=args.n_trials,
        timeout=args.timeout,
        feature_set=args.feature_set,
        lookback=args.lookback,
        horizon=args.horizon,
        val_source=args.val_source,
        seed=args.seed,
        optuna_score_type=args.optuna_score_type,
        n_best_results=args.n_best_results,
        freeze=args.freeze,
        training_profile=args.training_profile,
        baseline_params=args.baseline_params,
        submodel_dirs=args.submodel_dirs,
        tune_feature_set=args.tune_feature_set,
        feature_set_candidates=args.feature_set_candidates,
        freeze_hyperparams=args.freeze_hyperparams,
        submodel_norm_type=args.submodel_norm_type,
        submodel_feature_set=args.submodel_feature_set,
        tuning_epochs=args.tuning_epochs,
        tuning_patience=args.tuning_patience,
    )
