"""
LSTM Hyperparameter Tuning using Optuna
"""
import os
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import joblib
from tqdm import tqdm

from feature_engineering import prepare_features, select_features
from common.training_utils import CryptoDataset, set_seed
from lstm_config import ALL_FEATURE_SETS
from lstm_train_price import LSTMPriceModel
from common.data_utils import create_sequences
from common.id_utils import get_next_optuna_id, extract_data_id_from_path
from common.visualization import plot_optuna_results

warnings.filterwarnings('ignore')


class TuningConfig:
    TRAIN_PATH = 'datasets/prepared'
    FEATURE_SET = None
    CUSTOM_FEATURES = None
    VALIDATE_FEATURES = True
    GENERATE_FEATURES = True
    USE_SUBSET = True
    SUBSET_SIZE = 200000
    N_TRIALS = 100
    TIMEOUT = None
    N_JOBS = 1
    ENABLE_PRUNING = True
    PRUNING_WARMUP = 5
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    BATCH_SIZE_OPTIONS = [128, 256, 512]
    VAL_SPLIT = 0.2
    USE_TIME_SERIES_SPLIT = True
    N_SPLITS = 3
    SEED = 42
    FIXED_HORIZON = None


def suggest_hyperparameters(trial, is_prepared, prepared_metadata):
    """Define hyperparameter search space."""
    params = {
        'hidden_size':    trial.suggest_categorical('hidden_size', [32, 64, 96, 128, 192, 256]),
        'num_layers':     trial.suggest_int('num_layers', 1, 3),
        'dropout':        trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
        'learning_rate':  trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size':     trial.suggest_categorical('batch_size', TuningConfig.BATCH_SIZE_OPTIONS),
        'weight_decay':   trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'scheduler_type': 'none',
        'use_batch_norm': True,
    }

    if is_prepared and prepared_metadata:
        # Lookback/horizon are fixed by the prepared data â€” do not tune them
        params['lookback'] = prepared_metadata['lookback']
        params['horizon']  = prepared_metadata['horizon']
    else:
        params['lookback'] = trial.suggest_categorical('lookback', [12, 24, 36, 48, 72])
        if TuningConfig.FIXED_HORIZON is None:
            params['horizon'] = trial.suggest_categorical('horizon', [1, 3, 6, 12])
        else:
            params['horizon'] = TuningConfig.FIXED_HORIZON

    return params


class LSTMTunable(LSTMPriceModel):
    """Use the same architecture as final training and the inference package."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1, use_batch_norm=True):
        if not use_batch_norm:
            raise ValueError("The canonical price model requires batch normalization.")
        super().__init__(input_size, hidden_size, num_layers, dropout)



def objective(trial):
    set_seed(TuningConfig.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        from common.data_utils import load_data
        data, is_prepared, metadata = load_data(
            TuningConfig.TRAIN_PATH,
            TuningConfig.USE_SUBSET,
            TuningConfig.SUBSET_SIZE,
            feature_set=TuningConfig.FEATURE_SET,
            custom_features=TuningConfig.CUSTOM_FEATURES,
        )

        params = suggest_hyperparameters(trial, is_prepared, metadata)

        trial.set_user_attr('resolved_params', dict(params))

        if params['scheduler_type'] == 'cosine':
            params['scheduler_t0']   = trial.suggest_int('scheduler_t0', 5, 15)
            params['scheduler_mult'] = trial.suggest_int('scheduler_mult', 1, 2)
        elif params['scheduler_type'] == 'plateau':
            params['scheduler_patience'] = trial.suggest_int('scheduler_patience', 3, 8)
            params['scheduler_factor']   = trial.suggest_float('scheduler_factor', 0.3, 0.7)

        # ---------------------------------------------------------------- #
        # Data path A: prepared directory (train.csv, chronological train/val split)
        # ---------------------------------------------------------------- #
        if is_prepared:
            X_train = data['X_train']
            y_train = data['y_train']
            X_val   = data['X_val']     # chronological validation tail of train.csv
            y_val   = data['y_val']
            features = data.get('features_list') or metadata['features']

            # Log info once (trial 0)
            if trial.number == 0:
                print(f"\n[Tuning] Prepared data: {X_train.shape[0]:,} train / "
                      f"{X_val.shape[0]:,} val sequences, {len(features)} features")

        # ---------------------------------------------------------------- #
        # Data path B: raw CSV â€” generate features + scale on-the-fly
        # ---------------------------------------------------------------- #
        else:
            df = data

            if TuningConfig.GENERATE_FEATURES:
                df = prepare_features(df)

            df = df.replace([np.inf, -np.inf], np.nan).dropna()

            has_buy_pressure = 'buy_pressure' in df.columns

            features = select_features(
                df,
                feature_set=TuningConfig.FEATURE_SET,
                custom_features=TuningConfig.CUSTOM_FEATURES,
                validate=TuningConfig.VALIDATE_FEATURES,
                include_buy_pressure=has_buy_pressure,
            )
            features = [f for f in features if f != 'close']

            if len(features) < 5:
                raise ValueError(f"Too few features: {len(features)}")

            if TuningConfig.USE_TIME_SERIES_SPLIT:
                tscv   = TimeSeriesSplit(n_splits=TuningConfig.N_SPLITS)
                splits = list(tscv.split(df))
                train_idx, val_idx = splits[-1]
                train_df = df.iloc[train_idx].copy()
                val_df   = df.iloc[val_idx].copy()
            else:
                split_idx = int(len(df) * (1 - TuningConfig.VAL_SPLIT))
                train_df  = df.iloc[:split_idx].copy()
                val_df    = df.iloc[split_idx:].copy()

            feature_scaler = StandardScaler()
            train_df[features] = feature_scaler.fit_transform(train_df[features])
            val_df[features]   = feature_scaler.transform(val_df[features])

            train_df['close'] = np.log(train_df['close'])
            val_df['close']   = np.log(val_df['close'])

            price_scaler = StandardScaler()
            train_df['close'] = price_scaler.fit_transform(
                train_df['close'].values.reshape(-1, 1)).flatten()
            val_df['close'] = price_scaler.transform(
                val_df['close'].values.reshape(-1, 1)).flatten()

            X_train, y_train, _ = create_sequences(
                train_df, features, params['lookback'], params['horizon'])
            X_val, y_val, _     = create_sequences(
                val_df,   features, params['lookback'], params['horizon'])

        trial.set_user_attr('resolved_features', list(features))

        if len(X_train) < 100 or len(X_val) < 50:
            raise optuna.TrialPruned("Insufficient data for sequences")

        train_loader = DataLoader(
            CryptoDataset(X_train, y_train),
            batch_size=params['batch_size'], shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            CryptoDataset(X_val, y_val),
            batch_size=params['batch_size'], shuffle=False,
        )

        model = LSTMTunable(
            len(features), params['hidden_size'], params['num_layers'],
            params['dropout'], params['use_batch_norm'],
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
        )

        scheduler = None
        if params['scheduler_type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=params['scheduler_t0'], T_mult=params['scheduler_mult'])
        elif params['scheduler_type'] == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=params['scheduler_patience'],
                factor=params['scheduler_factor'], mode='min')

        best_val_loss = float('inf')
        patience = 0

        for epoch in range(TuningConfig.MAX_EPOCHS):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch).squeeze(), y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss += criterion(model(X_batch).squeeze(), y_batch).item()
            val_loss /= len(val_loader)

            if scheduler is not None:
                scheduler.step(val_loss) if params['scheduler_type'] == 'plateau' \
                    else scheduler.step()

            trial.report(val_loss, epoch)

            if TuningConfig.ENABLE_PRUNING and epoch >= TuningConfig.PRUNING_WARMUP:
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= TuningConfig.EARLY_STOPPING_PATIENCE:
                    break

        del model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_loss

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()


class ProgressCallback:
    def __init__(self):
        self.pbar = None
        self.best_value = float('inf')

    def __call__(self, study, trial):
        if self.pbar is None:
            self.pbar = tqdm(total=TuningConfig.N_TRIALS, desc="Trials")
        self.pbar.update(1)

        if trial.state == TrialState.COMPLETE:
            if trial.value < self.best_value:
                self.best_value = trial.value
                self.pbar.set_postfix(
                    {'best': f'{self.best_value:.6f}', 'trial': trial.number})
        elif trial.state == TrialState.PRUNED:
            self.pbar.set_postfix({'status': 'PRUNED', 'trial': trial.number})


def run_tuning(n_trials=None, timeout=None, study_name=None):

    if n_trials is None:
        n_trials = TuningConfig.N_TRIALS
    if timeout is None:
        timeout = TuningConfig.TIMEOUT

    data_id  = extract_data_id_from_path(TuningConfig.TRAIN_PATH)
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lstm_data')
    os.makedirs(base_dir, exist_ok=True)

    optuna_id  = get_next_optuna_id(base_dir)
    date_short = datetime.now().strftime('%d%m')

    if study_name is None:
        study_name = f"{optuna_id}_optuna_tuning_results_{data_id}_{date_short}"

    output_dir = os.path.join(base_dir, study_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'tuning.log'), encoding='utf-8'),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("OPTUNA HYPERPARAMETER TUNING")
    logger.info("=" * 80)
    logger.info(f"Optuna ID: {optuna_id}")
    logger.info(f"Data ID:   {data_id}")
    logger.info(f"Study:     {study_name}")
    logger.info(f"Output:    {output_dir}")
    logger.info(f"Trials:    {n_trials}")
    logger.info(f"Horizon:   {'TUNED' if TuningConfig.FIXED_HORIZON is None else f'FIXED (t+{TuningConfig.FIXED_HORIZON})'}")
    logger.info(f"Device:    {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    sampler = TPESampler(seed=TuningConfig.SEED)
    pruner  = (MedianPruner(n_startup_trials=5, n_warmup_steps=TuningConfig.PRUNING_WARMUP)
               if TuningConfig.ENABLE_PRUNING else None)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=sampler,
        pruner=pruner,
    )

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION START")
    logger.info("=" * 80)

    progress = ProgressCallback()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[progress],
            n_jobs=TuningConfig.N_JOBS,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        if progress.pbar:
            progress.pbar.close()

    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == TrialState.PRUNED]

    logger.info(f"\nTotal trials:    {len(study.trials)}")
    logger.info(f"Complete trials: {len(completed)}")
    logger.info(f"Pruned trials:   {len(pruned)}")

    if not completed:
        logger.error("\nNo trials completed successfully!")
        return None, None

    best = study.best_trial
    logger.info("\n" + "=" * 80)
    logger.info("BEST TRIAL")
    logger.info("=" * 80)
    logger.info(f"\nTrial:     {best.number}")
    logger.info(f"Val loss:  {best.value:.6f}")
    logger.info("\nParameters:")
    for k, v in best.params.items():
        logger.info(f"  {k:20s}: {v}")

    logger.info("\n" + "=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    sorted_trials = sorted(completed, key=lambda t: t.value)
    top_trials    = sorted_trials[:min(3, len(sorted_trials))]
    trial_names   = ['best', 'second_best', 'third_best']

    for idx, (trial, name) in enumerate(zip(top_trials, trial_names[:len(top_trials)])):
        params_to_save = trial.user_attrs.get('resolved_params', trial.params).copy()
        params_to_save['_metadata'] = {
            'rank':                idx + 1,
            'optuna_id':           optuna_id,
            'data_id':             data_id,
            'best_value':          float(trial.value),
            'trial_number':        trial.number,
            'timestamp':           datetime.now().strftime('%Y%m%d_%H%M%S'),
            'feature_set':         TuningConfig.FEATURE_SET,
            'custom_features':     trial.user_attrs.get('resolved_features', TuningConfig.CUSTOM_FEATURES),
            'validate_features':   TuningConfig.VALIDATE_FEATURES,
            'generate_features':   TuningConfig.GENERATE_FEATURES,
            'train_path':          TuningConfig.TRAIN_PATH,
            'fixed_horizon':       TuningConfig.FIXED_HORIZON,
            'validation_strategy': ('TimeSeriesSplit' if TuningConfig.USE_TIME_SERIES_SPLIT
                                    else 'Chronological'),
        }

        params_path = os.path.join(output_dir, f'{name}_params.json')
        with open(params_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        logger.info(f"  âś“ {name}_params.json (trial {trial.number}, loss: {trial.value:.6f})")

    joblib.dump(study, os.path.join(output_dir, 'study.pkl'))
    logger.info("  âś“ study.pkl")

    study.trials_dataframe().to_csv(os.path.join(output_dir, 'trials.csv'), index=False)
    logger.info("  âś“ trials.csv")

    plot_optuna_results(study, output_dir)

    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("OPTUNA TUNING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Optuna ID:   {optuna_id}\n")
        f.write(f"Data ID:     {data_id}\n")
        f.write(f"Study:       {study_name}\n")
        f.write(f"Data source: {TuningConfig.TRAIN_PATH}\n")
        f.write(f"Timestamp:   {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
        f.write(f"Total:       {len(study.trials)}\n")
        f.write(f"Complete:    {len(completed)}\n")
        f.write(f"Horizon:     {'TUNED' if TuningConfig.FIXED_HORIZON is None else f'FIXED (t+{TuningConfig.FIXED_HORIZON})'}\n\n")

        f.write(f"TOP {len(top_trials)} TRIAL(S):\n")
        for idx, (trial, name) in enumerate(zip(top_trials, trial_names[:len(top_trials)])):
            f.write(f"\n{idx + 1}. {name.upper()}:\n")
            f.write(f"   Trial:    {trial.number}\n")
            f.write(f"   Val loss: {trial.value:.6f}\n")
            f.write("   Parameters:\n")
            for k, v in trial.params.items():
                f.write(f"     {k:20s}: {v}\n")
    logger.info("  âś“ summary.txt")

    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("\nTrain with best params:")
    logger.info(f"  python lstm_model/lstm_train_price.py --train_path {TuningConfig.TRAIN_PATH} --load_optuna_params {output_dir}/best_params.json --epochs 100")
    logger.info("\nTrain all top 3 models:")
    logger.info("  Repeat final training with each selected *_params.json file.")
    if TuningConfig.FEATURE_SET:
        logger.info(f"  (Feature set '{TuningConfig.FEATURE_SET}' will be loaded from metadata)")
    logger.info("\n" + "=" * 80)

    return study, best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials',         type=int,   default=None)
    parser.add_argument('--timeout',          type=int,   default=None)
    parser.add_argument('--study_name',       type=str,   default=None)
    parser.add_argument('--subset_size',      type=int,   default=None)
    parser.add_argument('--feature_set',      type=str,   default=None, choices=ALL_FEATURE_SETS)
    parser.add_argument('--custom_features',  type=str,   nargs='+', default=None)
    parser.add_argument('--no_validate_features', action='store_true')
    parser.add_argument('--no_generate_features', action='store_true')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--fixed_horizon',    type=int,   default=None)

    args = parser.parse_args()

    if args.subset_size:        TuningConfig.SUBSET_SIZE     = args.subset_size
    if args.feature_set:        TuningConfig.FEATURE_SET     = args.feature_set
    if args.custom_features:    TuningConfig.CUSTOM_FEATURES = args.custom_features
    if args.no_validate_features: TuningConfig.VALIDATE_FEATURES = False
    if args.no_generate_features: TuningConfig.GENERATE_FEATURES = False
    TuningConfig.MAX_EPOCHS = args.epochs
    if args.train_path:         TuningConfig.TRAIN_PATH      = args.train_path
    if args.fixed_horizon is not None: TuningConfig.FIXED_HORIZON = args.fixed_horizon

    run_tuning(
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=args.study_name,
    )
