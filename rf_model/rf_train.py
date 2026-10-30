"""
Random Forest Price Prediction Training
"""
import os
import warnings
from datetime import datetime

import numpy as np

from rf_config import RFConfig
from common.data_utils import load_data
from common.id_utils import get_next_model_id, extract_data_id, detect_symbol
from common.training_utils import (
    set_seed, setup_logger,
    train_model as _train_model,
    save_rf_model, print_final_summary,
    compute_sample_weights,
)

warnings.filterwarnings('ignore')


def train(
        data_path,
        lookback=None,
        horizon=None,
        flattening_method=None,
        feature_set=None,
        custom_features=None,
        # RF hyperparameters
        n_estimators=None,
        max_depth=None,
        max_features=None,
        max_samples=None,
        min_samples_leaf=None,
        min_samples_split=None,
        min_impurity_decrease=None,
        min_weight_fraction_leaf=None,
        bootstrap=None,
        criterion=None,
        # target pipeline
        scale_target=None,
        target_scaler_type=None,
        winsorize_target=None,
        winsorize_percentile=None,
        # sample weights
        sample_weight_mode=None,
        sample_weight_alpha=None,
        sample_weight_min_w=None,
        sample_weight_percentile=None,
        sample_weight_low_w=None,
        data_source='train',
        # misc
        seed=42,
        verbose=True,
        load_optuna_params=None,
):
    set_seed(seed)

    # ------------------------------------------------------------------
    # Load Optuna params -- override only values not explicitly provided
    # ------------------------------------------------------------------
    if load_optuna_params is not None:
        optuna_params, optuna_meta = RFConfig.load_optuna_params(load_optuna_params)
        print(f"Loaded Optuna params from: {load_optuna_params}")

        if lookback is None:              lookback              = optuna_params.get('lookback')
        if horizon is None:               horizon               = optuna_params.get('horizon')
        if flattening_method is None:     flattening_method     = optuna_params.get('flattening_method')
        if n_estimators is None:          n_estimators          = optuna_params.get('n_estimators')
        if max_depth is None:             max_depth             = optuna_params.get('max_depth')
        if max_features is None:          max_features          = optuna_params.get('max_features')
        if max_samples is None:           max_samples           = optuna_params.get('max_samples')
        if min_samples_leaf is None:      min_samples_leaf      = optuna_params.get('min_samples_leaf')
        if min_samples_split is None:     min_samples_split     = optuna_params.get('min_samples_split')
        if min_impurity_decrease is None: min_impurity_decrease = optuna_params.get('min_impurity_decrease')
        if min_weight_fraction_leaf is None: min_weight_fraction_leaf = optuna_params.get('min_weight_fraction_leaf')
        if bootstrap is None:             bootstrap             = optuna_params.get('bootstrap')
        if criterion is None:             criterion             = optuna_params.get('criterion')
        if scale_target is None:          scale_target          = optuna_meta.get('scale_target')
        if target_scaler_type is None:    target_scaler_type    = optuna_meta.get('target_scaler_type')
        if feature_set is None:           feature_set           = optuna_meta.get('feature_set')

    # ------------------------------------------------------------------
    # Apply config defaults
    # ------------------------------------------------------------------
    if lookback is None:              lookback              = RFConfig.DEFAULT_LOOKBACK
    if horizon is None:               horizon               = RFConfig.DEFAULT_HORIZON
    if flattening_method is None:     flattening_method     = RFConfig.DEFAULT_FLATTENING
    if feature_set is None:           feature_set           = RFConfig.DEFAULT_FEATURE_SET
    if scale_target is None:          scale_target          = RFConfig.SCALE_TARGET
    if target_scaler_type is None:    target_scaler_type    = RFConfig.TARGET_SCALER_TYPE
    if winsorize_target is None:      winsorize_target      = RFConfig.WINSORIZE_TARGET
    if winsorize_percentile is None:  winsorize_percentile  = RFConfig.WINSORIZE_PERCENTILE
    if sample_weight_mode is not None and str(sample_weight_mode).lower() in ('none', 'uniform'):
        sample_weight_mode = None
    elif sample_weight_mode is None:
        sample_weight_mode = RFConfig.SAMPLE_WEIGHT_MODE
    if sample_weight_alpha is None:       sample_weight_alpha       = RFConfig.SAMPLE_WEIGHT_ALPHA
    if sample_weight_min_w is None:       sample_weight_min_w       = RFConfig.SAMPLE_WEIGHT_MIN_W
    if sample_weight_percentile is None:  sample_weight_percentile  = RFConfig.SAMPLE_WEIGHT_PERCENTILE
    if sample_weight_low_w is None:       sample_weight_low_w       = RFConfig.SAMPLE_WEIGHT_LOW_W

    # ------------------------------------------------------------------
    # Build RF params dict
    # ------------------------------------------------------------------
    rf_criterion = RFConfig.get_rf_criterion() if criterion is None else criterion

    # Validate explicit criterion
    if criterion is not None and criterion in RFConfig.INCOMPATIBLE_OBJECTIVES:
        raise ValueError(
            f"Criterion '{criterion}' is incompatible with log-return targets.\n"
            f"Use one of: {sorted(RFConfig.VALID_OBJECTIVES)}"
        )

    params = dict(RFConfig.DEFAULT_PARAMS)
    params['criterion'] = rf_criterion

    overrides = {
        'n_estimators':             n_estimators,
        'max_depth':                max_depth,
        'max_features':             max_features,
        'max_samples':              max_samples,
        'min_samples_leaf':         min_samples_leaf,
        'min_samples_split':        min_samples_split,
        'min_impurity_decrease':    min_impurity_decrease,
        'min_weight_fraction_leaf': min_weight_fraction_leaf,
        'bootstrap':                bootstrap,
    }
    for k, v in overrides.items():
        if v is not None:
            params[k] = v

    # oob_score & max_samples: honour bootstrap constraints in scikit-learn
    if not params.get('bootstrap', True):
        params['oob_score'] = False
        params['max_samples'] = None
    else:
        params['oob_score'] = RFConfig.USE_OOB_SCORE

    params['random_state'] = seed

    # ------------------------------------------------------------------
    print("=" * 80)
    print("RANDOM FOREST TRAINING")
    print("=" * 80)

    symbol = detect_symbol(os.path.basename(data_path))

    data, _, metadata = load_data(
        data_path,
        flattening_method=flattening_method,
        feature_set=feature_set,
        custom_features=custom_features,
        lookback=lookback,
        horizon=horizon,
        scale_target=scale_target,
        target_scaler_type=target_scaler_type,
        winsorize_target=winsorize_target,
        winsorize_percentile=winsorize_percentile,
        data_source=data_source,
    )

    data_id           = extract_data_id(metadata)
    features          = metadata['features']
    flat_features     = metadata['flat_features']
    lookback          = metadata['lookback']
    horizon           = metadata['horizon']
    flattening_method = metadata['flattening_method']
    clip_bounds       = data.get('clip_bounds')

    X_train        = data['X_train']
    y_train        = data['y_train']
    X_val          = data['X_val']
    y_val          = data['y_val']
    feature_scaler = data['feature_scaler']
    target_scaler  = data.get('target_scaler')

    print(f"\nData ID:    {data_id}")
    print(f"Features:   {len(features)} -> {len(flat_features)} flat")
    print(f"Flattening: {flattening_method}")
    print(f"Lookback:   {lookback}, Horizon: {horizon}")
    print(f"Train:      {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Criterion:       {params['criterion']}")
    print(f"Target scaler:   {'ENABLED (' + target_scaler.method + ')' if target_scaler else 'DISABLED'}")
    print(f"Winsorisation:   {'ENABLED (' + str(winsorize_percentile) + 'th pct)' if winsorize_target else 'DISABLED'}")
    print(f"Sample weights:  {'ENABLED (mode=' + str(sample_weight_mode) + ')' if sample_weight_mode else 'DISABLED'}")

    # ------------------------------------------------------------------
    # Compute sample weights from raw y_train (before target scaling)
    # target_scaler was already applied inside load_data, so we need the
    # raw values -- inverse-transform if scaler is active.
    # ------------------------------------------------------------------
    if sample_weight_mode is not None and target_scaler is not None:
        y_train_raw = target_scaler.inverse_transform(y_train)
    else:
        y_train_raw = y_train

    sample_weight = compute_sample_weights(
        y_train_raw,
        mode=sample_weight_mode,
        alpha=sample_weight_alpha,
        min_w=sample_weight_min_w,
        percentile=sample_weight_percentile,
        low_w=sample_weight_low_w,
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    model_id   = get_next_model_id()
    date_short = datetime.now().strftime('%d%m')
    run_name   = f"{model_id}_{symbol}_rf_model_{data_id}_{date_short}"
    output_dir = os.path.join(RFConfig.MODELS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("=" * 80)
    logger.info(f"RANDOM FOREST TRAINING - {symbol.upper()} - Horizon t+{horizon}")
    logger.info("=" * 80)
    logger.info(f"Model ID:   {model_id}")
    logger.info(f"Data ID:    {data_id}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Criterion:       {params['criterion']}")
    logger.info(f"Target scaler:   {'ENABLED (' + target_scaler.method + ')' if target_scaler else 'DISABLED'}")
    logger.info(f"Winsorisation:   {'ENABLED (' + str(winsorize_percentile) + 'th pct)' if winsorize_target else 'DISABLED'}")
    logger.info(f"Sample weights:  {'ENABLED (mode=' + str(sample_weight_mode) + ')' if sample_weight_mode else 'DISABLED'}")
    if clip_bounds:
        logger.info(f"Clip bounds:     [{clip_bounds['lower']:.6f}, {clip_bounds['upper']:.6f}]")
    logger.info("\nHyperparameters:")
    for key, value in sorted(params.items()):
        logger.info(f"  {key:25s}: {value}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model, train_metrics, val_metrics = _train_model(
        X_train, y_train, X_val, y_val,
        params, logger, verbose,
        target_scaler=target_scaler,
        sample_weight=sample_weight,
        reweight_misdir=RFConfig.REWEIGHT_MISDIR,
        misdir_penalty=RFConfig.MISDIR_PENALTY,
        misdir_min_quantile=RFConfig.MISDIR_MIN_QUANTILE,
    )

    logger.info("\nTRAIN METRICS:")
    logger.info(f"  MAE:      {train_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {train_metrics['rmse']:.6f}")
    logger.info(f"  R2:       {train_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {train_metrics['dir_acc']:.2f}%")
    logger.info("\nVAL METRICS:")
    logger.info(f"  MAE:      {val_metrics['mae']:.6f}")
    logger.info(f"  RMSE:     {val_metrics['rmse']:.6f}")
    logger.info(f"  R2:       {val_metrics['r2']:.4f}")
    logger.info(f"  Dir Acc:  {val_metrics['dir_acc']:.2f}%")

    # ------------------------------------------------------------------
    # Build config for saving
    # ------------------------------------------------------------------
    config = {
        'model_id':           model_id,
        'data_id':            data_id,
        'symbol':             symbol,
        'model_type':         'random_forest',
        'target_type':        'log_return',
        'features':           features,
        'flat_features':      flat_features,
        'n_features':         len(features),
        'n_flat_features':    len(flat_features),
        'lookback':           lookback,
        'horizon':            horizon,
        'flattening_method':  flattening_method,
        'feature_set':        feature_set,
        'params':             params,
        'train_metrics':      train_metrics,
        'val_metrics':        val_metrics,
        'timestamp':          datetime.now().strftime('%Y%m%d_%H%M%S'),
        'output_dir':         output_dir,
        'scale_target':       scale_target,
        'target_scaler_type': target_scaler_type if scale_target else None,
        'winsorize_target':   winsorize_target,
        'winsorize_percentile': winsorize_percentile if winsorize_target else None,
        'clip_bounds':        clip_bounds,
        'train_path':         data_path,
        'data_source':        data_source,
        # Persist the signal threshold so tests and the inference package reproduce the
        # simulation of this run even if RFConfig.DIR_ACC_THRESHOLD changes later.
        'dir_acc_threshold':  RFConfig.DIR_ACC_THRESHOLD,
        'signal_threshold':   RFConfig.DIR_ACC_THRESHOLD or 0.001,
        'oob_score':          float(model.oob_score_) if params.get('oob_score') and params.get('bootstrap') else None,
        'scaler_feature_cols': list(feature_scaler.feature_names_in_)
                               if hasattr(feature_scaler, 'feature_names_in_')
                               else None,
        'sample_weight_params': {
            'mode':        sample_weight_mode,
            'alpha':       sample_weight_alpha,
            'min_w':       sample_weight_min_w,
            'percentile':  sample_weight_percentile,
            'low_w':       sample_weight_low_w,
        },
        'reweight_misdir':      RFConfig.REWEIGHT_MISDIR,
        'misdir_penalty':       RFConfig.MISDIR_PENALTY       if RFConfig.REWEIGHT_MISDIR else None,
        'misdir_min_quantile':  RFConfig.MISDIR_MIN_QUANTILE  if RFConfig.REWEIGHT_MISDIR else None,
    }

    save_rf_model(model, feature_scaler, config, output_dir, logger,
                  target_scaler=target_scaler)
    print_final_summary(model_id, output_dir, train_metrics, val_metrics, logger)

    return model, feature_scaler, config


# Backward-compatible alias
train_model = train


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--load_optuna_params', type=str, default=None,
                        help='Path to Optuna best_params.json')
    parser.add_argument('--lookback', type=int, default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--flattening_method', type=str, default=None,
                        choices=['lag', 'agg', 'hybrid', 'signal', 'none'])
    parser.add_argument('--feature_set', type=str, default=None)
    parser.add_argument('--n_estimators', type=int, default=None)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--max_features', type=float, default=None)
    parser.add_argument('--max_samples', type=float, default=None)
    parser.add_argument('--min_samples_leaf', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=None)
    parser.add_argument('--min_impurity_decrease', type=float, default=None)
    parser.add_argument('--min_weight_fraction_leaf', type=float, default=None)
    parser.add_argument('--no_bootstrap', dest='bootstrap', action='store_false', default=None)
    parser.add_argument('--criterion', type=str, default=None,
                        choices=['squared_error', 'absolute_error', 'friedman_mse'])
    parser.add_argument('--scale_target', action='store_true', default=None)
    parser.add_argument('--no_scale_target', dest='scale_target', action='store_false')
    parser.add_argument('--target_scaler_type', type=str, default=None,
                        choices=['standard', 'robust'])
    parser.add_argument('--winsorize_target', action='store_true', default=None)
    parser.add_argument('--winsorize_percentile', type=float, default=None)
    parser.add_argument('--sample_weight_mode', type=str, default=None,
                        choices=['abs', 'abs_clip', 'rank', 'percentile', 'none', 'uniform'],
                        help='Sample weighting scheme (none / uniform = disabled)')
    parser.add_argument('--sample_weight_alpha', type=float, default=None,
                        help='Exponent for abs/abs_clip modes (default 1.0)')
    parser.add_argument('--sample_weight_min_w', type=float, default=None,
                        help='Weight floor fraction for abs_clip mode (default 0.1)')
    parser.add_argument('--sample_weight_percentile', type=float, default=None,
                        help='Threshold percentile for percentile mode (default 70.0)')
    parser.add_argument('--sample_weight_low_w', type=float, default=None,
                        help='Weight for below-threshold samples in percentile mode (default 0.1)')
    parser.add_argument('--data_source', type=str, default='train', choices=['train', 'agent'],
                        help="Dataset split to train on: 'train' (default) or 'agent'")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        load_optuna_params=args.load_optuna_params,
        lookback=args.lookback,
        horizon=args.horizon,
        flattening_method=args.flattening_method,
        feature_set=args.feature_set,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        max_samples=args.max_samples,
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        min_impurity_decrease=args.min_impurity_decrease,
        min_weight_fraction_leaf=args.min_weight_fraction_leaf,
        bootstrap=args.bootstrap,
        criterion=args.criterion,
        scale_target=args.scale_target,
        target_scaler_type=args.target_scaler_type,
        winsorize_target=args.winsorize_target,
        winsorize_percentile=args.winsorize_percentile,
        sample_weight_mode=args.sample_weight_mode,
        sample_weight_alpha=args.sample_weight_alpha,
        sample_weight_min_w=args.sample_weight_min_w,
        sample_weight_percentile=args.sample_weight_percentile,
        sample_weight_low_w=args.sample_weight_low_w,
        data_source=args.data_source,
        seed=args.seed,
        verbose=args.verbose,
    )
