"""
Data utilities for Signal Model pipeline.

Public API:
  - load_signal_data() — unified entry point: CSV → sequences → oracle target
  - create_sequences() — 3D sequence creation for LSTM
"""
import os
import json

import numpy as np
import pandas as pd
import joblib

from .target_utils import (
    compute_sigma_series,
    align_sigma,
    generate_oracle_targets,
    winsorize_targets,
)


def fill_sigma_nan(sigma_aligned: np.ndarray) -> np.ndarray:
    """Safely fill NaN/inf values in aligned sigma array with median of finite values."""
    sigma = np.asarray(sigma_aligned, dtype=np.float64).copy()
    finite_mask = np.isfinite(sigma)
    if np.any(finite_mask):
        sigma_fill = float(np.median(sigma[finite_mask]))
    else:
        sigma_fill = 0.001
    return np.where(finite_mask, sigma, sigma_fill)


# =========================================================================
# 3D SEQUENCE CREATION
# =========================================================================

def create_sequences(data_array, lookback):
    """
    Create 3D sequences for LSTM from 2D array.

    X[i] = data_array[i : i+lookback]

    Args:
        data_array: np.ndarray of shape (n_total, n_features)
        lookback:   int — sequence length

    Returns:
        np.ndarray of shape (n_samples, lookback, n_features)
    """
    n_total = len(data_array)
    n_samples = n_total - lookback
    if n_samples <= 0:
        raise ValueError(
            f"Insufficient data for sequences: {n_total} rows, "
            f"lookback={lookback}, need at least {lookback + 1}"
        )
    n_features = data_array.shape[1]
    X = np.zeros((n_samples, lookback, n_features), dtype=np.float32)
    for i in range(n_samples):
        X[i] = data_array[i:i + lookback]
    return X


def _create_log_return_target(close, lookback, horizon):
    """
    Create log-return targets aligned to sequences.

    For sample i (whose last visible candle is i+lookback-1):
      y[i] = log(close[i+lookback-1+horizon] / close[i+lookback-1])

    Returns y of shape (n_samples,) — samples with insufficient future
    data are set to NaN.
    """
    close = np.asarray(close, dtype=np.float64)
    n_total = len(close)
    n_samples = n_total - lookback
    y = np.full(n_samples, np.nan, dtype=np.float64)

    for i in range(n_samples):
        cur_idx = i + lookback - 1
        fut_idx = cur_idx + horizon
        if fut_idx < n_total and close[cur_idx] > 0:
            y[i] = np.log(close[fut_idx] / close[cur_idx])
    return y


def _coverage_info(requested_set, n_requested, resolved, missing):
    """Build the coverage descriptor returned alongside the resolved feature list."""
    n_requested = int(n_requested)
    return {
        'requested_set': requested_set,
        'n_requested': n_requested,
        'n_resolved': len(resolved),
        'coverage': (len(resolved) / n_requested) if n_requested else 1.0,
        'missing': list(missing),
    }


def _resolve_features(metadata, feature_set=None, custom_features=None, *,
                      min_coverage=1.0, allow_partial=False):
    """
    Resolve feature list from metadata or explicit args.

    Args:
        metadata:        parsed metadata.json of the prepared dataset
        feature_set:     name of a set from feature_engineering.get_feature_sets()
        custom_features: explicit feature list (wins over feature_set); used when
                         re-loading data for an already trained model
        min_coverage:    minimum fraction of requested features that must exist
                         in the dataset (only for the feature_set branch)
        allow_partial:   downgrade coverage errors to warnings

    Returns:
        (features, coverage_info) where coverage_info is
        {'requested_set', 'n_requested', 'n_resolved', 'coverage', 'missing'}

    Raises:
        ValueError: unknown feature set name, empty resolved list, or coverage
            below min_coverage while allow_partial is False.
    """
    import logging
    log = logging.getLogger('signal_data_utils')
    base = metadata.get('features', [])
    data_id = metadata.get('data_id', 'UNKNOWN')

    # --- explicit feature list (saved model): never validated against the registry
    # so that adding new features to feature_engineering.py cannot invalidate
    # models that ask for their own features by name.
    if custom_features:
        resolved = [f for f in custom_features if f in base]
        missing = [f for f in custom_features if f not in base]
        info = _coverage_info(None, len(custom_features), resolved, missing)
        if missing:
            msg = (
                f"{len(missing)} of {len(custom_features)} saved features are absent "
                f"from dataset '{data_id}': {missing}. Loading fewer features would "
                f"change input_size and fail later with an unreadable shape mismatch. "
                f"Use a dataset containing these features, or pass "
                f"allow_partial_feature_set=True (--allow_partial_feature_set 1) "
                f"to reproduce the old truncating behaviour."
            )
            if not allow_partial:
                raise ValueError(msg)
            log.warning(msg)
        return resolved, info

    if not feature_set or feature_set == metadata.get('feature_set'):
        return list(base), _coverage_info(feature_set, len(base), base, [])

    try:
        from feature_engineering import get_feature_sets
    except ImportError:
        log.warning(
            f"feature_engineering registry unavailable — cannot validate feature set "
            f"'{feature_set}'; falling back to all {len(base)} dataset features."
        )
        return list(base), _coverage_info(feature_set, len(base), base, [])

    has_bp = metadata.get('has_buy_pressure', 'buy_pressure' in base)
    sets = get_feature_sets(include_buy_pressure=has_bp)
    if feature_set not in sets:
        raise ValueError(
            f"Unknown feature set '{feature_set}' (dataset '{data_id}', "
            f"metadata feature_set='{metadata.get('feature_set')}'). "
            f"Available: {sorted(sets.keys())}"
        )

    requested = sets[feature_set]
    if not requested:
        raise ValueError(
            f"Feature set '{feature_set}' is an empty placeholder in "
            f"feature_engineering.get_feature_sets() - nothing to train on."
        )

    resolved = [f for f in requested if f in base]
    missing = [f for f in requested if f not in base]
    info = _coverage_info(feature_set, len(requested), resolved, missing)

    if info['coverage'] < min_coverage:
        msg = (
            f"Feature set '{feature_set}' requested {len(requested)} features but "
            f"dataset '{data_id}' provides only {len(resolved)} "
            f"(coverage {info['coverage']:.1%} < required {min_coverage:.1%}). "
            f"feature_set '{feature_set}' requires columns absent from dataset "
            f"'{data_id}' (dataset feature_set: '{metadata.get('feature_set')}', "
            f"missing: {missing}). Use --feature_set "
            f"{metadata.get('feature_set') or '<metadata.feature_set>'} "
            f"(--allow_partial_feature_set 1 only truncates the set and changes input_size)."
        )
        if not allow_partial:
            raise ValueError(msg)
        log.warning(msg)

    if not resolved:
        raise ValueError(
            f"Feature set '{feature_set}' resolved to zero features on dataset "
            f"'{data_id}'. Requested: {requested}"
        )
    return resolved, info


# =========================================================================
# MAIN LOADER
# =========================================================================

def load_signal_data(data_path, *, feature_set=None, custom_features=None,
                     submodel_feature_set=None,
                     lookback=48, horizon=1, val_source='split',
                     signal_threshold=0.001, window_sigma=20,
                     extra_features_paths=None,
                     canonical_oracle_params=None,
                     max_lookback=72, preferred_data_format='csv',
                     winsorize_percentile=None,
                     data_source='train',
                     feature_round_decimals=None,
                     allow_partial_feature_set: bool = False,
                     min_feature_coverage: float = 1.0):
    """
    Load prepared CSV data, create sequences, and generate oracle targets.

    Args:
        submodel_feature_set: optional feature set name for submodels (e.g. 'submodel_compat_eth',
            'submodel_compat_btc', 'technical_bp') or an explicit list of feature names
            (used when XGB / RF / Price submodels are present). When provided, parallel
            sequence matrices (X_train_submodel, X_val_submodel) are created for submodel
            feature feeding.
        val_source: 'split' | 'test' | 'sym' | 'agent'
        winsorize_percentile: optional float e.g. 99.0 for 1% extreme clipping
        data_source: 'train' (default) | 'agent'
            'train' — standard: loads train.csv as training data
            'agent' — loads agent.csv as training data (with 80/20 split).
                       Used for submodel pipelines to avoid data leakage
                       (LR submodels were trained on train.csv).
        feature_round_decimals: optional int to quantize/round input features.
        allow_partial_feature_set: when True, missing features only produce a
            warning instead of a ValueError (old, silently truncating behaviour).
        min_feature_coverage: minimum fraction of a named feature set that must
            exist in the dataset (default 1.0 = all features required).
    """
    if os.path.isfile(data_path):
        parent = os.path.dirname(data_path) or '.'
        if os.path.isfile(os.path.join(parent, 'metadata.json')):
            data_path = parent
        else:
            raise FileNotFoundError(
                f"load_signal_data expects a prepared directory, got a file: {data_path}"
            )
    with open(os.path.join(data_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    features, feature_coverage = _resolve_features(
        metadata, feature_set, custom_features,
        min_coverage=min_feature_coverage,
        allow_partial=allow_partial_feature_set)
    submodel_features = None
    submodel_coverage = None
    if submodel_feature_set is not None:
        is_name_list = isinstance(submodel_feature_set, (list, tuple))
        submodel_features, submodel_coverage = _resolve_features(
            metadata,
            feature_set=None if is_name_list else submodel_feature_set,
            custom_features=list(submodel_feature_set) if is_name_list else None,
            min_coverage=min_feature_coverage,
            allow_partial=allow_partial_feature_set)

    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    feature_scaler = None
    if os.path.exists(scaler_path):
        try:
            feature_scaler = joblib.load(scaler_path)
        except Exception:
            # Identity is only a correct stand-in when the dataset was never scaled;
            # for any scaled dataset a silent identity would feed unscaled inputs to the
            # model at test time, so the load error must surface.
            scalers_meta = metadata.get('scalers') or {}
            scaler_kind = scalers_meta.get('features')
            unscaled = ((isinstance(scaler_kind, str) and scaler_kind.lower() == 'none')
                        or scalers_meta.get('scale_csv') is False)
            if not unscaled:
                raise
            from sklearn.preprocessing import FunctionTransformer
            feature_scaler = FunctionTransformer(func=None, inverse_func=None, validate=False)

    # Select training data source
    if data_source == 'agent':
        train_csv_path = os.path.join(data_path, 'agent.csv')
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(
                f"agent.csv not found in {data_path}. "
                f"data_source='agent' requires agent.csv for submodel training."
            )
        train_df = pd.read_csv(train_csv_path)
        # Force split mode — val from agent split (agent.csv) itself (80/20)
        val_source = 'split'
    else:
        train_csv_path = os.path.join(data_path, 'train.csv')
        train_df = pd.read_csv(train_csv_path)

    if val_source == 'sym':
        _val_csv = 'sym.csv'
    elif val_source == 'agent':
        _val_csv = 'agent.csv'
    else:
        _val_csv = 'test.csv'
    test_df = pd.read_csv(os.path.join(data_path, _val_csv))

    sub_feats_avail = None
    if submodel_features is not None:
        sub_feats_avail = [f for f in submodel_features if f in train_df.columns]
        if len(sub_feats_avail) < len(submodel_features):
            import logging
            dropped = [f for f in submodel_features if f not in train_df.columns]
            logging.getLogger('signal_data_utils').warning(
                f"Submodel feature set '{submodel_feature_set}': {len(dropped)} feature(s) "
                f"listed in metadata.json are absent from the CSV: {dropped}"
            )
            submodel_coverage = _coverage_info(
                submodel_feature_set, submodel_coverage['n_requested'],
                sub_feats_avail, submodel_coverage['missing'] + dropped)

    def _extract_arrays(df):
        if 'close' not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'close' column for target calculation. "
                f"Available columns: {list(df.columns)}"
            )
        feat_data = df[features].values.astype(np.float32)
        if feature_round_decimals is not None and feature_round_decimals > 0:
            feat_data = np.round(feat_data, feature_round_decimals)
        close = df['close'].values.astype(np.float64)

        X = create_sequences(feat_data, lookback)
        n_samples = X.shape[0]

        X_sub = None
        if sub_feats_avail is not None:
            feat_data_sub = df[sub_feats_avail].values.astype(np.float32)
            if feature_round_decimals is not None and feature_round_decimals > 0:
                feat_data_sub = np.round(feat_data_sub, feature_round_decimals)
            X_sub = create_sequences(feat_data_sub, lookback)

        y_true_lr = _create_log_return_target(close, lookback, horizon)
        sigma_series = compute_sigma_series(close, window_sigma)
        sigma_aligned = align_sigma(sigma_series, lookback, n_samples)
        sigma_aligned = fill_sigma_nan(sigma_aligned)

        c_sigma_aligned = None
        if canonical_oracle_params is not None:
            c_window = canonical_oracle_params['window_sigma']
            c_sigma_series = compute_sigma_series(close, c_window)
            c_sigma_aligned = align_sigma(c_sigma_series, lookback, n_samples)
            c_sigma_aligned = fill_sigma_nan(c_sigma_aligned)

        valid = np.isfinite(y_true_lr)
        if max_lookback and max_lookback > lookback:
            trim = max_lookback - lookback
            if trim < len(valid):
                valid[:trim] = False

        close_aligned = close[np.arange(n_samples) + (lookback - 1)]

        return {
            'X': X,
            'y_true_lr': y_true_lr,
            'sigma_aligned': sigma_aligned,
            'c_sigma_aligned': c_sigma_aligned,
            'close_aligned': close_aligned,
            'valid': valid,
            'X_sub': X_sub,
            'close': close,
            'n_samples': n_samples,
        }

    def _build_targets(y_lr, sigma, c_sigma=None):
        oracle_buy, oracle_sell = generate_oracle_targets(
            y_lr, sigma, signal_threshold)
        target = np.stack([oracle_buy, oracle_sell], axis=1).astype(np.float32)

        target_canonical = None
        if canonical_oracle_params is not None and c_sigma is not None:
            c_thresh = canonical_oracle_params['signal_threshold']
            cb, cs = generate_oracle_targets(y_lr, c_sigma, c_thresh)
            target_canonical = np.stack([cb, cs], axis=1).astype(np.float32)
        return target, target_canonical

    if val_source == 'split':
        arrays = _extract_arrays(train_df)
        valid = arrays['valid']
        X_valid = arrays['X'][valid]
        y_lr_valid = arrays['y_true_lr'][valid]
        sigma_valid = arrays['sigma_aligned'][valid]
        c_sigma_valid = arrays['c_sigma_aligned'][valid] if arrays['c_sigma_aligned'] is not None else None
        close_valid = arrays['close_aligned'][valid]
        X_sub_valid = arrays['X_sub'][valid] if arrays['X_sub'] is not None else None

        split_idx = int(len(X_valid) * 0.8)
        y_lr_train = y_lr_valid[:split_idx]
        y_lr_val = y_lr_valid[split_idx:]

        if winsorize_percentile is not None and winsorize_percentile > 0:
            y_lr_train, y_lr_val, _ = winsorize_targets(
                y_lr_train, y_lr_val, percentile=winsorize_percentile)
            y_lr_valid_for_target = np.concatenate([y_lr_train, y_lr_val])
        else:
            y_lr_valid_for_target = y_lr_valid

        target_full, target_canon_full = _build_targets(
            y_lr_valid_for_target, sigma_valid, c_sigma_valid)

        X_train, X_val = X_valid[:split_idx], X_valid[split_idx:]
        target_train, target_val = target_full[:split_idx], target_full[split_idx:]
        y_true_train, y_true_val = y_lr_train, y_lr_val
        close_train, close_val = close_valid[:split_idx], close_valid[split_idx:]
        if X_sub_valid is not None:
            X_train_submodel = X_sub_valid[:split_idx]
            X_val_submodel = X_sub_valid[split_idx:]
        else:
            X_train_submodel = X_val_submodel = None

        if target_canon_full is not None:
            target_train_canonical = target_canon_full[:split_idx]
            target_val_canonical = target_canon_full[split_idx:]
        else:
            target_train_canonical = target_val_canonical = None
    else:
        arr_tr = _extract_arrays(train_df)
        arr_vl = _extract_arrays(test_df)
        val_tr = arr_tr['valid']
        val_vl = arr_vl['valid']
        y_lr_tr = arr_tr['y_true_lr'][val_tr]
        y_lr_vl = arr_vl['y_true_lr'][val_vl]

        if winsorize_percentile is not None and winsorize_percentile > 0:
            y_lr_tr, y_lr_vl, _ = winsorize_targets(
                y_lr_tr, y_lr_vl, percentile=winsorize_percentile)

        c_sig_tr = arr_tr['c_sigma_aligned'][val_tr] if arr_tr['c_sigma_aligned'] is not None else None
        c_sig_vl = arr_vl['c_sigma_aligned'][val_vl] if arr_vl['c_sigma_aligned'] is not None else None

        target_train, target_train_canonical = _build_targets(
            y_lr_tr, arr_tr['sigma_aligned'][val_tr], c_sig_tr)
        target_val, target_val_canonical = _build_targets(
            y_lr_vl, arr_vl['sigma_aligned'][val_vl], c_sig_vl)

        X_train = arr_tr['X'][val_tr]
        X_val = arr_vl['X'][val_vl]
        y_true_train = y_lr_tr
        y_true_val = y_lr_vl
        close_train = arr_tr['close_aligned'][val_tr]
        close_val = arr_vl['close_aligned'][val_vl]
        X_train_submodel = arr_tr['X_sub'][val_tr] if arr_tr['X_sub'] is not None else None
        X_val_submodel = arr_vl['X_sub'][val_vl] if arr_vl['X_sub'] is not None else None

    result = {
        'X_train': X_train,
        'target_train': target_train,
        'X_val': X_val,
        'target_val': target_val,
        'y_true_train': y_true_train,
        'y_true_val': y_true_val,
        'close_train': close_train,
        'close_val': close_val,
        'features': features,
        'feature_scaler': feature_scaler,
        'metadata': metadata,
        'feature_coverage': feature_coverage,
        'submodel_feature_coverage': submodel_coverage,
    }
    if X_train_submodel is not None:
        result['X_train_submodel'] = X_train_submodel
        result['X_val_submodel'] = X_val_submodel
        result['submodel_features'] = sub_feats_avail if sub_feats_avail is not None else submodel_features
    if canonical_oracle_params is not None:
        result['target_train_canonical'] = target_train_canonical
        result['target_val_canonical'] = target_val_canonical
    return result


def resolve_submodel_feature_set(
    submodel_dirs: list[str] | None,
    symbol: str = 'ETH',
    explicit_set: str | list[str] | None = None,
) -> str | list[str] | None:
    """
    Resolve the appropriate submodel feature set (CP-146).

    If explicit_set is provided, returns it directly.
    With any XGB / RF / Price submodel returns the ordered union of their feature
    lists (a list of names), because the compat sets only cover LR submodels.
    Otherwise auto-detects based on symbol and the submodel's n_features (e.g. 32 -> FE3).
    """
    if explicit_set:
        return explicit_set
    if not submodel_dirs:
        return None

    from .submodel_manager import (
        _resolve_model_dir, union_submodel_features, detect_submodel_kind)
    try:
        resolved = [_resolve_model_dir(d) for d in submodel_dirs]
    except FileNotFoundError:
        resolved = []  # SubmodelManager reports the unresolved directory later
    # No broad try/except here: a silent fallback to a compat set would feed
    # tree / price submodels the wrong columns.
    if any(detect_submodel_kind(d) != 'lr' for d in resolved):
        return union_submodel_features(resolved)

    is_fe3 = False
    try:
        from pathlib import Path
        import json
        first_dir = _resolve_model_dir(submodel_dirs[0])
        cfg_path = Path(first_dir) / 'model_config.json'
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                sm_cfg = json.load(f)
            if sm_cfg.get('n_features') == 32 or len(sm_cfg.get('features', [])) == 32:
                is_fe3 = True
    except Exception:
        pass

    if 'BTC' in str(symbol).upper():
        return 'submodel_compat_btc_fe3' if is_fe3 else 'submodel_compat_btc'
    return 'submodel_compat_eth_fe3' if is_fe3 else 'submodel_compat_eth'
