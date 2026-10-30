"""
Test data loading and reporting utilities for Random Forest model evaluation.

Handles loading held-out test splits (test.csv, agent.csv, val.csv),
causal alignment via align_features_and_target, and Parquet/CSV persistence.
"""
from __future__ import annotations

import os
import json
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from .flattening import flatten_features, align_features_and_target

__all__ = [
    'load_test_data_only',
    'load_test_csv',
    '_extract_close',
    'find_results_path',
    'load_results',
    'save_results',
    'build_rf_test_summary',
]


def _extract_close(df: pd.DataFrame) -> np.ndarray | None:
    """Return raw close-price series as float64 array, or None if unavailable."""
    if 'close' not in df.columns:
        return None
    return df['close'].to_numpy(dtype=np.float64)


def _resolve_features_from_metadata(
    base_features: list[str],
    feature_set: str | None,
    custom_features: list[str] | None,
    has_buy_pressure: bool,
    features: list[str] | None = None,
) -> list[str]:
    """
    Feature list for testing. `features` (model_config['features']) is authoritative: it is
    exactly what the model was trained on. The name-based fallback exists for callers that
    only know a feature_set and never trims silently (unknown name or missing column raises).
    """
    if features is not None:
        return list(features)

    from .data_utils import _resolve_features
    return _resolve_features(base_features, feature_set, custom_features, has_buy_pressure)


def load_test_data_only(
    data_path: str,
    flattening_method: str | None = None,
    feature_set: str | None = None,
    custom_features: list[str] | None = None,
    lookback: int | None = None,
    horizon: int | None = None,
    test_file_type: str = 'test',
    return_close: bool = False,
    features: list[str] | None = None,
) -> tuple[dict, dict] | tuple[dict, dict, np.ndarray | None]:
    """
    Load test data from a prepared dataset directory with strict causal alignment.

    `features` is the model's own feature list (model_config['features']); when given it
    takes precedence over feature_set / custom_features and every column must exist in
    the CSV (ValueError otherwise).
    """
    from .data_utils import create_log_return_target

    valid_splits = ('test', 'agent', 'sym', 'val', 'train')
    if test_file_type not in valid_splits:
        raise ValueError(
            f"Invalid test_file_type: {test_file_type!r}. Must be one of {valid_splits}"
        )

    metadata_path = os.path.join(data_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata.json in {data_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    csv_file = f'{test_file_type}.csv'
    test_csv = os.path.join(data_path, csv_file)
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Missing {csv_file} in {data_path}")

    test_df = pd.read_csv(test_csv)

    scaler_path = os.path.join(data_path, 'scaler_features.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(data_path, 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Feature scaler not found in {data_path}")
    feature_scaler = joblib.load(scaler_path)

    base_features = metadata['features']
    lookback = int(lookback if lookback is not None else metadata['lookback'])
    horizon = int(horizon if horizon is not None else metadata['horizon'])
    # Column presence, not the metadata flag: E114/E115 declare buy_pressure without it.
    has_buy_pressure = 'buy_pressure' in base_features

    features = _resolve_features_from_metadata(
        base_features, feature_set, custom_features, has_buy_pressure, features=features
    )
    missing_cols = [f for f in features if f not in test_df.columns]
    if missing_cols:
        raise ValueError(
            f"{csv_file} in {data_path} lacks model feature columns: {missing_cols}. "
            f"The model cannot be evaluated on a dataset that does not contain them."
        )

    if flattening_method is None:
        flattening_method = metadata.get('flattening_method', 'hybrid')

    X_test_flat, flat_features = flatten_features(
        test_df, features, lookback, method=flattening_method
    )
    y_test_raw = create_log_return_target(test_df, lookback, horizon)

    # Centralised 4-step alignment
    X_test_clean, y_test_clean, first_valid_idx = align_features_and_target(
        X_test_flat, y_test_raw, lookback, horizon
    )

    # Prepared dataset test.csv is already pre-scaled
    X_test = X_test_clean.values
    close_full = _extract_close(test_df)

    data = {
        'X_test': X_test,
        'y_test': y_test_clean,
        'close_full': close_full,
        'feature_scaler': feature_scaler,
        'first_valid_idx': first_valid_idx,
    }

    metadata.update({
        'flat_features': flat_features,
        'n_flat_features': len(flat_features),
        'n_features': len(features),
        'features': features,
        'lookback': lookback,
        'horizon': horizon,
        'flattening_method': flattening_method,
        'has_buy_pressure': has_buy_pressure,
        'test_file_type': test_file_type,
        'data_alignment_version': 2,
    })

    if return_close:
        return data, metadata, close_full
    return data, metadata


def load_test_csv(
    csv_path: str,
    config: dict,
    feature_scaler,
    return_close: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict] | tuple[np.ndarray, np.ndarray, dict, np.ndarray | None]:
    """Load and process a standalone raw CSV for testing."""
    from feature_engineering import prepare_features
    from .data_utils import create_log_return_target

    df = pd.read_csv(csv_path)
    close_full = _extract_close(df)

    df_feats = prepare_features(df)
    # Replace infinite values with NaN without dropping rows to maintain 1:1 alignment with close_full
    df_feats = df_feats.replace([np.inf, -np.inf], np.nan)

    lookback = int(config.get('lookback', 48))
    horizon = int(config.get('horizon', 12))
    method = config.get('flattening_method', 'hybrid')
    features = config.get('features', [])
    missing_cols = [f for f in features if f not in df_feats.columns]
    if missing_cols:
        raise ValueError(f"Model features could not be generated from {csv_path}: {missing_cols}")

    if feature_scaler is not None:
        n_in = getattr(feature_scaler, 'n_features_in_', None)
        if n_in == len(features):
            df_feats[features] = feature_scaler.transform(df_feats[features].values)

    X_flat, flat_features = flatten_features(df_feats, features, lookback, method=method)
    y_raw = create_log_return_target(df_feats, lookback, horizon)

    X_clean, y_clean, first_valid_idx = align_features_and_target(X_flat, y_raw, lookback, horizon)
    X_scaled = X_clean.values
    if feature_scaler is not None:
        n_in = getattr(feature_scaler, 'n_features_in_', None)
        if n_in == len(flat_features):
            X_scaled = feature_scaler.transform(X_clean.values)

    meta = {
        'flat_features': flat_features,
        'lookback': lookback,
        'horizon': horizon,
        'first_valid_idx': first_valid_idx,
        'data_alignment_version': 2,
    }

    if return_close:
        return X_scaled, y_clean, meta, close_full
    return X_scaled, y_clean, meta


def find_results_path(output_dir: str | Path) -> Path | None:
    """Find existing results file, preferring Parquet over CSV."""
    p_parquet = Path(output_dir) / 'results.parquet'
    if p_parquet.exists():
        return p_parquet
    p_csv = Path(output_dir) / 'results.csv'
    if p_csv.exists():
        return p_csv
    return None


def load_results(results_path: str | Path) -> pd.DataFrame:
    """Load results DataFrame from Parquet or CSV."""
    p = Path(results_path)
    if not p.exists():
        raise FileNotFoundError(f"Results file not found: {p}")
    if p.suffix.lower() == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_results(
    output_dir: str | Path,
    df: pd.DataFrame,
    *,
    results_format: str = 'parquet',
) -> Path:
    """Save results DataFrame as Parquet or CSV with graceful fallback."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if results_format == 'parquet':
        target = out / 'results.parquet'
        try:
            df.to_parquet(target, compression='snappy', index=False)
            return target
        except (ImportError, ValueError) as err:
            print(f"[WARN] Parquet engine unavailable ({err}), falling back to results.csv")
            target = out / 'results.csv'
            df.to_csv(target, index=False)
            return target
    else:
        target = out / 'results.csv'
        df.to_csv(target, index=False)
        return target


def build_rf_test_summary(
    model_id: str,
    test_id: str,
    metrics: dict,
    thresh_metrics: dict | None = None,
    sim_profile: dict | None = None,
    rank_score_dict: dict | None = None,
    test_file_type: str = 'test',
    results_format: str = 'parquet',
    results_path: Path | None = None,
) -> str:
    """Build standardized summary.txt text block for test run."""
    lines = [
        "=" * 70,
        "RANDOM FOREST MODEL TEST REPORT",
        "=" * 70,
        f"Model ID:        {model_id}",
        f"Test ID:         {test_id}",
        f"Date:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Test Split:      {test_file_type}",
        f"Results Format:  {results_format} ({results_path.name if results_path else 'results.' + results_format})",
        "",
        "-" * 70,
        "REGRESSION METRICS (LOG-RETURN)",
        "-" * 70,
        f"  MAE:                   {metrics['mae']:.6f}",
        f"  RMSE:                  {metrics['rmse']:.6f}",
        f"  R2:                    {metrics['r2']:.4f}",
        f"  Directional Accuracy:  {metrics.get('dir_acc', 0.0):.2f}%",
    ]

    if 'compression_ratio' in metrics:
        lines.append(f"  Compression Ratio (CR): {metrics['compression_ratio']:.4f}")
    if 'corr' in metrics:
        lines.append(f"  Correlation (Pearson):  {metrics['corr']:.4f}")

    if thresh_metrics and thresh_metrics.get('n_above', 0) > 0:
        lines.extend([
            "",
            "-" * 70,
            f"THRESHOLD METRICS (|y_pred| >= {thresh_metrics['threshold']:.4f})",
            "-" * 70,
            f"  Samples above threshold: {thresh_metrics['n_above']} ({thresh_metrics['pct_above']:.1f}%)",
            f"  Threshold Dir Acc:       {thresh_metrics.get('t_dir_acc', 0.0):.2f}%",
            f"  Threshold MAE:           {thresh_metrics.get('t_mae', 0.0):.6f}",
            f"  Threshold Bias BUY:      {thresh_metrics.get('t_bias_buy', 0.0):.1f}%",
        ])

    if sim_profile and 'variants' in sim_profile:
        lines.extend([
            "",
            "-" * 70,
            "SIMULATION V2 PROFILE (4 VARIANTS)",
            "-" * 70,
        ])
        for var_name, var_data in sim_profile['variants'].items():
            gross_m = var_data['gross']['model']
            gross_o = var_data['gross']['oracle']
            act = var_data['activity']
            lines.extend([
                f"\n  Variant: {var_name}",
                f"    Gross Final Value:    ${gross_m['final_value']:.2f} (Return: {gross_m['return_pct']:+.2f}%, Trades: {gross_m['n_trades']})",
                f"    Oracle Final Value:   ${gross_o['final_value']:.2f} (Return: {gross_o['return_pct']:+.2f}%)",
                f"    Log Capture Ratio:    {var_data['gross'].get('capture_log') if var_data['gross'].get('capture_log') is not None else 0.0:.4f}",
                f"    Strong Signals:       {act.get('pct_strong', 0.0):.1f}%",
                f"    Calibration Buy/Sell: {act.get('calib_buy') or 0.0:.3f} / {act.get('calib_sell') or 0.0:.3f}",
            ])
            if 'net' in var_data:
                net_m = var_data['net']['model']
                fee = var_data['config'].get('net_fee_bps', 2.0)
                lines.append(f"    Net ({fee} bps) Value:     ${net_m['final_value']:.2f} (Return: {net_m['return_pct']:+.2f}%)")

    if rank_score_dict:
        lines.extend([
            "",
            "=" * 70,
            "RANK SCORE v2 SUMMARY",
            "=" * 70,
            f"  Rank Score (0-100):     {rank_score_dict.get('rank_score', 0.0):.2f}",
            f"  Signal Class:           {rank_score_dict.get('signal_class', 'UNKNOWN')}",
        ])
        comps = rank_score_dict.get('rank_components', {})
        if comps:
            lines.extend([
                f"    Activity (25):        {comps.get('activity', 0.0):.2f}",
                f"    Calibration (25):     {comps.get('calibration', 0.0):.2f}",
                f"    Direction (20):       {comps.get('direction', 0.0):.2f}",
                f"    Capture (25):         {comps.get('capture', 0.0):.2f}",
                f"    CR Health (5):        {comps.get('cr_health', 0.0):.2f}",
            ])

    lines.append("=" * 70)
    return "\n".join(lines) + "\n"
