"""
LSTM Price Prediction Testing Script

Evaluates trained LSTM price models using Simulation Profile v2,
implied log-returns mu, Rank Score v2, Parquet persistence, and dollar visualizations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from common.id_utils import (
    extract_model_id,
    get_next_test_id,
    is_known_data_id,
    resolve_data_id,
    resolve_dataset_path,
    resolve_repo_path,
)
from common.metrics import (
    calculate_directional_accuracy,
    calculate_log_return_metrics,
    calculate_price_metrics,
    compute_rank_score,
)
from common.signals import compute_signals, compute_sigma_series, align_sigma
from common.simulation_profile import evaluate_lstm_simulation_v2
from common.target_utils import (
    compute_actual_log_returns,
    compute_implied_log_returns,
    convert_predictions_to_dollars,
)
from common.test_data_utils import (
    build_lstm_test_summary,
    generate_sticky_signal_pos,
    load_test_data,
    save_results,
)
from common.visualization import (
    plot_test_results,
    plot_log_returns_results,
    plot_simulation_equity,
)
from lstm_config import DATA_DIR
from lstm_train_price import LSTMPriceModel

warnings.filterwarnings('ignore')


def _load_model_config(model_dir: Path) -> dict:
    """Load model configuration from model_config.json with fallback to model_config.pkl."""
    json_path = model_dir / 'model_config.json'
    pkl_path = model_dir / 'model_config.pkl'

    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif pkl_path.exists():
        return joblib.load(pkl_path)
    raise FileNotFoundError(f"Missing model_config in {model_dir}")


def _resolve_test_path(test_path: str | Path | None, config: dict) -> str:
    """
    Dataset lookup order: CLI path -> config['train_path'] (repo-relative or legacy absolute;
    a file path uses its parent directory) -> data_preparation/prepared_data/<data_id>_*.
    ``prepared_dir_name`` is tried first and skipped when it does not resolve (a run that
    was pointed at train.csv stored the file name there). No dataset family is assumed.
    """
    if test_path is not None:
        return str(test_path)
    train_path = config.get('train_path')
    if train_path:
        candidate = resolve_repo_path(train_path)
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.is_dir() and (candidate / "metadata.json").is_file():
            return str(candidate)
    for ref in (config.get("prepared_dir_name"), config.get("data_id")):
        if not is_known_data_id(ref):
            continue
        try:
            return str(resolve_dataset_path(ref))
        except FileNotFoundError:
            continue
    raise ValueError("Cannot locate the test dataset: pass --test_path (model_config.json has no dataset reference).")


def test_model(
    model_dir: str | Path,
    test_path: str | Path | None = None,
    test_file_type: str = 'test',
    save_results_flag: bool = True,
    generate_plots: bool = True,
) -> dict:
    """
    Run standardized evaluation pipeline on trained LSTM price model.
    """
    model_dir = Path(model_dir)
    config = _load_model_config(model_dir)

    model_id = extract_model_id(str(model_dir))
    data_id = config.get('data_id', 'UNKNOWN')
    symbol = config.get('symbol', 'eth')
    lookback = int(config.get('lookback', 48))
    horizon = int(config.get('horizon', 12))
    features = config.get('features', [])
    use_log_prices = config.get('use_log_prices', True)

    # 1. Resolve test path (data_id is finalised after loading, when metadata is known)
    test_path = _resolve_test_path(test_path, config)

    # 2. Load scalers
    price_scaler = None
    for p_name in ['price_scaler.pkl', 'scaler_price.pkl']:
        p_path = model_dir / p_name
        if p_path.exists():
            price_scaler = joblib.load(p_path)
            break

    feature_scaler = None
    for f_name in ['feature_scaler.pkl', 'scaler_features.pkl']:
        f_path = model_dir / f_name
        if f_path.exists():
            feature_scaler = joblib.load(f_path)
            break

    # 3. Load test data (strictly enforcing CSV priority and exact length n = N - L - H)
    loaded = load_test_data(
        test_path=str(test_path),
        lookback=lookback,
        horizon=horizon,
        features=features,
        price_scaler=price_scaler,
        feature_scaler=feature_scaler,
        test_file_type=test_file_type,
        use_log_prices=use_log_prices,
    )

    # Directory name -> metadata -> config; 'UNKNOWN' never overwrites a valid ID.
    data_id = resolve_data_id(test_path, loaded.get('metadata'), config.get('data_id'))

    X_test = loaded['X_test']
    current_prices = loaded['current_prices']
    future_prices = loaded['future_prices']
    close_full = loaded['close_full']
    n_samples = loaded['n_samples']

    # 4. Model inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMPriceModel(
        input_size=len(features),
        hidden_size=int(config.get('hidden_size', 64)),
        num_layers=int(config.get('num_layers', 1)),
        dropout=float(config.get('dropout', 0.1)),
    ).to(device)

    model_path = model_dir / 'best_model.pth'
    if not model_path.exists():
        model_path = model_dir / 'lstm_price_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_dir}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            end = min(start + batch_size, len(X_test))
            x_batch = torch.from_numpy(X_test[start:end]).to(device)
            out = model(x_batch).squeeze(-1).cpu().numpy()
            preds.append(np.atleast_1d(out))
    y_pred_scaled = np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)

    # 5. Inversion to implied log-return mu and dollar space
    mu = compute_implied_log_returns(
        y_pred_scaled=y_pred_scaled,
        current_prices=current_prices,
        price_scaler=price_scaler,
        use_log_prices=use_log_prices,
    )
    r_true = compute_actual_log_returns(
        future_prices=future_prices,
        current_prices=current_prices,
    )
    predicted_dollars = convert_predictions_to_dollars(
        y_pred_scaled=y_pred_scaled,
        price_scaler=price_scaler,
        use_log_prices=use_log_prices,
    )

    # 6. Evaluation of Simulation Profile v2
    sim_v2 = evaluate_lstm_simulation_v2(
        y_true=r_true,
        y_pred=mu,
        close=close_full,
        lookback=lookback,
        horizon=horizon,
        threshold=0.001,
        window_sigma=100,
        asset=symbol,
        start_cash=100.0,
        primary_fee_bps=2.0,
        auxiliary_fee_bps=2.0,
        warmup=2000,
        position_mode='sticky',
    )

    # 7. Regression and Price metrics
    log_metrics = calculate_log_return_metrics(r_true, mu, threshold=0.001)
    price_metrics = calculate_price_metrics(future_prices, predicted_dollars)
    dir_acc = calculate_directional_accuracy(future_prices, predicted_dollars, current_prices)

    merged_metrics = {**sim_v2, **log_metrics}
    merged_metrics['mae_price'] = price_metrics['mae']
    merged_metrics['rmse_price'] = price_metrics['rmse']
    merged_metrics['mse_price'] = price_metrics['mse']
    merged_metrics['mape_price'] = price_metrics['mape']
    merged_metrics['r2_price'] = price_metrics['r2']
    merged_metrics['price_dir_acc'] = dir_acc

    # 8. Compute unified Rank Score
    rank_info = compute_rank_score(merged_metrics)
    merged_metrics.update(rank_info)

    # Add metadata
    test_id = get_next_test_id(DATA_DIR)
    merged_metrics['test_id'] = test_id
    merged_metrics['model_id'] = model_id
    merged_metrics['data_id'] = data_id
    merged_metrics['test_file_type'] = test_file_type
    merged_metrics['lookback'] = lookback
    merged_metrics['horizon'] = horizon
    merged_metrics['epochs_trained'] = config.get('epochs_trained')
    merged_metrics['best_epoch'] = config.get('best_epoch')
    merged_metrics['evaluated_at'] = datetime.now().isoformat()

    # 9. Build results DataFrame with authentic Gaussian signals
    sigma_series = compute_sigma_series(close_full, window=100)
    sigma = align_sigma(sigma_series, lookback=lookback, n_samples=n_samples, offset=lookback - 1)
    sig_buy, sig_sell = compute_signals(mu, sigma, threshold=0.001)

    # Extract positions from A60 gross
    model_sim = sim_v2.get('_model_sim', {})
    step_indices = model_sim.get('step_indices', np.arange(0, n_samples, horizon))
    positions = model_sim.get('positions', np.zeros(len(step_indices), dtype=np.int8))
    signal_pos = generate_sticky_signal_pos(step_indices, positions, n_samples, horizon)

    df_results = pd.DataFrame({
        'actual': future_prices,
        'predicted': predicted_dollars,
        'current': current_prices,
        'error_$': predicted_dollars - future_prices,
        'error_%': ((predicted_dollars - future_prices) / np.maximum(future_prices, 1e-8)) * 100.0,
        'actual_log': r_true,
        'predicted_log': mu,
        'signal_buy': sig_buy,
        'signal_sell': sig_sell,
        'signal_pos': signal_pos,
    })

    # 10. Persistence
    date_str = datetime.now().strftime('%d%m')
    results_dir = Path(DATA_DIR) / f"{test_id}_test_results_{model_id}_{data_id}_{date_str}"

    if save_results_flag:
        save_results(results_dir, df_results, merged_metrics, test_id, model_id=model_id)
        if generate_plots:
            plot_test_results(future_prices, predicted_dollars, current_prices, str(results_dir), horizon)
            plot_log_returns_results(r_true, mu, str(results_dir), horizon)
            plot_simulation_equity(sim_v2, str(results_dir), start_cash=100.0)

    summary_text = build_lstm_test_summary(merged_metrics, test_id, model_id)
    print(summary_text)

    return {
        'test_id': test_id,
        'model_id': model_id,
        'results_dir': str(results_dir),
        'rank_score': merged_metrics.get('rank_score'),
        'metrics': merged_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM Price Prediction model.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to trained model directory")
    parser.add_argument('--test_path', type=str, default=None, help="Path to prepared dataset or CSV file")
    parser.add_argument('--test_file_type', type=str, default='test', help="Split type (test, agent, sym)")
    parser.add_argument('--no_plots', action='store_true', help="Disable plot generation")

    args = parser.parse_args()
    test_model(
        model_dir=args.model_dir,
        test_path=args.test_path,
        test_file_type=args.test_file_type,
        generate_plots=not args.no_plots,
    )
