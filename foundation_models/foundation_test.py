#!/usr/bin/env python
"""
Główny skrypt ewaluacji testowej (OOS) dla modeli fundamentowych.

Obsługuje:
- Amazon Chronos-2 (w trybie Univariate oraz Multivariate Group Attention z 39 cechami E109)
- Google TimesFM 2.5 (w trybie Eager execution)
- Checkpointy po fine-tuningu
- Kanoniczny zbiór testowy test.csv (OOS)
- Pełną symulację handlową ze $100 kapitału (standard modułu LR)
- Generowanie kompletu wykresów diagnostycznych w formacie PNG

Użycie:
    # Chronos-2 Zero-Shot Univariate na E109 test.csv:
    python foundation_models/foundation_test.py --model_type chronos --model_name amazon/chronos-2 --input_mode single

    # Chronos-2 Zero-Shot Multivariate z 39 cechami Group Attention:
    python foundation_models/foundation_test.py --model_type chronos --model_name amazon/chronos-2 --input_mode features_max

    # Google TimesFM 2.5 Zero-Shot:
    python foundation_models/foundation_test.py --model_type timesfm --model_name google/timesfm-2.5-200m-pytorch
"""
import os
import sys
import argparse
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
COMMON_DIR = os.path.join(SCRIPT_DIR, "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from foundation_config import FoundationConfig
from common.data_loaders import (
    load_canonical_test_data,
    resample_prices,
    prepare_univariate_windows,
    prepare_multivariate_windows,
)
from common.metrics import calculate_metrics
from common.simulation_profile import evaluate_foundation_simulation_v2
from common.visualization import plot_simulation_equity, plot_test_visualizations
from common.reporting import print_evaluation_report, save_metrics_json, save_test_summary_txt
from common.id_utils import create_test_output_dir
from common.chronos_wrapper import ChronosWrapper
from common.timesfm_wrapper import TimesFMWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Ewaluacja testowa OOS modeli fundamentowych (Chronos-2 / TimesFM).")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["chronos", "timesfm"],
        default="chronos",
        help="Typ modelu: 'chronos' lub 'timesfm'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=FoundationConfig.DEFAULT_CHRONOS_MODEL,
        help="Nazwa modelu na HuggingFace lub ścieżka do lokalnego checkpointu.",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        choices=[FoundationConfig.INPUT_MODE_SINGLE, FoundationConfig.INPUT_MODE_FEATURES],
        default=FoundationConfig.INPUT_MODE_SINGLE,
        help="Tryb wejścia dla Chronos: 'single' (univariate) lub 'features_max' (39 cech technicznych).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=FoundationConfig.DEFAULT_DATA_DIR,
        help="Ścieżka do katalogu danych (domyślnie E109).",
    )
    parser.add_argument(
        "--resample",
        type=str,
        choices=["1m", "5m", "15m", "30m", "1h"],
        default="1m",
        help="Interwał czasowy świec (domyślnie 1m).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=FoundationConfig.LOOKBACK,
        help="Długość okna historii w świecach (domyślnie 48).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=FoundationConfig.HORIZON,
        help="Horyzont prognozy w przód (domyślnie 12).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=FoundationConfig.DEFAULT_BATCH_SIZE,
        help="Rozmiar paczki inferencyjnej (domyślnie 64).",
    )
    parser.add_argument(
        "--eval_stride",
        type=int,
        default=1,
        help="Krok przesuwania okna ewaluacji (1 = pełny gęsty przebieg OOS; horizon = okna niezależne).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Opcjonalny limit próbek testowych (np. dla szybkich smoke testów).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Urządzenie: 'cuda' lub 'cpu'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Katalog zapisu wyników i wykresów (domyślnie automatyczny ze znacznikiem czasu).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    clean_m_name = os.path.basename(args.model_name.rstrip("/\\"))
    FoundationConfig.ensure_dirs()
    if args.output_dir:
        out_dir = args.output_dir
        test_id = "MANUAL"
        os.makedirs(out_dir, exist_ok=True)
    else:
        test_id, out_dir = create_test_output_dir(
            base_dir=FoundationConfig.DATA_DIR,
            model_name=f"{args.model_type}_{clean_m_name}",
            data_id="E109",
        )

    print(f"[*] Wczytywanie kanonicznego zbioru testowego z: {args.data_dir}")
    test_df, metadata, scaler = load_canonical_test_data(args.data_dir)

    # Resampling cen jeśli zażądano
    prices = test_df["close"].to_numpy(dtype=np.float64)
    if args.resample != "1m":
        prices = resample_prices(prices, args.resample)
        print(f"[*] Po resamplingu ({args.resample}): {len(prices):,} zagregowanych świec.")

    # Przygotowanie sekwencji
    features_list = metadata.get("features", [])
    if args.input_mode == FoundationConfig.INPUT_MODE_FEATURES and args.model_type == "chronos" and features_list:
        print(f"[*] Przygotowanie danych w trybie MULTIVARIATE ({len(features_list)} cech technicznych z E109)...")
        scale_csv = bool((metadata.get("scalers") or {}).get("scale_csv", True))
        target_ctx, cov_ctx, y_true, curr_prices = prepare_multivariate_windows(
            df=test_df,
            feature_names=features_list,
            scaler=scaler,
            lookback=args.lookback,
            horizon=args.horizon,
            eval_stride=args.eval_stride,
            already_scaled=scale_csv,
        )
    else:
        print(f"[*] Przygotowanie danych w trybie UNIVARIATE (stacjonarne zwroty logarytmiczne)...")
        target_ctx, y_true, curr_prices = prepare_univariate_windows(
            prices=prices,
            lookback=args.lookback,
            horizon=args.horizon,
            eval_stride=args.eval_stride,
        )
        cov_ctx = None

    if args.max_samples is not None and args.max_samples < len(y_true):
        target_ctx = target_ctx[:args.max_samples]
        if cov_ctx is not None:
            cov_ctx = cov_ctx[:args.max_samples]
        y_true = y_true[:args.max_samples]
        curr_prices = curr_prices[:args.max_samples]

    n_samples = len(y_true)
    print(f"[*] Liczba próbek do ewaluacji: {n_samples:,} (Horyzont H={args.horizon}, Lookback W={args.lookback})")

    # Inicjalizacja wybranego modelu
    if args.model_type == "chronos":
        wrapper = ChronosWrapper(model_name=args.model_name, device=args.device)
    else:
        wrapper = TimesFMWrapper(
            model_name=args.model_name,
            context_len=args.lookback,
            horizon_len=args.horizon,
            device=args.device,
        )

    # Inferencja wsadowa
    predictions = []
    total_batches = (n_samples + args.batch_size - 1) // args.batch_size
    print(f"[*] Inferencja modelu {args.model_name} ({total_batches} batchy, batch_size={args.batch_size})...")

    for b_idx in range(total_batches):
        start_i = b_idx * args.batch_size
        end_i = min(start_i + args.batch_size, n_samples)
        b_target = target_ctx[start_i:end_i]

        if args.model_type == "chronos":
            if cov_ctx is not None:
                b_cov = cov_ctx[start_i:end_i]
                b_preds = wrapper.predict_batch_multivariate(b_target, b_cov, horizon=args.horizon)
            else:
                b_preds = wrapper.predict_batch_univariate(b_target, horizon=args.horizon)
        else:
            b_preds = wrapper.predict_batch(b_target, horizon=args.horizon)

        predictions.extend(b_preds)
        if (b_idx + 1) % max(1, total_batches // 5) == 0 or (b_idx + 1) == total_batches:
            print(f"    Przetworzono: {b_idx + 1}/{total_batches} batchy.")

    y_pred = np.asarray(predictions, dtype=np.float64)

    # Obliczenie metryk predykcyjnych
    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        strong_quantile=FoundationConfig.STRONG_QUANTILE,
        threshold=FoundationConfig.SIGNAL_THRESHOLD,
    )

    # Przeprowadzenie symulacji handlowej (Standard modułu LR / Kontrakt v2)
    resample_mins = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(args.resample, 1)
    warmup_steps = max(50, 2000 // resample_mins)

    sim_results = evaluate_foundation_simulation_v2(
        y_true=y_true,
        y_pred=y_pred,
        close_prices=prices,
        lookback=args.lookback,
        horizon=args.horizon,
        threshold=FoundationConfig.SIGNAL_THRESHOLD,
        window_sigma=FoundationConfig.SIGNAL_WINDOW_SIGMA,
        start_cash=FoundationConfig.SIM_START_CASH,
        primary_fee_bps=2.0,
        auxiliary_fee_bps=2.0,
        warmup=warmup_steps,
        position_mode="sticky",
    )
    metrics.update(sim_results)

    # Generowanie wykresów
    print(f"\n  Test ID:    {test_id}")
    print(f"  Katalog:    {out_dir}")
    print("  Generowanie wizualizacji...")
    if "_model_equity_curve" in sim_results:
        plot_simulation_equity(
            equity_curve=sim_results["_model_equity_curve"],
            output_dir=out_dir,
            start_cash=FoundationConfig.SIM_START_CASH,
            title=f"Simulation Equity: {clean_m_name} ({args.input_mode.upper()})",
        )
        print("    [+] simulation_equity.png")

    plot_test_visualizations(
        y_true=y_true,
        y_pred=y_pred,
        output_dir=out_dir,
        horizon=args.horizon,
    )
    print("    [+] log_returns_timeseries.png")
    print("    [+] log_returns_scatter.png")
    print("    [+] error_distribution.png")
    print("    [+] combined_results.png")

    # Prezentacja raportu i zapis JSON
    print_evaluation_report(
        model_name=args.model_name,
        input_mode=args.input_mode,
        metrics=metrics,
        horizon=args.horizon,
        lookback=args.lookback,
        resample=args.resample,
    )

    print("  Zapis wyników i predykcji (standard v2)...")
    json_path = save_metrics_json(metrics=metrics, output_dir=out_dir)
    print("    [+] metrics.json")

    npz_path = os.path.join(out_dir, "predictions.npz")
    np.savez_compressed(
        npz_path,
        y_true=y_true,
        y_pred=y_pred,
        prices=curr_prices,
    )
    print("    [+] predictions.npz (kompaktowy zapis dla replay)")

    summary_path = save_test_summary_txt(
        metrics=metrics,
        output_dir=out_dir,
        test_id=test_id,
        model_name=args.model_name,
        data_id="E109",
        horizon=args.horizon,
        input_mode=args.input_mode,
        test_file_name="test.csv",
    )
    print("    [+] summary.txt")
    print(f"\n[OK] Test {test_id} zakończony pomyślnie. Komplet wyników zapisano w: {out_dir}\n")


if __name__ == "__main__":
    main()
