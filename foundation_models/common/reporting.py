"""
Moduł raportowania wyników ewaluacji modeli fundamentowych.

Obsługuje:
- Prezentację sformatowanego podsumowania w konsoli (standard simulation v2: $gross ($net))
- Zapis pełnego słownika metryk do pliku metrics.json
- Generowanie wierszy tabelarycznych zgodnych ze standardem rankingów projektu
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional
import numpy as np


def print_evaluation_report(
    model_name: str,
    input_mode: str,
    metrics: Dict[str, Any],
    horizon: int = 12,
    lookback: int = 48,
    resample: str = "1m",
) -> None:
    """Prezentuje sformatowaną tabelę wyników w konsoli (standard v2)."""
    print("\n" + "=" * 75)
    print(f" RAPORT EWALUACJI MODELU FUNDAMENTOWEGO ({resample.upper()}, H={horizon}, W={lookback})")
    print(f" Model:               {model_name}")
    print(f" Tryb wejściowy:      {input_mode.upper()}")
    print("=" * 75)
    print(f"Liczba próbek OOS:    {metrics.get('samples', 0):,}")
    print(f"Błąd bezwzględny MAE: {metrics.get('mae', 0.0):.6f}")
    print(f"Błąd kwadratowy RMSE: {metrics.get('rmse', 0.0):.6f}")
    print(f"Compression Ratio CR: {metrics.get('compression_ratio', 0.0):.4f} (optymalnie ~0.5 - 1.2)")
    print("-" * 75)
    p_p = metrics.get("pearson_p", float("nan"))
    p_str = f"{p_p:.3e}" if (p_p == p_p and p_p is not None) else "N/A"
    print(f"Korelacja Pearsona r: {metrics.get('pearson_r', 0.0):+.4f} (p-value: {p_str})")
    s_p = metrics.get("spearman_p", float("nan"))
    s_str = f"{s_p:.3e}" if (s_p == s_p and s_p is not None) else "N/A"
    print(f"Korelacja Spearmana:  {metrics.get('spearman_rho', 0.0):+.4f} (p-value: {s_str})")
    print(f"Trafność kierunku:    {metrics.get('dir_acc', 50.0):.2f}% (baza losowa = 50.0%)")
    print(f"Trafność Strong:      {metrics.get('strong_dir_acc', 50.0):.2f}% (N={metrics.get('strong_count', 0)})")
    print("-" * 75)

    gross_val = metrics.get("sim_final_model", metrics.get("sim_final_absolute_60", 100.0))
    net_val = metrics.get("sim_final_model_net", metrics.get("sim_final_absolute_60_net"))
    net_str = f" (${net_val:.2f} net 2bps)" if net_val is not None else ""

    print("WYNIKI SYMULACJI (standard v2, $100 kapitału, long+short):")
    print(f"  Primary A60:        ${gross_val:.2f} gross{net_str}")
    if "sim_final_absolute_70" in metrics:
        a70_g = metrics.get("sim_final_absolute_70", 100.0)
        a70_n = metrics.get("sim_final_absolute_70_net")
        a70_n_str = f" (${a70_n:.2f} net)" if a70_n is not None else ""
        print(f"  Aux A70:            ${a70_g:.2f} gross{a70_n_str}")
    if "sim_final_zscore_expanding" in metrics:
        z_g = metrics.get("sim_final_zscore_expanding", 100.0)
        z_n = metrics.get("sim_final_zscore_expanding_net")
        z_n_str = f" (${z_n:.2f} net)" if z_n is not None else ""
        print(f"  Aux Z-Score (exp):  ${z_g:.2f} gross{z_n_str}")
    if "sim_final_percentile_expanding" in metrics:
        p_g = metrics.get("sim_final_percentile_expanding", 100.0)
        p_n = metrics.get("sim_final_percentile_expanding_net")
        p_n_str = f" (${p_n:.2f} net)" if p_n is not None else ""
        print(f"  Aux Pct 5% (exp):   ${p_g:.2f} gross{p_n_str}")

    n_trades = metrics.get("sim_n_trades_model", metrics.get("sim_n_trades", 0))
    print(f"  Liczba transakcji:  {n_trades}")
    pct_s = metrics.get("pct_strong")
    if pct_s is not None:
        print(f"  Aktywność (strong): {pct_s:.2f}%")
    cap_val = metrics.get("sim_capture_log")
    cap_str = f"{cap_val:.5f}" if cap_val is not None else "N/A"
    print(f"  Oracle capture log: {cap_str}")
    print(f"  Zunifikowany Rank:  {metrics.get('rank_score', 0.0):.2f} pt ({metrics.get('signal_class', 'UNKNOWN')})")
    print("=" * 75 + "\n")


def save_metrics_json(metrics: Dict[str, Any], output_dir: str, filename: str = "metrics.json") -> str:
    """Zapisuje słownik metryk do pliku JSON."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    clean_metrics = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (np.floating, float)):
            clean_metrics[k] = None if (v != v) else float(v)
        elif isinstance(v, (np.integer, int)):
            clean_metrics[k] = int(v)
        elif isinstance(v, np.ndarray):
            clean_metrics[k] = v.tolist()
        else:
            clean_metrics[k] = v

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_metrics, f, indent=2, ensure_ascii=False)

    return out_path


def format_ranking_row(
    model_id: str,
    dataset_id: str,
    metrics: Dict[str, Any],
    params_summary: str = "",
) -> str:
    """Generuje wiersz tabeli w standardzie rankingów v2 ($gross ($net))."""
    t_dir = f"{metrics.get('t_dir_acc', metrics.get('dir_acc', 50.0)):.1f}%"
    mae = f"{metrics.get('mae', 0.0):.4f}"
    cr = f"{metrics.get('compression_ratio', 0.0):.3f}"
    rank = f"{metrics.get('rank_score', 0.0):.2f}"

    gross_val = metrics.get("sim_final_model", metrics.get("sim_final_absolute_60", 100.0))
    net_val = metrics.get("sim_final_model_net", metrics.get("sim_final_absolute_60_net"))
    if net_val is not None:
        sim_cash = f"${gross_val:.2f} (${net_val:.2f})"
    else:
        sim_cash = f"${gross_val:.2f}"

    cap = metrics.get("sim_capture_log")
    sim_cap = f"{cap:.5f}" if cap is not None else "—"
    sig_cls = metrics.get("signal_class", "UNKNOWN")

    return (
        f"| {model_id} | {dataset_id} | {t_dir} | {mae} | {cr} | {rank} | "
        f"{sim_cash} | {sim_cap} | {sig_cls} | {params_summary} |"
    )


def save_test_summary_txt(
    metrics: Dict[str, Any],
    output_dir: str,
    test_id: str = "FMT001",
    model_name: str = "Chronos-2",
    data_id: str = "E109",
    horizon: int = 12,
    input_mode: str = "single",
    test_file_name: str = "test.csv",
) -> str:
    """Zapisuje sformatowane podsumowanie tekstowe summary.txt w standardzie LR/Signal v2."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.txt")

    p_p = metrics.get("pearson_p", float("nan"))
    p_str = f"{p_p:.4e}" if (p_p == p_p and p_p is not None) else "n/a"
    s_p = metrics.get("spearman_p", float("nan"))
    s_str = f"{s_p:.4e}" if (s_p == s_p and s_p is not None) else "n/a"
    cap_val = metrics.get("sim_capture_log")
    cap_str = f"{cap_val:.5f}" if cap_val is not None else "n/a"

    gross_val = metrics.get("sim_final_model", metrics.get("sim_final_absolute_60", 100.0))
    net_val = metrics.get("sim_final_model_net", metrics.get("sim_final_absolute_60_net"))
    if net_val is not None:
        cash_str = f"${gross_val:.2f} (${net_val:.2f} net 2bps)"
    else:
        cash_str = f"${gross_val:.2f}"

    lines = [
        "=" * 75,
        "FOUNDATION MODEL TEST SUMMARY (SIMULATION V2)",
        "=" * 75,
        "",
        f"Test ID:     {test_id}",
        f"Model:       {model_name}",
        f"Input Mode:  {input_mode}",
        f"Data ID:     {data_id} ({test_file_name})",
        f"Horizon:     t+{horizon}",
        f"Samples:     {metrics.get('samples', 0):,}",
        "",
        "METRICS (log-return space):",
        f"  MAE:       {metrics.get('mae', 0.0):.6f}",
        f"  RMSE:      {metrics.get('rmse', 0.0):.6f}",
        f"  MSE:       {metrics.get('mse', 0.0):.8f}",
        f"  R2:        {metrics.get('r2', 0.0):.4f}",
        f"  CR:        {metrics.get('compression_ratio', 0.0):.4f} (pred/true std)",
        f"  Pearson r: {metrics.get('pearson_r', 0.0):+.4f} (p={p_str})",
        f"  Spearman:  {metrics.get('spearman_rho', 0.0):+.4f} (p={s_str})",
        f"  Dir Acc:   {metrics.get('dir_acc', 50.0):.2f}%",
        f"  t_Dir Acc: {metrics.get('t_dir_acc', metrics.get('dir_acc', 50.0)):.2f}%",
        f"  Strong Acc:{metrics.get('strong_dir_acc', 50.0):.2f}% (N={metrics.get('strong_count', 0)})",
        "",
        "-" * 75,
        "SIGNAL + SIMULATION V2 ($100 capital, long+short):",
        "-" * 75,
        f"  Primary A60 Cash: {cash_str}",
        f"  Total Trades:     {metrics.get('sim_n_trades_model', metrics.get('sim_n_trades', 0))}",
        f"  Sim Steps:        {metrics.get('sim_n_steps', 0)}",
        f"  Pct Strong:       {metrics.get('pct_strong', 0.0):.2f}%",
        f"  Sim Capture Log:  {cap_str}",
        f"  Rank Score:       {metrics.get('rank_score', 0.0):.2f} pt",
        f"  Signal Class:     {metrics.get('signal_class', 'UNKNOWN')}",
        "=" * 75,
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return summary_path
