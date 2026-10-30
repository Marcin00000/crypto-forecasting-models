"""
Moduł wizualizacji wyników ewaluacji modeli fundamentowych.

Generuje wykresy PNG w standardzie projektu:
- simulation_equity.png: Krzywa kapitału SAMEGO MODELU w czasie (bez abstrakcyjnego Oracle)
- log_returns_timeseries.png: Przebieg rzeczywisty vs prognoza
- log_returns_scatter.png: Wykres rozrzutu z linią idealnej predykcji
- error_distribution.png: Histogram błędów resztowych
- combined_results.png: Panel 2x2 z kroczącą trafnością kierunku
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_simulation_equity(
    equity_curve: List[float],
    output_dir: str,
    start_cash: float = 100.0,
    title: str = "Simulation Equity: Model Curve",
) -> str:
    """
    Generates equity curve plot for the MODEL alone over time (starting from $100).
    Oracle curve is deliberately omitted to prevent Y-axis scale distortion.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    steps = list(range(len(equity_curve)))
    final_val = equity_curve[-1] if equity_curve else start_cash
    ret_pct = (final_val / start_cash - 1.0) * 100.0

    ax.plot(steps, equity_curve, color="#2ca02c" if final_val >= start_cash else "#d62728",
            linewidth=1.8, label=f"Model Equity (Final: ${final_val:.2f}, {ret_pct:+.2f}%)")
    ax.axhline(y=start_cash, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
               label=f"Start Capital (${start_cash:.2f})")

    ax.set_xlabel("Simulation Step", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "simulation_equity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_test_visualizations(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    horizon: int = 12,
    max_timeseries_points: int = 2000,
) -> None:
    """
    Generates complete diagnostic test plots in English (aligned with LR/Signal standard):
    - log_returns_timeseries.png
    - log_returns_scatter.png
    - error_distribution.png
    - combined_results.png
    """
    os.makedirs(output_dir, exist_ok=True)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]
    errors = yp - yt

    # 1. Full time series
    n_plot = min(n, max_timeseries_points)
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(yt[:n_plot], color="#1f77b4", label="Actual", linewidth=1.0, alpha=0.75)
    ax.plot(yp[:n_plot], color="#ff7f0e", label="Predicted", linewidth=1.2, alpha=0.85)
    ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Sample", fontsize=11)
    ax.set_ylabel("Log Return", fontsize=11)
    ax.set_title(f"Log Returns: Actual vs Predicted (n={n_plot:,}, t+{horizon})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "log_returns_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    if n > 5000:
        idx_scatter = np.random.choice(n, size=5000, replace=False)
        s_yt, s_yp = yt[idx_scatter], yp[idx_scatter]
    else:
        s_yt, s_yp = yt, yp

    ax.scatter(s_yt, s_yp, alpha=0.25, s=8, color="#2ca02c")
    lim_min = min(float(np.min(yt)), float(np.min(yp)))
    lim_max = max(float(np.max(yt)), float(np.max(yp)))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual Log Return", fontsize=11)
    ax.set_ylabel("Predicted Log Return", fontsize=11)
    ax.set_title(f"Scatter (n={n:,})", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "log_returns_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Error distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=60, color="#17becf", edgecolor="black", alpha=0.7)
    ax.axvline(x=0.0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax.axvline(x=float(np.mean(errors)), color="green", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(errors):.6f}")
    ax.set_xlabel("Error (log return)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Error Distribution", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Combined 2x2 plot
    n_grid = min(1000, n)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    # Top-left: time series
    axes[0, 0].plot(yt[:n_grid], label="Actual", color="#1f77b4", lw=1.0, alpha=0.7)
    axes[0, 0].plot(yp[:n_grid], label="Predicted", color="#ff7f0e", lw=1.0, alpha=0.85)
    axes[0, 0].axhline(y=0.0, color="black", ls="--", lw=0.8, alpha=0.3)
    axes[0, 0].set_title(f"Time Series (n={n_grid})")
    axes[0, 0].set_ylabel("Log Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: scatter
    axes[0, 1].scatter(s_yt, s_yp, alpha=0.2, s=6, color="#2ca02c")
    axes[0, 1].plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.2)
    axes[0, 1].set_title(f"Scatter (n={n:,})")
    axes[0, 1].set_xlabel("Actual")
    axes[0, 1].set_ylabel("Predicted")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: error distribution
    axes[1, 0].hist(errors, bins=60, color="#17becf", edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(x=0.0, color="r", ls="--", lw=1.5)
    axes[1, 0].set_title("Error Distribution")
    axes[1, 0].set_xlabel("Error")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: rolling directional accuracy
    nonzero = (yt != 0.0) & (yp != 0.0)
    correct = np.zeros(n)
    correct[nonzero] = (np.sign(yt[nonzero]) == np.sign(yp[nonzero])).astype(float)
    window = min(500, max(50, n // 20))
    if n > window:
        rolling_acc = np.convolve(correct, np.ones(window) / window, mode="valid") * 100.0
        axes[1, 1].plot(rolling_acc, color="#9467bd", lw=1.2)
        axes[1, 1].axhline(y=50.0, color="r", ls="--", lw=1.2, label="Random 50%")
        axes[1, 1].set_title(f"Rolling Dir Accuracy (w={window})")
        axes[1, 1].set_xlabel("Sample")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, f"Dir Acc: {correct.mean() * 100:.2f}%",
                        transform=axes[1, 1].transAxes, ha="center", va="center", fontsize=14)
        axes[1, 1].set_title("Directional Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_results.png"), dpi=150, bbox_inches="tight")
    plt.close()
