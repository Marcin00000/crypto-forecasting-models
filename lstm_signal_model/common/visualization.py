"""
Visualization utilities for LSTM Signal Model test pipeline.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_signal_test_results(y_pred, target_oracle, y_true_lr, output_dir,
                             horizon, min_signal_pct=60.0):
    """Create signal test visualizations (4 PNG files)."""
    print("\nGenerating visualizations...")
    y_pred = np.asarray(y_pred, dtype=np.float64)
    target_oracle = np.asarray(target_oracle, dtype=np.float64)
    n = len(y_pred)
    threshold = min_signal_pct / 100.0

    # 1. Time series — buy and sell channels
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    axes[0].plot(target_oracle[:, 0], 'b-', label='Oracle buy', lw=0.8, alpha=0.7)
    axes[0].plot(y_pred[:, 0], 'r-', label='Pred buy', lw=0.8, alpha=0.7)
    axes[0].axhline(y=threshold, color='gray', ls='--', alpha=0.5,
                    label=f'Threshold {min_signal_pct}%')
    axes[0].set_ylabel('p_buy')
    axes[0].set_title(f'Buy signal: Oracle vs Predicted (n={n}, t+{horizon})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(target_oracle[:, 1], 'b-', label='Oracle sell', lw=0.8, alpha=0.7)
    axes[1].plot(y_pred[:, 1], 'r-', label='Pred sell', lw=0.8, alpha=0.7)
    axes[1].axhline(y=threshold, color='gray', ls='--', alpha=0.5)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('p_sell')
    axes[1].set_title('Sell signal: Oracle vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_timeseries.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  signal_timeseries.png")

    # 2. Scatter — buy and sell
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ch, name in zip(axes, [0, 1], ['Buy', 'Sell']):
        ax.scatter(target_oracle[:, ch], y_pred[:, ch], alpha=0.2, s=8,
                   color='green' if ch == 0 else 'purple')
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect')
        ax.set_xlabel(f'Oracle {name.lower()}')
        ax.set_ylabel(f'Predicted {name.lower()}')
        ax.set_title(f'{name} scatter (n={n})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'signal_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  signal_scatter.png")

    # 3. Error distribution
    err_buy = y_pred[:, 0] - target_oracle[:, 0]
    err_sell = y_pred[:, 1] - target_oracle[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(err_buy, bins=60, alpha=0.7, edgecolor='black', color='steelblue')
    axes[0].axvline(x=0, color='r', ls='--', lw=2)
    axes[0].set_title('Error distribution — Buy')
    axes[0].set_xlabel('pred - oracle')
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(err_sell, bins=60, alpha=0.7, edgecolor='black', color='orange')
    axes[1].axvline(x=0, color='r', ls='--', lw=2)
    axes[1].set_title('Error distribution — Sell')
    axes[1].set_xlabel('pred - oracle')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  error_distribution.png")

    # 4. Combined 2x2
    _plot_combined(y_pred, target_oracle, err_buy, y_true_lr,
                   output_dir, n, threshold)


def plot_simulation_equity(sim_result, output_dir, start_cash=100.0):
    """Plot model equity curve and start-cash line (oracle omitted from PNG)."""
    if sim_result is None:
        return
    model_sim = sim_result.get('_model_sim', {})
    eq_m = model_sim.get('equity_curve')
    if eq_m is None:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq_m, 'r-', label='Model', lw=1.2)
    ax.axhline(y=start_cash, color='gray', ls='--', alpha=0.5, label='Start')
    ax.set_xlabel('Simulation step')
    ax.set_ylabel('Portfolio value ($)')
    ax.set_title('Simulation equity: Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simulation_equity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  simulation_equity.png")


def _plot_combined(y_pred, target_oracle, err_buy, y_true_lr,
                   output_dir, n, threshold):
    """Combined 2x2 plot."""
    n_grid = min(1000, n)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].plot(target_oracle[:n_grid, 0], 'b-', lw=0.8, alpha=0.7,
                    label='Oracle')
    axes[0, 0].plot(y_pred[:n_grid, 0], 'r-', lw=0.8, alpha=0.7,
                    label='Pred')
    axes[0, 0].set_title(f'Buy timeseries (n={n_grid})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(target_oracle[:, 0], y_pred[:, 0], alpha=0.2, s=8,
                       color='green')
    axes[0, 1].plot([0, 1], [0, 1], 'r--', lw=2)
    axes[0, 1].set_title('Buy scatter')
    axes[0, 1].set_xlabel('Target Buy')
    axes[0, 1].set_ylabel('Pred Buy')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(err_buy, bins=60, alpha=0.7, edgecolor='black',
                    color='orange')
    axes[1, 0].axvline(x=0, color='r', ls='--', lw=2)
    axes[1, 0].set_title('Buy error distribution')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(target_oracle[:, 1], y_pred[:, 1], alpha=0.2, s=8,
                       color='crimson')
    axes[1, 1].plot([0, 1], [0, 1], 'b--', lw=2)
    axes[1, 1].set_title('Sell scatter')
    axes[1, 1].set_xlabel('Target Sell')
    axes[1, 1].set_ylabel('Pred Sell')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  combined_results.png")


def plot_feature_importance(importances, output_dir, top_n=25):
    """Plot per-feature permutation importance (mean delta MAE with std error bars)."""
    if not importances:
        return
    print("\nGenerating feature importance plot...")
    sorted_items = sorted(
        importances.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
    names = [kv[0] for kv in reversed(sorted_items)]
    means = [kv[1][0] for kv in reversed(sorted_items)]
    stds = [kv[1][1] for kv in reversed(sorted_items)]

    colors = [
        '#ff7f0e' if name.startswith('submodel_') else '#1f77b4'
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    ax.barh(names, means, xerr=stds, capsize=3, color=colors,
            alpha=0.85, edgecolor='black', lw=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Mean Delta MAE (Degradation on Permutation)')
    ax.set_title(f'Permutation Importance (Top {len(names)} Features)')
    ax.grid(True, alpha=0.3, axis='x')

    if any(name.startswith('submodel_') for name in names):
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', edgecolor='black', label='Base Feature'),
            Patch(facecolor='#ff7f0e', edgecolor='black', label='Submodel Output')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  feature_importance.png")
