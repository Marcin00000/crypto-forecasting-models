"""
Visualization utilities for LSTM Log-Return pipeline.

Training curves, test result plots, and Optuna plots.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# TRAINING CURVES
# =========================================================================

def plot_training_curves(train_losses, val_losses, best_epoch, output_path):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: list of per-epoch training losses
        val_losses:   list of per-epoch validation losses
        best_epoch:   int — epoch with best val loss (1-indexed)
        output_path:  str — full path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7,
               label=f'Best Epoch ({best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =========================================================================
# TEST RESULTS
# =========================================================================

def plot_test_results(y_true_log, y_pred_log, output_dir, horizon):
    """
    Create test result visualizations for log returns.

    Generates 4 plots:
      1. Time series (actual vs predicted)
      2. Scatter plot
      3. Error distribution
      4. Combined 2x2 with rolling directional accuracy
    """
    print("\nGenerating visualizations...")
    n = len(y_true_log)

    # 1. Full time series
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(y_true_log, 'b-', label='Actual', linewidth=1.0, alpha=0.7)
    ax.plot(y_pred_log, 'r-', label='Predicted', linewidth=1.0, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Log Return')
    ax.set_title(f'Log Returns: Actual vs Predicted (n={n}, t+{horizon})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_returns_timeseries.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ log_returns_timeseries.png")

    # 2. Scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_log, y_pred_log, alpha=0.3, s=10, color='green')
    lo = min(y_true_log.min(), y_pred_log.min())
    hi = max(y_true_log.max(), y_pred_log.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Log Return')
    ax.set_ylabel('Predicted Log Return')
    ax.set_title(f'Scatter (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_returns_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ log_returns_scatter.png")

    # 3. Error distribution
    errors = y_pred_log - y_true_log
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=60, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
    ax.axvline(x=float(np.mean(errors)), color='g', linestyle='--', lw=2,
               label=f'Mean: {np.mean(errors):.6f}')
    ax.set_xlabel('Error (log return)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ error_distribution.png")

    # 4. Combined 2x2
    _plot_combined(y_true_log, y_pred_log, errors, output_dir, n)


def _plot_combined(y_true, y_pred, errors, output_dir, n):
    """Combined 2x2 plot with rolling directional accuracy."""
    n_grid = min(1000, n)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Time series subset
    axes[0, 0].plot(y_true[:n_grid], 'b-', label='Actual', lw=1.0, alpha=0.7)
    axes[0, 0].plot(y_pred[:n_grid], 'r-', label='Predicted', lw=1.0, alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', ls='--', lw=0.8, alpha=0.3)
    axes[0, 0].set_title(f'Time Series (n={n_grid})')
    axes[0, 0].set_ylabel('Log Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter
    axes[0, 1].scatter(y_true, y_pred, alpha=0.3, s=10, color='green')
    axes[0, 1].plot([lo, hi], [lo, hi], 'r--', lw=2)
    axes[0, 1].set_title(f'Scatter (n={n})')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    # Error histogram
    axes[1, 0].hist(errors, bins=60, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].grid(True, alpha=0.3)

    # Rolling directional accuracy
    actual_dir = np.sign(y_true)
    pred_dir = np.sign(y_pred)
    nonzero = (y_true != 0) & (y_pred != 0)
    correct = np.zeros(len(y_true))
    correct[nonzero] = (actual_dir[nonzero] == pred_dir[nonzero]).astype(float)
    window = min(500, n // 4)
    if window > 10:
        rolling = np.convolve(correct, np.ones(window) / window,
                              mode='valid') * 100
        axes[1, 1].plot(rolling, color='purple', linewidth=1.0)
        axes[1, 1].axhline(y=50, color='r', linestyle='--', lw=1.5, alpha=0.7)
        axes[1, 1].set_title(f'Rolling Dir Accuracy (w={window})')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, f'Dir Acc: {correct.mean() * 100:.2f}%',
                        transform=axes[1, 1].transAxes, ha='center',
                        va='center', fontsize=14)
        axes[1, 1].set_title('Directional Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ combined_results.png")


# =========================================================================
# OPTUNA RESULTS
# =========================================================================

def plot_optuna_results(study, output_dir):
    """
    Create Optuna study visualizations.

    Generates:
      1. Optimization history (trial values over time)
      2. Parameter importances (if enough completed trials)
      3. Top trials comparison bar chart
    """
    print("\nGenerating Optuna visualizations...")

    # 1. Optimization history
    fig, ax = plt.subplots(figsize=(12, 6))
    trials = [t for t in study.trials
              if t.state.name == 'COMPLETE']
    if trials:
        values = [t.value for t in trials]
        numbers = [t.number for t in trials]
        ax.plot(numbers, values, 'b.-', alpha=0.6, markersize=4)

        # Running best
        best_vals = []
        current_best = float('inf')
        for v in values:
            current_best = min(current_best, v)
            best_vals.append(current_best)
        ax.plot(numbers, best_vals, 'r-', linewidth=2.0, label='Best')

        ax.set_xlabel('Trial')
        ax.set_ylabel('Objective Value')
        ax.set_title(f'Optimization History ({len(trials)} trials)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optuna_history.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ optuna_history.png")

    # 2. Top trials comparison
    if len(trials) >= 3:
        sorted_trials = sorted(trials, key=lambda t: t.value)[:10]
        fig, ax = plt.subplots(figsize=(12, 6))
        trial_nums = [f"T{t.number}" for t in sorted_trials]
        trial_vals = [t.value for t in sorted_trials]
        bars = ax.barh(trial_nums[::-1], trial_vals[::-1], color='steelblue')
        ax.set_xlabel('Objective Value')
        ax.set_title('Top 10 Trials')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optuna_top_trials.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ optuna_top_trials.png")

    # 3. Parameter distributions (top param values for best trials)
    if len(trials) >= 5:
        try:
            _plot_param_distributions(trials, output_dir)
        except Exception as e:
            print(f"  ⚠ param distributions skipped: {e}")


def _plot_param_distributions(trials, output_dir):
    """Plot parameter value distributions from trials."""
    sorted_trials = sorted(trials, key=lambda t: t.value)
    params = sorted_trials[0].params
    n_params = len(params)

    if n_params == 0:
        return

    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx, param_name in enumerate(sorted(params.keys())):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        values = [t.params.get(param_name) for t in sorted_trials
                  if param_name in t.params]
        objectives = [t.value for t in sorted_trials
                      if param_name in t.params]

        if all(isinstance(v, (int, float)) for v in values if v is not None):
            ax.scatter(values, objectives, alpha=0.5, s=20, color='teal')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Objective')
        else:
            ax.barh(range(len(values[:10])),
                    [objectives[i] for i in range(min(10, len(values)))],
                    tick_label=[str(v)[:15] for v in values[:10]],
                    color='teal')
            ax.set_xlabel('Objective')
            ax.set_ylabel(param_name)
        ax.grid(True, alpha=0.3)

    # Hide unused
    for idx in range(n_params, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.suptitle('Parameter vs Objective', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optuna_param_distributions.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ optuna_param_distributions.png")


# =========================================================================
# FEATURE IMPORTANCE
# =========================================================================

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

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    ax.barh(names, means, xerr=stds, capsize=3, color='#1f77b4',
            alpha=0.85, edgecolor='black', lw=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Mean Delta MAE (Degradation on Permutation)')
    ax.set_title(f'Permutation Importance (Top {len(names)} Features)')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ feature_importance.png")
