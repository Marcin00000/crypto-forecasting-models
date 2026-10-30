"""
Visualization utilities for LSTM pipeline
"""
import os
import sys

# Configure console encoding for Windows environments
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error
import optuna
from optuna.trial import TrialState


def plot_training_curves(train_losses, val_losses, *args, **kwargs):
    """Plot training and validation loss curves supporting both 3-arg and 4-arg signatures."""
    if len(train_losses) == 0:
        return

    output_path = kwargs.get('output_path')
    best_val_loss = kwargs.get('best_val_loss')

    if len(args) == 1:
        output_path = args[0]
        best_val_loss = min(val_losses) if val_losses else 0.0
    elif len(args) >= 2:
        if isinstance(args[0], (int, float)):
            best_val_loss, output_path = args[0], args[1]
        else:
            output_path, best_val_loss = args[0], args[1]

    if not output_path:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train', alpha=0.8)
    ax.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    title = 'Training Curves'
    if best_val_loss is not None:
        try:
            title += f' - Best: {float(best_val_loss):.6f}'
        except Exception:
            pass
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def plot_test_results(y_true_price, y_pred_price, current_prices, output_dir, horizon):
    """Create comprehensive test result visualizations"""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Time series
    n = min(500, len(y_true_price))
    x = np.arange(n)
    axes[0, 0].plot(x, y_true_price[:n], 'b-', label='Actual', linewidth=2, alpha=0.8)
    axes[0, 0].plot(x, y_pred_price[:n], 'r-', label='Predicted', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title(f'Actual vs Predicted (first {n} samples, horizon t+{horizon})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    r2 = r2_score(y_true_price[:n], y_pred_price[:n])
    mae = mean_absolute_error(y_true_price[:n], y_pred_price[:n])
    axes[0, 0].text(0.02, 0.98, f'R² = {r2:.4f}\nMAE = ${mae:.2f}',
                    transform=axes[0, 0].transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 2. Scatter plot
    axes[0, 1].scatter(y_true_price, y_pred_price, alpha=0.3, s=10, color='green')
    min_val = min(y_true_price.min(), y_pred_price.min())
    max_val = max(y_true_price.max(), y_pred_price.max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect', alpha=0.7)
    axes[0, 1].set_xlabel('Actual Price ($)')
    axes[0, 1].set_ylabel('Predicted Price ($)')
    axes[0, 1].set_title('Predicted vs Actual')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Error distribution
    errors = y_pred_price - y_true_price
    axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
    axes[1, 0].axvline(x=np.mean(errors), color='g', linestyle='--', lw=2,
                       label=f'Mean: ${np.mean(errors):.2f}')
    axes[1, 0].set_xlabel('Error ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Percentage error
    pct_errors = (errors / (y_true_price + 1e-8)) * 100
    axes[1, 1].hist(pct_errors, bins=50, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
    axes[1, 1].axvline(x=np.mean(pct_errors), color='g', linestyle='--', lw=2,
                       label=f'Mean: {np.mean(pct_errors):.2f}%')
    axes[1, 1].set_xlabel('Error (%)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Percentage Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] results.png")

    # Full time series
    fig, ax = plt.subplots(figsize=(20, 8))
    x_full = np.arange(len(y_true_price))
    ax.plot(x_full, y_true_price, 'b-', label='Actual', lw=1.5, alpha=0.7)
    ax.plot(x_full, y_pred_price, 'r-', label='Predicted', lw=1.5, alpha=0.7)
    ax.fill_between(x_full, y_true_price, y_pred_price, alpha=0.2, color='gray')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'Full Time Series (horizon t+{horizon}, n={len(y_true_price)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] timeseries.png")

    # Directional accuracy
    if len(current_prices) > 0 and not np.all(current_prices == 0):
        actual_dir = np.sign(y_true_price - current_prices)
        pred_dir = np.sign(y_pred_price - current_prices)
        correct = actual_dir == pred_dir
        dir_acc = np.mean(correct) * 100

        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        n_dir = min(200, len(actual_dir))
        x_dir = np.arange(n_dir)
        colors = ['green' if correct[i] else 'red' for i in range(n_dir)]

        axes[0].bar(x_dir, actual_dir[:n_dir], alpha=0.5, color='blue', label='Actual', width=0.8)
        axes[0].scatter(x_dir, pred_dir[:n_dir], c=colors, s=50, label='Predicted', marker='x', linewidths=2)
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Direction (1=UP, -1=DOWN)')
        axes[0].set_title(f'Directional Prediction (first {n_dir} samples)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-1.5, 1.5])
        axes[0].text(0.02, 0.98, f'Accuracy: {dir_acc:.2f}%',
                     transform=axes[0].transAxes, va='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        actual_bin = (actual_dir > 0).astype(int)
        pred_bin = (pred_dir > 0).astype(int)
        cm = confusion_matrix(actual_bin, pred_bin)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'],
                    cbar_kws={'label': 'Count'})
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix - Direction')

        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / total * 100
                axes[1].text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                             ha='center', va='center', fontsize=10, color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'directional.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  [OK] directional.png")

        return dir_acc

    return None


def plot_optuna_results(study, output_dir):
    """Create Optuna optimization visualizations"""
    print("\nGenerating plots...")

    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'history.png'), dpi=150)
        plt.close()
        print("  [OK] history.png")
    except Exception as exc:
        print(f"  [WARN] history.png skipped: {exc}")

    try:
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'importances.png'), dpi=150)
        plt.close()
        print("  [OK] importances.png")
    except Exception as exc:
        print(f"  [WARN] importances.png skipped: {exc}")

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    losses = [t.value for t in completed]

    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(study.best_value, color='r', linestyle='--', lw=2,
               label=f'Best: {study.best_value:.6f}')
    plt.xlabel('Validation Loss')
    plt.ylabel('Frequency')
    plt.title('Trial Results Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution.png'), dpi=150)
    plt.close()
    print("  ✓ distribution.png")


def plot_simulation_equity(sim_result: dict, output_dir: str | os.PathLike, start_cash: float = 100.0) -> None:
    """Plot market simulation portfolio equity curve (Gross and Net) over time."""
    if not sim_result:
        return

    model_sim = sim_result.get('_model_sim', {})
    eq_gross = model_sim.get('equity_curve')
    if eq_gross is None or len(eq_gross) == 0:
        return

    # Try to retrieve net equity curve from A60 variant if available
    eq_net = None
    variants = sim_result.get('simulation_variants', {})
    a60 = variants.get('absolute_60', {})
    net_item = a60.get('net', {}).get('model', {})
    if isinstance(net_item, dict) and 'equity_curve' in net_item:
        eq_net = net_item.get('equity_curve')

    ret_g = sim_result.get('sim_return_model_pct', 0.0)
    ret_n = sim_result.get('sim_return_model_net_pct', 0.0)
    trades = sim_result.get('sim_n_trades_model', model_sim.get('n_trades', 0))

    fig, ax = plt.subplots(figsize=(14, 6))
    x_steps = np.arange(len(eq_gross))
    ax.plot(x_steps, eq_gross, 'b-', label=f'Model Gross ({ret_g:+.2f}%)', lw=1.5, alpha=0.85)
    if eq_net is not None and len(eq_net) == len(eq_gross):
        ax.plot(x_steps, eq_net, 'r-', label=f'Model Net ({ret_n:+.2f}%)', lw=1.5, alpha=0.85)

    ax.axhline(y=start_cash, color='gray', ls='--', alpha=0.7, label=f'Start Cash (${start_cash:.1f})')
    ax.set_xlabel('Simulation Step (decision intervals)')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'Market Simulation Equity: Model Gross vs Net (A60, Trades={trades})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_file = os.path.join(str(output_dir), 'simulation_equity.png')
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ simulation_equity.png")


def plot_log_returns_results(y_true_log: np.ndarray, y_pred_log: np.ndarray, output_dir: str | os.PathLike, horizon: int) -> None:
    """Create standard return-space visualizations for implied log-returns."""
    n = min(len(y_true_log), len(y_pred_log))
    y_true_log = np.asarray(y_true_log[:n], dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log[:n], dtype=np.float64)
    errors = y_pred_log - y_true_log
    out_dir_str = str(output_dir)

    # 1. Full time series
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(y_true_log, 'b-', label='Actual log-return (r_true)', linewidth=1.0, alpha=0.7)
    ax.plot(y_pred_log, 'r-', label='Implied log-return (μ)', linewidth=1.0, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Log Return')
    ax.set_title(f'Implied vs Actual Log Returns (n={n}, horizon t+{horizon})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_str, 'log_returns_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ log_returns_timeseries.png")

    # 2. Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_log, y_pred_log, alpha=0.3, s=10, color='green')
    lo = min(float(y_true_log.min()), float(y_pred_log.min()))
    hi = max(float(y_true_log.max()), float(y_pred_log.max()))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual Log Return')
    ax.set_ylabel('Implied Log Return')
    ax.set_title(f'Scatter: Implied vs Actual Log Return (n={n})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_str, 'log_returns_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ log_returns_scatter.png")

    # 3. Error distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=60, alpha=0.7, edgecolor='black', color='orange')
    ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
    ax.axvline(x=float(np.mean(errors)), color='g', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.6f}')
    ax.set_xlabel('Error (log return)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution (Implied Log Returns)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_str, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ error_distribution.png")

    # 4. Combined 2x2
    n_grid = min(1000, n)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].plot(y_true_log[:n_grid], 'b-', label='Actual', lw=1.0, alpha=0.7)
    axes[0, 0].plot(y_pred_log[:n_grid], 'r-', label='Predicted (μ)', lw=1.0, alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', ls='--', lw=0.8, alpha=0.3)
    axes[0, 0].set_title(f'Time Series (first {n_grid} samples)')
    axes[0, 0].set_ylabel('Log Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_true_log, y_pred_log, alpha=0.3, s=10, color='green')
    axes[0, 1].plot([lo, hi], [lo, hi], 'r--', lw=2)
    axes[0, 1].set_title(f'Scatter (n={n})')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(errors, bins=60, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].grid(True, alpha=0.3)

    actual_dir = np.sign(y_true_log)
    pred_dir = np.sign(y_pred_log)
    nonzero = (y_true_log != 0) & (y_pred_log != 0)
    correct = np.zeros(len(y_true_log))
    correct[nonzero] = (actual_dir[nonzero] == pred_dir[nonzero]).astype(float)
    window = min(500, max(10, n // 4))
    if window > 10:
        rolling = np.convolve(correct, np.ones(window) / window, mode='valid') * 100
        axes[1, 1].plot(rolling, color='purple', linewidth=1.0)
        axes[1, 1].axhline(y=50, color='r', linestyle='--', lw=1.5, alpha=0.7)
        axes[1, 1].set_title(f'Rolling Directional Accuracy (w={window})')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_str, 'combined_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ combined_results.png")
