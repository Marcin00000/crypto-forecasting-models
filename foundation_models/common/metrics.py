"""
Metryki ewaluacyjne dla modeli fundamentowych szeregów czasowych.

Implementacja w 100% zgodna z definicjami w module LSTM Log-Return:
- Błędy regresyjne: MAE, RMSE, MSE, R², Pseudo-Huber
- Statystyka kierunkowa: DirAcc %, Strong DirAcc % (top 20%)
- Współczynnik kompresji wariancji predykcji: Compression Ratio (CR)
- Korelacja liniowa i rangowa: Pearson r (z p-value), Spearman rho (z p-value)
"""
from __future__ import annotations

import math
from typing import Dict, Any, Optional

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def pseudo_huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Oblicza stratę Pseudo-Hubera: delta^2 * (sqrt(1 + (r/delta)^2) - 1)."""
    r = y_pred - y_true
    return float(np.mean(delta**2 * (np.sqrt(1.0 + (r / delta)**2) - 1.0)))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strong_quantile: float = 0.80,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Oblicza pełny zestaw metryk predykcyjnych i statystycznych dla log-returnów.

    Args:
        y_true: Rzeczywiste zwroty logarytmiczne
        y_pred: Przewidywane zwroty logarytmiczne
        strong_quantile: Odcięcie dla najsilniejszych sygnałów (np. 0.80 = top 20%)
        threshold: Opcjonalny próg ekonomiczny T
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    n = min(len(yt), len(yp))
    yt, yp = yt[:n], yp[:n]

    errors = yp - yt
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((yt - np.mean(yt))**2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    ph1 = pseudo_huber_loss(yt, yp, delta=1.0)
    ph10 = pseudo_huber_loss(yt, yp, delta=10.0)

    true_std = float(np.std(yt))
    pred_std = float(np.std(yp))
    compression_ratio = float(pred_std / max(true_std, 1e-12))

    # Pearson r i Spearman rho z p-value
    if HAS_SCIPY and n > 2 and true_std > 1e-12 and pred_std > 1e-12:
        r_pearson, p_pearson = pearsonr(yp, yt)
        r_spearman, p_spearman = spearmanr(yp, yt)
    else:
        if true_std > 1e-12 and pred_std > 1e-12:
            r_pearson = float(np.corrcoef(yp, yt)[0, 1])
        else:
            r_pearson = 0.0
        p_pearson = float("nan")
        r_spearman = float("nan")
        p_spearman = float("nan")

    # Trafność kierunku (Directional Accuracy)
    nonzero_mask = (yt != 0.0) & (yp != 0.0)
    if nonzero_mask.sum() > 0:
        correct = np.sign(yt[nonzero_mask]) == np.sign(yp[nonzero_mask])
        dir_acc = float(np.mean(correct) * 100.0)
    else:
        dir_acc = 50.0

    # Strong Directional Accuracy (dla najwyższych modułów predykcji)
    abs_preds = np.abs(yp)
    strong_cut = float(np.quantile(abs_preds, strong_quantile)) if n > 0 else 0.0
    strong_mask = nonzero_mask & (abs_preds >= strong_cut)
    if strong_mask.sum() > 0:
        strong_correct = np.sign(yt[strong_mask]) == np.sign(yp[strong_mask])
        strong_dir_acc = float(np.mean(strong_correct) * 100.0)
        strong_count = int(strong_mask.sum())
    else:
        strong_dir_acc = 50.0
        strong_count = 0

    # Metryki progowe (jeśli zdefiniowano threshold)
    t_dir_acc = dir_acc
    if threshold is not None and threshold > 0:
        t_mask = np.abs(yt) >= threshold
        if t_mask.sum() >= 10:
            t_correct = np.sign(yt[t_mask]) == np.sign(yp[t_mask])
            t_dir_acc = float(np.mean(t_correct) * 100.0)

    return {
        "samples": n,
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "pseudo_huber": ph1,
        "pseudo_huber_large": ph10,
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "error_median": float(np.median(errors)),
        "true_mean": float(np.mean(yt)),
        "true_std": true_std,
        "pred_mean": float(np.mean(yp)),
        "pred_std": pred_std,
        "compression_ratio": compression_ratio,
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_rho": float(r_spearman),
        "spearman_p": float(p_spearman),
        "dir_acc": dir_acc,
        "t_dir_acc": t_dir_acc,
        "strong_dir_acc": strong_dir_acc,
        "strong_count": strong_count,
        "strong_threshold": strong_cut,
    }
