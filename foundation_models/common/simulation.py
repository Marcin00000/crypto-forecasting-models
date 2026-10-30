"""
Symulacja rynkowa i probabilistyczny model sygnałowy dla modeli fundamentowych.

Wdrożenie w 100% ustandaryzowane ze specyfikacją modułu LSTM Log-Return (CP-153):
- Probabilistyczne sygnały BUY / SELL z rozkładu normalnego Gaussa:
    signal_buy  = P(r >  T | mu, sigma) * 100
    signal_sell = P(r < -T | mu, sigma) * 100
- Progi causal-expanding bez dostępu do przyszłych danych (brak look-ahead bias)
- Symulacja portfela Long/Short ze startowym kapitałem $100
- Prawidłowe naliczanie opłat: 2 nogi przy flipie pozycji, 1 noga przy wejściu/wyjściu z flat
- Pełny 5-komponentowy rank_score (0-100 pkt) z twardym gatingiem zapaści predykcji (DEAD)
"""
from __future__ import annotations

import math
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from scipy.special import ndtr


def compute_sigma_series(close: np.ndarray, window: int = 100) -> np.ndarray:
    """Kroczące odchylenie standardowe 1-krokowych zwrotów logarytmicznych."""
    s = pd.Series(np.asarray(close, dtype=np.float64))
    log_ret = np.log(s / s.shift(1))
    sigma = log_ret.rolling(window=window, min_periods=window).std()
    return sigma.to_numpy()


def align_sigma(
    sigma_series: np.ndarray, lookback: int, n_samples: int, fill_nan: bool = True
) -> np.ndarray:
    """
    Dopasowuje zmienność do ostatniej widocznej świecy w oknie próbki i.
    Dla próbki i ostatnia świeca to i + lookback - 1.

    fill_nan=False zachowuje NaN z okresu rozgrzewki (kontrakt profilu v2 zgodny z LR:
    kroki bez historii zmienności są nieaktywne).
    """
    idx = np.arange(n_samples) + (lookback - 1)
    max_idx = len(sigma_series) - 1
    idx = np.clip(idx, 0, max_idx)
    sigma_aligned = sigma_series[idx]
    if not fill_nan:
        return sigma_aligned

    finite = sigma_aligned[np.isfinite(sigma_aligned)]
    if len(finite) > 0:
        fill_val = float(np.median(finite))
        sigma_aligned = np.where(np.isfinite(sigma_aligned), sigma_aligned, fill_val)
    else:
        sigma_aligned = np.where(np.isfinite(sigma_aligned), sigma_aligned, 0.001)

    return sigma_aligned


def fill_sigma_nan(sigma: np.ndarray) -> np.ndarray:
    """Uzupełnia wartości NaN w aligned sigma medianą wartości skończonych."""
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    finite = sigma_arr[np.isfinite(sigma_arr)]
    if len(finite):
        fill = float(np.median(finite))
        return np.where(np.isfinite(sigma_arr), sigma_arr, fill)
    return sigma_arr


def compute_signals(mu: np.ndarray, sigma: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generuje probabilistyczne sygnały BUY i SELL w skali [0, 100].

    signal_buy  = P(r >  threshold | mu, sigma) * 100
    signal_sell = P(r < -threshold | mu, sigma) * 100
    """
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    signal_buy = (1.0 - ndtr((threshold - mu_arr) / sigma_safe)) * 100.0
    signal_sell = ndtr((-threshold - mu_arr) / sigma_safe) * 100.0
    return np.clip(signal_buy, 0.0, 100.0), np.clip(signal_sell, 0.0, 100.0)


def _expanding_zscore_threshold(values: np.ndarray, cutoff: float, warmup: int) -> np.ndarray:
    """Causal expanding mean/std bazujący wyłącznie na obserwacjach ściśle przed t."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    sums = np.concatenate(([0.0], np.cumsum(clean)))
    sums2 = np.concatenate(([0.0], np.cumsum(clean * clean)))
    counts = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    idx = np.arange(len(values))
    count = counts[idx]
    valid = count >= int(warmup)
    mean = np.divide(sums[idx], count, out=np.full(len(values), np.nan), where=valid)
    second = np.divide(sums2[idx], count, out=np.full(len(values), np.nan), where=valid)
    std = np.sqrt(np.maximum(second - mean * mean, 0.0))
    return mean + float(cutoff) * std


def _expanding_quantile_threshold(values: np.ndarray, percentile_cutoff: float, warmup: int) -> np.ndarray:
    """Causal expanding quantile przesunięty tak, by krok t nigdy nie był uwzględniany w progu."""
    quantile = 1.0 - float(percentile_cutoff) / 100.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile_cutoff musi mieścić się w przedziale [0, 100].")
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .expanding(min_periods=int(warmup))
        .quantile(quantile, interpolation="linear")
        .shift(1)
        .to_numpy(dtype=np.float64)
    )


def compute_signal_thresholds(
    signal_buy: np.ndarray,
    signal_sell: np.ndarray,
    *,
    mode: str = "absolute",
    min_signal: float = 60.0,
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    history: str = "full",
    warmup: int = 0,
) -> Dict[str, Any]:
    """Wyznacza serie progów odcięcia dla sygnałów kupna i sprzedaży."""
    buy = np.asarray(signal_buy, dtype=np.float64)
    sell = np.asarray(signal_sell, dtype=np.float64)
    n = min(len(buy), len(sell))
    buy, sell = buy[:n], sell[:n]

    if mode not in {"absolute", "percentile", "zscore"}:
        raise ValueError(f"Nieobsługiwany tryb progów: {mode}")
    if history not in {"full", "expanding"}:
        raise ValueError(f"Nieobsługiwany tryb historii progów: {history}")
    if int(warmup) < 0:
        raise ValueError("warmup musi być nieujemny")

    if mode == "absolute":
        buy_t = np.full(n, float(min_signal), dtype=np.float64)
        sell_t = np.full(n, float(min_signal), dtype=np.float64)
    elif history == "expanding":
        if mode == "zscore":
            buy_t = _expanding_zscore_threshold(buy, zscore_cutoff, warmup)
            sell_t = _expanding_zscore_threshold(sell, zscore_cutoff, warmup)
        else:
            buy_t = _expanding_quantile_threshold(buy, percentile_cutoff, warmup)
            sell_t = _expanding_quantile_threshold(sell, percentile_cutoff, warmup)
    else:
        finite_buy = buy[np.isfinite(buy)]
        finite_sell = sell[np.isfinite(sell)]
        if mode == "percentile":
            q = 1.0 - float(percentile_cutoff) / 100.0
            b_val = float(np.quantile(finite_buy, q)) if len(finite_buy) else np.nan
            s_val = float(np.quantile(finite_sell, q)) if len(finite_sell) else np.nan
        else:
            b_val = float(np.mean(finite_buy) + float(zscore_cutoff) * np.std(finite_buy)) if len(finite_buy) else np.nan
            s_val = float(np.mean(finite_sell) + float(zscore_cutoff) * np.std(finite_sell)) if len(finite_sell) else np.nan
        buy_t = np.full(n, b_val, dtype=np.float64)
        sell_t = np.full(n, s_val, dtype=np.float64)

    eligible = np.arange(n) >= int(warmup)
    eligible &= np.isfinite(buy) & np.isfinite(sell)
    eligible &= np.isfinite(buy_t) & np.isfinite(sell_t)
    return {
        "buy": buy_t,
        "sell": sell_t,
        "eligible": eligible,
        "mode": mode,
        "history": history,
        "warmup": int(warmup),
    }


def _last_finite(values) -> Optional[float]:
    if values is None:
        return None
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite[-1]) if finite.size else None


def _compute_thresholds(
    signal_buy: np.ndarray,
    signal_sell: np.ndarray,
    *,
    mode: str = "absolute",
    min_signal: float = 60.0,
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
) -> Tuple[float, float]:
    """Statyczne progi na całym oknie (zachowane dla wstecznej kompatybilności pojedynczych testów)."""
    if mode == "percentile":
        p_buy = float(np.percentile(signal_buy, 100.0 - percentile_cutoff))
        p_sell = float(np.percentile(signal_sell, 100.0 - percentile_cutoff))
        return p_buy, p_sell
    elif mode == "zscore":
        b_mean, b_std = float(np.mean(signal_buy)), float(np.std(signal_buy))
        s_mean, s_std = float(np.mean(signal_sell)), float(np.std(signal_sell))
        return b_mean + zscore_cutoff * b_std, s_mean + zscore_cutoff * s_std
    else:
        return float(min_signal), float(min_signal)


def compute_signal_metrics(
    mu: np.ndarray,
    y_true: np.ndarray,
    sigma: np.ndarray,
    threshold: float,
    min_signal: float = 60.0,
    threshold_mode: str = "absolute",
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    threshold_history: str = "full",
    warmup: int = 0,
    eligible: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Metryki jakości sygnałów: aktywność, kalibracja probabilistyczna i kierunkowość."""
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = min(len(mu), len(y_true), len(sigma))
    mu, y_true, sigma = mu[:n], y_true[:n], sigma[:n]

    signal_buy, signal_sell = compute_signals(mu, sigma, threshold)
    thresholds = compute_signal_thresholds(
        signal_buy, signal_sell, mode=threshold_mode, min_signal=min_signal,
        percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
        history=threshold_history, warmup=warmup,
    )
    mask = np.asarray(thresholds["eligible"] if eligible is None else eligible, dtype=bool)[:n]
    n_el = int(mask.sum())
    buy_el = signal_buy[mask]
    sell_el = signal_sell[mask]
    yt_el = y_true[mask]
    buy_t = thresholds["buy"][mask]
    sell_t = thresholds["sell"][mask]

    strong_buy = buy_el >= buy_t
    strong_sell = sell_el >= sell_t
    strong_any = strong_buy | strong_sell
    pct_strong = float(strong_any.mean() * 100) if n_el else 0.0

    calib_buy = float((yt_el[strong_buy] > threshold).mean()) if strong_buy.any() else None
    calib_sell = float((yt_el[strong_sell] < -threshold).mean()) if strong_sell.any() else None

    if strong_any.any():
        pred_dir = np.where(buy_el >= sell_el, 1.0, -1.0)[strong_any]
        true_dir = np.sign(yt_el[strong_any])
        nz = true_dir != 0
        dir_acc_strong = float((pred_dir[nz] == true_dir[nz]).mean() * 100) if nz.any() else None
    else:
        dir_acc_strong = None

    return {
        "mu_std": float(np.std(mu[np.isfinite(sigma)])) if n else 0.0,
        "signal_buy_std": float(np.std(buy_el)) if n_el else 0.0,
        "signal_sell_std": float(np.std(sell_el)) if n_el else 0.0,
        "signal_min_strong": float(min_signal),
        "threshold_mode_eval": threshold_mode,
        "threshold_history_eval": threshold_history,
        "threshold_warmup_eval": int(warmup),
        "thresh_buy_eval": _last_finite(thresholds["buy"]),
        "thresh_sell_eval": _last_finite(thresholds["sell"]),
        "signal_window_sigma_valid_pct": float(np.isfinite(sigma).mean() * 100) if n else 0.0,
        "pct_strong": pct_strong,
        "pct_strong_buy": float(strong_buy.mean() * 100) if n_el else 0.0,
        "pct_strong_sell": float(strong_sell.mean() * 100) if n_el else 0.0,
        "n_strong_buy": int(strong_buy.sum()),
        "n_strong_sell": int(strong_sell.sum()),
        "eligible_n": n_el,
        "calib_buy": calib_buy,
        "calib_sell": calib_sell,
        "n_strong": int(strong_any.sum()),
        "dir_acc_strong": dir_acc_strong,
        "_buy_thresholds": thresholds["buy"],
        "_sell_thresholds": thresholds["sell"],
        "_threshold_eligible": mask,
    }


def simulate_long_short(
    mu: np.ndarray,
    y_true: np.ndarray,
    sigma: np.ndarray,
    threshold: float,
    min_signal: float = 60.0,
    horizon: int = 12,
    start_cash: float = 100.0,
    fee_bps: float = 0.0,
    threshold_mode: str = "absolute",
    percentile_cutoff: float = 5.0,
    zscore_cutoff: float = 1.6,
    position_mode: str = "sticky",
    threshold_history: str = "full",
    warmup: int = 0,
    thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Symulacja portfela Long/Short ze $100 kapitału początkowego.

    Prowadzona na niepokrywających się krokach horyzontu (step = horizon).
    Poprawnie nalicza opłaty: 2 nogi przy odwróceniu pozycji (+1 -> -1), 1 noga przy wejściu/wyjściu.
    """
    mu = np.asarray(mu, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = min(len(mu), len(y_true), len(sigma))

    signal_buy, signal_sell = compute_signals(mu[:n], sigma[:n], threshold)
    if thresholds is None:
        thresholds = compute_signal_thresholds(
            signal_buy, signal_sell, mode=threshold_mode, min_signal=min_signal,
            percentile_cutoff=percentile_cutoff, zscore_cutoff=zscore_cutoff,
            history=threshold_history, warmup=warmup,
        )

    step = max(int(horizon), 1)
    fee_factor = float(fee_bps) * 1e-4
    cash = float(start_cash)
    position = prev_position = n_trades = n_fee_legs = n_steps = 0
    n_long = n_short = n_flat = 0
    equity_curve = [cash]
    positions_history = [0]

    for i in range(0, n, step):
        if not thresholds["eligible"][i] or not np.isfinite(sigma[i]):
            position = 0
        else:
            b, s = signal_buy[i], signal_sell[i]
            buy_th = thresholds["buy"][i]
            sell_th = thresholds["sell"][i]
            if b >= buy_th and b >= s:
                position = 1
            elif s >= sell_th:
                position = -1
            else:
                if position_mode == "reset":
                    position = 0

        if position != prev_position:
            n_trades += 1
            legs = 1 if (prev_position == 0 or position == 0) else 2
            n_fee_legs += legs
            if fee_bps > 0:
                cash *= (1.0 - fee_factor) ** legs
            prev_position = position

        r = y_true[i]
        if np.isfinite(r) and position != 0:
            cash *= float(np.exp(position * r))

        n_steps += 1
        if position == 1:
            n_long += 1
        elif position == -1:
            n_short += 1
        else:
            n_flat += 1

        equity_curve.append(cash)
        positions_history.append(position)

    return {
        "final_value": cash,
        "return_pct": (cash / start_cash - 1.0) * 100.0,
        "n_trades": n_trades,
        "n_fee_legs": n_fee_legs,
        "n_steps": n_steps,
        "n_long": n_long,
        "n_short": n_short,
        "n_flat": n_flat,
        "equity_curve": equity_curve,
        "positions_history": positions_history,
        "buy_thresh": _last_finite(thresholds["buy"]),
        "sell_thresh": _last_finite(thresholds["sell"]),
        "_buy_thresholds": thresholds["buy"],
        "_sell_thresholds": thresholds["sell"],
        "_threshold_eligible": thresholds["eligible"],
    }


def derive_sim_capture_log(
    sim_final_model: float,
    sim_final_oracle: float,
    sim_start_cash: float = 100.0,
) -> Optional[float]:
    """Współczynnik przechwycenia wyroczni w skali logarytmicznej (v2)."""
    if sim_final_oracle <= sim_start_cash * 1.001:
        return None
    if sim_final_model <= 0:
        return 0.0
    return math.log(sim_final_model / sim_start_cash) / math.log(sim_final_oracle / sim_start_cash)


# =========================================================================
# RANK SCORE (kontrakt ustandaryzowany z modułem LR)
# =========================================================================
RANK_CR_DEAD, RANK_PCT_STRONG_DEAD, RANK_PCT_STRONG_FULL = 0.1, 1.0, 20.0
RANK_CALIB_FLOOR, RANK_CALIB_FULL, RANK_DIRACC_FULL = 0.5, 0.7, 60.0
RANK_CR_HEALTH_SIGMA, RANK_CAPTURE_LOG_FULL = 0.5, 0.10
RANK_W_ACTIVITY, RANK_W_CALIBRATION, RANK_W_DIRECTION = 25.0, 25.0, 20.0
RANK_W_CAPTURE, RANK_W_CR_HEALTH = 25.0, 5.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def compute_rank_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Oblicza ustandaryzowany rank_score (0-100 pkt) w oparciu o 5 składowych:
    Activity (25), Calibration (25), Direction (20), Capture (25), CR Health (5).
    """
    cr = metrics.get("compression_ratio")
    pct_strong = metrics.get("pct_strong")

    if pct_strong is None or cr is None:
        return {"rank_score": None, "signal_class": "UNKNOWN_SIGNAL"}

    # Gating zapaści modelu: płaski model lub brak silnych sygnałów
    if cr < RANK_CR_DEAD or pct_strong < RANK_PCT_STRONG_DEAD:
        return {
            "rank_score": 0.0, "signal_class": "DEAD",
            "rank_components": {
                "activity": 0.0, "calibration": 0.0, "direction": 0.0, "capture": 0.0, "cr_health": 0.0,
            },
        }

    # 1. Aktywność sygnałowa
    activity = _clamp01(pct_strong / RANK_PCT_STRONG_FULL) * RANK_W_ACTIVITY

    # 2. Kalibracja probabilistyczna
    calibs = [c for c in (metrics.get("calib_buy"), metrics.get("calib_sell")) if c is not None]
    if calibs:
        avg_calib = sum(calibs) / len(calibs)
        calib_norm = _clamp01((avg_calib - RANK_CALIB_FLOOR) / (RANK_CALIB_FULL - RANK_CALIB_FLOOR))
    else:
        calib_norm = 0.0
    calibration = calib_norm * RANK_W_CALIBRATION

    # 3. Trafność kierunku na silnych sygnałach
    dir_acc_strong = metrics.get("dir_acc_strong")
    if dir_acc_strong is not None:
        direction = _clamp01((dir_acc_strong - 50.0) / (RANK_DIRACC_FULL - 50.0)) * RANK_W_DIRECTION
    else:
        direction = 0.0

    # 4. Przechwycenie wyroczni (skala logarytmiczna)
    capture_log = metrics.get("sim_capture_log")
    if capture_log is not None:
        capture = _clamp01(capture_log / RANK_CAPTURE_LOG_FULL) * RANK_W_CAPTURE
    else:
        cap_ratio = metrics.get("sim_capture_ratio")
        capture = _clamp01(cap_ratio) * RANK_W_CAPTURE if cap_ratio is not None else 0.0

    # 5. Zdrowie kompresji wariancji (CR wokół 1.0)
    cr_health = math.exp(-0.5 * ((cr - 1.0) / RANK_CR_HEALTH_SIGMA) ** 2) * RANK_W_CR_HEALTH

    # Bez zaokrągleń i z klasą zależną od kalibracji – identycznie jak w rdzeniu LR.
    rank_score = float(activity + calibration + direction + capture + cr_health)

    if not calibs:
        signal_class = "UNKNOWN_SIGNAL"
    elif all(c >= 0.55 for c in calibs):
        signal_class = "TRADEABLE_GOOD"
    else:
        signal_class = "TRADEABLE_UNCAL"

    return {
        "rank_score": rank_score,
        "signal_class": signal_class,
        "ranking_method": "absolute_60_gross",
        "rank_metric_version": 2 if capture_log is not None else 1,
        "rank_components": {
            "activity": float(activity), "calibration": float(calibration),
            "direction": float(direction), "capture": float(capture),
            "cr_health": float(cr_health),
        },
    }


def evaluate_signal_and_simulation(
    y_true: np.ndarray, y_pred: np.ndarray, close_prices: np.ndarray,
    lookback: int = 48, horizon: int = 12, threshold: float = 0.001,
    window_sigma: int = 100, min_signal: float = 60.0, start_cash: float = 100.0,
    fee_bps: float = 0.0, threshold_mode: str = "absolute",
    percentile_cutoff: float = 5.0, zscore_cutoff: float = 1.6,
    position_mode: str = "sticky",
) -> Dict[str, Any]:
    """
    Funkcja pomostowa kompatybilności — przekierowuje do pełnego profilu simulation_v2.
    """
    from .simulation_profile import evaluate_foundation_simulation_v2
    return evaluate_foundation_simulation_v2(
        y_true=y_true,
        y_pred=y_pred,
        close_prices=close_prices,
        lookback=lookback,
        horizon=horizon,
        threshold=threshold,
        window_sigma=window_sigma,
        start_cash=start_cash,
        primary_fee_bps=2.0 if fee_bps == 0.0 else fee_bps,
        auxiliary_fee_bps=2.0 if fee_bps == 0.0 else fee_bps,
        position_mode=position_mode,
    )
