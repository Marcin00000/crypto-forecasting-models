"""
Moduł ładowania i przygotowywania danych dla modeli fundamentowych.

Obsługuje:
- zbiory danych przygotowane pod projekt (domyślnie E109_prep_eth_universal_max_1m),
- chronologiczny podział train.csv 80/20 in-sample na zbiór treningowy i walidacyjny,
- domyślny zbiór testowy test.csv (OOS),
- tryb Univariate (pojedynczy szereg zwrotów logarytmicznych),
- tryb Multivariate (zwroty + 39 cech technicznych z E109 dla mechanizmu Group Attention),
- poprawny resampling wielointerwałowy pobierający faktyczną cenę zamknięcia świecy.
"""
from __future__ import annotations

import os
import json
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

from foundation_config import FoundationConfig


def get_data_paths(data_dir: Optional[str] = None) -> Dict[str, str]:
    """Zwraca słownik zweryfikowanych ścieżek do plików zbioru danych."""
    base_dir = data_dir or FoundationConfig.DEFAULT_DATA_DIR
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"Nie znaleziono katalogu danych: {base_dir}. "
            "Podaj istniejący katalog zbioru (metadata.json, train.csv, test.csv)."
        )

    return {
        "data_dir": base_dir,
        "train": os.path.join(base_dir, FoundationConfig.DEFAULT_TRAIN_FILE),
        "test": os.path.join(base_dir, FoundationConfig.DEFAULT_TEST_FILE),
        "metadata": os.path.join(base_dir, FoundationConfig.DEFAULT_METADATA_FILE),
        "scaler": os.path.join(base_dir, FoundationConfig.DEFAULT_SCALER_FILE),
    }


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Wczytuje plik metadata.json zbioru danych."""
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resample_prices(prices: np.ndarray, resample_str: str) -> np.ndarray:
    """
    Poprawnie agreguje (resampluje) ceny zamknięcia świec 1-minutowych.

    Wybiera dokładnie ostatnią minutę każdego zagregowanego interwału
    (czyli indeks (stride - 1)::stride), naprawiając błąd z audytu.
    """
    resample_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
    stride = resample_map.get(resample_str.lower(), 1)
    if stride <= 1:
        return prices

    # Pobieramy zamknięcia świec zagregowanych
    return prices[(stride - 1)::stride]


def load_raw_dataframe(csv_path: str) -> pd.DataFrame:
    """Wczytuje plik CSV i weryfikuje obecność kolumny cenowej close."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Brak pliku danych: {csv_path}")

    df = pd.read_csv(csv_path)
    close_col = None
    for c in ["close", "Close", "c", "CLOSE"]:
        if c in df.columns:
            close_col = c
            break

    if close_col is None:
        raise ValueError(f"Brak kolumny 'close' w {csv_path}. Dostępne: {list(df.columns)}")

    if close_col != "close":
        df = df.rename(columns={close_col: "close"})

    return df


def prepare_univariate_windows(
    prices: np.ndarray,
    lookback: int,
    horizon: int,
    eval_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tworzy okna stacjonarnych zwrotów logarytmicznych dla trybu pojedynczego (Univariate).

    Returns:
        contexts: np.ndarray o kształcie (N, lookback)
        targets: np.ndarray o kształcie (N,) - skumulowany log-return na horyzoncie H
        current_prices: np.ndarray o kształcie (N,) - cena bazowa na końcu kontekstu
    """
    log_returns = np.diff(np.log(prices))
    n_samples = (len(log_returns) - lookback - horizon) // eval_stride + 1
    if n_samples <= 0:
        raise ValueError(f"Zbyt krótki szereg cenowy ({len(prices)} świec) dla lookback={lookback}, horizon={horizon}")

    contexts = []
    targets = []
    current_prices = []

    for idx in range(0, len(log_returns) - lookback - horizon + 1, eval_stride):
        ctx = log_returns[idx : idx + lookback]
        future = log_returns[idx + lookback : idx + lookback + horizon]
        cum_ret = float(np.sum(future))
        curr_p = float(prices[idx + lookback])

        contexts.append(ctx)
        targets.append(cum_ret)
        current_prices.append(curr_p)

    return (
        np.asarray(contexts, dtype=np.float32),
        np.asarray(targets, dtype=np.float64),
        np.asarray(current_prices, dtype=np.float64),
    )


def prepare_multivariate_windows(
    df: pd.DataFrame,
    feature_names: List[str],
    scaler: Any,
    lookback: int,
    horizon: int,
    eval_stride: int = 1,
    already_scaled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Tworzy okna dla trybu Multivariate (Group Attention):
    szereg zwrotów cenowych + macierz 39 przeskalowanych cech technicznych z E109.

    Returns:
        target_contexts: np.ndarray (N, lookback) - zwroty logarytmiczne
        covariate_contexts: np.ndarray (N, lookback, n_features) - przeskalowane cechy techniczne
        targets: np.ndarray (N,) - skumulowany zwrot horyzontu
        current_prices: np.ndarray (N,) - cena bazowa
    """
    prices = df["close"].to_numpy(dtype=np.float64)
    log_returns = np.diff(np.log(prices))

    # Wybór i skalowanie cech technicznych
    available_features = [f for f in feature_names if f in df.columns]
    if not available_features:
        raise ValueError("Żadna z zadanych cech technicznych nie występuje w DataFrame!")

    feat_matrix = df[available_features].to_numpy(dtype=np.float32)
    # Zbiór z scale_csv=true ma cechy już przeskalowane. Drugi transform przesuwa każdą kolumnę.
    if scaler is not None and not already_scaled:
        try:
            feat_matrix = scaler.transform(feat_matrix)
        except Exception as exc:
            raise ValueError(
                "Skalowanie cech nie powiodło się (scaler nie pasuje do kolumn). "
                "Dla zbioru z scale_csv=true podaj already_scaled=True."
            ) from exc

    # Wyrównanie cech z log-returns (indeks 1..N odpowiada pierwszemu log-return)
    feat_aligned = feat_matrix[1:]
    n_samples = (len(log_returns) - lookback - horizon) // eval_stride + 1
    if n_samples <= 0:
        raise ValueError(f"Zbyt krótki zbiór danych dla trybu Multivariate!")

    target_contexts = []
    covariate_contexts = []
    targets = []
    current_prices = []

    for idx in range(0, len(log_returns) - lookback - horizon + 1, eval_stride):
        ctx_ret = log_returns[idx : idx + lookback]
        ctx_cov = feat_aligned[idx : idx + lookback]
        future = log_returns[idx + lookback : idx + lookback + horizon]
        cum_ret = float(np.sum(future))
        curr_p = float(prices[idx + lookback])

        target_contexts.append(ctx_ret)
        covariate_contexts.append(ctx_cov)
        targets.append(cum_ret)
        current_prices.append(curr_p)

    return (
        np.asarray(target_contexts, dtype=np.float32),
        np.asarray(covariate_contexts, dtype=np.float32),
        np.asarray(targets, dtype=np.float64),
        np.asarray(current_prices, dtype=np.float64),
    )


def load_train_val_data(
    data_dir: Optional[str] = None,
    val_ratio: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Any]:
    """
    Wczytuje train.csv i dokonuje chronologicznego podziału 80/20 in-sample
    na zbiór treningowy i walidacyjny (standard val_source='split' z modułu LR).
    """
    paths = get_data_paths(data_dir)
    metadata = load_metadata(paths["metadata"])
    scaler = None
    if os.path.exists(paths["scaler"]):
        try:
            scaler = joblib.load(paths["scaler"])
        except Exception:
            scaler = None

    train_full_df = load_raw_dataframe(paths["train"])
    n_rows = len(train_full_df)
    val_size = int(n_rows * val_ratio)
    train_df = train_full_df.iloc[:-val_size].reset_index(drop=True)
    val_df = train_full_df.iloc[-val_size:].reset_index(drop=True)

    return train_df, val_df, metadata, scaler


def load_canonical_test_data(
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Wczytuje kanoniczny zbiór testowy test.csv (OOS) wraz z metadanymi i scalerem.
    """
    paths = get_data_paths(data_dir)
    metadata = load_metadata(paths["metadata"])
    scaler = None
    if os.path.exists(paths["scaler"]):
        try:
            scaler = joblib.load(paths["scaler"])
        except Exception:
            scaler = None

    test_df = load_raw_dataframe(paths["test"])
    return test_df, metadata, scaler
