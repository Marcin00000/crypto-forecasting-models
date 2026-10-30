"""
Centralna konfiguracja modułu Foundation Models (Chronos-2 & TimesFM 2.5).

Wzorowana na lstm_config.py i signal_config.py z zachowaniem
pełnej kompatybilności z geometrią danych E109.
"""
from __future__ import annotations

import os


class FoundationConfig:
    """Parametry i stałe konfiguracyjne dla modeli fundamentowych."""

    # -------------------------------------------------------------------------
    # DANE I ŚCIEŻKI
    # -------------------------------------------------------------------------
    DEFAULT_DATA_DIR = os.path.join(
        "data_preparation", "prepared_data", "E109_prep_eth_universal_max_1m"
    )
    DEFAULT_TRAIN_FILE = "train.csv"
    DEFAULT_TEST_FILE = "test.csv"
    DEFAULT_METADATA_FILE = "metadata.json"
    DEFAULT_SCALER_FILE = "scaler_features.pkl"

    DATA_DIR = os.path.join("foundation_models", "foundation_data")
    CHECKPOINTS_DIR = os.path.join("foundation_models", "checkpoints")
    RESULTS_DIR = os.path.join("foundation_models", "results")

    @classmethod
    def ensure_dirs(cls) -> None:
        """Upewnia się, że katalogi danych, checkpointów i wyników istnieją."""
        for d in [cls.DATA_DIR, cls.CHECKPOINTS_DIR, cls.RESULTS_DIR]:
            os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # GEOMETRIA SEKWENCJI (zgodna z E109)
    # -------------------------------------------------------------------------
    LOOKBACK = 48    # 48 świec historii (1m)
    HORIZON = 12     # 12 świec horyzontu w przód (t+12)
    VAL_SPLIT_RATIO = 0.20  # Podział 80/20 in-sample na train.csv

    # -------------------------------------------------------------------------
    # TRYBY WEJŚCIA DLA CHRONOS
    # -------------------------------------------------------------------------
    INPUT_MODE_SINGLE = "single"          # 1-kanałowy stacjonarny zwrot logarytmiczny
    INPUT_MODE_FEATURES = "features_max"  # Szereg zwrotów + 39 cech technicznych z E109

    # -------------------------------------------------------------------------
    # MODELE BAZOWE (HUGGING FACE)
    # -------------------------------------------------------------------------
    DEFAULT_CHRONOS_MODEL = "amazon/chronos-2"
    FALLBACK_CHRONOS_BOLT = "amazon/chronos-bolt-small"
    DEFAULT_TIMESFM_MODEL = "google/timesfm-2.5-200m-pytorch"

    # -------------------------------------------------------------------------
    # SYMULACJA RYNKOWA (Standard modułu LR)
    # -------------------------------------------------------------------------
    SIGNAL_WINDOW_SIGMA = 100     # Okno kroczącego odchylenia do estymacji zmienności rynku
    SIGNAL_THRESHOLD = 0.001      # Ekonomiczny próg ruchu ceny T
    SIGNAL_MIN_STRONG = 60.0      # Próg odcięcia silnych sygnałów (w skali 0-100)
    SIM_START_CASH = 100.0        # Kapitał początkowy symulacji ($100)
    DEFAULT_FEE_BPS = 0.0         # Prowizja transakcyjna (bps)
    STRONG_QUANTILE = 0.80        # Kwantyl do metryk strong directional accuracy (top 20%)

    # -------------------------------------------------------------------------
    # PARAMETRY FINE-TUNINGU
    # -------------------------------------------------------------------------
    DEFAULT_LR = 1e-4
    DEFAULT_WEIGHT_DECAY = 1e-2
    DEFAULT_EPOCHS = 5
    DEFAULT_BATCH_SIZE = 64
