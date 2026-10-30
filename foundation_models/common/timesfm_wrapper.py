"""
Wrapper dla modelu Google TimesFM 2.5 (Time Series Foundation Model).

Obsługuje:
- Model google/timesfm-2.5-200m-pytorch w środowisku PyTorch
- Obejście braku natywnego Tritona na systemie Windows (tryb Eager execution, wyłączenie dynamo)
- Wsadową inferencję na stacjonarnych zwrotach logarytmicznych
- Sumowanie przewidywanych 1-krokowych stóp zwrotu w horyzoncie H=12
"""
from __future__ import annotations

import os
import sys
import inspect
from typing import Optional, List

import numpy as np
import torch

# Całkowite wyłączenie torch._dynamo / Inductor (brak natywnego Tritona na Windows)
try:
    import torch._dynamo
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


class TimesFMWrapper:
    """Wrapper inferencyjny dla modelu Google TimesFM 2.5."""

    def __init__(
        self,
        model_name: str = "google/timesfm-2.5-200m-pytorch",
        context_len: int = 48,
        horizon_len: int = 12,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        """Ładuje i konfiguruje model TimesFM w trybie Eager."""
        try:
            import timesfm
            from huggingface_hub import snapshot_download

            if "2.0-500m" in self.model_name or "1.0-200m" in self.model_name:
                self.model_name = "google/timesfm-2.5-200m-pytorch"

            print(f"[*] Inicjalizacja Google TimesFM 2.5 na urządzeniu '{self.device}'...")

            if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                tfm_cls = getattr(timesfm, "TimesFM_2p5_200M_torch")
                tfm = tfm_cls()

                if not os.path.exists(self.model_name):
                    print(f"[*] Pobieranie wag '{self.model_name}' z HuggingFace...")
                    ckpt_dir = snapshot_download(repo_id=self.model_name)
                else:
                    ckpt_dir = self.model_name

                # Wyłączenie torch_compile w celu ominięcia braku Tritona na Windows (czysty tryb Eager)
                tfm.load_checkpoint(ckpt_dir, torch_compile=False)

                target_device = "cuda" if (self.device in ["gpu", "cuda"] and torch.cuda.is_available()) else "cpu"
                if hasattr(tfm, "model") and hasattr(tfm.model, "to"):
                    tfm.model.to(target_device)
                if hasattr(tfm, "to"):
                    try:
                        tfm.to(target_device)
                    except Exception:
                        pass

                # Konfiguracja horyzontu i kompilacja decode w trybie Eager
                if hasattr(timesfm, "ForecastConfig"):
                    fc = timesfm.ForecastConfig(
                        max_context=max(512, self.context_len),
                        max_horizon=max(128, self.horizon_len),
                    )
                    tfm.compile(fc)

                return tfm
            else:
                raise AttributeError("Brak klasy TimesFM_2p5_200M_torch w bibliotece timesfm.")

        except ImportError:
            raise ImportError(
                "Pakiet 'timesfm' nie jest zainstalowany. Zainstaluj: pip install timesfm"
            )
        except Exception as e:
            raise RuntimeError(f"Błąd inicjalizacji TimesFM 2.5: {e}")

    def predict_batch(
        self,
        batch_contexts: np.ndarray,
        horizon: int = 12,
    ) -> np.ndarray:
        """
        Generuje prognozy stóp zwrotu za pomocą TimesFM 2.5.

        Args:
            batch_contexts: np.ndarray o kształcie (batch_size, context_length)
            horizon: liczba kroków w przód

        Returns:
            np.ndarray o kształcie (batch_size,) ze skumulowanym zwrotem horyzontu
        """
        # Konwersja na listę szeregów 1D np.ndarray float32 akceptowaną przez TimesFM
        inputs = [np.asarray(batch_contexts[i], dtype=np.float32) for i in range(len(batch_contexts))]

        try:
            # Poprawna sygnatura w TimesFM 2.5: forecast(horizon, inputs)
            point_forecast, _ = self.model.forecast(horizon=horizon, inputs=inputs)
            pred_steps = point_forecast[:, :horizon]
            cum_returns = np.sum(pred_steps, axis=-1)
            return np.asarray(cum_returns, dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f"Błąd inferencji TimesFM: {e}")
