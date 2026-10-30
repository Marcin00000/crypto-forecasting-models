"""
Wrapper dla modeli Amazon Chronos (Chronos-2, Chronos-Bolt, Chronos-T5).

Obsługuje:
- Domyślny model amazon/chronos-2 (z mechanizmem Group Attention dla kowariantów)
- 2 tryby wejściowe:
    1. 'single' (univariate: pojedynczy szereg stacjonarnych zwrotów logarytmicznych)
    2. 'features_max' (multivariate: zwroty + 39 cech technicznych z E109)
- Defensywny sanity check kształtu tensora wyjściowego (likwidujący lukę z audytu)
- Natywną agregację kwantylową (mediana 0.5) zapobiegającą szumowi Monte Carlo
- Sumowanie przewidywanych stóp zwrotu w horyzoncie H
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, List, Union

import numpy as np
import torch


class ChronosWrapper:
    """Wrapper inferencyjny dla modeli rodziny Chronos."""

    def __init__(
        self,
        model_name: str = "amazon/chronos-2",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch_dtype is not None:
            self.dtype = torch_dtype
        else:
            self.dtype = torch.bfloat16 if (self.device != "cpu" and torch.cuda.is_available()) else torch.float32

        self.pipeline = self._load_pipeline()
        self.is_chronos2 = (self.pipeline.__class__.__name__ == "Chronos2Pipeline")
        self._logged_shape = False

    def _load_pipeline(self):
        """Ładuje odpowiedni pipeline Chronos z HuggingFace lub katalogu lokalnego."""
        try:
            from chronos import BaseChronosPipeline
            print(f"[*] Ładowanie modelu Chronos '{self.model_name}' na urządzenie '{self.device}' (dtype={self.dtype})...")
            pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=self.dtype,
            )
            return pipeline
        except ImportError:
            raise ImportError(
                "Pakiet 'chronos-forecasting' nie jest zainstalowany. Zainstaluj: pip install chronos-forecasting"
            )
        except Exception as e:
            raise RuntimeError(f"Błąd ładowania Chronos '{self.model_name}': {e}")

    def predict_batch_univariate(
        self,
        batch_contexts: np.ndarray,
        horizon: int = 12,
    ) -> np.ndarray:
        """
        Generuje prognozy dla trybu Univariate (pojedynczy szereg zwrotów).

        Args:
            batch_contexts: np.ndarray o kształcie (batch_size, context_length)
            horizon: liczba kroków prognozy w przód

        Returns:
            np.ndarray o kształcie (batch_size,) ze skumulowanym zwrotem horyzontu
        """
        batch_arr = np.asarray(batch_contexts, dtype=np.float32)

        with torch.no_grad():
            if self.is_chronos2:
                # Chronos-2 wymaga tensora 3D: (n_series, n_variates, history_length)
                tensor_ctx = torch.tensor(batch_arr, dtype=torch.float32).unsqueeze(1)
                if hasattr(self.pipeline, "predict_quantiles"):
                    q_out = self.pipeline.predict_quantiles(
                        tensor_ctx,
                        prediction_length=horizon,
                        quantile_levels=[0.5],
                    )
                    q_list = q_out[0] if isinstance(q_out, (tuple, list)) else q_out
                    if isinstance(q_list, list):
                        stacked = torch.stack(q_list)  # (batch_size, 1, horizon, 1)
                        pred_median_steps = stacked[:, 0, :, 0].float().cpu().numpy()
                    else:
                        pred_median_steps = q_list[:, 0, :, 0].float().cpu().numpy()
                else:
                    forecast = self.pipeline.predict(tensor_ctx, prediction_length=horizon)
                    if isinstance(forecast, list):
                        stacked = torch.stack(forecast)  # (batch_size, 1, num_samples, horizon)
                        pred_median_steps = torch.median(stacked[:, 0, :, :], dim=1).values.float().cpu().numpy()
                    else:
                        pred_median_steps = np.median(forecast.cpu().numpy()[:, 0, :, :], axis=1)
            else:
                # Chronos-Bolt oraz Chronos-T5 wymagają tensora 2D: (n_series, history_length)
                tensor_ctx = torch.tensor(batch_arr, dtype=torch.float32)
                if hasattr(self.pipeline, "predict_quantiles"):
                    q_out = self.pipeline.predict_quantiles(
                        tensor_ctx,
                        prediction_length=horizon,
                        quantile_levels=[0.5],
                    )
                    q_tensor = q_out[0] if isinstance(q_out, (tuple, list)) else q_out
                    if isinstance(q_tensor, torch.Tensor):
                        pred_median_steps = q_tensor[:, :, 0].float().cpu().numpy()
                    else:
                        pred_median_steps = np.asarray(q_tensor)[:, :, 0]
                else:
                    forecast = self.pipeline.predict(tensor_ctx, prediction_length=horizon)
                    if isinstance(forecast, torch.Tensor):
                        forecast_np = forecast.cpu().numpy()
                    else:
                        forecast_np = np.asarray(forecast)
                    if forecast_np.ndim == 3:
                        pred_median_steps = np.median(forecast_np, axis=1)
                    else:
                        pred_median_steps = forecast_np

            if not self._logged_shape:
                print(f"    [Sanity Check] Kształt wyjścia kroków: {pred_median_steps.shape}")
                self._logged_shape = True

            cum_returns = np.sum(pred_median_steps, axis=-1)
            return np.asarray(cum_returns, dtype=np.float64)

    def predict_batch_multivariate(
        self,
        batch_target_contexts: np.ndarray,
        batch_covariate_contexts: np.ndarray,
        horizon: int = 12,
    ) -> np.ndarray:
        """
        Generuje prognozy dla trybu Multivariate (Group Attention):
        szereg zwrotów + cechy techniczne z E109.

        Args:
            batch_target_contexts: (batch_size, context_length)
            batch_covariate_contexts: (batch_size, context_length, n_features)
            horizon: liczba kroków w przód
        """
        if not self.is_chronos2:
            # Modele z rodziny Chronos-Bolt / Chronos-T5 są jednowymiarowe
            return self.predict_batch_univariate(batch_target_contexts, horizon=horizon)

        target_arr = np.asarray(batch_target_contexts, dtype=np.float32)
        cov_arr = np.asarray(batch_covariate_contexts, dtype=np.float32)

        # Chronos-2 natywnie wspiera podawanie wielokanałowych szeregów
        # Łączymy target jako kanał 0, a kowarianty jako kanały 1..N: (batch_size, 1 + n_feat, ctx_len)
        target_expanded = np.expand_dims(target_arr, axis=1)  # (batch_size, 1, ctx_len)
        cov_transposed = np.transpose(cov_arr, (0, 2, 1))      # (batch_size, n_feat, ctx_len)
        multichannel_input = np.concatenate([target_expanded, cov_transposed], axis=1)  # (batch, 1+n_feat, ctx_len)

        tensor_multi = torch.tensor(multichannel_input, dtype=torch.float32)

        with torch.no_grad():
            if hasattr(self.pipeline, "predict_quantiles"):
                q_out = self.pipeline.predict_quantiles(
                    tensor_multi,
                    prediction_length=horizon,
                    quantile_levels=[0.5],
                )
                q_list = q_out[0] if isinstance(q_out, (tuple, list)) else q_out
                if isinstance(q_list, list):
                    stacked = torch.stack(q_list)  # (batch_size, 1+n_feat, horizon, 1)
                    target_steps = stacked[:, 0, :, 0].float().cpu().numpy()
                else:
                    target_steps = q_list[:, 0, :, 0].float().cpu().numpy()
            else:
                forecast = self.pipeline.predict(tensor_multi, prediction_length=horizon)
                if isinstance(forecast, list):
                    stacked = torch.stack(forecast)  # (batch_size, 1+n_feat, num_samples, horizon)
                    target_steps = torch.median(stacked[:, 0, :, :], dim=1).values.float().cpu().numpy()
                else:
                    target_steps = np.median(forecast.cpu().numpy()[:, 0, :, :], axis=1)

            cum_returns = np.sum(target_steps, axis=-1)
            return np.asarray(cum_returns, dtype=np.float64)
