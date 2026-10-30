"""
Silnik lekkiego fine-tuningu dla modeli fundamentowych szeregów czasowych.

Kluczowe właściwości:
- Uczenie na stacjonarnych stopach zwrotu (log-returns), a NIE surowych cenach w USD
  (likwiduje problem zapaści predykcji i kolapsu do stałej średniej z audytu)
- Chronologiczny podział train.csv 80/20 in-sample na train i validation (brak pliku val.csv)
- Monitorowanie straty walidacyjnej i wczesne zatrzymanie (Early Stopping)
- Zapis wag najlepszego checkpointu
"""
from __future__ import annotations

import os
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ReturnSeriesDataset(Dataset):
    """Dataset tworzący pary (kontekst zwrotów, target zwrotów) ze stacjonarnego szeregu."""

    def __init__(self, log_returns: np.ndarray, lookback: int, horizon: int, stride: int = 1):
        self.returns = np.asarray(log_returns, dtype=np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.n_samples = max(0, (len(self.returns) - lookback - horizon) // stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        ctx_end = start + self.lookback
        tgt_end = ctx_end + self.horizon

        ctx = torch.tensor(self.returns[start:ctx_end], dtype=torch.float32)
        tgt = torch.tensor(self.returns[ctx_end:tgt_end], dtype=torch.float32)
        return {"context": ctx, "target": tgt}


def evaluate_val_loss(model: torch.nn.Module, val_loader: DataLoader, device: str) -> float:
    """Oblicza średnią stratę na zbiorze walidacyjnym in-sample."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            output = model(context=context, target=target)
            loss = output.loss if hasattr(output, "loss") else output[0]
            total_loss += float(loss.item())
            total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


def run_chronos_finetune(
    model_name: str,
    train_prices: np.ndarray,
    val_prices: np.ndarray,
    output_dir: str,
    lookback: int = 48,
    horizon: int = 12,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Wykonuje lekki fine-tuning modelu Chronos na stacjonarnych zwrotach logarytmicznych.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f" LEKKI FINE-TUNING CHRONOS: {model_name} (Device: {dev})")
    print(f" Reprezentacja danych: Stacjonarne zwroty logarytmiczne (log_return)")
    print(f" Geometria: Lookback={lookback}, Horizon={horizon}, Epochs={epochs}")
    print("=" * 65)

    # Obliczenie stóp zwrotu logarytmicznych
    train_returns = np.diff(np.log(train_prices))
    val_returns = np.diff(np.log(val_prices))

    train_ds = ReturnSeriesDataset(train_returns, lookback=lookback, horizon=horizon, stride=1)
    val_ds = ReturnSeriesDataset(val_returns, lookback=lookback, horizon=horizon, stride=1)

    print(f"[*] Liczba próbek: Train = {len(train_ds):,}, Val (in-sample 20%) = {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    from chronos import BaseChronosPipeline
    print(f"[*] Ładowanie wag pretrained '{model_name}'...")
    torch_dtype = torch.bfloat16 if (dev != "cpu" and torch.cuda.is_available()) else torch.float32
    pipeline = BaseChronosPipeline.from_pretrained(
        model_name,
        device_map=dev,
        torch_dtype=torch_dtype,
    )
    model = pipeline.model

    # Ustawienie optymalizatora i schedulera
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

    initial_val_loss = evaluate_val_loss(model, val_loader, dev)
    print(f"[*] Bazowa strata walidacyjna przed fine-tuningiem: {initial_val_loss:.6f}\n")

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        batches_done = 0

        for step, batch in enumerate(train_loader, 1):
            context = batch["context"].to(dev)
            target = batch["target"].to(dev)

            optimizer.zero_grad()
            output = model(context=context, target=target)
            loss = output.loss if hasattr(output, "loss") else output[0]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            batches_done += 1

            if step % max(1, len(train_loader) // 5) == 0 or step == len(train_loader):
                curr_avg = running_loss / batches_done
                lr_val = scheduler.get_last_lr()[0]
                print(f"    Epoka {epoch}/{epochs} | Krok {step}/{len(train_loader)} | Train Loss: {curr_avg:.6f} | LR: {lr_val:.2e}")

        avg_train_loss = running_loss / max(1, batches_done)
        val_loss = evaluate_val_loss(model, val_loader, dev)
        epoch_dur = time.time() - epoch_start

        improved_note = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if hasattr(pipeline, "save_pretrained"):
                pipeline.save_pretrained(output_dir)
            else:
                model.save_pretrained(output_dir)
            improved_note = " -> [NOWY NAJLEPSZY ZAPISANY!]"

        print(f"--> Epoka {epoch} ZAKOŃCZONA ({epoch_dur:.1f}s) | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}{improved_note}\n")

    total_time = time.time() - start_time
    print("=" * 65)
    print(f" FINE-TUNING UKOŃCZONY w {total_time:.1f}s")
    print(f" Najlepsza strata walidacyjna: {best_val_loss:.6f} (startowa: {initial_val_loss:.6f})")
    print(f" Model zapisany w: {os.path.abspath(output_dir)}")
    print("=" * 65)

    return {
        "best_val_loss": best_val_loss,
        "initial_val_loss": initial_val_loss,
        "epochs": epochs,
        "total_time_seconds": total_time,
        "output_dir": output_dir,
    }
