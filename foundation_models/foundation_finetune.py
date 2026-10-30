#!/usr/bin/env python
"""
Skrypt CLI do lekkiego fine-tuningu modelu Chronos-2 na stacjonarnych stopach zwrotu.

Wzorowany na pipeline treningowym projektu:
- Korzysta ze zbioru E109_prep_eth_universal_max_1m (train.csv)
- Stosuje chronologiczny podział in-sample 80% train / 20% validation (brak pliku val.csv)
- Uczy się na szeregach log-return (likwidując zapaść predykcji)
- Zapisuje checkpoint z najlepszą stratą walidacyjną do foundation_models/checkpoints/

Użycie:
    python foundation_models/foundation_finetune.py --epochs 5 --batch_size 64
"""
import os
import sys
import argparse

# Dodanie ścieżki głównej i podkatalogu common do sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
COMMON_DIR = os.path.join(SCRIPT_DIR, "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)

from foundation_config import FoundationConfig
from common.data_loaders import load_train_val_data
from common.finetune_engine import run_chronos_finetune


def parse_args():
    parser = argparse.ArgumentParser(description="Lekki fine-tuning Chronos-2 na danych E109 (log-returns).")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=FoundationConfig.DEFAULT_DATA_DIR,
        help="Katalog ze zbiorem danych (domyślnie E109).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=FoundationConfig.DEFAULT_CHRONOS_MODEL,
        help="Identyfikator modelu bazowego (domyślnie amazon/chronos-2).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(FoundationConfig.CHECKPOINTS_DIR, "chronos2_e109_finetuned"),
        help="Katalog docelowy zapisu checkpointu.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=FoundationConfig.LOOKBACK,
        help="Długość okna historii w świecach 1m (domyślnie 48).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=FoundationConfig.HORIZON,
        help="Horyzont predykcji w przód (domyślnie 12).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=FoundationConfig.DEFAULT_EPOCHS,
        help="Liczba epok treningowych (domyślnie 5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=FoundationConfig.DEFAULT_BATCH_SIZE,
        help="Rozmiar paczki (domyślnie 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=FoundationConfig.DEFAULT_LR,
        help="Współczynnik uczenia Learning Rate (domyślnie 1e-4).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=FoundationConfig.DEFAULT_WEIGHT_DECAY,
        help="Waga regularyzacji L2 (domyślnie 1e-2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Urządzenie: 'cuda' lub 'cpu' (domyślnie auto-detekcja).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[*] Ładowanie danych treningowych ze zbioru: {args.data_dir}")
    train_df, val_df, metadata, _ = load_train_val_data(args.data_dir, val_ratio=FoundationConfig.VAL_SPLIT_RATIO)

    train_prices = train_df["close"].to_numpy()
    val_prices = val_df["close"].to_numpy()

    run_chronos_finetune(
        model_name=args.model_name,
        train_prices=train_prices,
        val_prices=val_prices,
        output_dir=args.output_dir,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )


if __name__ == "__main__":
    main()
