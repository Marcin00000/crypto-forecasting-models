# LSTM Log-Return — instrukcja użycia

[Opis modelu](OVERVIEW.md) · [Wspólne wymagania i dane](../USAGE.md)

## Przygotowanie

Kod treningowy znajduje się w `lstm_logreturn_model/`, a niezależny pakiet inferencyjny w `lstm_model_package/`.
Przykład zakłada własne świece w `datasets/ohlcv.csv` i własne artefakty w `models/lstm_logreturn/example/`.
Są to przykładowe lokalizacje; repozytorium nie zawiera tych plików.

Katalog modelu powinien zawierać:

- `model_config.json` opisujący wytrenowany model;
- wagi: `lstm_model.pth`;
- skaler cech zgodny z treningiem, zwykle `feature_scaler.pkl` lub `scaler_features.pkl`.

Skaler celu jest potrzebny, jeżeli był używany w treningu. Kalibrator jest opcjonalny i wymaga zgodnego sposobu użycia.

## Inferencja

```python
import pandas as pd
from lstm_model_package import LSTMModelPackage

df = pd.read_csv("datasets/ohlcv.csv")
package = LSTMModelPackage.from_directory("models/lstm_logreturn/example", device="cpu")
batch = package.predict_batch(df, data_mode="raw")
latest = package.predict_live(df, data_mode="raw")
print(batch.tail())
print(latest)
```

Dane muszą być uporządkowane w czasie i obejmować historię potrzebną do obliczenia cech.
`batch` opisuje kolejne okna danych, a `latest` ostatnią dostępną świecę.

## Trening na własnych danych

Dostępne opcje:

```powershell
python lstm_logreturn_model/lstm_train.py --help
```

Przykład dla przygotowanego zbioru:

```powershell
python lstm_logreturn_model/lstm_train.py --data_path datasets/prepared --epochs 100
```

Polecenie uruchamia rzeczywisty trening i zapisuje artefakty.
Format zbioru musi odpowiadać loaderowi tej rodziny. [Pipeline przygotowania danych](../data_preparation/USAGE.md) jest dostępny w repozytorium.

## Typowe problemy

- Brak konfiguracji, wag lub skalera: należy dostarczyć kompletny katalog modelu.
- Niezgodne kolumny albo kształt wag: dane i konfiguracja muszą odpowiadać treningowi.
- Za krótka historia: należy zwiększyć bufor danych, uwzględniając rozgrzewkę wskaźników.
- Wynik liczbowy bez użytecznych sygnałów: jakość wymaga osobnej oceny poza próbką;
  poprawne wczytanie modelu nie potwierdza skuteczności prognozy.

## Ewaluacja i Optuna

Ewaluator: `lstm_logreturn_model/lstm_test.py`, z argumentami `--model_dir` i `--test_path`.
Program strojenia: `lstm_logreturn_model/lstm_hyperparameter_tuning.py`.

[Pełny proces](../WORKFLOW.md) opisuje trening, test i użycie tego samego modelu.
[Instrukcja Optuny](../OPTUNA.md) wyjaśnia dobór parametrów i trening końcowy.
