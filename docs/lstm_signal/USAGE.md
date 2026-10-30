# LSTM Signal — instrukcja użycia

[Opis modelu](OVERVIEW.md) · [Wspólne wymagania i dane](../USAGE.md)

## Przygotowanie

Kod treningowy znajduje się w `lstm_signal_model/`, a niezależny pakiet inferencyjny w `signal_model_package/`.
Przykład zakłada własne świece w `datasets/ohlcv.csv` i własne artefakty w `models/lstm_signal/example/`.
Są to przykładowe lokalizacje; repozytorium nie zawiera tych plików.

Katalog modelu powinien zawierać:

- `model_config.json` opisujący wytrenowany model;
- wagi: `signal_model.pth`;
- skaler cech zgodny z treningiem, zwykle `feature_scaler.pkl` lub `scaler_features.pkl`.

Pakiet obsługuje modele z dwoma wyjściami (`output_mode=dual`). Modele fuzji wymagają również katalogów wszystkich używanych podmodeli.

## Inferencja

```python
import pandas as pd
from signal_model_package import SignalModelPackage

df = pd.read_csv("datasets/ohlcv.csv")
package = SignalModelPackage.from_directory("models/lstm_signal/example", device="cpu")
batch = package.predict_batch(df, data_mode="raw")
latest = package.predict_live(df, data_mode="raw")
print(batch.tail())
print(latest)
```

Dane muszą być uporządkowane w czasie i obejmować historię potrzebną do obliczenia cech.
`batch` opisuje kolejne okna danych, a `latest` ostatnią dostępną świecę.

### Model z fuzją podmodeli

Dla modelu wytrenowanego z podmodelami należy przekazać ich katalogi w kolejności zgodnej z treningiem.
Liczba, rodziny i konfiguracje podmodeli muszą odpowiadać zapisanej konfiguracji modelu Signal.

```python
package = SignalModelPackage.from_directory(
    "models/lstm_signal/example_fusion",
    device="cpu",
    submodel_dirs=["models/lstm_logreturn/example", "models/xgb/example"],
)
```

Lista jest przykładem dla modelu wymagającego właśnie tych dwóch podmodeli.
Nie zastępuje odczytania konfiguracji konkretnego modelu.

## Trening na własnych danych

Dostępne opcje:

```powershell
python lstm_signal_model/signal_train.py --help
```

Przykład dla przygotowanego zbioru:

```powershell
python lstm_signal_model/signal_train.py --data_path datasets/prepared --epochs 100
```

Polecenie uruchamia rzeczywisty trening i zapisuje artefakty.
Format zbioru musi odpowiadać loaderowi tej rodziny. Pipeline tworzenia zbiorów nie jest
dołączony do obecnego wydania.

## Typowe problemy

- Brak konfiguracji, wag lub skalera: należy dostarczyć kompletny katalog modelu.
- Niezgodne kolumny albo kształt wag: dane i konfiguracja muszą odpowiadać treningowi.
- Za krótka historia: należy zwiększyć bufor danych, uwzględniając rozgrzewkę wskaźników.
- Wynik liczbowy bez użytecznych sygnałów: jakość wymaga osobnej oceny poza próbką;
  poprawne wczytanie modelu nie potwierdza skuteczności prognozy.
