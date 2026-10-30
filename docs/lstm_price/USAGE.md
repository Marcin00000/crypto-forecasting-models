# LSTM Price — instrukcja użycia

[Opis modelu](OVERVIEW.md) · [Wspólne wymagania i dane](../USAGE.md)

## Przygotowanie

Kod treningowy znajduje się w `lstm_model/`, a niezależny pakiet inferencyjny w `lstm_price_model_package/`.
Przykład zakłada własne świece w `datasets/ohlcv.csv` i własne artefakty w `models/lstm_price/example/`.
Są to przykładowe lokalizacje; repozytorium nie zawiera tych plików.

Katalog modelu powinien zawierać:

- `model_config.json` opisujący wytrenowany model;
- wagi: `best_model.pth` (obsługiwana jest również nazwa `lstm_price_model.pth`);
- skaler cech zgodny z treningiem, zwykle `feature_scaler.pkl` lub `scaler_features.pkl`.

Jeżeli trening wykorzystywał skaler ceny, wymagany jest zgodny `price_scaler.pkl`. Konfiguracja określa także użycie logarytmu ceny.

## Inferencja

```python
import pandas as pd
from lstm_price_model_package import LSTMPriceModelPackage

df = pd.read_csv("datasets/ohlcv.csv")
package = LSTMPriceModelPackage.from_directory("models/lstm_price/example", device="cpu")
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
python lstm_model/lstm_train_price.py --help
```

Przykład dla przygotowanego zbioru:

```powershell
python lstm_model/lstm_train_price.py --train_path datasets/prepared --epochs 100
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
