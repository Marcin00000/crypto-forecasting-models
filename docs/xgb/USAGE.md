# XGBoost — instrukcja użycia

[Opis modelu](OVERVIEW.md) · [Wspólne wymagania i dane](../USAGE.md)

## Przygotowanie

Kod treningowy znajduje się w `xgb_model/`, a niezależny pakiet inferencyjny w `xgb_model_package/`.
Przykład zakłada własne świece w `datasets/ohlcv.csv` i własne artefakty w `models/xgb/example/`.
Są to przykładowe lokalizacje; repozytorium nie zawiera tych plików.

Katalog modelu powinien zawierać:

- `model_config.json` opisujący wytrenowany model;
- wagi: `xgb_model.json`;
- skaler cech zgodny z treningiem, zwykle `feature_scaler.pkl` lub `scaler_features.pkl`.

Parametry skalowania celu mogą być zapisane w konfiguracji. Jeżeli model korzysta z osobnego skalera celu, należy zachować go razem z modelem.

## Inferencja

```python
import pandas as pd
from xgb_model_package import XGBModelPackage

df = pd.read_csv("datasets/ohlcv.csv")
package = XGBModelPackage.from_directory("models/xgb/example")
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
python xgb_model/xgb_train.py --help
```

Przykład dla przygotowanego zbioru:

```powershell
python xgb_model/xgb_train.py --data_path datasets/prepared
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
