# Od danych do treningu, ewaluacji i inferencji

Proces wymaga własnych danych i zainstalowanego środowiska opisanego w [USAGE](USAGE.md).
Pliki wejściowe, wagi i wyniki pozostają lokalnie. Polecenia uruchamia się z katalogu repozytorium.

## 1. Przygotowanie danych

Przykład zakłada plik `datasets/eth_1m.csv` ze świecami ETH co minutę: `open_time` w milisekundach Unix
oraz OHLCV. Dla cech presji kupna potrzebny jest także `taker_buy_base_asset_volume`.

```powershell
$env:PYTHONUTF8 = "1"
python data_preparation/prep_cli.py datasets/eth_1m.csv --models universal --output_format both --output_dir datasets/prepared --split_mode percentage --train_pct 80 --test_pct 20 --feature_set max --lookback 48 --horizon 12 --data_id E999
```

Dla tego przykładu powstaje `datasets/prepared/E999_prep_eth_universal_max_1m`.
Nazwa zależy od wejścia i ustawień; należy użyć ścieżki wypisanej przez program.
[Instrukcja przygotowania danych](data_preparation/USAGE.md) opisuje formaty i podział.

## 2. Opcjonalne strojenie

```powershell
python xgb_model/xgb_hyperparameter_tuning.py --data_path datasets/prepared/E999_prep_eth_universal_max_1m --n_trials 20
```

Strojenie wybiera parametry, ale nie zastępuje treningu końcowego.
[Optuna](OPTUNA.md) opisuje pięć programów i wczytywanie ich wyników.
Końcowego testu nie należy używać do wyboru parametrów.

## 3. Trening końcowy

```powershell
python xgb_model/xgb_train.py --data_path datasets/prepared/E999_prep_eth_universal_max_1m --feature_set max --lookback 48 --horizon 12
```

Z wynikiem strojenia:

```powershell
python xgb_model/xgb_train.py --data_path datasets/prepared/E999_prep_eth_universal_max_1m --load_optuna_params xgb_model/xgb_data/study/best_params.json
```

`study` jest nazwą przykładową. Należy wskazać rzeczywisty plik zapisany przez Optunę.
Trener wypisuje katalog z konfiguracją, wagami i skalerami. Ten sam katalog służy do ewaluacji i inferencji.

## 4. Ewaluacja wytrenowanego modelu

```powershell
python xgb_model/xgb_test.py --model_dir xgb_model/xgb_models/X001_eth_xgb_model_E999_DDMM --test_path datasets/prepared/E999_prep_eth_universal_max_1m
```

Cała przykładowa ścieżka modelu, łącznie z umownym sufiksem `DDMM`, wymaga zastąpienia rzeczywistym wynikiem treningu.
Ewaluator oblicza metryki i wyniki symulacji; nie jest to test jednostkowy biblioteki.

| Rodzina | Ewaluator | Argument danych |
|---|---|---|
| Log-Return | `lstm_logreturn_model/lstm_test.py` | `--test_path` |
| Signal | `lstm_signal_model/signal_test.py` | `--data_path` |
| XGBoost | `xgb_model/xgb_test.py` | `--test_path` |
| Random Forest | `rf_model/rf_test.py` | `--test_path` |
| LSTM Price | `lstm_model/lstm_test_price.py` | `--test_path` |
| Foundation Models | `foundation_models/foundation_test.py` | `--data_dir` |

Pięć pierwszych programów przyjmuje także `--model_dir`. Foundation Models korzystają z `--model_type`
i `--model_name`; drugi argument może wskazywać katalog checkpointu po dostrajaniu.
Opcje każdego programu pokazuje `--help`.

## 5. Inferencja tym samym modelem

```python
import pandas as pd
from xgb_model_package import XGBModelPackage

package = XGBModelPackage.from_directory("xgb_model/xgb_models/X001_eth_xgb_model_E999_DDMM")
raw = pd.read_csv("datasets/eth_1m.csv")
signals = package.predict_batch(raw, data_mode="raw")
print(signals.tail())
```

Pakiet liczy prognozy, ale sam nie wydziela testu ani nie ocenia generalizacji.
Do oceny jakości należy użyć odłożonego okresu i przyjętej metody ewaluacji.

## Zakres sprawdzenia

Potwierdzono importy i interfejsy CLI, przygotowanie danych syntetycznych oraz testy mapowania
parametrów Price i zgodności architektury z pakietem. W przygotowaniu wydania nie uruchamiano
treningu ani prób Optuny. Pełny wynik jakościowy wymaga wykonania procesu na własnych danych.
