# Crypto Forecasting Models

Kod treningu modeli szeregów czasowych kryptowalut i niezależne pakiety inferencyjne.
Repozytorium udostępnia sześć rodzin modeli i wspólny silnik cech `feature_engineering.py`.
Wagi, zbiory danych oraz wyniki eksperymentów są przechowywane osobno.

| Rodzina | Ulotka | Instrukcja | Kod / pakiet |
|---|---|---|---|
| LSTM Log-Return | [Opis](docs/lstm_logreturn/OVERVIEW.md) | [Użycie](docs/lstm_logreturn/USAGE.md) | `lstm_logreturn_model/` · `lstm_model_package/` |
| LSTM Signal | [Opis](docs/lstm_signal/OVERVIEW.md) | [Użycie](docs/lstm_signal/USAGE.md) | `lstm_signal_model/` · `signal_model_package/` |
| XGBoost | [Opis](docs/xgb/OVERVIEW.md) | [Użycie](docs/xgb/USAGE.md) | `xgb_model/` · `xgb_model_package/` |
| Random Forest | [Opis](docs/rf/OVERVIEW.md) | [Użycie](docs/rf/USAGE.md) | `rf_model/` · `rf_model_package/` |
| LSTM Price | [Opis](docs/lstm_price/OVERVIEW.md) | [Użycie](docs/lstm_price/USAGE.md) | `lstm_model/` · `lstm_price_model_package/` |
| Foundation Models | [Opis](docs/foundation/OVERVIEW.md) | [Użycie](docs/foundation/USAGE.md) | `foundation_models/` |

## Początek pracy

[Wspólna instrukcja](docs/USAGE.md) opisuje środowisko, dane, artefakty i interpretację wyników.
[Spis dokumentacji](docs/README.md) prowadzi do ulotek i przykładów każdej rodziny.

```powershell
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
```

PyTorch instaluje się osobno w wariancie odpowiednim dla CPU lub GPU.
Dodatkowe biblioteki Foundation Models opisuje instrukcja tej rodziny.
Dane, wagi i skalery do przykładów muszą zostać dostarczone przez użytkownika.
Repozytorium zawiera przygotowanie danych, trening, ewaluatory i pięć programów Optuny.
Dane oraz testy jednostkowe pozostają poza tym wydaniem.

## Prywatność i zmiany

[Polityka publikacji](docs/PRIVACY.md) opisuje wykluczenia, lokalną kontrolę przed push
i wymagane sprawdzenie CI. Repozytorium nie zawiera automatycznego eksportera ani katalogu `tools`.
Zmiany trafiają na `main` przez pull request po sprawdzeniu prywatności.

## Pełny proces

[Przygotowanie danych, trening, test i inferencja](docs/WORKFLOW.md) prowadzi przez kolejne etapy.
[Optuna](docs/OPTUNA.md) opisuje strojenie i wczytywanie parametrów.
