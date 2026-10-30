# Dokumentacja modeli

Repozytorium udostępnia kod treningu i inferencji dla sześciu rodzin modeli szeregów czasowych.
Każda rodzina ma ulotkę `OVERVIEW.md` oraz instrukcję `USAGE.md`.

| Rodzina | Ulotka: zastosowanie i możliwości | Instrukcja |
|---|---|---|
| LSTM Log-Return | [OVERVIEW](lstm_logreturn/OVERVIEW.md) | [USAGE](lstm_logreturn/USAGE.md) |
| LSTM Signal | [OVERVIEW](lstm_signal/OVERVIEW.md) | [USAGE](lstm_signal/USAGE.md) |
| XGBoost | [OVERVIEW](xgb/OVERVIEW.md) | [USAGE](xgb/USAGE.md) |
| Random Forest | [OVERVIEW](rf/OVERVIEW.md) | [USAGE](rf/USAGE.md) |
| LSTM Price | [OVERVIEW](lstm_price/OVERVIEW.md) | [USAGE](lstm_price/USAGE.md) |
| Foundation Models | [OVERVIEW](foundation/OVERVIEW.md) | [USAGE](foundation/USAGE.md) |

[Wspólne przygotowanie środowiska i danych](USAGE.md) opisuje wymagania oraz znaczenie wyników.
[Ochrona prywatności przy publikacji](PRIVACY.md) opisuje kontrolę plików i historii Git.

Przykłady wymagają własnych danych i odpowiednich wag. Nie stanowią wyników eksperymentów
ani deklaracji skuteczności inwestycyjnej.

## Praca od danych do prognozy

- [Pełny proces](WORKFLOW.md)
- [Optuna](OPTUNA.md)
- [Przygotowanie danych: opis](data_preparation/OVERVIEW.md) i [instrukcja](data_preparation/USAGE.md)
