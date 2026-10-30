# Optuna — strojenie modeli

Repozytorium zawiera pięć programów strojenia. Plik `best_params.json` opisuje parametry,
a nie wytrenowany model. Po strojeniu wykonuje się trening końcowy, ewaluację i inferencję.

| Rodzina | Program | Argument danych | Budżet sieci |
|---|---|---|---|
| Log-Return | `lstm_logreturn_model/lstm_hyperparameter_tuning.py` | `--data_path` | domyślnie 100 epok |
| Signal | `lstm_signal_model/signal_hyperparameter_tuning.py` | `--data_path` | `--tuning_epochs 100` |
| XGBoost | `xgb_model/xgb_hyperparameter_tuning.py` | `--data_path` | zespół drzew |
| Random Forest | `rf_model/rf_hyperparameter_tuning.py` | `--data_path` | zespół drzew |
| LSTM Price | `lstm_model/lstm_hyperparameter_tuning.py` | `--train_path` | `--epochs 100` |

`--n_trials` określa liczbę prób. Wcześniejsze zatrzymanie lub pruning mogą skrócić próbę.
Foundation Models nie mają osobnego programu Optuny w tym wydaniu; dostępne są dostrajanie Chronos i ewaluacja.

## Uruchomienie

Przykłady korzystają z [przygotowanych danych](data_preparation/USAGE.md). Każde polecenie uruchamia rzeczywiste uczenie.

```powershell
python lstm_logreturn_model/lstm_hyperparameter_tuning.py --data_path datasets/prepared/E999_prep_eth_universal_max_1m --n_trials 20
python lstm_signal_model/signal_hyperparameter_tuning.py --data_path datasets/prepared/E999_prep_eth_universal_max_1m --n_trials 20 --tuning_epochs 100
python xgb_model/xgb_hyperparameter_tuning.py --data_path datasets/prepared/E999_prep_eth_universal_max_1m --n_trials 20
python rf_model/rf_hyperparameter_tuning.py --data_path datasets/prepared/E999_prep_eth_universal_max_1m --n_trials 20
python lstm_model/lstm_hyperparameter_tuning.py --train_path datasets/prepared/E999_prep_eth_universal_max_1m --n_trials 20 --epochs 100
```

Rodziny uruchamia się w osobnych procesach. Parametry wybiera się na treningu i walidacji;
końcowy zbiór testowy powinien pozostać poza tym wyborem.

## Trening z wybranymi parametrami

Trenery wszystkich pięciu rodzin przyjmują `--load_optuna_params` z plikiem parametrów.
Nie przekazuje się w tym miejscu obiektu badania `study.pkl`.

```powershell
python lstm_model/lstm_train_price.py --train_path datasets/prepared/E999_prep_eth_universal_max_1m --load_optuna_params lstm_model/lstm_data/study/best_params.json --epochs 100
```

`study` jest nazwą przykładową; należy użyć katalogu wypisanego przez strojenie.

## Zgodność LSTM Price

Publiczny tuner Price używa architektury obecnego trenera i pakietu inferencyjnego,
z normalizacją batch i bez dodatkowego schedulera. Zapisuje rozstrzygnięte parametry i listę cech.
Przy wczytywaniu `learning_rate` jest mapowane na `lr`; geometria okna i cechy są zachowywane.
Jawne opcje CLI mają pierwszeństwo w trenerze Price.

Starsze pliki opisujące inny wariant architektury lub scheduler są odrzucane, aby nie sugerować
możliwości wiernego odtworzenia wyniku nieobsługiwanym trenerem.
Pozostałe rodziny zachowują własne reguły wczytywania i pierwszeństwa parametrów.

## Prywatność

Wyniki strojenia mogą zawierać konfiguracje, ścieżki wejściowe, logi i historię prób.
Pozostają w ignorowanych katalogach lokalnych. Publikacja wybranych wyników wymaga osobnego przeglądu.
