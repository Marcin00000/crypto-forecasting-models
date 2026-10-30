# Crypto Forecasting Models

Kod modeli predykcyjnych dla szeregów czasowych kryptowalut oraz niezależne pakiety inferencyjne.
Repozytorium zawiera rdzeń treningu i inferencji. Wagi modeli, zbiory danych, wyniki eksperymentów
oraz zestawy testowe są przechowywane osobno.

| Rodzina | Kod treningowy / integracja | Pakiet inferencyjny |
|---|---|---|
| LSTM Log-Return | lstm_logreturn_model/ | lstm_model_package/ |
| LSTM Signal | lstm_signal_model/ | signal_model_package/ |
| XGBoost | xgb_model/ | xgb_model_package/ |
| Random Forest | rf_model/ | rf_model_package/ |
| LSTM Price | lstm_model/ | lstm_price_model_package/ |
| Chronos i TimesFM | foundation_models/ | wrappery w foundation_models/common/ |

Wspólny silnik cech znajduje się w feature_engineering.py. Pakiety inferencyjne zawierają
własne moduły cech, aby mogły być używane niezależnie od kodu treningowego.

## Środowisko

Kod sprawdzono importowo w Pythonie 3.14 z bibliotekami wymienionymi w requirements.txt.
PyTorch należy zainstalować w wersji odpowiedniej dla CPU lub używanej karty graficznej.
Instalacja pozostałych zależności:

~~~powershell
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
~~~

Integracje Foundation Models wymagają dodatkowo bibliotek chronos-forecasting lub timesfm
oraz ich zależności, zgodnych z wybranym modelem bazowym. Pobranie wag tych modeli odbywa się
dopiero podczas tworzenia odpowiedniego wrappera.

## Trening

Kod przyjmuje przygotowane dane dostarczone przez użytkownika. Katalog danych powinien zawierać
train.csv oraz pliki opisujące kolumny i skalowanie wymagane przez wybrany loader.
Pełny pipeline przygotowania danych jest poza zakresem tego wydania.

~~~powershell
python lstm_logreturn_model/lstm_train.py --help
python lstm_signal_model/signal_train.py --help
python xgb_model/xgb_train.py --help
python rf_model/rf_train.py --help
python lstm_model/lstm_train_price.py --help
python foundation_models/foundation_finetune.py --help
~~~

Przykład treningu sieci na dostarczonych danych:

~~~powershell
python lstm_logreturn_model/lstm_train.py --data_path datasets/prepared --epochs 100
~~~

Każdy program treningowy należy uruchamiać w osobnym procesie, z katalogu głównego repozytorium.
Moduły rodzin używają lokalnych pakietów common o tej samej nazwie.

## Inferencja

~~~python
from xgb_model_package import XGBModelPackage

package = XGBModelPackage.from_directory("models/xgb/my_model")
signals = package.predict_batch(ohlcv_dataframe, data_mode="raw")
~~~

Dla pozostałych rodzin dostępne są LSTMModelPackage, SignalModelPackage, RFModelPackage
oraz LSTMPriceModelPackage w odpowiadających im pakietach.
Każdy model wymaga konfiguracji, wag i odpowiednich skalerów.
Signal z architekturą submodel_fusion wymaga również przekazania listy katalogów submodeli.
Szczegóły parametrów opisują docstringi klas i metod.

## Aktualizacja publicznego rdzenia

Narzędzie tools/update_core.py odczytuje wyłącznie pliki z zatwierdzonej listy tools/core_files.json.
Nie korzysta z historii ani konfiguracji Git katalogu źródłowego.

Podgląd zmian:

~~~powershell
python tools/update_core.py --source ../research-source
~~~

Zastosowanie zmian lokalnie:

~~~powershell
python tools/update_core.py --source ../research-source --apply
~~~

Zastosowanie zmian, commit i publikacja do origin:

~~~powershell
python tools/update_core.py --source ../research-source --publish
~~~

Przed publikacją należy przejrzeć różnice w kodzie. Automatyczna kontrola blokuje typowe
prywatne ścieżki, adresy e-mail i formaty tokenów; nie zastępuje przeglądu treści.
Nowe pliki i zależności wymagają jawnego rozszerzenia listy eksportu.
Własne zmiany w publicznych kopiach plików zatrzymują aktualizację i wymagają ręcznego scalenia.
Usunięcia plików nie są propagowane automatycznie.

Narzędzie aktualizujące jest jedynym skryptem utrzymaniowym w tym wydaniu.
Skrypty eksperymentów, ewaluatory, tuning, testy i rozbudowana dokumentacja nie są częścią rdzenia.
Pomocnicze obliczenia metryk i wykresów wymagane przez trening pozostają w modułach common.
