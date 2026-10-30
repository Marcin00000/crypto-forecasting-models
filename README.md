# Crypto Forecasting Models

Kod modeli predykcyjnych dla szeregĂłw czasowych kryptowalut oraz niezaleĹĽne pakiety inferencyjne.
Repozytorium zawiera rdzeĹ„ treningu i inferencji. Wagi modeli, zbiory danych, wyniki eksperymentĂłw
oraz zestawy testowe sÄ… przechowywane osobno.

| Rodzina | Kod treningowy / integracja | Pakiet inferencyjny |
|---|---|---|
| LSTM Log-Return | lstm_logreturn_model/ | lstm_model_package/ |
| LSTM Signal | lstm_signal_model/ | signal_model_package/ |
| XGBoost | xgb_model/ | xgb_model_package/ |
| Random Forest | rf_model/ | rf_model_package/ |
| LSTM Price | lstm_model/ | lstm_price_model_package/ |
| Chronos i TimesFM | foundation_models/ | wrappery w foundation_models/common/ |

WspĂłlny silnik cech znajduje siÄ™ w feature_engineering.py. Pakiety inferencyjne zawierajÄ…
wĹ‚asne moduĹ‚y cech, aby mogĹ‚y byÄ‡ uĹĽywane niezaleĹĽnie od kodu treningowego.

## Ĺšrodowisko

Kod sprawdzono importowo w Pythonie 3.14 z bibliotekami wymienionymi w requirements.txt.
PyTorch naleĹĽy zainstalowaÄ‡ w wersji odpowiedniej dla CPU lub uĹĽywanej karty graficznej.
Instalacja pozostaĹ‚ych zaleĹĽnoĹ›ci:

~~~powershell
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
~~~

Integracje Foundation Models wymagajÄ… dodatkowo bibliotek chronos-forecasting lub timesfm
oraz ich zaleĹĽnoĹ›ci, zgodnych z wybranym modelem bazowym. Pobranie wag tych modeli odbywa siÄ™
dopiero podczas tworzenia odpowiedniego wrappera.

## Trening

Kod przyjmuje przygotowane dane dostarczone przez uĹĽytkownika. Katalog danych powinien zawieraÄ‡
train.csv oraz pliki opisujÄ…ce kolumny i skalowanie wymagane przez wybrany loader.
PeĹ‚ny pipeline przygotowania danych jest poza zakresem tego wydania.

~~~powershell
python lstm_logreturn_model/lstm_train.py --help
python lstm_signal_model/signal_train.py --help
python xgb_model/xgb_train.py --help
python rf_model/rf_train.py --help
python lstm_model/lstm_train_price.py --help
python foundation_models/foundation_finetune.py --help
~~~

PrzykĹ‚ad treningu sieci na dostarczonych danych:

~~~powershell
python lstm_logreturn_model/lstm_train.py --data_path datasets/prepared --epochs 100
~~~

KaĹĽdy program treningowy naleĹĽy uruchamiaÄ‡ w osobnym procesie, z katalogu gĹ‚Ăłwnego repozytorium.
ModuĹ‚y rodzin uĹĽywajÄ… lokalnych pakietĂłw common o tej samej nazwie.

## Inferencja

~~~python
from xgb_model_package import XGBModelPackage

package = XGBModelPackage.from_directory("models/xgb/my_model")
signals = package.predict_batch(ohlcv_dataframe, data_mode="raw")
~~~

Dla pozostaĹ‚ych rodzin dostÄ™pne sÄ… LSTMModelPackage, SignalModelPackage, RFModelPackage
oraz LSTMPriceModelPackage w odpowiadajÄ…cych im pakietach.
KaĹĽdy model wymaga konfiguracji, wag i odpowiednich skalerĂłw.
Signal z architekturÄ… submodel_fusion wymaga rĂłwnieĹĽ przekazania listy katalogĂłw submodeli.
SzczegĂłĹ‚y parametrĂłw opisujÄ… docstringi klas i metod.

## Aktualizacja publicznego rdzenia

NarzÄ™dzie tools/update_core.py odczytuje wyĹ‚Ä…cznie pliki z zatwierdzonej listy tools/core_files.json.
Nie korzysta z historii ani konfiguracji Git katalogu ĹşrĂłdĹ‚owego.

PodglÄ…d zmian:

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

Przed publikacjÄ… naleĹĽy przejrzeÄ‡ rĂłĹĽnice w kodzie. Automatyczna kontrola blokuje typowe
prywatne Ĺ›cieĹĽki, adresy e-mail i formaty tokenĂłw; nie zastÄ™puje przeglÄ…du treĹ›ci.
Nowe pliki i zaleĹĽnoĹ›ci wymagajÄ… jawnego rozszerzenia listy eksportu.
WĹ‚asne zmiany w publicznych kopiach plikĂłw zatrzymujÄ… aktualizacjÄ™ i wymagajÄ… rÄ™cznego scalenia.
UsuniÄ™cia plikĂłw nie sÄ… propagowane automatycznie.

NarzÄ™dzie aktualizujÄ…ce jest jedynym skryptem utrzymaniowym w tym wydaniu.
Skrypty eksperymentĂłw, ewaluatory, tuning, testy i rozbudowana dokumentacja nie sÄ… czÄ™Ĺ›ciÄ… rdzenia.
Pomocnicze obliczenia metryk i wykresĂłw wymagane przez trening pozostajÄ… w moduĹ‚ach common.
