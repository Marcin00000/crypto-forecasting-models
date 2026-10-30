# Foundation Models — instrukcja użycia

[Opis rodziny](OVERVIEW.md) · [Wspólne wymagania](../USAGE.md)

## Przygotowanie

Integracje znajdują się w `foundation_models/common/`.
Wymagają PyTorch oraz biblioteki dostawcy: `chronos-forecasting` dla Chronos lub `timesfm`
i `huggingface_hub` dla TimesFM. Ich wersje muszą obsługiwać wybrane modele bazowe.

Utworzenie wrappera może pobrać wagi z Hugging Face. Wymaga wtedy dostępu do sieci i miejsca
na dysku. Obsługa CPU jest dostępna, lecz czas wykonania zależy od modelu i sprzętu.
Przykłady nie wymagają umieszczania tokenów w kodzie.

## Chronos: pojedynczy szereg

Wejście ma kształt `(liczba_okien, długość_historii)`.
Poniższy szereg jest syntetyczny i pokazuje wyłącznie format danych.

```python
import numpy as np
from foundation_models.common.chronos_wrapper import ChronosWrapper

contexts = (0.001 * np.sin(np.arange(48) / 5.0))[None, :]
model = ChronosWrapper(model_name="amazon/chronos-2", device="cpu")
forecast = model.predict_batch_univariate(contexts, horizon=12)
print(forecast)
```

Wynik zawiera jedną prognozę skumulowanego zwrotu dla każdego okna.
Tryb wielowymiarowy Chronos-2 wymaga poprawnie wyrównanych kowariantów; jego kontrakt opisuje
docstring `predict_batch_multivariate`.

## TimesFM

```python
import numpy as np
from foundation_models.common.timesfm_wrapper import TimesFMWrapper

contexts = (0.001 * np.sin(np.arange(48) / 5.0))[None, :]
model = TimesFMWrapper(device="cpu")
forecast = model.predict_batch(contexts, horizon=12)
print(forecast)
```

Także tutaj wynik dotyczy zwrotu zagregowanego w horyzoncie prognozy.
Przekazanie surowej ceny zamiast zwrotów zmienia interpretację wyniku i nie odtwarza
przyjętego sposobu użycia wrapperów.

## Dostrajanie Chronos

Interfejs programu:

```powershell
python foundation_models/foundation_finetune.py --help
```

Przykład dla danych przygotowanych do tego loadera:

```powershell
python foundation_models/foundation_finetune.py --data_dir datasets/prepared --epochs 100
```

To rzeczywisty trening z zapisem checkpointu. Loader korzysta z danych treningowych
i wydziela chronologicznie część walidacyjną. Sama obecność pliku CSV nie wystarcza:
wymagane są właściwe kolumny i zgodna interpretacja danych.

## Typowe problemy

Brak biblioteki dostawcy lub niezgodna wersja API wymaga dopasowania środowiska.
Błąd pobierania wag należy odróżnić od błędu kształtu wejścia.
Wyniki na danych syntetycznych są przykładem technicznym, a nie miarą jakości modelu.

## Ewaluacja

`foundation_models/foundation_test.py` przyjmuje `--model_type`, `--model_name` i `--data_dir`.
Dla checkpointu po dostrajaniu `--model_name` wskazuje jego katalog.
Interfejs: `python foundation_models/foundation_test.py --help`.
W tym wydaniu nie ma osobnego programu Optuny dla Foundation Models.
