# Wspólne zasady użycia

## Środowisko

Polecenia uruchamia się z katalogu głównego repozytorium, najlepiej w osobnym środowisku Python.
Kontrola importów kodu została wykonana w Pythonie 3.14. Zależności są zapisane w `requirements.txt`.
PyTorch instaluje się osobno, w wariancie zgodnym z CPU lub dostępnym GPU.

```powershell
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
```

Rodziny neuronowe wymagają PyTorch. Integracje Foundation Models wymagają dodatkowych bibliotek
opisanych w ich instrukcji. Kod nie instaluje zależności automatycznie.

## Dane do inferencji

Pakiety inferencyjne przyjmują `pandas.DataFrame`. Dla `data_mode="raw"` podstawą są
świece OHLCV: `open`, `high`, `low`, `close`, `volume`, uporządkowane chronologicznie.
Cena musi być dodatnia, a wymagane kolumny kompletne i pozbawione NaN.
Czas świecy może być przekazany w kolumnie `open_time`.

Część zestawów cech wykorzystuje dodatkowo `taker_buy_base_asset_volume`.
Należy dostarczyć ją, jeżeli wymaga jej wybrany model, aby zachować zgodność z treningiem.
Interwał świec oraz sposób przygotowania danych powinny odpowiadać treningowi modelu.

Pakiet wylicza cechy i stosuje dostępne skalery. Danych w trybie `raw` nie skaluje się wcześniej.
Tryb `prepared` jest przeznaczony dla danych już przygotowanych zgodnie z konfiguracją modelu.
Nie służy do automatycznego rozpoznawania dowolnych tabel.

Do wyznaczenia cech i zmienności potrzeba historii dłuższej niż pojedyncze okno modelu.
Pakiety udostępniają `min_rows_required`. Wynik wsadowy może mieć mniej wierszy od wejścia,
ponieważ początkowe świece służą do rozgrzewki wskaźników.

## Artefakty modelu

Do inferencji potrzebny jest katalog z konfiguracją, wagami i skalerami zgodnymi z treningiem.
Szczegółowe nazwy plików podają instrukcje rodzin. Artefaktów nie dołącza się do publicznego repo.

Pliki pickle/joblib należy ładować wyłącznie z zaufanego źródła: ich odczyt może wykonywać kod.
Sama obecność poprawnego rozszerzenia nie potwierdza bezpieczeństwa ani zgodności modelu.

## Dwa sposoby inferencji

`predict_batch` zwraca prognozy dla kolejnych okien historii.
`predict_live` zwraca wynik dla ostatniej dostępnej świecy z przekazanego bufora.
Obie metody wymagają wystarczającej historii; `predict_live` nie pobiera danych z giełdy.

Sygnały `signal_buy` i `signal_sell` mają skalę 0–100. Nie są kwotą transakcji,
udziałem kapitału ani poleceniem wysłania zlecenia. Ich znaczenie zależy od rodziny modelu.
Wartość 80 nie jest gwarancją 80% trafnych lub zyskownych transakcji.

Wrappery Foundation Models zwracają prognozowany zwrot, a nie taki sam zestaw sygnałów.

## Trening i ocena

Programy treningowe wymagają przygotowanych zbiorów i metadanych obsługiwanych przez dany loader.
Katalog `datasets/prepared` w przykładach oznacza dane dostarczone przez użytkownika.
Pełny pipeline przygotowania danych i zestawy ewaluacyjne nie są częścią tego wydania.

Domyślnym przykładem treningu sieci jest 100 epok. XGBoost i Random Forest mają odrębny sposób
sterowania wielkością zespołu drzew. Rodziny treningowe uruchamia się w osobnych procesach,
ponieważ korzystają z lokalnych pakietów `common` o jednakowej nazwie.

Uruchomienie modelu i uzyskanie liczby nie potwierdza jakości strategii. Ocena wymaga osobnego
okresu danych, punktu odniesienia oraz uwzględnienia kosztów i sposobu wykonania transakcji.
