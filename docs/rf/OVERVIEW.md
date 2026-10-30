# Random Forest — ulotka

## Co to jest?

Model regresyjny uśredniający prognozy wielu drzew decyzyjnych.
Poszczególne drzewa korzystają z losowania obserwacji lub cech, co pozwala ograniczać zależność
wyniku od pojedynczego drzewa. Model operuje na tabelarycznej reprezentacji historii rynku.

## Do czego służy?

Do budowy prognoz logarytmicznych zwrotów oraz jako punkt porównania dla XGBoost i modeli neuronowych.
Pozwala badać nieliniowe związki cech bez korzystania z sieci rekurencyjnej.
W przeciwieństwie do boostingu drzewa nie są budowane jako kolejne poprawki błędów poprzednich drzew.

## Co udostępnia?

Kod treningowy i niezależny `rf_model_package`.
Pakiet przygotowuje cechy, wyznacza prognozę zwrotu i przelicza ją na sygnały BUY/SELL.
Inferencja nie wymaga PyTorch ani XGBoost.

## Jak interpretować wynik?

Uśrednianie prognoz nie gwarantuje stabilności po zmianie warunków rynkowych.
Bardzo niski błąd na zbiorze treningowym może współistnieć ze słabym wynikiem poza próbą.
Sygnały powstają z dodatkowego założenia o zmienności i progu ruchu ceny; nie są
bezpośrednią instrukcją otwarcia pozycji.

[Instrukcja użycia](USAGE.md) · [Pozostałe modele](../README.md)
