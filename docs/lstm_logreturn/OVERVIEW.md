# LSTM Log-Return — ulotka

## Co to jest?

Rodzina modeli sekwencyjnych prognozujących logarytmiczną stopę zwrotu ceny w zadanym horyzoncie.
Model analizuje okno historycznych cech, dzięki czemu może uwzględniać kolejność i kontekst zmian rynku.
Zależnie od konfiguracji wykorzystywane są m.in. warstwy LSTM lub GRU oraz dodatkowe mechanizmy przetwarzania sekwencji.

## Do czego służy?

Do badania oczekiwanego kierunku i wielkości zmiany ceny, porównywania zestawów cech
oraz budowania warstwy sygnałowej dla systemu transakcyjnego.
Prognoza zwrotu jest przeliczana na sygnały BUY i SELL z wykorzystaniem zmienności oraz progu ruchu ceny.

## Co udostępnia?

Kod treningowy oraz niezależny pakiet `lstm_model_package`.
Pakiet oblicza prognozy wsadowo albo dla ostatniej świecy z dostarczonej historii.
Wynik obejmuje prognozę zwrotu `mu`, oszacowanie zmienności `sigma` i sygnały w skali 0–100.

## Jak interpretować wynik?

To model wielkości zwrotu, a nie bezpośredni klasyfikator decyzji kupna i sprzedaży.
Wysoki sygnał wynika z przyjętej transformacji prognozy i zmienności.
Model stale przewidujący zwroty bliskie zeru może mieć niewielki błąd liczbowy, a jednocześnie
nie dostarczać użytecznej informacji kierunkowej.

[Instrukcja użycia](USAGE.md) · [Pozostałe modele](../README.md)
