# LSTM Signal — ulotka

## Co to jest?

Model sekwencyjny, który bezpośrednio przewiduje dwa wyniki sygnałowe: dla kupna i dla sprzedaży.
Zamiast najpierw prognozować cenę lub zwrot, uczy się reprezentacji związanej z docelowymi sygnałami.

## Do czego służy?

Do badania jakości sygnałów transakcyjnych oraz łączenia informacji z różnych źródeł.
Warianty samodzielne korzystają z cech rynkowych. Warianty fuzji mogą dodatkowo wykorzystywać
prognozy podmodeli Log-Return, XGBoost, Random Forest lub LSTM Price.

## Co udostępnia?

Trening i niezależny `signal_model_package` z inferencją wsadową oraz wynikiem dla ostatniej świecy.
Pakiet zwraca `prob_buy` i `prob_sell` w skali 0–1 oraz odpowiadające im sygnały w skali 0–100.
Dwa wyjścia nie muszą sumować się do jedności.

## Jak interpretować wynik?

Wartości są wynikami wyuczonymi dla przyjętej definicji sygnału.
Nie oznaczają automatycznie skalibrowanego prawdopodobieństwa zyskownej transakcji.
Fuzja wymaga poprawnych i zgodnych podmodeli; brak jednego z nich nie jest równoważny modelowi samodzielnemu.
Ocena powinna obejmować stabilność sygnałów i zachowanie poza okresem uczenia.

[Instrukcja użycia](USAGE.md) · [Pozostałe modele](../README.md)
