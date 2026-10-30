# LSTM Price — ulotka

## Co to jest?

Sieć LSTM przewidująca przyszły poziom ceny na podstawie okna historycznych cech.
Wewnętrzna reprezentacja celu może wykorzystywać logarytm i skalowanie, ale interpretacją prognozy
pozostaje poziom ceny. Zestawienie tej prognozy z ceną bieżącą pozwala obliczyć implikowany zwrot.

## Do czego służy?

Do badania prognoz poziomu ceny i porównywania ich z modelami bezpośrednio prognozującymi zwrot.
Umożliwia sprawdzenie, czy informacja o przyszłej cenie przekłada się na użyteczne sygnały kierunkowe.

## Co udostępnia?

Trening oraz niezależny `lstm_price_model_package`.
Pakiet odwraca odpowiednie transformacje celu, udostępnia prognozowaną cenę,
implikowany logarytmiczny zwrot oraz sygnały BUY/SELL.
Działa wsadowo i na ostatnim oknie dostarczonej historii.

## Jak interpretować wynik?

Cena często zmienia się niewiele między kolejnymi świecami. Prognoza bliska cenie bieżącej
może więc mieć mały błąd, nawet gdy słabo rozpoznaje kierunek następnego ruchu.
Istotnym punktem odniesienia jest prognoza utrzymania ostatniej ceny.
Przydatność transakcyjną należy oceniać osobno od błędu prognozy ceny.

[Instrukcja użycia](USAGE.md) · [Pozostałe modele](../README.md)
