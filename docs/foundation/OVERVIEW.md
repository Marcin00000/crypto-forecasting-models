# Foundation Models — ulotka

## Co to jest?

Integracje gotowych modeli szeregów czasowych Chronos i TimesFM.
Punktem wyjścia są wagi udostępnione przez dostawcę, dzięki czemu można badać prognozowanie
bez wcześniejszego treningu własnej sieci od początku.

## Do czego służą?

Do budowy punktów odniesienia dla modeli trenowanych lokalnie oraz badania przenoszenia
wiedzy między szeregami czasowymi. W projekcie wejściem są szeregi logarytmicznych zwrotów.
Domyślne integracje dotyczą Chronos-2 oraz TimesFM 2.5.

## Co udostępnia kod?

Wrappery do prognoz wsadowych oraz osobny program dostrajania Chronos.
Chronos udostępnia tryb pojedynczego szeregu, a integracja Chronos-2 również tryb z kowariantami.
TimesFM przyjmuje partie pojedynczych szeregów.
Wrappery agregują prognozy kroków w prognozę zwrotu całego horyzontu.

## Jak interpretować wynik?

Wynikiem wrappera jest liczba opisująca prognozowany zwrot, a nie zestaw sygnałów
`signal_buy` i `signal_sell` taki jak w pozostałych pakietach.
Ocena i przekształcenie prognozy w decyzję transakcyjną wymagają dodatkowej logiki.

Wstępne wytrenowanie nie potwierdza przydatności modelu do konkretnego rynku, interwału
lub okresu. Porównanie powinno używać tych samych danych i zasad oceny co modele lokalne.

[Instrukcja użycia](USAGE.md) · [Pozostałe modele](../README.md)
