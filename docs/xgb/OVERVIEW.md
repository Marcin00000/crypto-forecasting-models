# XGBoost — ulotka

## Co to jest?

Model regresyjny złożony z drzew budowanych kolejno tak, aby poprawiały prognozę całego zespołu.
Historia rynku jest zamieniana na tabelę cech, np. przez opóźnienia i statystyki okna.
Model przewiduje przyszły logarytmiczny zwrot.

## Do czego służy?

Do modelowania zależności nieliniowych w danych tabelarycznych i porównywania ich z podejściami sekwencyjnymi.
To przydatny wariant do badania znaczenia cech i wpływu sposobu reprezentacji historii.
Nie wymaga rekurencyjnej sieci neuronowej.

## Co udostępnia?

Kod treningowy oraz niezależny `xgb_model_package`.
Pakiet odtwarza sposób tworzenia cech, wczytuje model i generuje prognozę zwrotu oraz sygnały BUY/SELL.
Obsługuje obliczenia wsadowe i pojedynczy wynik dla ostatniej świecy.

## Jak interpretować wynik?

Sygnały 0–100 powstają z prognozy zwrotu, zmienności rynku i progu zmiany ceny.
Niska wartość błędu regresji nie przesądza o opłacalności sygnałów.
Znaczenie cechy w drzewach nie jest dowodem zależności przyczynowej.
Wyniki zależą od danych, konstrukcji cech i kontroli przeuczenia.

[Instrukcja użycia](USAGE.md) · [Pozostałe modele](../README.md)
