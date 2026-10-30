# Przygotowanie danych — ulotka

Pipeline przekształca świece OHLCV w zestaw wejściowy modeli. Wylicza cechy, dzieli dane
chronologicznie, dopasowuje skalery i zapisuje opis zbioru.

Tryb uniwersalny udostępnia CSV oraz sekwencje NPY dla odpowiednich loaderów.
Dostępne są podziały procentowe, według dat i długości okresów, również z dodatkowymi splitami walidacyjnymi.
Geometria okna, cechy i sposób skalowania powinny pozostać zgodne między treningiem i ewaluacją.

Podział czasowy sam nie wyklucza późniejszego wykorzystania testu przy wyborze modeli.
Pipeline przyjmuje dane użytkownika i nie pobiera automatycznie historii giełdowej.
Dane źródłowe, skalery i gotowe zbiory pozostają poza repozytorium.

[Instrukcja](USAGE.md) · [Cały proces](../WORKFLOW.md)
