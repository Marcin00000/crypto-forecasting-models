# Przygotowanie danych — instrukcja

[Opis pipeline'u](OVERVIEW.md) · [Cały proces](../WORKFLOW.md)

CSV powinien zawierać `open_time` w milisekundach Unix oraz `open`, `high`, `low`, `close`, `volume`.
Dla cech presji kupna potrzebne jest `taker_buy_base_asset_volume`. Dane muszą mieć poprawne
jednostki i ceny; nazwa pliku pomaga rozpoznać symbol, a odstępy czasu określają interwał.

```powershell
python data_preparation/prep_cli.py datasets/eth_1m.csv --models universal --output_format both --output_dir datasets/prepared --split_mode percentage --train_pct 80 --test_pct 20 --feature_set max --lookback 48 --horizon 12 --data_id E999
```

W przykładzie powstaje `datasets/prepared/E999_prep_eth_universal_max_1m`.
Należy zawsze użyć ścieżki wypisanej przez program, ponieważ nazwa zależy od danych i ustawień.
Wynik zawiera CSV splitów, sekwencje NPY, metadane i skalery. Sposób skalowania CSV opisują metadane;
nie należy ponownie skalować gotowych cech poza zgodnym loaderem.

Okno i horyzont trzeba zachować spójnie w przygotowaniu danych, treningu i ewaluacji.
Rozgrzewka wskaźników oraz podział zmniejszają liczbę dostępnych próbek; krótki CSV może nie wystarczyć.

`--split_mode percentage` używa procentów, `dates` granic dat, a `months` okresów opisanych w interfejsie.
Dodatkowe splity wymagają uwzględnienia ich w `--data_types`. Podział 80/20 jest przykładem technicznym.

```powershell
python data_preparation/prep_cli.py --help
```

Pliki zapisuje się lokalnie, np. w ignorowanym `datasets/`. Ponowne użycie istniejącego ID jest
blokowane bez jawnego `--overwrite`. Gotowe dane i metadane nie są przeznaczone do automatycznej publikacji.
