# Ochrona prywatności przy publikacji

Repozytorium zawiera kod i dokumentację. Dane, wagi, wyniki, poświadczenia i notatki robocze
pozostają poza publikowaną zawartością.

## Warstwy ochrony

- `.gitignore` wyklucza typowe dane robocze, poświadczenia i artefakty modeli.
- Hook `.githooks/pre-push` kontroluje nazwy plików w commitach po zatwierdzonej bazie
  oraz uruchamia Gitleaks na pełnej dostępnej historii Git.
- Workflow `Privacy` powtarza kontrolę na GitHubie. Gałąź `main` wymaga
  pull requestu i poprawnego wyniku sprawdzenia `privacy`.

Wbudowane secret scanning i push protection GitHub pozostają włączone.
CI działa po przesłaniu danych, więc nie zastępuje kontroli lokalnej.

## Konfiguracja lokalna

Wymagany jest Gitleaks 8.30.1 z oficjalnego wydania. Hook korzysta z programu w PATH,
ze zmiennej `GITLEAKS_BINARY` albo z lokalnej kopii `.local/security-bin/gitleaks.exe`.

```powershell
git config --local core.hooksPath .githooks
```

Następne polecenia `git push` uruchamiają kontrolę. Brak skanera lub błąd zatrzymuje push.
Klonowanie nie aktywuje hooków automatycznie; ustawienie dotyczy konkretnego checkoutu.

Sam skan historii:

```powershell
gitleaks git . --config .gitleaks.toml --log-opts="--all" --redact --no-banner
```

Wyniki mają maskowane wartości. Raportów z możliwymi sekretami nie publikuje się jako artefaktów CI.
Reguły domyślne są rozszerzone o katalogi użytkowników i adresy e-mail w plikach.
Kontrola nazw blokuje dane, wagi, katalogi robocze i wycofany katalog `tools`.

Bazą kontroli nazw jest ostatni commit sprzed polityki, wcześniej przejrzany pod kątem prywatności.
Jego identyfikator znajduje się w hooku. Skan sekretów obejmuje również wcześniejszą historię.
Automatyczne reguły nie zastępują przeglądu treści, konfiguracji, nowych formatów danych i odnośników.
Ignorowanie pliku nie usuwa go z istniejących commitów.

## Aktualizacja kodu i metadane

Automatyczny eksporter i publikator zostały wycofane. Zmiany przygotowuje się w osobnej gałęzi,
przegląda różnice i publikuje przez pull request. Zgodność kopii kodu nie uprawnia do publikacji
danych i lokalnych ustawień. Autor commitów powinien korzystać z wybranego publicznego
adresu GitHub noreply; dokumentacja nie powinna zawierać prywatnych danych kontaktowych.

Katalog `tools` usunięto z bieżącej wersji. Pozostaje w poprzednich commitach.
Nie przepisywano historii: wcześniejszy przegląd nie wykazał w nim sekretów.

Źródła: [Gitleaks](https://github.com/gitleaks/gitleaks),
[push protection GitHub](https://docs.github.com/en/code-security/concepts/secret-security/push-protection).
