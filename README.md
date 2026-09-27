# padel-match-analyzer

Analiza meczów padla z nagrania jednej stałej kamery: śledzenie piłki, liczenie uderzeń każdego gracza i prowadzenie wyniku. Projekt i etapy prac opisuje [PLAN.md](PLAN.md).

## Instalacja

Wymagany Python 3.11+ i [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Użycie

```
uv run padel info mecz.mp4    # rozdzielczość, klatkaż, długość nagrania
```

Nagrania trzymaj w katalogu `data/` (jest poza gitem).

## Testy

```
uv run pytest
uv run ruff check .
```
