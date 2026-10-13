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

### Prototyp: piłka i uderzenia per gracz

Wymaga dodatkowych zależności (`uv sync --extra vision`) i pliku z punktami kortu (przykład: `examples/yt_calibration.json`, poniżej).

```
uv run padel demo mecz.mp4 --calib kalibracja.json --start 30 --duration 120 -o data/demo
```

Wynik w `data/demo/`: `annotated.mp4` (piłka ze śladem, gracze z licznikami uderzeń, momenty uderzeń), `ball.csv`, `events.csv`, `players.csv`, `stats.json`.

Detekcja jest najwolniejsza (na CPU ok. 6 minut na minutę nagrania 360p) i zapisuje się w `detections.pkl`. Kolejne uruchomienie z `--reuse` (i ewentualnie `--no-video`) liczy tylko tor piłki i uderzenia, co zajmuje sekundy.

Plik kalibracji to co najmniej 4 punkty podłogi kortu widoczne na obrazie, z ich położeniem w metrach (kort 10 × 20 m, bliska linia końcowa y=0, siatka y=10):

```json
{"points": [
  {"image": [316, 343], "court": [5.0, 3.05]},
  {"image": [92, 222],  "court": [0.0, 10.0]},
  {"image": [540, 222], "court": [10.0, 10.0]},
  {"image": [196, 183], "court": [0.0, 20.0]},
  {"image": [441, 183], "court": [10.0, 20.0]}
]}
```

Nagrania trzymaj w katalogu `data/` (jest poza gitem).

## Testy

```
uv run pytest
uv run ruff check .
```
