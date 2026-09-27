# Plan: Padel Match Analyzer

## Cel

Skrypt w Pythonie, który z nagrania meczu padla robi pięć rzeczy:

1. wykrywa i śledzi piłkę,
2. zlicza uderzenia każdego z 4 graczy,
3. dzieli mecz na wymiany i proponuje, kto zdobył punkt (propozycje zatwierdzasz w oknie przeglądu),
4. prowadzi wynik: punkty, gemy i sety,
5. zapisuje wideo z naniesionymi informacjami oraz statystyki w JSON i CSV.

## Ustalenia

| Temat | Decyzja |
|---|---|
| Źródło | Plik wideo z 1 stałej kamery, ≥ 60 fps |
| Sprzęt | Komputer z GPU NVIDIA (CUDA) |
| Forma | Skrypt CLI w Pythonie (`padel …`) |
| Punktacja | Półautomatyczna: system proponuje, Ty zatwierdzasz w przeglądzie po analizie |
| Gracze | Ręczne oznaczenie na pierwszej klatce (imię i drużyna), dalej śledzenie |
| Uderzenia | Na start liczba uderzeń per gracz; dane zapisywane tak, żeby później dodać typy uderzeń |
| Zasady | Z przewagami, tie-break przy 6:6, mecz do 2 wygranych setów |
| Piłka | Własny model, trenowany na klatkach oznaczonych pomocniczym narzędziem |
| Dane | Twoje nagrania (duże pliki poza gitem, w `data/`) |
| Język | Kod i identyfikatory po angielsku, dokumentacja po polsku |

## Nagrywanie: zalecenia

Obecnie kamera stoi gdzieś między „za kortem nisko” a „za kortem wysoko”. Lepiej ustawić ją **jak najwyżej**:

- **Pozycja:** za tylną szybą, na środku osi kortu, na wysokości **3–4 m**. Wystarczy statyw z tyczką lub mocowanie na konstrukcji kortu.
- **Dlaczego wysoko:**
  - gracze bliżej kamery nie zasłaniają przeciwników,
  - daleka połowa kortu nie jest „ściśnięta”, więc przeliczanie pozycji na metry jest dokładniejsze,
  - piłka częściej leci na tle kortu, a nie ludzi za szybą.
- **Kadr:**
  - cały kort razem z daleką tylną szybą,
  - zapas u góry na loby,
  - wszystkie 4 narożniki kortu widoczne na obrazie.
- **Ustawienia:**
  - 1080p lub 4K przy 60 fps (lub więcej),
  - ręczna ostrość i ekspozycja; w trybie „pro” krótki czas naświetlania, około 1/1000 s, żeby piłka nie była rozmyta,
  - bez słońca czy lamp prosto w obiektyw.
- **Stabilność:** kamera nie może się ruszać. Kalibracja kortu robiona jest raz i obowiązuje dla całego nagrania.

Po M1 porównamy obecne nagrania z nagraniem testowym wg tych zaleceń.

**Wnioski z pierwszego zrzutu (kamera nisko za szybą, szeroki kąt):**

- Bliskie narożniki kortu są poza kadrem, a bliska połowa zajmuje większość obrazu. Kalibracja nie może więc wymagać 4 narożników (zob. etap 1).
- Na górze kadru widać smugi i refleksy na szybie. Trzeba przetrzeć szybę albo postawić kamerę nad nią, bo takie plamy łatwo pomylić z piłką.
- Daleka para graczy jest mała (około 60–80 px wysokości). Tym bardziej warto podnieść kamerę i cofnąć kadr tak, żeby objął bliską linię końcową.
- Zrzut pochodzi z ekranu telefonu (czarne pasy po bokach). Do analizy potrzebny jest oryginalny plik wideo, nie nagranie ekranu.

## Architektura

Analiza przebiega etapami. Każdy etap zapisuje wynik do katalogu wyjściowego. Przy ponownym uruchomieniu gotowe etapy są pomijane: detekcja (najwolniejsza część) liczy się raz, a heurystyki uderzeń i punktów można stroić w sekundach.

```
mecz.mp4
  │
  ├─[1] kalibracja kortu (klik narożników) ───────────► calibration.json
  ├─[2] gracze: YOLO11 + ByteTrack + ręczne ID ───────► players.jsonl
  ├─[3] piłka: detektor + filtr trajektorii ──────────► ball.csv
  ├─[4] uderzenia: zmiany trajektorii + bliskość ─────► hits.jsonl
  ├─[5] wymiany + propozycja zwycięzcy punktu ────────► rallies.json
  ├─[6] przegląd (okno OpenCV, klawiatura) ───────────► review.json
  └─[7] silnik punktacji + render ────────────────────► score.json, stats.json/csv, annotated.mp4
```

### 1. Kalibracja kortu

- Kort padla ma 10 × 20 m, siatka jest w połowie, a linie serwisowe leżą 6,95 m od siatki.
- Na pierwszej klatce klikasz **dowolne widoczne punkty charakterystyczne** podłogi kortu i dla każdego wybierasz, czym jest: narożnik, podstawa słupka siatki, przecięcie linii serwisowej z linią środkową lub boczną. Potrzeba co najmniej 4 punktów, z których żadne 3 nie leżą na jednej prostej (np. 2 dalekie narożniki + 2 podstawy słupków). Bliskie narożniki często są poza kadrem, więc nie mogą być wymagane. Z tego liczona jest homografia (`cv2.findHomography`), czyli przeliczenie punktu na obrazie na metry na korcie.
- **Ograniczenie:** homografia działa tylko dla punktów na podłodze, czyli stóp graczy i miejsc odbicia piłki. Nie działa dla piłki w locie, bo nie znamy jej wysokości.
- Weryfikacja: linie modelu kortu są rysowane na obrazie i od razu widać, czy się pokrywają.
- Kalibrację można użyć ponownie dla kolejnych nagrań, jeśli kamera się nie ruszyła.

### 2. Gracze

- Osoby wykrywa YOLO11 (Ultralytics), a śledzi je ByteTrack.
- Filtr zostawia tylko osoby, których stopy (środek dolnej krawędzi ramki) mieszczą się w obrysie kortu z marginesem. Odpada publiczność, sędzia i ludzie za szybą.
- Na pierwszej klatce klikasz każdego gracza i podajesz imię oraz drużynę (A albo B).
- Tożsamość idzie za śledzonym graczem, a nie za stroną kortu, więc zmiana stron nie psuje przypisań.
- Gdy gracz zniknie (zasłonięcie), nowy ślad jest dopasowywany do „zgubionego” gracza po pozycji na korcie, stronie siatki i kolorach stroju. Niepewne przypisania trafiają do przeglądu.

### 3. Piłka

Detektor ma wspólny interfejs `BallDetector`, więc implementacje można wymieniać.

- **Etap A, punkt odniesienia:**
  - gotowy YOLO11 z klasą „sports ball”,
  - szybki start i pierwszy pomiar jakości.
- **Etap B, własny model:**
  - `padel label` wybiera zróżnicowane klatki z nagrań i podpowiada pozycję piłki z bieżącego modelu,
  - Ty klikasz, poprawiasz albo oznaczasz „brak piłki”; etykiety zapisują się w formacie YOLO,
  - pętla: około 300 klatek → trening → lepsze podpowiedzi → kolejne 300 (łącznie mniej więcej 1–2 h pracy),
  - douczanie YOLO11 w wysokiej rozdzielczości (`imgsz=1280`), bo piłka ma na obrazie kilka–kilkanaście pikseli.
- **Etap C, tylko jeśli B nie wystarczy:**
  - TrackNetV3, sieć patrząca na 3 kolejne klatki naraz, zaprojektowana do małych, rozmytych piłek (tenis, badminton),
  - trenujemy ją na tych samych etykietach.
- **Obróbka trajektorii:**
  - odrzucanie detekcji fizycznie niemożliwych (skoki przez pół obrazu),
  - filtr Kalmana,
  - wypełnianie krótkich luk interpolacją (do około 8 klatek przy 60 fps),
  - każdy punkt oznaczony jako wykryty albo interpolowany.

### 4. Uderzenia

- **Kandydat na zdarzenie:** nagła zmiana kierunku lub prędkości piłki (pik przyspieszenia na trajektorii).
- **Klasyfikacja zdarzenia:**
  - *uderzenie*, gdy piłka jest w zasięgu gracza (ramka gracza powiększona o zasięg rakiety); przypisujemy je najbliższemu graczowi,
  - *odbicie* od podłogi lub szyby, gdy zmiana kierunku zachodzi z dala od graczy. Miejsce odbicia od podłogi przeliczamy na metry, więc wiemy, na której połowie wylądowała piłka.
- **Reguła padla:** drużyny muszą uderzać na zmianę. Jeśli wyjdą dwa „uderzenia” tej samej drużyny z rzędu, jedno z nich to najpewniej odbicie od szyby i zostaje odrzucone to mniej prawdopodobne.
- Progi są w pliku konfiguracyjnym (YAML) i dobierane na zestawie ewaluacyjnym.
- **Przygotowanie pod typy uderzeń.** Dla każdego uderzenia zapisujemy:
  - klatkę, czas i gracza,
  - ramkę gracza i jego pozycję na korcie (m),
  - wektor prędkości piłki przed i po uderzeniu,
  - wysokość piłki względem sylwetki gracza,
  - punkty kluczowe pozy (YOLO11-pose) w oknie ±15 klatek.

  Na tych danych wytrenujemy później klasyfikator (forehand, backhand, wolej, smecz, lob, serwis) bez ponownego przetwarzania wideo.

### 5. Wymiany i propozycja punktu

- **Początek wymiany:** serwis, czyli pierwsze uderzenie po przerwie (kilka sekund bez piłki w grze), gdy serwujący stoi za linią serwisową. Silnik punktacji wie, która drużyna serwuje.
- **Koniec wymiany:** brak piłki i uderzeń przez dłużej niż X sekund.
- **Propozycja zwycięzcy, wersja 1:**
  - po ostatnim uderzeniu jest odbicie na połowie przeciwnika, a przeciwnik nie odbił → punkt dla drużyny, która uderzyła ostatnia,
  - piłka nie dotarła na połowę przeciwnika (siatka) → punkt dla przeciwników,
  - pozostałe sytuacje (aut, szyba po stronie przeciwnika bez odbicia od podłogi) dostają propozycję z niską pewnością.
- Każda propozycja ma pewność 0–1. Niepewne są wyróżnione w przeglądzie.

### 6. Silnik punktacji

- Czysty Python bez zależności od wizji komputerowej, w pełni pokryty testami.
- Zasady:
  - punkty 0, 15, 30, 40, przewaga, gem,
  - set do 6 gemów z przewagą 2,
  - tie-break do 7 (przewaga 2) przy 6:6,
  - mecz do 2 wygranych setów,
  - serw zmienia się co gem.
- Golden point i super tie-break są przewidziane w konfiguracji, ale domyślnie wyłączone.
- Wynik jest zawsze liczony od nowa z listy zatwierdzonych punktów, więc cofanie i poprawki są trywialne.

### 7. Przegląd i wyniki

- `padel review out/` otwiera okno OpenCV, które odtwarza kolejne wymiany w pętli z trajektorią piłki i propozycją. Klawisze:

  | Klawisz | Działanie |
  |---|---|
  | `Enter` | akceptuj propozycję |
  | `1` / `2` | punkt dla drużyny A / B |
  | `0` | to nie był punkt (rozgrzewka, let) |
  | `←` / `→` | poprzednia / następna wymiana |
  | `spacja` | pauza |
  | `+` / `-` | szybciej / wolniej |
  | `q` | zapisz i wyjdź (można wrócić później) |

- `padel render out/` tworzy:
  - `annotated.mp4` z ramkami graczy (imię i licznik uderzeń), śladem piłki i tablicą wyniku,
  - `stats.json` i `stats.csv` z uderzeniami per gracz, listą wymian (czas, liczba uderzeń, zwycięzca) i wynikiem końcowym.

## CLI

```
padel info mecz.mp4               # rozdzielczość, fps, długość
padel calibrate mecz.mp4          # klikanie kortu → calibration.json
padel label mecz.mp4              # oznaczanie piłki z podpowiedziami
padel train-ball                  # trening detektora piłki
padel analyze mecz.mp4 -o out/    # etapy 2–5 (z cache)
padel review out/                 # zatwierdzanie punktów
padel render out/                 # wideo + statystyki
padel eval out/ --gt gt.json      # porównanie z ręcznie oznaczoną prawdą
```

## Struktura repozytorium

```
pyproject.toml
config/default.yaml          # progi i parametry heurystyk
src/padel/
  cli.py                     # typer
  video.py                   # odczyt klatek, metadane
  court/                     # kalibracja, homografia, model kortu
  players/                   # detekcja, tracking, tożsamość
  ball/                      # BallDetector, implementacje, trajektoria
  events/                    # uderzenia, odbicia, wymiany
  scoring/                   # silnik punktacji
  review/                    # okno przeglądu
  render/                    # nakładki na wideo, eksport statystyk
  labeling/                  # narzędzie do oznaczania piłki
  eval/                      # metryki
tests/
data/     (gitignore)        # nagrania, etykiety, wyniki
models/   (gitignore)        # wagi modeli
```

## Stos technologiczny

- Python 3.11+ ze środowiskiem zarządzanym przez `uv`,
- Ultralytics (YOLO11 detekcja i poza, ByteTrack) i PyTorch z CUDA,
- OpenCV, NumPy, SciPy,
- pydantic (modele danych i konfiguracja), typer (CLI),
- pytest i ruff.

## Ewaluacja

Bez pomiaru nie wiadomo, czy zmiana coś poprawia. Dlatego z jednego Twojego nagrania (około 10 minut) robimy **zestaw ewaluacyjny** i nigdy nie trenujemy na nim modelu. Zawiera:

- około 200 klatek z oznaczoną piłką,
- listę uderzeń (klatka i gracz), oznaczaną klawiszem podczas odtwarzania w narzędziu,
- listę punktów.

| Metryka | Wstępny cel MVP |
|---|---|
| Piłka: recall (tolerancja ~5 px) | ≥ 85% |
| Uderzenia: F1 (tolerancja ±3 klatki) | ≥ 85% |
| Uderzenia: poprawny gracz | ≥ 90% |
| Propozycja zwycięzcy punktu | ≥ 70% (resztę poprawiasz w przeglądzie) |

Cele są wstępne i skorygujemy je po pierwszych pomiarach.

## Kamienie milowe

| # | Zakres | Gotowe, gdy |
|---|---|---|
| M0 ✅ | Szkielet: `pyproject`, ruff, pytest, CLI, `padel info` | `padel info` wypisuje fps i rozdzielczość Twojego pliku |
| M1 | Kalibracja kortu i homografia | Linie kortu z modelu pokrywają się z obrazem |
| M2 | Gracze: detekcja, tracking, filtr kortu, ręczne ID, odzyskiwanie ID | Wideo z ramkami i imionami bez zamiany tożsamości na 10-minutowym fragmencie |
| M3 | Piłka: punkt odniesienia → narzędzie do oznaczania → własny model → trajektoria | Recall piłki ≥ 85% na zestawie ewaluacyjnym |
| M4 | Uderzenia z przypisaniem do graczy i zapisem kontekstu (poza) | F1 uderzeń ≥ 85% |
| M5 | Silnik punktacji (testy), wymiany, propozycje zwycięzcy | Testy silnika zielone, propozycje z pewnością w `rallies.json` |
| M6 | Przegląd, render i eksport statystyk | Pełny przebieg: wideo → zatwierdzony wynik → `annotated.mp4` + statystyki |
| M7 | *Później:* typy uderzeń, golden point, automatyczna kalibracja, ewentualnie interfejs webowy | — |

## Ryzyka

| Ryzyko | Jak ograniczamy |
|---|---|
| Piłka zasłonięta (gracze, siatka, refleksy na szybie) | Interpolacja luk, TrackNet (kontekst 3 klatek), wysoka kamera |
| Piłka wylatuje z kadru przy lobach | Zapas u góry kadru; luka w trajektorii nie kończy wymiany od razu |
| Uderzenia po dalekiej stronie są małe i trudne | Wysoka kamera, trening na klatkach z daleką stroną, 4K jeśli możliwe |
| Brak wysokości piłki (tylko 2D) myli odbicie od szyby z uderzeniem | Reguła naprzemiennych uderzeń drużyn, przegląd niepewnych przypadków |
| Podobnie ubrani gracze | Ręczne ID i dopasowanie po pozycji; przegląd niepewnych przypisań |
| Automatyczny werdykt punktu bywa błędny | Tryb półautomatyczny z pewnością i szybkim przeglądem |

## Otwarte kwestie

- Wysokość i pozycja kamery: porównamy obecne nagrania z nagraniem wg zaleceń powyżej.
- Czy na nagraniach widać cały kort łącznie z daleką tylną szybą? Sprawdzimy w M1 na Twoim pliku.
- Gdzie trzymać duże pliki (nagrania, wagi)? Domyślnie lokalnie w `data/` i `models/`, poza gitem.
