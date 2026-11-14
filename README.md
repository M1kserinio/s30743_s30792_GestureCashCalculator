# GestureCashCalculator

## Instalacja

Python 3.11 albo 3.12 (MediaPipe 0.10.14 nie ma paczek dla nowszych).
OpenCV musi być w wersji 4 (`opencv-python<5`), bo w OpenCV 5 nie ma już `cv2.HOGDescriptor`.

**Windows** (PowerShell, w folderze projektu):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

- Drugą linijkę PowerShell może zablokować („running scripts is disabled”). Wtedy raz:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` i jeszcze raz `.venv\Scripts\activate`.
- Trzecia linijka instaluje PyTorch z obsługą karty NVIDIA. Musi być **przed** `requirements.txt`,
  bo zwykłe `pip install torch` na Windowsie daje wersję tylko na procesor. Dokładne polecenie
  dla swojej karty: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
  (Windows, Pip, Python, CUDA). Bez karty NVIDIA trzecią linijkę się pomija.
- Ostatnia linijka ma wypisać `True`. `False` = trening pójdzie na procesorze (RT-DETR wtedy w praktyce się nie da).
- Ścieżka do projektu bez polskich liter (np. `C:\inz\...`). OpenCV na Windowsie nie otwiera plików,
  gdy w ścieżce jest `ą`, `ł` itp., i zwraca pusty obraz bez żadnego błędu.

**macOS / Linux**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Duże pliki (zbiory i modele) są na Dysku Google:
https://drive.google.com/drive/u/1/folders/1LgeCDNymjKiNIBoaZ6EgkzROMyp1XqLI

`roboflow.zip`, `datasets/` i `models/` nie idą przez gita. `roboflow.zip` trzeba
skopiować ręcznie do głównego folderu projektu (bez rozpakowywania).
Podział `splits/pieniadze_split.csv` jest w repo, więc na każdym komputerze wychodzi ten sam.

## Banknoty: co porównujemy

Detektory i klasyfikatory robią co innego, więc mają osobne tabele:

- **detektory** (całe zdjęcie → ramki z nominałem): YOLO11 n/s/m i RT-DETR-l
  (transformer, bez NMS),
- **klasyfikatory** (wycięty banknot → nominał albo „brak”): HOG + SVM,
  histogram kolorów + SVM (dwa klasyczne baseline’y), własna prosta CNN,
  MobileNetV3-Small (lekka sieć) i ResNet-18,
- **całe potoki** na tych samych pełnych zdjęciach: sam detektor kontra
  „lokalizator (YOLO z jedną klasą banknot) + klasyfikator”.

Wszystko liczy się na jednym podziale `splits/pieniadze_split.csv`.
Podział jest po sesjach (podobne ujęcia i lustrzane odbicia są zawsze razem),
a nie losowo po plikach. Szczegóły i powód są w `PRZEBIEG_PROGRAMU.md`, w części 11.

```text
objects/detection/    detektory (YOLO i RT-DETR): trening, ocena, FPS, detektor w kamerze
objects/classifiers/  klasyfikatory: wycinki, uczenie, predykcja, ocena; svm.py = HOG i kolory + SVM
objects/cnn/ resnet/ mobilenet/   budowa i trening trzech sieci
objects/pipelines/    porównanie potoków
objects/robustness/   zmiany warunków (światło, tło, kąt, zasłonięcie, zagięcie, brud...)
objects/session_groups.py, dataset_splitter.py   podział po sesjach
objects/negatives.py  zdjęcia bez banknotu (klasa „brak”, fałszywe alarmy)
objects/metrics.py    mAP, dopasowanie ramek, przedziały ufności
```

## Pełny trening od zera

Podział jest nowy, więc **wszystkie stare modele trzeba wytrenować od nowa**.
Na Windowsie całość robi jeden plik (najlepiej puścić na noc):

```powershell
.\trenuj_wszystko.bat
```

Albo krok po kroku (to samo, co jest w `.bat`):

```bash
python -m scripts.prepare_dataset roboflow.zip   # podział + zdjęcia bez banknotu
python -m scripts.classifiers.prepare            # wycinki banknotów + klasa "brak"
python -m scripts.robustness.prepare             # test pod 16 warunkami

python -m scripts.classifiers.train              # 5 klasyfikatorów
python -m scripts.detectors.train                # yolo11n/s/m, lokalizator, rtdetr-l (najdłużej)

python -m scripts.classifiers.eval               # tabela klasyfikatorów
python -m scripts.detectors.eval                 # tabela detektorów, wybiera models/pieniadze_best.pt
python -m scripts.pipelines.eval                 # detektor kontra lokalizator + klasyfikator
python -m scripts.robustness.eval                # odporność potoków i klasyfikatorów
```

Trening da się zawęzić do wybranych modeli, np. gdy jeden trzeba poprawić:

```bash
python -m scripts.classifiers.train resnet cnn   # do wyboru: hog kolory cnn mobilenet resnet
python -m scripts.detectors.train rtdetr         # do wyboru: yolo lokalizator rtdetr
```

Ocena zawsze bierze wszystkie modele, które są w `models/`.
Gdy zabraknie pamięci karty (`CUDA out of memory`), zmniejsz `BATCH` albo `RTDETR_BATCH` w `objects/config.py`.

Opcjonalnie, **przed** `prepare_dataset`: `python -m scripts.record_background sesja_01`
nagrywa klatki bez banknotów (same dłonie, stół) do klasy „brak”.
Nowe zdjęcia oznaczają nowy podział i trening od nowa.

Stare polecenia:

| było | jest |
|---|---|
| `scripts.yolo.train` | `scripts.detectors.train yolo` |
| `scripts.yolo.eval` | `scripts.detectors.eval` |
| `scripts.hog.train`, `scripts.cnn.train`, `scripts.resnet.train` | `scripts.classifiers.train hog` / `cnn` / `resnet` |
| `scripts.hog.eval`, `scripts.cnn.eval`, `scripts.resnet.eval` | `scripts.classifiers.eval` |

- `datasets/odpornosc/<nazwa>/images` + `labels`: własne prawdziwe zdjęcia
  (zgięte, brudne) w formacie YOLO. Ocena odporności policzy je
  jako kolejny warunek.

## Gdzie jest odpowiedź na które pytanie

| Pytanie | Plik w `results/` |
|---|---|
| podział i przeciek | `study2_split_leaks.csv`, `study2_dataset_stats.csv` |
| tabela klasyfikatorów | `study2_classifiers_summary.csv`, `_per_class.csv`, `_confusion.png` |
| tabela detektorów | `study2_detectors_summary.csv`, `_per_class.csv`, `_confusion.png`, `_map_vs_fps.png` |
| potoki | `study2_pipelines_summary.csv`, `_conditions.csv`, `_confusion.png` |
| odporność | `study2_robustness_*.csv`, `study2_robustness_*.png`, `study2_robustness_examples.png` |
| z dłonią / bez | `study2_classifiers_conditions.csv`, `study2_pipelines_conditions.csv` |
| na czym mierzony był czas | `study2_hardware.csv` |

Krzywe uczenia detektorów są w `runs/pieniadze/<model>/results.png`,
a krzywe precision-recall na teście w `runs/pieniadze/eval/<model>_test/`.

## Aplikacja

```bash
python Main.py
```

Aplikacja nie wystartuje bez modeli gestów `gest_results/B_MLP_model.pkl` i `C_SVC_model.pkl`, a tych nie ma w repo.
Gdy ich brakuje: `cd gestures`, `python train_b.py`, `python train_c.py`, `cd ..`.

Na górnym pasku jest `Model: <detektor> | <klasyfikator>`, np. `Model: yolo11s (best) | ResNet-18`.
Na starcie działa detektor, który wygrał w `scripts.detectors.eval`, i nominał podaje on sam.

- `y` — następny detektor: `yolo11n` → `yolo11s` → `yolo11m` → `rtdetr-l` (tylko te, które są w `models/`),
- `n` albo gest `L_THUMB` — kto podaje nominał: sam detektor → HOG+SVM → Kolory+SVM → Prosta CNN →
  MobileNetV3 → ResNet-18 → znowu detektor. Gdy klasyfikator uzna, że w ramce nie ma banknotu
  (klasa „brak”), ramka nie wchodzi do sumy,
- `Tab` — tryb gestów A/B/C, `q` albo `Esc` — wyjście.
