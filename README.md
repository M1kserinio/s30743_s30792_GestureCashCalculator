# GestureCashCalculator

## Instalacja

Wymagany Python 3.11 albo 3.12.

**Do uruchomienia gotowej aplikacji:** pobierz `models.zip` z Dysku Google i rozpakuj w głównym katalogu projektu. Powinien powstać folder `models`.
Duże pliki, w tym modele, pobierz z [Dysku Google](https://drive.google.com/drive/u/1/folders/1LgeCDNymjKiNIBoaZ6EgkzROMyp1XqLI).

W PowerShell, w katalogu projektu, wykonaj:

```powershell
pip install -r requirements.txt
```
**Do ponownego treningu:** pobierz też do głównego katalogu `roboflow.zip` (nie rozpakowuj) oraz wagi bazowe `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` i `rtdetr-l.pt`.

## Pełny trening od zera


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
## Aplikacja

```bash
python Main.py
```

Do obsługi gestów potrzebne są dodatkowo `gest_results/B_MLP_model.pkl` i `C_SVC_model.pkl`. Nie są częścią `models.zip`; można je wytrenować skryptami `gestures/train_b.py` i `gestures/train_c.py`.

Na górnym pasku jest `Model: <detektor> | <klasyfikator>`, np. `Model: yolo11s (best) | ResNet-18`.
Na starcie działa detektor, który wygrał w `scripts.detectors.eval`, i nominał podaje on sam.

- `y` — następny detektor: `yolo11n` → `yolo11s` → `yolo11m` → `rtdetr-l` (tylko te, które są w `models/`),
- `n` albo gest `L_THUMB` — kto podaje nominał: sam detektor → HOG+SVM → Kolory+SVM → Prosta CNN →
  MobileNetV3 → ResNet-18 → znowu detektor. Gdy klasyfikator uzna, że w ramce nie ma banknotu
  (klasa „brak”), ramka nie wchodzi do sumy,
- `Tab` — tryb gestów A/B/C, `q` albo `Esc` — wyjście.
