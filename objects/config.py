from pathlib import Path

#WSZYSTKIE USTAWIENIA STUDY 2 SA TUTAJ

#sciezki liczone od folderu projektu, zeby dzialalo niezaleznie skad odpalamy skrypt
ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "datasets" / "roboflow_export"
DATASET_DIR = ROOT / "datasets" / "pieniadze"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = ROOT / "runs" / "pieniadze"
MODELS_DIR = ROOT / "models"
BEST_MODEL = MODELS_DIR / "pieniadze_best.pt"
RESULTS_DIR = ROOT / "results"

#podzial zbioru, reszta (70%) idzie do train
TEST_PART = 0.1
VAL_PART = 0.2

#trening - na probny trening: YOLO_MODELS = ["yolo11n"] i EPOCHS = 20
YOLO_MODELS = ["yolo11n", "yolo11s", "yolo11m"]  #n (najszybszy) -> s -> m (najdokladniejszy)
EPOCHS = 100
BATCH = 16  #jak zabraknie pamieci to zmniejszyc do 8
IMGSZ = 640  #(jak monety slabo wyjda to sprobowac 960)
SEED = 0  #ten sam seed wszedzie -> przy powtorzeniu wychodza te same wyniki

CONFIDENCE = 0.5
HISTORY_SIZE = 10  #z ilu ostatnich klatek bierzemy najczestszy odczyt kwoty

#nazwa klasy z Roboflow -> ile to zlotych
CLASS_VALUES = {
    #banknoty
    "10zl": 10.0,
    "20zl": 20.0,
    "50zl": 50.0,
    "100zl": 100.0,
    "200zl": 200.0,
    "500zl": 500.0,
    #monety
    "1gr": 0.01,
    "2gr": 0.02,
    "5gr": 0.05,
    "10gr": 0.10,
    "20gr": 0.20,
    "50gr": 0.50,
    "1zl": 1.0,
    "2zl": 2.0,
    "5zl": 5.0,
}


def model_path(model_name):
    #gdzie lezy wytrenowany model, np. models/pieniadze_yolo11n.pt
    return MODELS_DIR / f"pieniadze_{model_name}.pt"


def best_device():
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
