from pathlib import Path

#WSZYSTKIE USTAWIENIA STUDY 2 SA TUTAJ

#sciezki liczone od folderu projektu, zeby dzialalo niezaleznie skad odpalamy skrypt
ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "datasets" / "roboflow_export"
DATASET_DIR = ROOT / "datasets" / "pieniadze"
DATA_YAML = DATASET_DIR / "data.yaml"
NEGATIVES_DIR = ROOT / "datasets" / "brak_banknotu"  #wycinki tla z naszych zdjec, osobno dla train/val/test
ROBUSTNESS_DIR = ROOT / "datasets" / "odpornosc"
RUNS_DIR = ROOT / "runs" / "pieniadze"
MODELS_DIR = ROOT / "models"
BEST_MODEL = MODELS_DIR / "pieniadze_best.pt"
LOCATOR_MODEL = MODELS_DIR / "pieniadze_locator.pt"
RESULTS_DIR = ROOT / "results"

#podzial jest zapisany w repo, zeby kazdy model (i kazdy z nas) dostal dokladnie te same zdjecia
SPLIT_FILE = ROOT / "splits" / "pieniadze_split.csv"

#podzial zbioru, reszta (~70%) idzie do train
TEST_PART = 0.15
VAL_PART = 0.15
#od tego podobienstwa (cechy ResNet-18, cosinus) dwa zdjecia uznajemy za te sama sesje
#sprawdzone na oko: powyzej 0.88 to zawsze to samo miejsce i ta sama chwila, ponizej zwykle inne sceny
SESSION_SIMILARITY = 0.88

#na szybki probny trening wystarczy YOLO_MODELS = ["yolo11n"] i EPOCHS = 20
YOLO_MODELS = ["yolo11n", "yolo11s", "yolo11m"]  #n (najszybszy) -> s -> m (najdokladniejszy)
RTDETR_MODELS = ["rtdetr-l"]  #transformer, bez NMS
DETECTOR_MODELS = YOLO_MODELS + RTDETR_MODELS
LOCATOR_BASE = "yolo11s"  #z tego robimy lokalizator jednej klasy "banknot" do potoku 2-etapowego
EPOCHS = 100
BATCH = 16  #blad "CUDA out of memory" -> zmniejszyc do 8
RTDETR_BATCH = 8  #RT-DETR zajmuje duzo wiecej pamieci karty niz YOLO, w razie bledu -> 4
IMGSZ = 640
SEED = 0  #ten sam seed wszedzie -> przy powtorzeniu wychodza te same wyniki

CONFIDENCE = 0.44
NMS_IOU = 0.60
HISTORY_SIZE = 10  #z ilu ostatnich klatek bierzemy najczestszy odczyt kwoty

#klasa "brak banknotu" (tlo, sama reka). Detektor dostaje takie zdjecia bez ramek, klasyfikatory jako 7. klase
BACKGROUND_CLASS = "brak"
BACKGROUND_PART = 0.1  #ile zdjec tla dorzucamy do train/val detektora (ultralytics radzi do 10%)

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
    BACKGROUND_CLASS: 0.0,
}
BOX_COLORS = {
    "10zl": (0, 255, 0),
    "20zl": (255, 0, 0),
    "50zl": (0, 255, 255),
    "100zl": (0, 0, 255),
    "200zl": (0, 165, 255),
    "500zl": (255, 0, 255),
    "1gr": (255, 255, 0),
    "2gr": (255, 128, 0),
    "5gr": (128, 255, 0),
    "10gr": (255, 0, 128),
    "20gr": (128, 0, 255),
    "50gr": (0, 128, 255),
    "1zl": (128, 255, 255),
    "2zl": (255, 128, 255),
    "5zl": (128, 128, 255),
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


def benchmark_devices():
    #czas mierzymy zawsze na CPU (jest na kazdym komputerze), a jak jest karta, to jeszcze na niej
    devices = ["cpu"]
    if best_device() != "cpu":
        devices.append(best_device())
    return devices


def device_name(device):
    #ultralytics chce "0", zwykly PyTorch "cuda"
    return "cuda" if device == "0" else device
