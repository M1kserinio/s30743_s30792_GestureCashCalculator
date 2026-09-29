import shutil
import zipfile

import yaml

from objects.config import CLASS_VALUES, DATA_YAML, DATASET_DIR, EXPORT_DIR
from objects.dataset_splitter import DatasetSplitter
from objects.dataset_stats import DatasetStats

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


class RoboflowDataset:
    def __init__(self, zip_path):
        self.zip_path = zip_path  #zip pobrany z Roboflow w formacie YOLOv8
        self.class_names = []

    def prepare(self):
        self.unzip()
        self.class_names = self.read_class_names()
        self.check_class_names()

        samples = self.find_images()
        if not samples:
            print(f"Brak zdjec w {EXPORT_DIR} - czy eksport jest w formacie YOLOv8?")
            return

        split = DatasetSplitter().split(samples)
        self.copy_files(split)
        self.save_data_yaml()
        DatasetStats(self.class_names).report(split)
        print(f"\nGotowe: {DATA_YAML}")

    def unzip(self):
        #czyscimy stary eksport, zeby nie pomieszac zdjec z dwoch wersji
        if EXPORT_DIR.exists():
            shutil.rmtree(EXPORT_DIR)
        with zipfile.ZipFile(self.zip_path) as zf:
            zf.extractall(EXPORT_DIR)

    def read_class_names(self):
        #Roboflow zapisuje nazwy klas w data.yaml, kolejnosc = numer klasy w plikach .txt
        with open(EXPORT_DIR / "data.yaml") as f:
            data = yaml.safe_load(f)
        return data["names"]

    def check_class_names(self):
        for name in self.class_names:
            if name not in CLASS_VALUES:
                print(f"UWAGA: klasa '{name}' nie ma kwoty w CLASS_VALUES (objects/config.py) - w apce bedzie 0 zl")

    def find_images(self):
        #Roboflow daje foldery train/valid/test, w kazdym images/ i labels/
        #bierzemy wszystko razem i dzielimy od nowa w DatasetSplitter
        samples = []
        for image in sorted(EXPORT_DIR.glob("*/images/*")):
            if image.suffix.lower() not in IMAGE_EXTS:
                continue
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            samples.append((image, label))
        return samples

    def copy_files(self, split):
        #zawsze od zera, zeby nie zostaly pliki z poprzedniego uruchomienia
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)

        for split_name, samples in split.items():
            images_dir = DATASET_DIR / split_name / "images"
            labels_dir = DATASET_DIR / split_name / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            for image, label in samples:
                shutil.copy(image, images_dir / image.name)
                #brak .txt = zdjecie bez pieniedzy (tlo), YOLO to rozumie
                if label.exists():
                    shutil.copy(label, labels_dir / label.name)

    def save_data_yaml(self):
        #plik dla YOLO: gdzie sa zdjecia i jak sie nazywaja klasy
        #"path" jako pelna sciezka, bo sciezki z Roboflow (../train/images) potrafia sie rozjechac
        data = {
            "path": str(DATASET_DIR),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": self.class_names,
        }
        with open(DATA_YAML, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
