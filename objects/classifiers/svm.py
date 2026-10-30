import pickle
import time

import cv2
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import datasets

from objects.classifiers.config import CLASSIFICATION_DATASET_DIR, CNN_IMAGE_SIZE, COLOR_MODEL, HOG_MODEL
from objects.classifiers.fit import check_dataset
from objects.config import MODELS_DIR, SEED


def hog_features(bgr_image, image_size=128):
    """Kierunki krawedzi w malych komorkach obrazu (HOG). Podejscie sprzed sieci splotowych, koloru nie widzi."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor((image_size, image_size), (16, 16), (8, 8), (8, 8), 9)
    return hog.compute(gray).reshape(-1)


def color_features(bgr_image, image_size=128):
    """Jakie kolory i w jakich ilosciach sa na wycinku. Ksztaltu i napisow ta metoda w ogole nie widzi.

    Polskie banknoty roznia sie barwa (20 zl fioletowy, 50 zl niebieski, 100 zl zielony, 200 zl zolty),
    wiec juz sam rozklad kolorow sporo mowi. Za to kazda zmiana swiatla przesuwa ten rozklad.
    """
    image = cv2.resize(bgr_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #odcien x nasycenie: 18 przedzialow barwy po 10 stopni i 4 poziomy nasycenia
    hue_saturation = cv2.calcHist([hsv], [0, 1], None, [18, 4], [0, 180, 0, 256]).ravel()
    brightness = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    histogram = np.concatenate([hue_saturation, brightness])
    #pierwiastek z udzialow, zeby maly fioletowy pasek nie zginal przy duzej bialej plamie
    return np.sqrt(histogram / histogram.sum())


#nazwa cech zapisana w pickle -> funkcja, ktora je liczy (predict.py bierze ja stad)
FEATURES = {
    "hog": hog_features,
    "kolory": color_features,
}


class SvmTrainer:
    """Klasyczne metody (bez sieci): stale cechy z kazdego wycinka i SVM na nich."""

    def __init__(self, title, features, kernel, path):
        self.title = title
        self.features = features
        self.kernel = kernel
        self.path = path

    def train(self):
        check_dataset()
        MODELS_DIR.mkdir(exist_ok=True)
        print(f"\n=== {self.title} ===")
        start = time.perf_counter()
        train_x, train_y, class_names = self._split("train")
        #SVM sam z siebie nie daje prawdopodobienstw, a potrzebujemy pewnosci (prog w apce, mAP w potokach).
        #Kalibracja Platta na 5 foldach - to samo co dawne SVC(probability=True), ktore sklearn wycofuje
        model = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(kernel=self.kernel, random_state=SEED), method="sigmoid", cv=5, ensemble=False),
        )
        model.fit(train_x, train_y)
        with open(self.path, "wb") as f:
            pickle.dump({
                "model": model,
                "features": self.features,
                "class_names": class_names,
                "image_size": CNN_IMAGE_SIZE,
                "training_seconds": time.perf_counter() - start,
            }, f)
        print(f"Zapisano {self.path}")

    def _split(self, split):
        folder = datasets.ImageFolder(CLASSIFICATION_DATASET_DIR / split)
        features = []
        labels = []
        for path, class_id in folder.samples:
            image = cv2.imread(path)
            features.append(FEATURES[self.features](image, CNN_IMAGE_SIZE))
            labels.append(class_id)
        return np.asarray(features), np.asarray(labels), folder.classes


class HogTrainer(SvmTrainer):
    def __init__(self):
        #liniowy SVM: cech HOG jest kilka tysiecy, w takiej przestrzeni klasy zwykle da sie rozdzielic plaszczyzna
        super().__init__("HOG + SVM", "hog", "linear", HOG_MODEL)


class ColorTrainer(SvmTrainer):
    def __init__(self):
        #jadro RBF: cech jest tylko 80, liniowa granica miedzy kolorami bylaby za prosta
        super().__init__("Histogram kolorow + SVM", "kolory", "rbf", COLOR_MODEL)
