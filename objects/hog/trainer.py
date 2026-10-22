import pickle
import time

import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torchvision import datasets

from objects.classifiers.config import CLASSIFICATION_DATASET_DIR, CNN_IMAGE_SIZE, HOG_MODEL
from objects.classifiers.fit import check_dataset
from objects.config import MODELS_DIR, SEED
from objects.hog.features import hog_features


class HogTrainer:
    def train(self):
        check_dataset()
        MODELS_DIR.mkdir(exist_ok=True)
        print("\n=== HOG + SVM ===")
        start = time.perf_counter()
        train_x, train_y, class_names = self._split("train")
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", probability=True, random_state=SEED),
        )
        model.fit(train_x, train_y)
        with open(HOG_MODEL, "wb") as f:
            pickle.dump({
                "model": model,
                "class_names": class_names,
                "image_size": CNN_IMAGE_SIZE,
                "training_seconds": time.perf_counter() - start,
            }, f)
        print(f"Zapisano {HOG_MODEL}")

    def _split(self, split):
        folder = datasets.ImageFolder(CLASSIFICATION_DATASET_DIR / split)
        features = []
        labels = []
        for path, class_id in folder.samples:
            image = cv2.imread(path)
            features.append(hog_features(image, CNN_IMAGE_SIZE))
            labels.append(class_id)
        return np.asarray(features), np.asarray(labels), folder.classes
