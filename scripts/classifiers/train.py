"""Trening klasyfikatorow na wycinkach banknotow (z klasa "brak").

python -m scripts.classifiers.train              wszystkie piec
python -m scripts.classifiers.train resnet cnn   tylko wybrane
"""

import sys

from objects.classifiers.svm import ColorTrainer, HogTrainer
from objects.cnn.trainer import CnnTrainer
from objects.mobilenet.trainer import MobileNetTrainer
from objects.resnet.trainer import ResNetTrainer

TRAINERS = {
    "hog": HogTrainer,
    "kolory": ColorTrainer,
    "cnn": CnnTrainer,
    "mobilenet": MobileNetTrainer,
    "resnet": ResNetTrainer,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(TRAINERS)
    unknown = [name for name in names if name not in TRAINERS]
    if unknown:
        sys.exit(f"Nie znam: {' '.join(unknown)}. Do wyboru: {' '.join(TRAINERS)}")
    for name in names:
        TRAINERS[name]().train()
    print("\nTrening zakonczony. Dalej: python -m scripts.classifiers.eval")
