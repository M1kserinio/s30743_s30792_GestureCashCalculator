"""Trening HOG+SVM, prostej CNN i ResNet-18 na wycinkach banknotow."""

from objects.cnn.trainer import CnnTrainer
from objects.hog.trainer import HogTrainer
from objects.resnet.trainer import ResNetTrainer


if __name__ == "__main__":
    HogTrainer().train()
    CnnTrainer().train()
    ResNetTrainer().train()
    print("\nTrening zakonczony. Uruchom: python -m scripts.classifiers.eval")
