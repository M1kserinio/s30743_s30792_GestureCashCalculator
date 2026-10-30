from pathlib import Path

from objects.config import MODELS_DIR, ROOT


CLASSIFICATION_DATASET_DIR = ROOT / "datasets" / "banknote_crops"
CLASSIFICATION_METADATA = CLASSIFICATION_DATASET_DIR / "metadata.csv"

CNN_MODEL = MODELS_DIR / "banknote_cnn.pt"
RESNET_MODEL = MODELS_DIR / "banknote_resnet18.pt"
MOBILENET_MODEL = MODELS_DIR / "banknote_mobilenet_v3.pt"
HOG_MODEL = MODELS_DIR / "banknote_hog_svm.pkl"
COLOR_MODEL = MODELS_DIR / "banknote_color_svm.pkl"

#nazwa w tabeli -> plik wag. Ta sama kolejnosc w tabelach, w ocenie potokow i w odpornosci
CLASSIFIERS = [
    ("HOG+SVM", HOG_MODEL),
    ("Kolory+SVM", COLOR_MODEL),
    ("Prosta CNN", CNN_MODEL),
    ("MobileNetV3", MOBILENET_MODEL),
    ("ResNet-18", RESNET_MODEL),
]

CNN_IMAGE_SIZE = 128
RESNET_IMAGE_SIZE = 224
MOBILENET_IMAGE_SIZE = 224
CLASSIFIER_BATCH = 32
CNN_EPOCHS = 40
RESNET_EPOCHS = 20
MOBILENET_EPOCHS = 20


def classification_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
