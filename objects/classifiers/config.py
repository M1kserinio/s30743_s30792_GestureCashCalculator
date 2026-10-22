from pathlib import Path

from objects.config import MODELS_DIR, ROOT


CLASSIFICATION_DATASET_DIR = ROOT / "datasets" / "banknote_crops"
CLASSIFICATION_METADATA = CLASSIFICATION_DATASET_DIR / "metadata.csv"

CNN_MODEL = MODELS_DIR / "banknote_cnn.pt"
RESNET_MODEL = MODELS_DIR / "banknote_resnet18.pt"
HOG_MODEL = MODELS_DIR / "banknote_hog_svm.pkl"

CNN_IMAGE_SIZE = 128
RESNET_IMAGE_SIZE = 224
CLASSIFIER_BATCH = 32
CNN_EPOCHS = 40
RESNET_EPOCHS = 20


def classification_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
