from objects.classifiers.config import (
    RESNET_EPOCHS,
    RESNET_IMAGE_SIZE,
    RESNET_MODEL,
    classification_device,
)
from objects.classifiers.fit import fine_tune, seed_everything
from objects.config import MODELS_DIR
from objects.resnet.model import make_resnet18


class ResNetTrainer:
    def train(self, epochs=RESNET_EPOCHS):
        seed_everything()
        MODELS_DIR.mkdir(exist_ok=True)
        print("\n=== ResNet-18 ===")
        fine_tune(
            lambda class_count: make_resnet18(class_count, pretrained=True), "fc",
            epochs, RESNET_MODEL, "resnet18", RESNET_IMAGE_SIZE, classification_device(),
        )
        print(f"Zapisano najlepszy model: {RESNET_MODEL}")
