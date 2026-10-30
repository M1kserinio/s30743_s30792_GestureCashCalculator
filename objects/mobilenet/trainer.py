from objects.classifiers.config import (
    MOBILENET_EPOCHS,
    MOBILENET_IMAGE_SIZE,
    MOBILENET_MODEL,
    classification_device,
)
from objects.classifiers.fit import check_dataset, fine_tune, seed_everything
from objects.config import MODELS_DIR
from objects.mobilenet.model import make_mobilenet


class MobileNetTrainer:
    def train(self, epochs=MOBILENET_EPOCHS):
        check_dataset()
        seed_everything()
        MODELS_DIR.mkdir(exist_ok=True)
        print("\n=== MobileNetV3-Small ===")
        #glowa to caly "classifier" (dwie warstwy liniowe), tak jak "fc" w ResNet
        fine_tune(
            lambda class_count: make_mobilenet(class_count, pretrained=True), "classifier",
            epochs, MOBILENET_MODEL, "mobilenet_v3_small", MOBILENET_IMAGE_SIZE, classification_device(),
        )
        print(f"Zapisano najlepszy model: {MOBILENET_MODEL}")
