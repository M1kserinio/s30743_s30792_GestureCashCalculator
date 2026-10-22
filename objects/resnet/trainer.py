import time

import torch

from objects.classifiers.config import (
    RESNET_EPOCHS,
    RESNET_IMAGE_SIZE,
    RESNET_MODEL,
    classification_device,
)
from objects.classifiers.fit import check_dataset, fit_epochs, loaders, seed_everything
from objects.config import MODELS_DIR
from objects.resnet.model import make_resnet18


class ResNetTrainer:
    def train(self, epochs=RESNET_EPOCHS):
        check_dataset()
        seed_everything()
        MODELS_DIR.mkdir(exist_ok=True)
        device = classification_device()
        print("\n=== ResNet-18 ===")
        train_loader, val_loader, class_names = loaders(RESNET_IMAGE_SIZE)
        model = make_resnet18(len(class_names), pretrained=True).to(device)

        frozen_epochs = max(1, epochs // 3)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
        start = time.perf_counter()
        best_accuracy = fit_epochs(
            model, optimizer, train_loader, val_loader, frozen_epochs, RESNET_MODEL,
            "resnet18", class_names, RESNET_IMAGE_SIZE, device,
            training_start=start,
        )

        for parameter in model.parameters():
            parameter.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        fit_epochs(
            model, optimizer, train_loader, val_loader, epochs - frozen_epochs, RESNET_MODEL,
            "resnet18", class_names, RESNET_IMAGE_SIZE, device,
            training_start=start, best_accuracy=best_accuracy, first_epoch=frozen_epochs,
        )
        print(f"Zapisano najlepszy model: {RESNET_MODEL}")
