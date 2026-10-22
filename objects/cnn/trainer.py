import time

import torch

from objects.classifiers.config import CNN_EPOCHS, CNN_IMAGE_SIZE, CNN_MODEL, classification_device
from objects.classifiers.fit import check_dataset, fit_epochs, loaders, seed_everything
from objects.cnn.model import SimpleCNN
from objects.config import MODELS_DIR


class CnnTrainer:
    def train(self, epochs=CNN_EPOCHS):
        check_dataset()
        seed_everything()
        MODELS_DIR.mkdir(exist_ok=True)
        device = classification_device()
        print("\n=== Prosta CNN ===")
        train_loader, val_loader, class_names = loaders(CNN_IMAGE_SIZE)
        model = SimpleCNN(len(class_names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        fit_epochs(
            model, optimizer, train_loader, val_loader, epochs, CNN_MODEL,
            "cnn", class_names, CNN_IMAGE_SIZE, device,
            training_start=time.perf_counter(),
        )
        print(f"Zapisano najlepszy model: {CNN_MODEL}")
