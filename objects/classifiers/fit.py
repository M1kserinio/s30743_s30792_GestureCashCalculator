import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from objects.classifiers.config import CLASSIFICATION_DATASET_DIR, CLASSIFIER_BATCH
from objects.config import SEED


def check_dataset():
    required = [CLASSIFICATION_DATASET_DIR / split for split in ("train", "val", "test")]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            "Brak zbioru cropow. Uruchom python -m scripts.classifiers.prepare"
        )


def loaders(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.15),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder(CLASSIFICATION_DATASET_DIR / "train", train_transform)
    val = datasets.ImageFolder(CLASSIFICATION_DATASET_DIR / "val", eval_transform)
    if train.classes != val.classes:
        raise ValueError("Train i val nie maja tych samych klas")
    return (
        DataLoader(train, batch_size=CLASSIFIER_BATCH, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=CLASSIFIER_BATCH, shuffle=False, num_workers=0),
        train.classes,
    )


def fit_epochs(model, optimizer, train_loader, val_loader, epochs, path,
               model_type, class_names, image_size, device, training_start,
               best_accuracy=-1.0, first_epoch=0):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(first_epoch, first_epoch + epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)

        accuracy = _accuracy(model, val_loader, device)
        mean_loss = total_loss / len(train_loader.dataset)
        print(f"Epoka {epoch + 1}: loss={mean_loss:.4f}, val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": image_size,
                "val_accuracy": accuracy,
                "training_seconds": time.perf_counter() - training_start,
            }, path)
    return best_accuracy


def _accuracy(model, loader, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predictions == labels).sum().item())
    return correct / len(loader.dataset)


def seed_everything():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
