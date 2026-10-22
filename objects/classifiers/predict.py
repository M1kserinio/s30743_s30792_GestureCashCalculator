import pickle

import cv2
import numpy as np

from objects.cnn.model import SimpleCNN
from objects.hog.features import hog_features
from objects.resnet.model import make_resnet18


class BanknoteClassifier:
    """Wspolny interfejs predykcji dla HOG, prostej CNN i ResNet-18."""

    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.kind = model_path.suffix.lower()
        self.model = None
        self.class_names = []
        self.image_size = None
        self._load()

    def _load(self):
        if self.kind == ".pkl":
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
            self.model = saved["model"]
            self.class_names = saved["class_names"]
            self.image_size = saved["image_size"]
            return

        import torch

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint["image_size"]
        if checkpoint["model_type"] == "cnn":
            self.model = SimpleCNN(len(self.class_names))
        elif checkpoint["model_type"] == "resnet18":
            self.model = make_resnet18(len(self.class_names), pretrained=False)
        else:
            raise ValueError(f"Nieznany typ modelu: {checkpoint['model_type']}")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    def predict(self, bgr_image):
        if self.kind == ".pkl":
            features = hog_features(bgr_image, self.image_size).reshape(1, -1)
            class_id = int(self.model.predict(features)[0])
            confidence = float(np.max(self.model.predict_proba(features)))
            return self.class_names[class_id], confidence

        import torch

        tensor = self.image_tensor(bgr_image, self.image_size).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        class_id = int(probabilities.argmax().item())
        return self.class_names[class_id], float(probabilities[class_id].item())

    @staticmethod
    def image_tensor(bgr_image, image_size):
        import torch

        image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std
