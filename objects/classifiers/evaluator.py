import csv
import pickle
import time
from collections import Counter

import cv2
import torch
from sklearn.metrics import accuracy_score, f1_score

from objects.classifiers.config import (
    CLASSIFICATION_DATASET_DIR,
    CLASSIFICATION_METADATA,
    CNN_MODEL,
    HOG_MODEL,
    RESNET_MODEL,
    classification_device,
)
from objects.classifiers.predict import BanknoteClassifier
from objects.config import CLASS_VALUES
from objects.results_writer import ResultsWriter


class ClassificationEvaluator:
    def __init__(self):
        self.device = classification_device()
        self.writer = ResultsWriter()

    def evaluate_all(self):
        self.evaluate_models("classifiers", [
            ("HOG+SVM", HOG_MODEL),
            ("Prosta CNN", CNN_MODEL),
            ("ResNet-18", RESNET_MODEL),
        ])

    def evaluate_models(self, prefix, models):
        rows = self._test_rows()
        models = [(name, path) for name, path in models if path.exists()]
        if not models:
            raise FileNotFoundError(
                "Brak wag tego modelu. Najpierw go wytrenuj."
            )

        summary = []
        conditions = []
        confusion = []
        for name, path in models:
            print(f"\n=== Ewaluacja: {name} ===")
            classifier = BanknoteClassifier(path, self.device)
            predictions, latency_ms = self._predict(classifier, rows)
            true_labels = [row["class"] for row in rows]

            summary.append({
                "model": name,
                "samples": len(rows),
                "accuracy": round(accuracy_score(true_labels, predictions), 4),
                "f1_macro": round(f1_score(true_labels, predictions, average="macro"), 4),
                "amount_mae_pln": round(self._amount_mae(true_labels, predictions), 2),
                "latency_ms_per_crop": round(latency_ms, 2),
                "training_seconds": round(self._training_seconds(path), 1),
            })
            conditions.extend(self._condition_rows(name, rows, predictions))
            confusion.extend(self._confusion_rows(name, true_labels, predictions))

        self.writer.save_csv(f"study2_{prefix}_summary.csv", summary)
        self.writer.save_csv(f"study2_{prefix}_conditions.csv", conditions)
        self.writer.save_csv(f"study2_{prefix}_confusion.csv", confusion)

    def _predict(self, classifier, rows):
        predictions = []
        start = time.perf_counter()
        for row in rows:
            image = cv2.imread(str(CLASSIFICATION_DATASET_DIR / row["crop_path"]))
            label, _ = classifier.predict(image)
            predictions.append(label)
        elapsed = time.perf_counter() - start
        return predictions, elapsed * 1000 / len(rows)

    def _condition_rows(self, model_name, rows, predictions):
        output = []
        for field in ("object_count", "source"):
            values = sorted({row[field] for row in rows})
            for value in values:
                indices = [i for i, row in enumerate(rows) if row[field] == value]
                true = [rows[i]["class"] for i in indices]
                predicted = [predictions[i] for i in indices]
                output.append({
                    "model": model_name,
                    "condition": field,
                    "value": value,
                    "samples": len(indices),
                    "accuracy": round(accuracy_score(true, predicted), 4),
                    "f1_macro": round(f1_score(true, predicted, average="macro"), 4),
                    "amount_mae_pln": round(self._amount_mae(true, predicted), 2),
                })
        return output

    def _confusion_rows(self, model_name, true_labels, predictions):
        counts = Counter(zip(true_labels, predictions))
        labels = sorted(set(true_labels) | set(predictions))
        return [
            {
                "model": model_name,
                "actual": actual,
                "predicted": predicted,
                "count": counts[(actual, predicted)],
            }
            for actual in labels
            for predicted in labels
        ]

    def _amount_mae(self, true_labels, predictions):
        errors = [
            abs(CLASS_VALUES[actual] - CLASS_VALUES[predicted])
            for actual, predicted in zip(true_labels, predictions)
        ]
        return sum(errors) / len(errors)

    def _training_seconds(self, path):
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f).get("training_seconds", 0.0)
        return torch.load(path, map_location="cpu", weights_only=False).get("training_seconds", 0.0)

    def _test_rows(self):
        if not CLASSIFICATION_METADATA.exists():
            raise FileNotFoundError(
                "Brak metadata.csv. Uruchom python -m scripts.classifiers.prepare"
            )
        with open(CLASSIFICATION_METADATA, newline="") as f:
            rows = [row for row in csv.DictReader(f) if row["split"] == "test"]
        if not rows:
            raise ValueError("Zbior testowy cropow jest pusty")
        return rows
