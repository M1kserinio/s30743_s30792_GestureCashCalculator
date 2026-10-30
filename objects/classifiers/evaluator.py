import csv
import pickle
import time
from collections import Counter

import cv2
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from objects.classifiers.config import CLASSIFICATION_DATASET_DIR, CLASSIFICATION_METADATA, CLASSIFIERS
from objects.classifiers.predict import BanknoteClassifier
from objects.config import BACKGROUND_CLASS, CLASS_VALUES, benchmark_devices, device_name
from objects.metrics import bootstrap_interval
from objects.results_writer import ResultsWriter
from objects.test_tags import read_test_tags

WARMUP = 10  #pierwsze predykcje sa duzo wolniejsze, nie licze ich


class ClassificationEvaluator:
    def __init__(self):
        self.writer = ResultsWriter()

    def evaluate_all(self):
        rows = self._test_rows()
        models = [(name, path) for name, path in CLASSIFIERS if path.exists()]
        if not models:
            raise FileNotFoundError("Brak wag klasyfikatorow. Najpierw: python -m scripts.classifiers.train")

        #wszystkie wycinki od razu w pamieci, zeby czas byl czasem modelu, a nie czytania z dysku
        crops = [cv2.imread(str(CLASSIFICATION_DATASET_DIR / row["crop_path"])) for row in rows]
        true_labels = [row["class"] for row in rows]
        tags = read_test_tags()

        summary = []
        per_class = []
        conditions = []
        confusion = []
        matrices = {}
        for name, path in models:
            print(f"\n=== Ewaluacja: {name} ===")
            #wynik zawsze z CPU, czas z kazdego urzadzenia osobno
            classifier = BanknoteClassifier(path, "cpu")
            predictions = [classifier.predict(crop)[0] for crop in crops]

            summary.append(self._summary_row(name, path, classifier, rows, true_labels, predictions, crops))
            per_class.extend(self._per_class_rows(name, true_labels, predictions))
            conditions.extend(self._condition_rows(name, rows, predictions, tags))
            labels, matrix = self._confusion(true_labels, predictions)
            matrices[name] = (labels, matrix)
            confusion.extend(
                {"model": name, "actual": actual, "predicted": predicted, "count": matrix[i][j]}
                for i, actual in enumerate(labels) for j, predicted in enumerate(labels)
            )

        self.writer.save_csv("study2_classifiers_summary.csv", summary)
        self.writer.save_csv("study2_classifiers_per_class.csv", per_class)
        self.writer.save_csv("study2_classifiers_conditions.csv", conditions)
        self.writer.save_csv("study2_classifiers_confusion.csv", confusion)
        self.writer.save_confusion_plot("study2_classifiers_confusion.png", matrices)
        self.writer.save_hardware()

    def _summary_row(self, name, path, classifier, rows, true_labels, predictions, crops):
        correct = [actual == predicted for actual, predicted in zip(true_labels, predictions)]
        low, high = bootstrap_interval(correct, [row["image_name"] for row in rows])
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="macro", zero_division=0)
        params = classifier.parameter_count()

        row = {
            "model": name,
            "params_mln": round(params / 1_000_000, 3) if params else "",
            "samples": len(rows),
            "accuracy": round(accuracy_score(true_labels, predictions), 4),
            "accuracy_ci95_low": low,
            "accuracy_ci95_high": high,
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "f1_macro": round(f1, 4),
            "false_alarm_rate": self._false_alarm_rate(true_labels, predictions),
            "missed_as_background": self._missed_rate(true_labels, predictions),
            "amount_mae_pln": round(self._amount_mae(true_labels, predictions), 2),
        }
        row.update(self._timing(path, crops))
        row["training_seconds"] = round(self._training_seconds(path), 1)
        return row

    def _false_alarm_rate(self, true_labels, predictions):
        #wycinek tla, ktory model nazwal banknotem = falszywy alarm
        background = [p for t, p in zip(true_labels, predictions) if t == BACKGROUND_CLASS]
        if not background:
            return ""
        return round(sum(p != BACKGROUND_CLASS for p in background) / len(background), 4)

    def _missed_rate(self, true_labels, predictions):
        #prawdziwy banknot, ktory model uznal za tlo - w kamerze ta ramka by zniknela
        banknotes = [p for t, p in zip(true_labels, predictions) if t != BACKGROUND_CLASS]
        return round(sum(p == BACKGROUND_CLASS for p in banknotes) / len(banknotes), 4)

    def _timing(self, path, crops):
        row = {}
        for device in benchmark_devices():
            name = device_name(device)
            if path.suffix == ".pkl" and name != "cpu":
                #SVM ze sklearn liczy tylko na CPU
                row[f"ms_{name}"] = ""
                row[f"fps_{name}"] = ""
                continue
            classifier = BanknoteClassifier(path, name)
            for crop in crops[:WARMUP]:
                classifier.predict(crop)
            start = time.perf_counter()
            for crop in crops:
                classifier.predict(crop)
            ms = (time.perf_counter() - start) * 1000 / len(crops)
            row[f"ms_{name}"] = round(ms, 2)
            row[f"fps_{name}"] = round(1000 / ms, 1)
        return row

    def _per_class_rows(self, model_name, true_labels, predictions):
        labels = sorted(set(true_labels))
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, labels=labels, zero_division=0)
        return [
            {
                "model": model_name,
                "class": label,
                "precision": round(precision[i], 4),
                "recall": round(recall[i], 4),
                "f1": round(f1[i], 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        ]

    def _condition_rows(self, model_name, rows, predictions, tags):
        output = []
        fields = {"object_count": [row["object_count"] for row in rows],
                  "source": [row["source"] for row in rows]}
        #reczne opisy (np. wzor stary/nowy) z splits/warunki_test.csv
        for field in sorted({field for row in rows for field in tags.get(row["image_name"], {})}):
            fields[field] = [tags.get(row["image_name"], {}).get(field) for row in rows]

        for field, values in fields.items():
            for value in sorted({v for v in values if v is not None}):
                indices = [i for i, v in enumerate(values) if v == value]
                true = [rows[i]["class"] for i in indices]
                predicted = [predictions[i] for i in indices]
                output.append({
                    "model": model_name,
                    "condition": field,
                    "value": value,
                    "samples": len(indices),
                    "accuracy": round(accuracy_score(true, predicted), 4),
                    "f1_macro": round(f1_score(true, predicted, average="macro", zero_division=0), 4),
                    "amount_mae_pln": round(self._amount_mae(true, predicted), 2),
                })
        return output

    def _confusion(self, true_labels, predictions):
        counts = Counter(zip(true_labels, predictions))
        labels = sorted(set(true_labels) | set(predictions))
        return labels, [[counts[(actual, predicted)] for predicted in labels] for actual in labels]

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
