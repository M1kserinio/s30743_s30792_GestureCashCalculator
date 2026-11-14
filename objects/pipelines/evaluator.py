from collections import Counter

import cv2
import yaml

from objects.classifiers.config import CLASSIFIERS
from objects.config import (BEST_MODEL, CLASS_VALUES, CONFIDENCE, DATA_YAML, DATASET_DIR, DETECTOR_MODELS,
                            LOCATOR_MODEL, NEGATIVES_DIR, benchmark_devices, device_name, model_path)
from objects.detection.fps_meter import FpsMeter
from objects.labels import load_labeled_folder
from objects.metrics import bootstrap_interval, false_alarm_stats, match_any_class, mean_average_precision
from objects.pipelines.pipelines import DetectorPipeline, TwoStagePipeline
from objects.results_writer import ResultsWriter

#do mAP potrzebna jest cala lista ramek, takze slabych. Ponizej 0.05 ramki prawie nic juz nie zmieniaja,
#a potok 2-etapowy musialby klasyfikowac setki smieciowych wycinkow
MIN_CONF = 0.05


class PipelineEvaluator:
    #Detektor kontra "znajdz banknot + klasyfikator" na tych samych pelnych zdjeciach testowych.
    #Wszystkie potoki sa liczone tym samym kodem (objects/metrics.py), wiec liczby stoja w jednej tabeli.
    #Detektor tez tu jest, ale jego mAP z ultralytics (tabela detektorow) moze sie roznic o pare setnych,
    #bo ultralytics bierze ramki od pewnosci 0.001, a tu od 0.05

    def __init__(self):
        self.writer = ResultsWriter()
        with open(DATA_YAML) as f:
            self.class_names = yaml.safe_load(f)["names"]

    def evaluate_all(self):
        names, images, truths = self.load_split(DATASET_DIR / "test")
        negatives = [cv2.imread(str(path)) for path in sorted((NEGATIVES_DIR / "test").glob("*.jpg"))]
        fps_meter = FpsMeter(images[:50])

        summary = []
        conditions = []
        confusion = []
        matrices = {}
        for title, kind, build in self.specs():
            print(f"\n=== Potok: {title} ===")
            pipeline = build("cpu")
            predictions = [pipeline.run(image, conf=MIN_CONF) for image in images]
            row, pairs = self.scores(predictions, truths)
            row = {"pipeline": title, "type": kind, **row}
            row.update(false_alarm_stats([pipeline.run(image) for image in negatives]))
            row.update(self.timing(pipeline, build, fps_meter))
            summary.append(row)

            conditions += self.condition_rows(title, names, predictions, truths)
            labels, matrix = self.confusion(pairs)
            matrices[title] = (labels, matrix)
            confusion += [{"pipeline": title, "actual": actual, "predicted": predicted, "count": matrix[i][j]}
                          for i, actual in enumerate(labels) for j, predicted in enumerate(labels)]

        if not summary:
            print("Brak modeli - potrzebny jest detektor (models/pieniadze_*.pt) albo lokalizator i klasyfikatory")
            return
        self.writer.save_csv("study2_pipelines_summary.csv", summary)
        self.writer.save_csv("study2_pipelines_conditions.csv", conditions)
        self.writer.save_csv("study2_pipelines_confusion.csv", confusion)
        self.writer.save_confusion_plot("study2_pipelines_confusion.png", matrices)
        self.writer.save_hardware()

    def specs(self):
        #(nazwa w tabeli, rodzaj, funkcja budujaca potok na danym urzadzeniu)
        specs = []
        for name in DETECTOR_MODELS:
            if model_path(name).exists():
                specs.append((name, "sam detektor",
                              lambda device, n=name: DetectorPipeline(model_path(n), device)))

        locator, locator_name = self.locator()
        for classifier_name, path in CLASSIFIERS:
            if locator and path.exists():
                specs.append((f"{locator_name} + {classifier_name}", "lokalizator + klasyfikator",
                              lambda device, p=path: TwoStagePipeline(locator, p, device)))
        return specs

    def locator(self):
        if LOCATOR_MODEL.exists():
            return LOCATOR_MODEL, "lokalizator"
        if BEST_MODEL.exists():
            #tak dziala dzis aplikacja: ramka z najlepszego detektora, jego nominal ignorujemy
            print("Brak models/pieniadze_locator.pt - jako lokalizatora uzywam pieniadze_best.pt")
            return BEST_MODEL, "best (jako lokalizator)"
        return None, None

    def load_split(self, folder):
        return load_labeled_folder(folder, self.class_names)

    def scores(self, predictions, truths):
        #mAP z pelnej listy ramek, reszta przy progu z aplikacji (CONFIDENCE)
        map50, map50_95 = mean_average_precision(predictions, truths, self.class_names)
        row = {"map50": round(map50, 4), "map50_95": round(map50_95, 4)}
        point, pairs = self.operating_point(predictions, truths)
        row.update(point)
        return row, pairs

    def operating_point(self, predictions, truths):
        kept = [[p for p in found if p["conf"] >= CONFIDENCE] for found in predictions]
        pairs = [pair for found, real in zip(kept, truths) for pair in match_any_class(found, real)]
        correct = sum(1 for actual, predicted in pairs if actual == predicted)
        found_count = sum(len(found) for found in kept)
        real_count = sum(len(real) for real in truths)
        precision = correct / found_count if found_count else 0.0
        recall = correct / real_count if real_count else 0.0

        #to, czego naprawde chce uzytkownik kalkulatora: czy suma na zdjeciu sie zgadza
        sums_ok, sum_errors = [], []
        for found, real in zip(kept, truths):
            error = abs(sum(CLASS_VALUES[p["label"]] for p in found) - sum(CLASS_VALUES[t["label"]] for t in real))
            sums_ok.append(error < 0.01)
            sum_errors.append(error)
        low, high = bootstrap_interval(sums_ok, range(len(sums_ok)))

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall + 1e-9), 4),
            "wrong_denomination": sum(1 for a, p in pairs if "tlo" not in (a, p) and a != p),
            "missed": sum(1 for _, p in pairs if p == "tlo"),
            "extra_boxes": sum(1 for a, _ in pairs if a == "tlo"),
            "amount_accuracy": round(sum(sums_ok) / len(sums_ok), 4),
            "amount_ci95_low": low,
            "amount_ci95_high": high,
            "amount_mae_pln": round(sum(sum_errors) / len(sum_errors), 2),
        }, pairs

    def timing(self, cpu_pipeline, build, fps_meter):
        row = {}
        for device in benchmark_devices():
            #nowy potok dla kazdego urzadzenia, bo YOLO zapamietuje urzadzenie z pierwszego predict
            pipeline = cpu_pipeline if device == "cpu" else build(device)
            ms, fps = fps_meter.measure(pipeline.run)
            row[f"ms_{device_name(device)}"] = round(ms, 1)
            row[f"fps_{device_name(device)}"] = round(fps, 1)
        return row

    def condition_rows(self, title, names, predictions, truths):
        fields = {
            "source": ["bez_dloni_mix" if name.startswith("mix_banknotow") else "z_dlonia" for name in names],
            "object_count": [len(real) for real in truths],
        }

        rows = []
        for field, values in fields.items():
            for value in sorted({v for v in values if v is not None}):
                indices = [i for i, v in enumerate(values) if v == value]
                point, _ = self.operating_point([predictions[i] for i in indices], [truths[i] for i in indices])
                rows.append({"pipeline": title, "condition": field, "value": value, "images": len(indices),
                             "f1": point["f1"], "amount_accuracy": point["amount_accuracy"],
                             "amount_mae_pln": point["amount_mae_pln"]})
        return rows

    def confusion(self, pairs):
        counts = Counter(pairs)
        labels = self.class_names + ["tlo"]
        return labels, [[counts[(actual, predicted)] for predicted in labels] for actual in labels]
