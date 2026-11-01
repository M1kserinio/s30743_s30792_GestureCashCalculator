import shutil
from collections import Counter

import cv2
import yaml

from objects.config import (BEST_MODEL, CONFIDENCE, DATA_YAML, DATASET_DIR, DETECTOR_MODELS, IMGSZ, NEGATIVES_DIR,
                            NMS_IOU, RUNS_DIR, benchmark_devices, device_name, model_path)
from objects.detection.fps_meter import FpsMeter
from objects.detection.models import architecture, load_detector
from objects.labels import load_labeled_folder
from objects.metrics import false_alarm_stats, match_any_class
from objects.pipelines.pipelines import DetectorPipeline
from objects.results_writer import ResultsWriter


class DetectorEvaluator:
    #Tabela detektorow (YOLO, RT-DETR): mAP, precision, recall, F1, macierz pomylek, FPS

    def __init__(self):
        #FPS zawsze na CPU + na GPU jak jest
        self.devices = benchmark_devices()
        self.writer = ResultsWriter()

    def evaluate_all(self):
        model_names = [name for name in DETECTOR_MODELS if model_path(name).exists()]
        if not model_names:
            print("Brak modeli w models/ - najpierw uruchom python -m scripts.detectors.train")
            return

        with open(DATA_YAML) as f:
            self.class_names = yaml.safe_load(f)["names"]
        _, self.images, self.truths = load_labeled_folder(DATASET_DIR / "test", self.class_names)
        self.negatives = [cv2.imread(str(path)) for path in sorted((NEGATIVES_DIR / "test").glob("*.jpg"))]
        #do pomiaru FPS wystarczy 50 zdjec z testu
        fps_meter = FpsMeter(self.images[:50])

        summary = []
        per_class = []
        confusion = []
        matrices = {}
        for model_name in model_names:
            print(f"\n=== Ewaluacja {model_name} ===")
            row, classes, labels, matrix = self.evaluate(model_name, fps_meter)
            summary.append(row)
            per_class += classes
            matrices[model_name] = (labels, matrix)
            confusion += [{"model": model_name, "actual": actual, "predicted": predicted, "count": matrix[i][j]}
                          for i, actual in enumerate(labels) for j, predicted in enumerate(labels)]

        self.writer.save_csv("study2_detectors_summary.csv", summary)
        self.writer.save_csv("study2_detectors_per_class.csv", per_class)
        self.writer.save_csv("study2_detectors_confusion.csv", confusion)
        self.writer.save_confusion_plot("study2_detectors_confusion.png", matrices)
        self.writer.save_map_vs_fps_plot(summary, [device_name(d) for d in self.devices],
                                         "study2_detectors_map_vs_fps.png")
        self.writer.save_hardware()

        best = self.pick_best(summary)
        shutil.copy(model_path(best["model"]), BEST_MODEL)
        print(f"\nNajlepszy: {best['model']} -> {BEST_MODEL} (tego uzywa apka)")

    def evaluate(self, model_name, fps_meter):
        model = load_detector(model_path(model_name))
        #liczymy parametry przed val, bo val laczy warstwy (fuse) i liczba by sie zmienila
        params = sum(p.numel() for p in model.model.parameters())
        kind = architecture(model)

        #val tylko do wyboru najlepszego modelu, test to wynik do pracy
        #(gdybysmy wybierali po tescie, to wynik na tescie bylby zawyzony)
        val = self.run_val(model, model_name, "val")
        test = self.run_val(model, model_name, "test")
        #precision i recall z ultralytics sa przy progu pewnosci, przy ktorym F1 wychodzi najwyzsze
        precision, recall = float(test.box.mp), float(test.box.mr)

        row = {
            "model": model_name,
            "architecture": kind,
            "params_mln": round(params / 1_000_000, 2),
            "val_map50_95": round(float(val.box.map), 4),
            "test_precision": round(precision, 4),
            "test_recall": round(recall, 4),
            "test_f1": round(2 * precision * recall / (precision + recall + 1e-9), 4),
            "test_map50": round(float(test.box.map50), 4),
            "test_map50_95": round(float(test.box.map), 4),
        }

        #macierz i falszywe alarmy liczymy sami, przy progu z aplikacji. Macierz z ultralytics bierze ramki
        #od pewnosci 0.001, wiec wychodza w niej tysiace ramek "z tla", ktorych apka nigdy by nie pokazala
        detector = DetectorPipeline(model_path(model_name), self.devices[-1])
        pairs = [pair for image, real in zip(self.images, self.truths)
                 for pair in match_any_class(detector.run(image), real)]
        row.update(false_alarm_stats([detector.run(image) for image in self.negatives]))

        for device in self.devices:
            ms, fps = fps_meter.measure(self.frame_runner(model_name, device))
            row[f"ms_{device_name(device)}"] = round(ms, 1)
            row[f"fps_{device_name(device)}"] = round(fps, 1)

        labels, matrix = self.confusion(pairs)
        return row, self.per_class_rows(model, model_name, test), labels, matrix

    def frame_runner(self, model_name, device):
        #nowy model dla kazdego urzadzenia, bo YOLO zapamietuje urzadzenie z pierwszego predict
        #progi takie jak w apce; RT-DETR parametr iou po prostu ignoruje, bo nie ma NMS
        model = load_detector(model_path(model_name))
        return lambda image: model.predict(image, imgsz=IMGSZ, conf=CONFIDENCE, iou=NMS_IOU,
                                           device=device, verbose=False)

    def run_val(self, model, model_name, split):
        #wykresy (krzywe PR itd.) tylko dla testu
        return model.val(data=str(DATA_YAML), split=split, imgsz=IMGSZ, plots=(split == "test"),
                         project=str(RUNS_DIR / "eval"), name=f"{model_name}_{split}", exist_ok=True)

    def per_class_rows(self, model, model_name, test):
        #ap_class_index to numery klas ktore sa w tescie - jak jakiejs nie ma, to nie ma jej na liscie
        rows = []
        for i, class_id in enumerate(test.box.ap_class_index):
            rows.append({
                "model": model_name,
                "class": model.names[int(class_id)],
                "precision": round(float(test.box.p[i]), 4),
                "recall": round(float(test.box.r[i]), 4),
                "ap50": round(float(test.box.ap50[i]), 4),
                "ap50_95": round(float(test.box.ap[i]), 4),
            })
        return rows

    def confusion(self, pairs):
        #[prawdziwa][przewidziana], "tlo" = zgubiony banknot (kolumna) albo ramka na pustym miejscu (wiersz)
        counts = Counter(pairs)
        labels = self.class_names + ["tlo"]
        return labels, [[counts[(actual, predicted)] for predicted in labels] for actual in labels]

    def pick_best(self, summary):
        #najlepszy wybieramy po val (nie po tescie)
        best = summary[0]
        for row in summary:
            if row["val_map50_95"] > best["val_map50_95"]:
                best = row
        return best
