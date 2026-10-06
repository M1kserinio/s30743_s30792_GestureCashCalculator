import shutil

import cv2
from ultralytics import YOLO

from objects.config import BEST_MODEL, DATA_YAML, DATASET_DIR, IMGSZ, RUNS_DIR, YOLO_MODELS, best_device, model_path
from objects.fps_meter import FpsMeter
from objects.results_writer import ResultsWriter


class YoloEvaluator:
    def __init__(self):
        #FPS zawsze na CPU (dziala na kazdym komputerze) + na GPU jak jest
        self.devices = ["cpu"]
        if best_device() != "cpu":
            self.devices.append(best_device())
        self.writer = ResultsWriter()

    def evaluate_all(self):
        model_names = [name for name in YOLO_MODELS if model_path(name).exists()]
        if not model_names:
            print("Brak modeli w models/ - najpierw uruchom python -m scripts.train_yolo")
            return

        fps_meter = FpsMeter(self.load_test_images())
        summary = []
        per_class = []
        for model_name in model_names:
            print(f"\n=== Ewaluacja {model_name} ===")
            row, classes = self.evaluate(model_name, fps_meter)
            summary.append(row)
            per_class += classes

        self.writer.save_csv("study2_yolo_summary.csv", summary)
        self.writer.save_csv("study2_yolo_per_class.csv", per_class)
        self.writer.save_map_vs_fps_plot(summary, [self.device_name(d) for d in self.devices])

        best = self.pick_best(summary)
        shutil.copy(model_path(best["model"]), BEST_MODEL)
        print(f"\nNajlepszy: {best['model']} -> {BEST_MODEL} (tego uzywa apka)")

    def load_test_images(self):
        #do pomiaru FPS bierzemy max 50 zdjec z testu
        paths = sorted((DATASET_DIR / "test" / "images").glob("*"))[:50]
        return [cv2.imread(str(path)) for path in paths]

    def evaluate(self, model_name, fps_meter):
        model = YOLO(str(model_path(model_name)))
        #liczymy parametry przed val, bo val laczy warstwy (fuse) i liczba by sie zmienila
        params = sum(p.numel() for p in model.model.parameters())

        #val tylko do wyboru najlepszego modelu, test to wynik do pracy
        #(gdybysmy wybierali po tescie, to wynik na tescie bylby zawyzony)
        val = self.run_val(model, model_name, "val")
        test = self.run_val(model, model_name, "test")

        row = {
            "model": model_name,
            "params_mln": round(params / 1_000_000, 2),
            "val_map50_95": round(float(val.box.map), 4),
            "test_precision": round(float(test.box.mp), 4),
            "test_recall": round(float(test.box.mr), 4),
            "test_map50": round(float(test.box.map50), 4),
            "test_map50_95": round(float(test.box.map), 4),
        }
        for device in self.devices:
            ms, fps = fps_meter.measure(model_path(model_name), device)
            row[f"ms_{self.device_name(device)}"] = round(ms, 1)
            row[f"fps_{self.device_name(device)}"] = round(fps, 1)

        return row, self.per_class_rows(model, model_name, test)

    def run_val(self, model, model_name, split):
        #wykresy (macierz pomylek, krzywe PR) do testow
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

    def pick_best(self, summary):
        #najlepszy wybieramy po val (nie po tescie)
        best = summary[0]
        for row in summary:
            if row["val_map50_95"] > best["val_map50_95"]:
                best = row
        return best

    def device_name(self, device):
        return "cuda" if device == "0" else device
