import cv2
import numpy as np
from sklearn.metrics import f1_score

from objects.classifiers.config import CLASSIFIERS
from objects.classifiers.predict import BanknoteClassifier
from objects.config import ROBUSTNESS_DIR, best_device, device_name
from objects.labels import crop_with_margin
from objects.pipelines.evaluator import MIN_CONF, PipelineEvaluator
from objects.results_writer import ResultsWriter
from objects.robustness.conditions import Conditions


class RobustnessEvaluator:
    #Te same modele na zmienionych zdjeciach testowych: tabela model x warunek i mapy ciepla.
    #Potoki (detektor albo lokalizator + klasyfikator) dostaja cale zdjecie.
    #Klasyfikatory osobno dostaja wycinki z prawdziwych ramek, zeby bylo widac, ile traci sam klasyfikator,
    #a ile to wina tego, ze lokalizator nie znalazl banknotu

    def __init__(self):
        self.pipelines = PipelineEvaluator()
        self.writer = ResultsWriter()
        #tu liczy sie tylko trafnosc, wiec bierzemy najszybsze urzadzenie (wynik od niego nie zalezy)
        self.device = best_device()

    def evaluate_all(self):
        folders = self.condition_folders()
        if not folders:
            print("Brak zmienionych zdjec - najpierw python -m scripts.robustness.prepare")
            return

        pipelines = [(title, build(self.device)) for title, _, build in self.pipelines.specs()]
        classifiers = [(name, BanknoteClassifier(path, device_name(self.device)))
                       for name, path in CLASSIFIERS if path.exists()]
        groups = {name: group for name, (group, _) in Conditions([]).all.items()}

        pipeline_rows, classifier_rows = [], []
        for folder in folders:
            print(f"\n=== Warunek: {folder.name} ===")
            group = groups[folder.name]
            _, images, truths = self.pipelines.load_split(folder)
            for title, pipeline in pipelines:
                predictions = [pipeline.run(image, conf=MIN_CONF) for image in images]
                scores, _ = self.pipelines.scores(predictions, truths)
                pipeline_rows.append({"pipeline": title, "condition": folder.name, "group": group,
                                      "images": len(images), "map50": scores["map50"], "f1": scores["f1"],
                                      "recall": scores["recall"], "precision": scores["precision"],
                                      "amount_accuracy": scores["amount_accuracy"]})

            crops, labels = self.crops(images, truths)
            for name, classifier in classifiers:
                predicted = [classifier.predict(crop)[0] for crop in crops]
                classifier_rows.append({"classifier": name, "condition": folder.name, "group": group,
                                        "crops": len(crops),
                                        "accuracy": round(float(np.mean([p == t for p, t in zip(predicted, labels)])), 4),
                                        "f1_macro": round(float(f1_score(labels, predicted, labels=sorted(set(labels)),
                                                                         average="macro", zero_division=0)), 4)})

        self.add_drop(pipeline_rows, "pipeline", "f1")
        self.add_drop(classifier_rows, "classifier", "accuracy")
        if pipeline_rows:
            self.writer.save_csv("study2_robustness_pipelines.csv", pipeline_rows)
            self.save_heatmap("study2_robustness_pipelines_f1.png", "Potoki: F1 (IoU 0.5) przy zmianie warunkow",
                              pipeline_rows, "pipeline", "f1")
            self.save_heatmap("study2_robustness_pipelines_amount.png", "Potoki: zgodna suma na zdjeciu",
                              pipeline_rows, "pipeline", "amount_accuracy")
        if classifier_rows:
            self.writer.save_csv("study2_robustness_classifiers.csv", classifier_rows)
            self.save_heatmap("study2_robustness_classifiers.png",
                              "Klasyfikatory: accuracy na wycinkach z prawdziwych ramek",
                              classifier_rows, "classifier", "accuracy")

    def condition_folders(self):
        folders = [ROBUSTNESS_DIR / name for name in Conditions([]).all]
        return [folder for folder in folders if (folder / "images").is_dir()]

    def crops(self, images, truths):
        crops, labels = [], []
        for image, real in zip(images, truths):
            for truth in real:
                crop = crop_with_margin(image, *truth["box"])
                if crop.size:
                    crops.append(crop)
                    labels.append(truth["label"])
        return crops, labels

    def add_drop(self, rows, key, metric):
        #ile model traci wzgledem niezmienionego testu - latwiej to porownac niz same wartosci
        base = {row[key]: row[metric] for row in rows if row["condition"] == "oryginal"}
        for row in rows:
            row[f"{metric}_drop"] = round(base[row[key]] - row[metric], 4) if row[key] in base else ""

    def save_heatmap(self, file_name, title, rows, key, metric):
        #warunki w wierszach, modele w kolumnach - warunkow jest duzo wiecej, wiec tak miesci sie na stronie A4
        models = list(dict.fromkeys(row[key] for row in rows))
        conditions = list(dict.fromkeys(row["condition"] for row in rows))
        values = {(row[key], row["condition"]): row[metric] for row in rows}
        table = [[values.get((model, condition), 0.0) for model in models] for condition in conditions]
        self.writer.save_heatmap(file_name, title, conditions, models, table)
