from objects.config import CLASS_VALUES
from objects.labels import read_labels
from objects.results_writer import ResultsWriter

SPLITS = ["train", "val", "test"]


class DatasetStats:
    def __init__(self, class_names):
        self.class_names = class_names

    def report(self, split, negatives):
        stats = {}
        for split_name in SPLITS:
            stats[split_name] = self.count_objects(split[split_name])

        self.save_csv(split, stats, negatives)
        self.check_missing(stats)

    def count_objects(self, samples):
        counts = {name: 0 for name in self.class_names}
        images = 0
        for image, label in samples:
            boxes = read_labels(label)
            if boxes:
                images += 1
            for class_id, *_ in boxes:
                counts[self.class_names[class_id]] += 1
        return {"counts": counts, "images": images}

    def save_csv(self, split, stats, negatives):
        #ile obiektow kazdej klasy jest w train/val/test - do rozdzialu o danych
        rows = []
        for split_name in SPLITS:
            for name in self.class_names:
                rows.append({"split": split_name, "class": name, "value_pln": CLASS_VALUES.get(name),
                             "objects": stats[split_name]["counts"][name]})
            rows.append({"split": split_name, "class": "images_with_banknotes", "value_pln": "",
                         "objects": stats[split_name]["images"]})
            rows.append({"split": split_name, "class": "background_images_in_detector_set", "value_pln": 0,
                         "objects": negatives[split_name]["detector_backgrounds"]})
            rows.append({"split": split_name, "class": "no_banknote_images", "value_pln": 0,
                         "objects": negatives[split_name]["negatives"]})
        ResultsWriter().save_csv("study2_dataset_stats.csv", rows)

    def check_missing(self, stats):
        #jak jakiejs klasy nie ma w val albo test, to nie da sie jej uczciwie ocenic
        for name in self.class_names:
            if stats["val"]["counts"][name] == 0 or stats["test"]["counts"][name] == 0:
                print(f"UWAGA: klasy '{name}' brakuje w val albo test - oznacz wiecej zdjec z tym nominalem")
