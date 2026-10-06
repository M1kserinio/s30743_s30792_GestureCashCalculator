from objects.config import CLASS_VALUES
from objects.results_writer import ResultsWriter

SPLITS = ["train", "val", "test"]


class DatasetStats:
    def __init__(self, class_names):
        self.class_names = class_names

    def report(self, split):
        stats = {}
        for split_name in SPLITS:
            stats[split_name] = self.count_objects(split[split_name])

        self.save_csv(split, stats)
        self.check_missing(stats)

    def count_objects(self, samples):
        # class_id x y width height
        counts = {name: 0 for name in self.class_names}
        empty = 0
        for image, label in samples:
            lines = []
            if label.exists():
                lines = [line for line in label.read_text().splitlines() if line.strip()]
            if not lines:
                empty += 1
            for line in lines:
                class_id = int(line.split()[0])
                counts[self.class_names[class_id]] += 1
        return {"counts": counts, "empty": empty}


    def save_csv(self, split, stats):
        #ta sama tabela do csv - przyda sie do rozdzialu o danych
        rows = []
        for split_name in SPLITS:
            for name in self.class_names:
                rows.append({"split": split_name, "class": name, "value_pln": CLASS_VALUES.get(name),
                             "objects": stats[split_name]["counts"][name]})
            rows.append({"split": split_name, "class": "background_images", "value_pln": 0,
                         "objects": stats[split_name]["empty"]})
            rows.append({"split": split_name, "class": "all_images", "value_pln": "",
                         "objects": len(split[split_name])})
        ResultsWriter().save_csv("study2_dataset_stats.csv", rows)

    def check_missing(self, stats):
        #jak jakiejs klasy nie ma w val albo test, to nie da sie jej uczciwie ocenic
        for name in self.class_names:
            if stats["val"]["counts"][name] == 0 or stats["test"]["counts"][name] == 0:
                print(f"UWAGA: klasy '{name}' brakuje w val albo test - oznacz wiecej zdjec z tym nominalem")
