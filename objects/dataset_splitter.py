import csv
import random

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from objects.config import SEED, SPLIT_FILE, TEST_PART, VAL_PART
from objects.labels import read_labels
from objects.results_writer import ResultsWriter
from objects.session_groups import SessionGroups

SPLITS = ["train", "val", "test"]


class DatasetSplitter:
    def split(self, samples):
        saved = self.load_saved(samples)
        if saved:
           # print(f"Podzial wziety z {SPLIT_FILE} (ten sam dla wszystkich modeli)")
            groups, indices = self.saved_groups_and_indices(samples)
            self.save_leak_report(samples, groups, indices)
            return saved

        images = [image for image, _ in samples]
        collage_boxes = [read_labels(label) if self.is_collage(image) else None for image, label in samples]
        sessions = SessionGroups(images, collage_boxes)
        groups = np.asarray(sessions.groups())
        strata = np.asarray([self.stratum(image, label) for image, label in samples])
        print(f"Zdjec: {len(samples)}, grup (sesji albo pojedynczych zdjec): {len(set(groups))}")

        indices = self.assign(strata, groups)
        split = {name: [samples[i] for i in indices[name]] for name in SPLITS}
        self.save(split, samples, groups)
        self.save_leak_report(samples, groups, indices)
        return split

    def assign(self, strata, groups):
        #StratifiedGroupKFold: cala grupa idzie do jednego zbioru, a nominaly rozkladaja sie mniej wiecej rowno
        everything = np.arange(len(strata))
        test = self.one_part(everything, strata, groups, round(1 / TEST_PART))
        rest = np.setdiff1d(everything, test)
        val = self.one_part(rest, strata, groups, round(len(rest) / len(everything) / VAL_PART))
        train = np.setdiff1d(rest, val)
        return {"train": list(train), "val": list(val), "test": list(test)}

    def one_part(self, indices, strata, groups, parts):
        folds = StratifiedGroupKFold(n_splits=parts, shuffle=True, random_state=SEED)
        _, chosen = next(folds.split(indices, strata[indices], groups[indices]))
        return indices[chosen]

    def is_collage(self, image):
        #mix_banknotow = banknoty wklejone na obce tlo, bez dloni
        return image.name.startswith("mix_banknotow")

    def stratum(self, image, label):
        #wedlug czego pilnujemy proporcji: nominal zdjecia z dlonia, a kolaze jako osobna grupa
        if self.is_collage(image):
            return "mix"
        return str(read_labels(label)[0][0])

    def save(self, split, samples, groups):
        group_of = {image.name: int(group) for (image, _), group in zip(samples, groups)}
        rows = [{"image": image.name, "group": group_of[image.name], "split": name}
                for name in SPLITS for image, _ in split[name]]
        SPLIT_FILE.parent.mkdir(exist_ok=True)
        with open(SPLIT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "group", "split"])
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (SPLITS.index(row["split"]), row["image"])))
        print(f"Zapisano podzial: {SPLIT_FILE}")

    def load_saved(self, samples):
        if not SPLIT_FILE.exists():
            return None
        with open(SPLIT_FILE, newline="") as f:
            saved = {row["image"]: row["split"] for row in csv.DictReader(f)}
        missing = [image.name for image, _ in samples if image.name not in saved]
        if missing:
            #nowe zdjecia = nowy podzial = wszystkie modele trzeba uczyc od nowa
            print(f"UWAGA: {len(missing)} zdjec nie ma w {SPLIT_FILE.name}, licze podzial od nowa")
            return None
        split = {name: [] for name in SPLITS}
        for image, label in samples:
            split[saved[image.name]].append((image, label))
        return split

    def saved_groups_and_indices(self, samples):
        with open(SPLIT_FILE, newline="") as f:
            saved = {row["image"]: row for row in csv.DictReader(f)}
        groups = np.asarray([int(saved[image.name]["group"]) for image, _ in samples])
        indices = {
            split: [i for i, (image, _) in enumerate(samples) if saved[image.name]["split"] == split]
            for split in SPLITS
        }
        return groups, indices

    def save_leak_report(self, samples, groups, split_indices):
        rows = []
        for first, second in [("train", "test"), ("train", "val"), ("val", "test")]:
            rows.append(self.leak_row(
                samples, groups, f"{first}-{second} (po sesjach)",
                split_indices[first], split_indices[second],
            ))

        everything = list(range(len(samples)))
        random.Random(SEED).shuffle(everything)
        test_size = len(split_indices["test"])
        rows.append(self.leak_row(
            samples, groups, "train-test (losowo po plikach, dla porownania)",
            everything[test_size:], everything[:test_size],
        ))
        ResultsWriter().save_csv("study2_split_leaks.csv", rows)

    def leak_row(self, samples, groups, name, first, second):
        first_groups = {groups[i] for i in first}
        twins = np.asarray([groups[i] in first_groups for i in second])
        collages = np.asarray([self.is_collage(samples[i][0]) for i in second])
        return {
            "zbiory": name,
            "zdjec_w_drugim": len(second),
            "z_blizniakiem_w_pierwszym": int(twins.sum()),
            "w_tym_zdjecia_z_dlonia": int((twins & ~collages).sum()),
            "w_tym_kolaze": int((twins & collages).sum()),
        }
