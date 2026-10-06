import random

from objects.config import SEED, TEST_PART, VAL_PART


class DatasetSplitter:
    def split(self, samples):
        groups = self.make_groups(samples)
        print(f"Zdjec: {len(samples)}, grup (sesji albo pojedynczych zdjec): {len(groups)}")

        #losowa kolejnosc grup, ale zawsze ta sama dzieki SEED
        names = sorted(groups)
        random.Random(SEED).shuffle(names)

        #najpierw zapelniamy test -> val i reszta do train ( train na koncu bo gdyby bylo na odwrot. to przy malej lizbie sesji test moglby byc pusty)
        split = {"train": [], "val": [], "test": []}
        for name in names:
            if len(split["test"]) < TEST_PART * len(samples):
                split["test"] += groups[name]
            elif len(split["val"]) < VAL_PART * len(samples):
                split["val"] += groups[name]
            else:
                split["train"] += groups[name]
        return split

    def make_groups(self, samples): # para zdjecie-etykieta
        groups = {}
        for image, label in samples:
            name = self.group_name(image)
            if name not in groups:
                groups[name] = []
            groups[name].append((image, label))
        return groups

    def group_name(self, image):
        original = image.name.split(".rf.")[0]
        first_part = original.split("_")[0]
        if first_part[1:].isdigit():
            return first_part
        return original
