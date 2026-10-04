import random

from objects.config import SEED, TEST_PART, VAL_PART


class DatasetSplitter:
    def split(self, samples):
        groups = self.make_groups(samples)
        print(f"Zdjec: {len(samples)}, grup (sesji albo pojedynczych zdjec): {len(groups)}")

        #losowa kolejnosc grup, ale zawsze ta sama dzieki SEED
        names = sorted(groups)
        random.Random(SEED).shuffle(names)

        #najpierw zapelniamy test, potem val, reszta do train
        #gdyby zaczac od train, to przy malej liczbie sesji test moglby zostac pusty
        split = {"train": [], "val": [], "test": []}
        for name in names:
            if len(split["test"]) < TEST_PART * len(samples):
                split["test"] += groups[name]
            elif len(split["val"]) < VAL_PART * len(samples):
                split["val"] += groups[name]
            else:
                split["train"] += groups[name]
        return split

    def make_groups(self, samples):
        #zdjecia z tej samej grupy zawsze trafiaja do tego samego zbioru
        groups = {}
        for image, label in samples:
            name = self.group_name(image)
            if name not in groups:
                groups[name] = []
            groups[name].append((image, label))
        return groups

    def group_name(self, image):
        #Roboflow zmienia nazwe: s01_biurko_12.jpg -> s01_biurko_12_jpg.rf.<losowy_hash>.jpg
        #wiec oryginalna nazwa to wszystko przed ".rf."
        original = image.name.split(".rf.")[0]

        #nazwa zaczyna sie od sesji (s01_, s02_...) -> cala sesja idzie do jednego zbioru
        #inaczej prawie takie same zdjecia z jednej sesji bylyby w train i w test -> za wysokie mAP
        first_part = original.split("_")[0]
        if first_part.startswith("s") and first_part[1:].isdigit():
            return first_part

        #bez sesji kazde zdjecie jest osobno
        #(kopie z augmentacji maja ta sama oryginalna nazwe, wiec i tak zostaja razem)
        return original
