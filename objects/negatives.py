import random
import shutil

import cv2

from objects.config import BACKGROUND_PART, DATASET_DIR, NEGATIVES_DIR, SEED
from objects.labels import read_labels, to_pixels

PATCH_MIN_SIDE = 0.2  #pas tla musi miec co najmniej 20% krotszego boku zdjecia
PATCH_GAP = 0.03      #odstep od ramek, zeby w tle nie zostal rog banknotu
PATCH_MAX_RATIO = 2   #dluzszy bok najwyzej 2x krotszy, inaczej to tylko cienki pasek


class Negatives:
    """Zdjecia bez banknotu: klasa "brak" dla klasyfikatorow i proba do liczenia falszywych alarmow.

    Osobnych zdjec bez banknotow nie mamy, wiec wycinamy je z naszych: kawalek zdjecia obok banknotow
    (reka, stol, sciana). Wycinek zostaje w zbiorze zdjecia, z ktorego jest, wiec nie ma przecieku miedzy train i test.
    """

    def prepare(self, split):
        if NEGATIVES_DIR.exists():
            shutil.rmtree(NEGATIVES_DIR)

        counts = {}
        for split_name, samples in split.items():
            output_dir = NEGATIVES_DIR / split_name
            output_dir.mkdir(parents=True)
            for image, label in samples:
                self.save_patch(image, label, output_dir)
            counts[split_name] = {
                "negatives": len(list(output_dir.glob("*.jpg"))),
                "detector_backgrounds": self.add_to_detector_set(split_name, len(samples)),
            }
            print(f"{split_name}: {counts[split_name]['negatives']} zdjec bez banknotu")
        return counts

    def save_patch(self, image_path, label_path, output_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            return
        patch = self.background_patch(image, read_labels(label_path))
        if patch is not None:
            cv2.imwrite(str(output_dir / f"{image_path.stem}_tlo.jpg"), patch)

    def background_patch(self, image, boxes):
        #najwiekszy pas obok wszystkich ramek naraz: z lewej, z prawej, nad albo pod nimi
        image_h, image_w = image.shape[:2]
        corners = [to_pixels(x, y, w, h, image_w, image_h) for _, x, y, w, h in boxes]
        gap = PATCH_GAP * min(image_w, image_h)
        left = int(min(c[0] for c in corners) - gap)
        top = int(min(c[1] for c in corners) - gap)
        right = int(max(c[2] for c in corners) + gap)
        bottom = int(max(c[3] for c in corners) + gap)
        strips = [
            (0, 0, left, image_h),
            (right, 0, image_w, image_h),
            (0, 0, image_w, top),
            (0, bottom, image_w, image_h),
        ]
        x1, y1, x2, y2 = max(strips, key=lambda s: min(s[2] - s[0], s[3] - s[1]))
        short = min(x2 - x1, y2 - y1)
        if short < PATCH_MIN_SIDE * min(image_w, image_h):
            return None

        #za dlugi pas przycinamy od srodka
        if x2 - x1 > PATCH_MAX_RATIO * short:
            middle = (x1 + x2) // 2
            x1, x2 = middle - short, middle + short
        if y2 - y1 > PATCH_MAX_RATIO * short:
            middle = (y1 + y2) // 2
            y1, y2 = middle - short, middle + short
        return image[y1:y2, x1:x2]

    def add_to_detector_set(self, split_name, images_in_split):
        #do train i val detektora dokladamy ~10% zdjec tla bez ramek, zeby uczyl sie, czego nie zaznaczac
        #test zostaje bez tla: mAP liczymy na samych banknotach, a falszywe alarmy osobno na brak_banknotu/test
        if split_name == "test":
            return 0
        negatives = sorted((NEGATIVES_DIR / split_name).glob("*.jpg"))
        count = min(len(negatives), round(BACKGROUND_PART * images_in_split))
        for path in random.Random(SEED).sample(negatives, count):
            shutil.copy(path, DATASET_DIR / split_name / "images" / path.name)
        return count
