import shutil
import zlib

import cv2
import numpy as np

from objects.config import DATASET_DIR, NEGATIVES_DIR, RESULTS_DIR, ROBUSTNESS_DIR, SEED
from objects.labels import read_labels, to_pixels
from objects.results_writer import ResultsWriter
from objects.robustness.conditions import Conditions


class RobustnessDataset:
    #Zapisuje zbior testowy pod kazdym warunkiem do datasets/odpornosc/<warunek>/ (images + labels jak w YOLO).
    #Zmienione zdjecia sa na dysku, wiec kazdy model dostaje dokladnie te same i mozna je obejrzec

    def prepare(self):
        test_images = sorted((DATASET_DIR / "test" / "images").glob("*"))
        backgrounds = [cv2.imread(str(path)) for path in sorted((NEGATIVES_DIR / "test").glob("*.jpg"))]
        if not test_images or not backgrounds:
            raise SystemExit("Brak testu albo zdjec bez banknotow - najpierw python -m scripts.prepare_dataset")

        conditions = Conditions(backgrounds)
        examples = {}
        example_index = self.example_index(test_images)
        for name, (_, change) in conditions.all.items():
            folder = ROBUSTNESS_DIR / name
            shutil.rmtree(folder, ignore_errors=True)
            (folder / "images").mkdir(parents=True)
            (folder / "labels").mkdir()
            for index, image_path in enumerate(test_images):
                #los zalezy od warunku i zdjecia, a nie od kolejnosci - to samo zdjecie zawsze zmienia sie tak samo
                rng = np.random.default_rng([SEED, zlib.crc32(name.encode()), index])
                image = cv2.imread(str(image_path))
                changed, boxes = change(image, self.boxes(image_path, image), rng)
                cv2.imwrite(str(folder / "images" / image_path.name), changed)
                self.save_labels(folder / "labels" / f"{image_path.stem}.txt", boxes, changed)
                if index == example_index:
                    examples[name] = (changed, boxes)
            print(f"{name}: {len(test_images)} zdjec")

        self.save_examples(examples)

    def example_index(self, test_images):
        #na obrazku z przykladami lepiej widac zmiany na zdjeciu z reka niz na kolazu
        for index, path in enumerate(test_images):
            if not path.name.startswith("mix_banknotow"):
                return index
        return 0

    def boxes(self, image_path, image):
        label_path = DATASET_DIR / "test" / "labels" / f"{image_path.stem}.txt"
        image_h, image_w = image.shape[:2]
        return [(class_id, *to_pixels(x, y, w, h, image_w, image_h)) for class_id, x, y, w, h in read_labels(label_path)]

    def save_labels(self, label_path, boxes, image):
        image_h, image_w = image.shape[:2]
        lines = []
        for class_id, x1, y1, x2, y2 in boxes:
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{class_id} {(x1 + x2) / 2 / image_w:.6f} {(y1 + y2) / 2 / image_h:.6f} "
                         f"{(x2 - x1) / image_w:.6f} {(y2 - y1) / image_h:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    def save_examples(self, examples):
        #jedno zdjecie pod kazdym warunkiem, z ramkami po przeliczeniu - do pracy i do sprawdzenia, czy ramki pasuja
        plt = ResultsWriter().plt()
        columns = 4
        rows = (len(examples) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (name, (image, boxes)) in zip(axes.flat, examples.items()):
            image = image.copy()
            for _, x1, y1, x2, y2 in boxes:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(name, fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "study2_robustness_examples.png", dpi=120)
        plt.close(fig)
        print(f"Zapisano {RESULTS_DIR / 'study2_robustness_examples.png'}")
