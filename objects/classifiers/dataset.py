import csv
import random
import shutil

import cv2
import yaml

from objects.classifiers.config import CLASSIFICATION_DATASET_DIR, CLASSIFICATION_METADATA
from objects.config import BACKGROUND_CLASS, DATA_YAML, DATASET_DIR, NEGATIVES_DIR, SEED
from objects.labels import crop_with_margin, read_labels, to_pixels


class ClassificationDataset:
    """Wycinanie banknotow z ramek YOLO do osobnego zbioru klasyfikacyjnego."""

    def prepare(self):
        if not DATA_YAML.exists():
            raise FileNotFoundError(
                "Brak datasets/pieniadze/data.yaml. Najpierw uruchom "
                "python -m scripts.prepare_dataset roboflow.zip"
            )

        if CLASSIFICATION_DATASET_DIR.exists():
            shutil.rmtree(CLASSIFICATION_DATASET_DIR)
        CLASSIFICATION_DATASET_DIR.mkdir(parents=True)

        with open(DATA_YAML) as f:
            class_names = yaml.safe_load(f)["names"]

        rows = []
        for split in ("train", "val", "test"):
            banknotes = self._prepare_split(split, class_names)
            rows.extend(banknotes)
            rows.extend(self._prepare_background(split, len(banknotes) // len(class_names)))

        with open(CLASSIFICATION_METADATA, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        print(f"Gotowe: {len(rows)} wycinkow w {CLASSIFICATION_DATASET_DIR}")
        print(f"Metadane: {CLASSIFICATION_METADATA}")

    def _prepare_split(self, split, class_names):
        rows = []
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        for image_path in sorted(images_dir.glob("*")):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            annotations = read_labels(label_path)
            source = "bez_dloni_mix" if image_path.stem.startswith("mix_banknotow") else "z_dlonia"
            image_h, image_w = image.shape[:2]

            for box_index, (class_id, x, y, width, height) in enumerate(annotations):
                crop = crop_with_margin(image, *to_pixels(x, y, width, height, image_w, image_h))
                if crop.size == 0:
                    continue

                class_name = class_names[class_id]
                rows.append(self._save(crop, split, class_name, f"{image_path.stem}_{box_index}.jpg", {
                    "image_name": image_path.name,
                    "object_count": len(annotations),
                    "source": source,
                }))
        return rows

    def _prepare_background(self, split, count):
        #klasa "brak": tyle wycinkow tla, ile srednio jest wycinkow jednego nominalu w tym zbiorze
        #dzieki niej klasyfikator moze powiedziec "tu nie ma banknotu" i odrzucic falszywa ramke detektora
        negatives = sorted((NEGATIVES_DIR / split).glob("*.jpg"))
        rows = []
        for path in random.Random(SEED).sample(negatives, min(count, len(negatives))):
            crop = cv2.imread(str(path))
            if crop is None:
                continue
            rows.append(self._save(crop, split, BACKGROUND_CLASS, path.name, {
                "image_name": path.name,
                "object_count": 0,
                "source": "tlo",
            }))
        return rows

    def _save(self, crop, split, class_name, file_name, info):
        output_dir = CLASSIFICATION_DATASET_DIR / split / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_name
        cv2.imwrite(str(output_path), crop)
        return {
            "split": split,
            "crop_path": str(output_path.relative_to(CLASSIFICATION_DATASET_DIR)),
            "image_name": info["image_name"],
            "class": class_name,
            "object_count": info["object_count"],
            "source": info["source"],
        }
