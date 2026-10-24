import csv
import shutil

import cv2
import yaml

from objects.classifiers.config import CLASSIFICATION_DATASET_DIR, CLASSIFICATION_METADATA
from objects.config import DATA_YAML, DATASET_DIR


class ClassificationDataset:

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
            rows.extend(self._prepare_split(split, class_names))

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
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            annotations = self._read_labels(label_path)
            source = "bez_dloni_mix" if image_path.stem.startswith("mix_banknotow") else "z_dlonia"

            for box_index, (class_id, x, y, width, height) in enumerate(annotations):
                crop = self._crop(image, x, y, width, height)
                if crop.size == 0:
                    continue

                class_name = class_names[class_id]
                output_dir = CLASSIFICATION_DATASET_DIR / split / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{image_path.stem}_{box_index}.jpg"
                cv2.imwrite(str(output_path), crop)

                rows.append({
                    "split": split,
                    "crop_path": str(output_path.relative_to(CLASSIFICATION_DATASET_DIR)),
                    "image_name": image_path.name,
                    "class": class_name,
                    "object_count": len(annotations),
                    "source": source,
                })
        return rows

    def _read_labels(self, label_path):
        rows = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            rows.append((int(parts[0]), *(float(value) for value in parts[1:])))
        return rows

    def _crop(self, image, x, y, width, height):
        image_h, image_w = image.shape[:2]
        margin = 0.04
        x1 = max(0, int((x - width / 2 - margin * width) * image_w))
        y1 = max(0, int((y - height / 2 - margin * height) * image_h))
        x2 = min(image_w, int((x + width / 2 + margin * width) * image_w))
        y2 = min(image_h, int((y + height / 2 + margin * height) * image_h))
        return image[y1:y2, x1:x2]
