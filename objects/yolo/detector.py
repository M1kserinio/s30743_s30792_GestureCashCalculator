import filecmp
import cv2
from ultralytics import YOLO

from objects.config import (
    BEST_MODEL,
    CLASS_VALUES,
    CONFIDENCE,
    IMGSZ,
    NMS_IOU,
    YOLO_MODELS,
    best_device,
    model_path,
    BOX_COLORS,
)


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(str(BEST_MODEL))
        self.device = best_device()
        self.model_options = [
            (name, model_path(name))
            for name in YOLO_MODELS
            if model_path(name).exists()
        ]
        self.current_model_index = next(
            (
                index
                for index, (_, path) in enumerate(self.model_options)
                if filecmp.cmp(BEST_MODEL, path, shallow=False)
            ),
            -1,
        )
        best_name = (
            self.model_options[self.current_model_index][0]
            if self.current_model_index >= 0
            else "best"
        )
        self.model_name = f"{best_name} (best)"

    def next_model(self):
        """Przełącza na kolejny wytrenowany model YOLO z folderu."""
        if not self.model_options:
            return
        self.current_model_index = (self.current_model_index + 1) % len(self.model_options)
        self.model_name, path = self.model_options[self.current_model_index]
        self.model = YOLO(str(path))
        print(f"Przelaczono model YOLO na: {self.model_name}")

    def detection(self, img, money_zone=None):
        """Wykrywa banknoty. Jeśli podano money_zone (x1, y1, x2, y2), odrzuca te poza strefą."""
        result = self.model.predict(
            img,
            imgsz=IMGSZ,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device=self.device,
            verbose=False,
        )[0]

        detected_objects = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Jeśli zdefiniowano strefę dla pieniędzy, sprawdzamy czy środek banknotu w niej leży
            if money_zone:
                zx1, zy1, zx2, zy2 = money_zone
                if not (zx1 <= cx <= zx2 and zy1 <= cy <= zy2):
                    continue

            label = self.model.names[int(box.cls[0])]
            detected_objects.append({
                "label": label,
                "value": CLASS_VALUES.get(label, 0.0),
                "conf": float(box.conf[0]),
                "box": (x1, y1, x2, y2),
            })

        return detected_objects

    def draw(self, img, detected_objects):
        """Rysuje ramki i napisy tylko dla przekazanych (przefiltrowanych) obiektów."""
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["box"]
            color = BOX_COLORS.get(obj["label"], (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{obj['label']} ({obj['conf']:.2f})"
            cv2.putText(img, label_text, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 2)