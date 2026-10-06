import filecmp

import cv2
from ultralytics import YOLO

from objects.config import (
    BEST_MODEL,
    CLASS_VALUES,
    CONFIDENCE,
    HISTORY_SIZE,
    IMGSZ,
    YOLO_MODELS,
    best_device,
    model_path,
)


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(str(BEST_MODEL))
        self.device = best_device()
        self.history = []  #ostatnie odczyty (lewa, prawa)
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
        if not self.model_options:
            raise FileNotFoundError("Brak wytrenowanych wariantow YOLO w folderze models/")

        self.current_model_index = (self.current_model_index + 1) % len(self.model_options)
        self.model_name, path = self.model_options[self.current_model_index]
        self.model = YOLO(str(path))
        self.history.clear()
        print(f"model YOLO: {self.model_name}")

    def detection(self, img, zones):
        result = self.model.predict(img, imgsz=IMGSZ, conf=CONFIDENCE, device=self.device, verbose=False)[0]

        detected_objects = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = self.model.names[int(box.cls[0])]

            detected_objects.append({
                "label": label,
                "value": CLASS_VALUES.get(label, 0.0),  #nieznana klasa = 0 zl
                "conf": float(box.conf[0]),
                "box": (x1, y1, x2, y2),
                "zone": self.get_zone((x1 + x2) / 2, zones)
            })

        return detected_objects

    def get_zone(self, middle, zones):
        #strefa po srodku banknotu
        x_left, x_right = zones
        if middle < x_left:
            return "LEFT"
        if middle > x_right:
            return "RIGHT"
        return "MIDDLE"  #pieniadze na srodku nie licza sie do zadnej strony

    def zone_values(self, detected_objects):
        left = 0.0
        right = 0.0
        for obj in detected_objects:
            if obj["zone"] == "LEFT":
                left += obj["value"]
            elif obj["zone"] == "RIGHT":
                right += obj["value"]

        #round bo w pythonie 0.1 + 0.2 = 0.30000000000000004, a nizej porownujemy odczyty
        self.history.append((round(left, 2), round(right, 2)))
        if len(self.history) > HISTORY_SIZE:
            self.history.pop(0)

        #zwracamy najczestszy odczyt z ostatnich klatek, a nie aktualny
        #bo YOLO czasem na jedna klatke gubi banknot i kwota by skakala
        return max(self.history, key=self.history.count)

    def draw(self, img, detected_objects):
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f"{obj['label']} {obj['conf']:.2f}", (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 255), 2)
