from objects.classifiers.predict import BanknoteClassifier
from objects.config import BACKGROUND_CLASS, CONFIDENCE, IMGSZ, NMS_IOU, device_name
from objects.detection.models import load_detector
from objects.labels import crop_with_margin


class DetectorPipeline:
    #Sam detektor: ramka i nominal w jednym przejsciu sieci

    def __init__(self, weights, device):
        self.model = load_detector(weights)
        self.device = device

    def run(self, image, conf=CONFIDENCE):
        return [{"box": box, "label": self.model.names[class_id], "conf": score, "score": score}
                for box, score, class_id in find_boxes(self.model, image, conf, self.device)]


class TwoStagePipeline:
    #Najpierw "gdzie jest jakis banknot" (lokalizator), potem "jaki to nominal", Klasyfikator zna tez klase "brak", wiec moze odrzucic ramke postawiona na rece albo na scianie.

    def __init__(self, locator_weights, classifier_path, device):
        self.locator = load_detector(locator_weights)
        self.classifier = BanknoteClassifier(classifier_path, device_name(device))
        self.device = device

    def run(self, image, conf=CONFIDENCE):
        objects = []
        for box, score, _ in find_boxes(self.locator, image, conf, self.device):
            crop = crop_with_margin(image, *box)
            if crop.size == 0:
                continue
            label, label_conf = self.classifier.predict(crop)
            if label == BACKGROUND_CLASS:
                continue
            #"conf" = pewnosc lokalizatora (na niej jest prog jak w apce)
            #"score" = pewnosc calego potoku do mAP: i ze to banknot, i ze ten nominal
            objects.append({"box": box, "label": label, "conf": score, "score": score * label_conf})
        return objects


def find_boxes(model, image, conf, device):
    result = model.predict(image, imgsz=IMGSZ, conf=conf, iou=NMS_IOU, device=device, verbose=False)[0]
    return [(tuple(float(v) for v in box.xyxy[0].tolist()), float(box.conf[0]), int(box.cls[0]))
            for box in result.boxes]
