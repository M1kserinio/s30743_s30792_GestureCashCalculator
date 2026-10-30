from ultralytics import RTDETR, YOLO


def load_detector(weights):
    """Wczytuje wagi YOLO albo RT-DETR.

    RT-DETR trzeba otworzyc klasa RTDETR: YOLO() wczytaloby te wagi, ale potem robilo NMS
    na wyjsciu, ktore NMS w ogole nie potrzebuje. Typ bierzemy z samego pliku, nie z nazwy,
    bo pieniadze_best.pt moze byc kopia dowolnego z nich.
    """
    model = YOLO(str(weights))
    if is_rtdetr(model):
        return RTDETR(str(weights))
    return model


def is_rtdetr(model):
    return type(model.model).__name__ == "RTDETRDetectionModel"


def architecture(model):
    if is_rtdetr(model):
        return "transformer (bez NMS)"
    return "CNN + NMS"
