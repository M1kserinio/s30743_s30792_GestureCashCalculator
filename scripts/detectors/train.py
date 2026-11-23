#Trening detektorow na tym samym zbiorze.

#python -m scripts.detectors.train               yolo, lokalizator, rtdetr - po kolei
#python -m scripts.detectors.train rtdetr        tylko wybrane (RT-DETR trwa najdluzej, mozna go puscic osobno)

#yolo        - yolo11n, yolo11s, yolo11m
#lokalizator - yolo11s z jedna klasa "banknot", pierwszy etap potoku lokalizator + klasyfikator
#rtdetr      - rtdetr-l (transformer, bez NMS)

import sys

from objects.detection.trainer import DetectorTrainer

if __name__ == "__main__":
    trainer = DetectorTrainer()
    steps = {
        "yolo": trainer.train_yolo,
        "lokalizator": trainer.train_locator,
        "rtdetr": trainer.train_rtdetr,
    }
    names = sys.argv[1:] or list(steps)
    unknown = [name for name in names if name not in steps]
    if unknown:
        sys.exit(f"Nie znam: {' '.join(unknown)}. Do wyboru: {' '.join(steps)}")
    for name in names:
        steps[name]()
    print("\nTrening zakonczony. Dalej: python -m scripts.detectors.eval")
