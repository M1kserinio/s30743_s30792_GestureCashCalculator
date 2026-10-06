import cv2

from app.cash_counter import CashCounter
from app.ui import UI
from gestures.gesture_module import Gesture
from objects.config import BEST_MODEL
from objects.object_detector import ObjectDetector


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.gesture_system = Gesture()
        #YOLO tylko jak jest wytrenowany model, inaczej apka dziala jak wczesniej
        self.object_system = ObjectDetector() if BEST_MODEL.exists() else None
        self.counter = CashCounter(use_detector=self.object_system is not None)
        self.ui = UI()

        self.split_left = 0.15
        self.split_right = 0.85

    def run(self):
        while True:
            ok, img = self.cap.read()
            if not ok:
                break

            self.process_frame(img)
            self.ui.show(img)
            if self.ui.quit_pressed():
                break

        self.cap.release()
        self.ui.close()

    def process_frame(self, img):
        zones = self.get_zones(img)

        detected_objects = []
        if self.object_system:
            #YOLO przed mediapipe, bo mediapipe rysuje kropki dloni na img i YOLO by je widzial
            detected_objects = self.object_system.detection(img, zones)
            left, right = self.object_system.zone_values(detected_objects)
            self.counter.set_detected(left, right)

        hand_info = self.gesture_system.detection(img, zones, manual=False)
        operation = self.gesture_system.detect_info(hand_info)
        self.counter.apply(operation)

        #ramki pieniedzy rysujemy rowniez na koncu
        if self.object_system:
            self.object_system.draw(img, detected_objects)
        self.ui.draw(img, zones, self.counter, operation)

    def get_zones(self, img):
        #granice stref w pikselach: lewa strefa < x_left, prawa strefa > x_right
        w = img.shape[1]
        return w * self.split_left, w * self.split_right
