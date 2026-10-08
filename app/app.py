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

        self.modes = ("A", "B", "C")
        self.mode = "A"

        self.STABLE_FRAMES = 5
        self.candidate_ops = []  # wynik z ostatniej klatki
        self.candidate_count = 0  # ile klatek z rzędu kandydat się powtarza
        self.shown_ops = []  # wynik aktualnie pokazywany na ekranie

    def remove_duplicate_hands(self, hands, min_dist=0.05):
        """
        MediaPipe przy max_num_hands=2 potrafi zwrócić tę samą dłoń dwa razy (dwie prawie
        nakładające się detekcje). Jeśli średnie położenie landmarków dwóch dłoni (x, y w skali
        0-1 obrazu) jest bliżej niż min_dist, uznajemy je za tę samą dłoń i zostawiamy pierwszą.
        """
        unique = []
        centers = []
        for hand in hands:
            pts = hand["landmarks"].landmark
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            if all(((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5 >= min_dist for ux, uy in centers):
                unique.append(hand)
                centers.append((cx, cy))
        return unique

    def stabilize(self, operations):
        """Zwraca wynik do pokazania - zmienia go dopiero, gdy nowy utrzyma się STABLE_FRAMES klatek."""
        if operations == self.candidate_ops:
            self.candidate_count += 1
        else:
            self.candidate_ops = operations
            self.candidate_count = 1
        if self.candidate_count >= self.STABLE_FRAMES:
            self.shown_ops = operations
        return self.shown_ops

    def next_mode(self):
        """Przełącza tryb na kolejny z listy (po C wraca do A)."""
        idx = self.modes.index(self.mode)
        self.mode = self.modes[(idx + 1) % len(self.modes)]

    def to_operation(self, hand):
        """
        Zamienia wynik z Gesture.detect_gesture (np. {"side": "Left", "gesture": "SPREAD_HAND"})
        na format oczekiwany przez draw_ui (np. "L_SPREAD").
        Modele zwracają etykiety z sufiksem "_HAND" (OK_HAND, SPREAD_HAND, TIGHT_HAND),
        a draw_ui ich nie ma, więc go obcinamy. Brak gestu (None, np. tryb C) -> "NONE".
        """
        gesture = hand["gesture"]
        if gesture is None or gesture == "NONE":
            return "NONE"
        prefix = "L" if hand["side"] == "Left" else "R"
        return f"{prefix}_{gesture.removesuffix('_HAND')}"

    def run(self):
        while True:
            ok, img = self.cap.read()
            if not ok:
                break

            self.process_frame(img)
            self.ui.show(img)
            if self.ui.quit_pressed():
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("\t"):
                self.next_mode()

        self.cap.release()
        self.ui.close()

    def process_frame(self, img):
        zones = self.get_zones(img)

        detected_objects = []
        if self.object_system:
            # YOLO przed mediapipe, bo mediapipe rysuje kropki dloni na img i YOLO by je widzial
            detected_objects = self.object_system.detection(img, zones)
            left, right = self.object_system.zone_values(detected_objects)
            self.counter.set_detected(left, right)

        hand_gesture = self.gesture_system.detect_gesture(img, mode=self.mode)
        hand_gesture = self.remove_duplicate_hands(hand_gesture)
        operations = self.stabilize([self.to_operation(hand) for hand in hand_gesture])
        self.counter.apply(operations[0] if operations else "NONE")

        # ramki pieniedzy rysujemy dopiero na koncu, z tego samego powodu co wyzej
        if self.object_system:
            self.object_system.draw(img, detected_objects)
        self.ui.draw(img, zones, self.counter, operations, self.mode)

    def get_zones(self, img):
        #granice stref w pikselach: lewa strefa < x_left, prawa strefa > x_right
        w = img.shape[1]
        return w * self.split_left, w * self.split_right
