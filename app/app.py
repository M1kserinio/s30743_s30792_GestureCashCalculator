import threading
import time
import traceback

import cv2

from app.cash_counter import CashCounter
from app.ui import UI
from gestures.gesture_module import Gesture
from objects.config import BEST_MODEL
from objects.object_detector import ObjectDetector


class Camera:
    """Wątek tylko dla kamery"""

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self.running = True
        self.lock = threading.Lock()
        self.new = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame
            self.new.set()
            if not ok:
                break

    def read(self):
        """zwraca klatke"""
        self.new.wait(timeout=1) #tutaj czeka na klatke max sekunde w razie jakby cos sie zacielo
        self.new.clear()
        with self.lock:
            return self.ok and self.frame is not None, self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


class Detection:
    """watek dla liczenia"""

    def __init__(self, det):
        self.detection = det #sposob detekcji
        self.frame = None
        self.result = None
        self.new = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        self.frame = frame
        self.new.set()

    def _loop(self):
        while True:
            self.new.wait()
            self.new.clear()
            try:
                self.result = self.detection(self.frame)
            except Exception:
                traceback.print_exc()  # bez tego nie pojawia sie bledy


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.camera = Camera(self.cap)
        self.gesture_system = Gesture()
        #YOLO tylko jak jest wytrenowany model, inaczej apka dziala jak wczesniej
        self.object_system = ObjectDetector() if BEST_MODEL.exists() else None
        self.counter = CashCounter(use_detector=self.object_system is not None)
        self.ui = UI()

        self.split_left = 0.15
        self.split_right = 0.85

        self.modes = ("A", "B", "C")
        self.mode = "A"

        self.HOLD_TIME = 0.7  # ile sekund trzeba trzymac gest, zeby odpalila sie jego komenda
        self.hold_start = {}  # gest -> moment, w ktorym zaczal byc trzymany
        self.fired = set()  # gesty, ktorych komenda juz sie zrobila (zeby nie bylo w kolko)

        # kazdy gest ma swoja komende
        self.commands = {
            "L_OK": self.cmd_l_ok, "R_OK": self.cmd_r_ok,
            "L_SPREAD": self.cmd_l_spread, "R_SPREAD": self.cmd_r_spread,
            "L_TIGHT": self.cmd_l_tight, "R_TIGHT": self.cmd_r_tight,
            "L_FIST": self.cmd_l_fist, "R_FIST": self.cmd_r_fist,
            "L_POINT": self.cmd_l_point, "R_POINT": self.cmd_r_point,
            "L_THUMB": self.cmd_l_thumb, "R_THUMB": self.cmd_r_thumb,
            "L_TWO": self.cmd_l_two, "R_TWO": self.cmd_r_two,
        }

        # Tu robi sie detekcja na najnowszej klatce
        self.model_switch = False
        self.hands_worker = Detection(self.detect_hands)
        self.objects_worker = Detection(self.detect_objects) if self.object_system else None

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

    def update_commands(self, operations):
        """Odpala komende gestu, gdy jest trzymany co najmniej HOLD_TIME sekund (raz na jedno trzymanie)."""
        now = time.monotonic()
        # zapominamy gesty, ktorych juz nie widac - po ponownym pokazaniu liczymy czas od zera
        self.hold_start = {op: t for op, t in self.hold_start.items() if op in operations}
        self.fired = {op for op in self.fired if op in operations}

        for op in operations:
            if op == "NONE":
                continue
            if op not in self.hold_start:
                self.hold_start[op] = now
            elif now - self.hold_start[op] >= self.HOLD_TIME and op not in self.fired:
                self.fired.add(op)
                self.commands[op]()

    # komendy
    def cmd_l_ok(self):
        print("L_OK")

    def cmd_r_ok(self):
        print("R_OK")

    def cmd_l_spread(self):
        print("L_SPREAD")

    def cmd_r_spread(self):
        print("R_SPREAD")

    def cmd_l_tight(self):
        print("L_TIGHT")

    def cmd_r_tight(self):
        print("R_TIGHT")

    def cmd_l_fist(self):
        print("L_FIST")

    def cmd_r_fist(self):
        print("R_FIST")

    def cmd_l_point(self):
        print("L_POINT")

    def cmd_r_point(self):
        print("R_POINT")

    def cmd_l_thumb(self):
        print("L_THUMB")

    def cmd_r_thumb(self):
        print("R_THUMB")

    def cmd_l_two(self):
        print("L_TWO")

    def cmd_r_two(self):
        print("R_TWO")

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
            ok, img = self.camera.read()
            if not ok:
                break

            self.process_frame(img)
            self.ui.show(img)

            # jedno waitKey na petle - drugie wywolanie "zjadaloby" czesc nacisniec klawiszy (np. TAB)
            key = cv2.waitKey(1) & 0xFF
            if self.ui.quit_pressed(key):
                break
            if self.object_system and self.ui.model_switch_pressed():
                self.model_switch = True  # zmiane robi watek YOLO, zeby nie zmieniac modelu w trakcie detekcji
            if key == ord("\t"):
                self.next_mode()

        self.camera.stop()
        self.cap.release()
        self.ui.close()

    def detect_hands(self, frame):
        """Wykrywa reke i zwraca operacje dla obu z nich"""
        hands = self.gesture_system.detect_gesture(frame, mode=self.mode)
        hands = self.remove_duplicate_hands(hands)
        return [self.to_operation(hand) for hand in hands]

    def detect_objects(self, frame):
        """Twoje yolo i tak zmienisz"""
        if self.model_switch:
            self.model_switch = False
            self.object_system.next_model()
        objects = self.object_system.detection(frame, self.get_zones(frame))
        left, right = self.object_system.zone_values(objects)
        self.counter.set_detected(left, right)
        return objects

    def process_frame(self, img):
        zones = self.get_zones(img)

        # kopie, bo ponizej rysujemy na img, a watki czytaja klatke rownolegle
        self.hands_worker.submit(img.copy())
        if self.object_system:
            self.objects_worker.submit(img.copy())

        operations = self.hands_worker.result or []  # pokazywane od razu (ostatni gotowy wynik)
        self.update_commands(operations)  # komenda dopiero po przytrzymaniu

        model_name = self.object_system.model_name if self.object_system else None
        if self.object_system:
            self.object_system.draw(img, self.objects_worker.result or [])
        self.ui.draw(img, zones, self.counter, operations, self.mode, model_name)

    def get_zones(self, img):
        #granice stref w pikselach: lewa strefa < x_left, prawa strefa > x_right
        w = img.shape[1]
        return w * self.split_left, w * self.split_right