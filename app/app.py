import threading
import time
import traceback
import cv2

from app.cash_counter import CashCounter
from app.ui import UI
from gestures.gesture_module import Gesture
from objects.classifiers.config import CNN_MODEL, RESNET_MODEL
from objects.config import BEST_MODEL, CLASS_VALUES
from objects.yolo.detector import ObjectDetector


class Camera:
    """Oddzielnie kamera w wątku"""
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = False
        self.running = True
        self.lock = threading.Lock() #klucz lockujący na wątek
        self.new = threading.Event() #używane do sycnhronizacji wątków, nowa klatka
        threading.Thread(target=self._loop, daemon=True).start() #start wątku

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame #zapamiętanie stanu i klatki
            self.new.set() #nowa klatka
            if not ok:
                break

    def read(self):
        self.new.wait(timeout=1) #czeka na nową klatkę przez max sekunde
        self.new.clear() #jak dostanie to resetuje nową
        with self.lock:
            return self.ok and self.frame is not None, self.frame #zwraca stan i klatkę

    def stop(self):
        self.running = False


class Detection:
    """Wątek dla detekcji dłoni lub YOLO."""
    def __init__(self, det_fn):
        self.det_fn = det_fn #funckja obecnego weykrywania
        self.frame = None
        self.result = None
        self.new = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, frame):
        self.frame = frame
        self.new.set()

    def _loop(self):
        while True:
            self.new.wait() #czeka na nową klatkę bez timeoutu bo detekcja dłużej zajmuje
            self.new.clear()
            try:
                self.result = self.det_fn(self.frame)  #wywołanie funkcji detekcji dłoni lub YOLO
            except Exception:
                traceback.print_exc()


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.camera = Camera(self.cap)
        self.gesture_system = Gesture()
        self.object_system = ObjectDetector() if BEST_MODEL.exists() else None
        self.counter = CashCounter()
        self.ui = UI()

        self.ui_system = 1           # Styl 2 jako aktywny
        self.modes = ("A", "B", "C")
        self.mode = "A"

        self.HOLD_TIME = 0.6
        self.hold_start = {}
        self.fired = set()

        self.r_filters = { #prawa ręka służy do pokazywania szczególnych banknotów
            "R_POINT": "10",
            "R_TWO": "20",
            "R_SPREAD": "50",
            "R_TIGHT": "100",
            "R_FIST": "200",
            "R_THUMB": "500",
            "R_OK": "ALL",
        }

        self.recognizer_options = ("YOLO", "CNN", "ResNet-18")
        self.recognizer_paths = {"CNN": CNN_MODEL, "ResNet-18": RESNET_MODEL}
        self.recognizer_index = 0
        self.recognizer_name = "YOLO"
        self.classifier = None

        self.commands = { #lewa to operacja na banknotach lub apce
            "L_OK": self.counter.save_group,
            "L_POINT": self.counter.calculate_add,
            "L_TWO": self.counter.calculate_subtract,
            "L_FIST": self.toggle_commands_help,
            "L_THUMB": self.next_recognizer,
            "L_SPREAD": self.next_mode,
            "L_TIGHT": self.next_ui_system,
        }

        self.hands_worker = Detection(self.detect_hands) #workery do mediapipe i yolo
        self.objects_worker = Detection(self.detect_objects) if self.object_system else None

    def toggle_commands_help(self):
        """Do L_FIST instrukcja"""
        self.ui.show_help = not self.ui.show_help

    def next_yolo_model(self):
        """Klawisz y: kolejny wariant YOLO, bez zmiany jego kodu."""
        if self.object_system:
            self.object_system.next_model()

    def next_recognizer(self):
        """L_THUMB: kto nadaje nominał ramkom znalezionym przez YOLO."""
        self.recognizer_index = (self.recognizer_index + 1) % len(self.recognizer_options)
        self.recognizer_name = self.recognizer_options[self.recognizer_index]
        path = self.recognizer_paths.get(self.recognizer_name)
        if path is None or not path.exists():
            self.classifier = None
            if path is not None:
                print(f"Brak wag {self.recognizer_name}: {path}")
            else:
                print("Nominał z YOLO")
            return

        from objects.classifiers.config import classification_device
        from objects.classifiers.predict import BanknoteClassifier

        self.classifier = BanknoteClassifier(path, classification_device())
        print(f"Nominał rozpoznaje: {self.recognizer_name}")

    def label_with_recognizer(self, frame, objects):
        if self.classifier is None:
            return objects
        height, width = frame.shape[:2]
        labeled = []
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            item = dict(obj)
            if crop.size:
                label, confidence = self.classifier.predict(crop)
                item["label"] = label
                item["conf"] = confidence
                item["value"] = CLASS_VALUES.get(label, 0.0)
            labeled.append(item)
        return labeled

    def next_ui_system(self):
        """Do L_TIGHT zmiana stylu UI"""
        self.ui_system = 1 if self.ui_system >= 3 else self.ui_system + 1

    def next_mode(self):
        """Do L_SPREAD zmiana trybu detekcji gestów A/B/C"""
        idx = self.modes.index(self.mode)
        self.mode = self.modes[(idx + 1) % len(self.modes)]

    def to_operation(self, hand):
        """Tworze operacje, czyli zalezy ktora łapa i potem dodaje gest"""
        gesture = hand["gesture"]
        if not gesture or gesture == "NONE":
            return "NONE"

        # Obrot bo MediaPipe zaklada odwrocony obraz
        prefix = "R" if hand["side"] == "Left" else "L"
        clean_gesture = gesture.removesuffix("_HAND")
        return f"{prefix}_{clean_gesture}"

    def update_gestures(self, operations):
        now = time.monotonic() #zapamiętanie czasu w sekundach
        self.hold_start = {op: t for op, t in self.hold_start.items() if op in operations} #sprawdzenie czy gest jest nadal aktywny, jeśli nie to usuwa z hold_start
        self.fired = {op for op in self.fired if op in operations} #sprawdzenie czy gest jest nadal aktywny, jeśli nie to usuwa z fired

        active_r_gestures = [op for op in operations if op.startswith("R_")] #aktywne gesty prawej ręki
        if active_r_gestures and active_r_gestures[0] in self.r_filters: #ustawia filtr na podstawie aktywnego gestu R
            self.counter.set_r_filter(self.r_filters[active_r_gestures[0]])
        else:
            self.counter.set_r_filter(None)

        for op in operations:
            if not op.startswith("L_"):
                continue
            if op not in self.hold_start: #jeśli gest nie był wcześniej aktywny, to zapisuje czas jego rozpoczęcia
                self.hold_start[op] = now
            elif now - self.hold_start[op] >= self.HOLD_TIME and op not in self.fired: #jeśli gest jest aktywny dłużej niż HOLD_TIME i nie został jeszcze wywołany, to wywołuje komendę
                self.fired.add(op) #zapamiętuje, że komenda została wywołana
                if op in self.commands:
                    self.commands[op]() #wywołanie komendy
        #Ogólnie to cała ta funkcja for nade mną jest po to, żeby niechacy nie nie wywołać komendy, dlateo zapisuje czas i po sekundzie dopiero aktywuje komende

    def detect_hands(self, frame):
        """wykrywa dlonie"""
        h, w = frame.shape[:2]
        hands = self.gesture_system.detect_gesture(frame, mode=self.mode)
        valid_hands = []

        #Dla UI 1 i 2 (kwadraty i strefy) sprawdza czy lapy sa w danym obszarze

        for hand in hands:
            pts = hand["landmarks"].landmark
            cx, cy = (sum(p.x for p in pts) / len(pts)) * w, (sum(p.y for p in pts) / len(pts)) * h

            if self.ui_system == 1:
                if cx > (w // 2) or cy > (h - 75):
                    continue
            elif self.ui_system == 2:
                # Wymiary powiększonych kwadratów dłoni
                box_w, box_h = int(w * 0.24), int(h * 0.32)
                m_x, m_y = 15, 15

                in_left_box = (m_x <= cx <= m_x + box_w) and (h - box_h - m_y <= cy <= h - m_y)
                in_right_box = (w - box_w - m_x <= cx <= w - m_x) and (h - box_h - m_y <= cy <= h - m_y)
                if not (in_left_box or in_right_box):
                    continue

            valid_hands.append(hand)

        return [self.to_operation(h) for h in valid_hands]

    def detect_objects(self, frame):
        """Wykrywa banknoty"""
        h, w = frame.shape[:2]
        money_zone = None

        if self.ui_system == 1:
            money_zone = (w // 2, 0, w, h - 75)

        objects = self.object_system.detection(frame, money_zone=money_zone)

        # W UI 2 banknoty mogą leżeć wszędzie, byle nie wewnątrz kwadratów na dłonie
        if self.ui_system == 2:
            box_w, box_h = int(w * 0.24), int(h * 0.32)
            m_x, m_y = 15, 15

            # Sprawdza czy środek banknotu leży w obszarze kwadratów dłoni
            def is_in_hand_box(bx, by):
                in_l = (m_x <= bx <= m_x + box_w) and (h - box_h - m_y <= by <= h - m_y)
                in_r = (w - box_w - m_x <= bx <= w - m_x) and (h - box_h - m_y <= by <= h - m_y)
                return in_l or in_r

            # Filtruje banknoty, które leżą w obszarze kwadratów dłoni
            objects = [o for o in objects if not is_in_hand_box(
                (o["box"][0] + o["box"][2]) / 2, #czy jest w srodku pierwszego
                (o["box"][1] + o["box"][3]) / 2 #albo drugiego, x oraz y podzielic na 2 to srodek
            )]

        self.counter.set_objects(self.label_with_recognizer(frame, objects))
        return objects

    def run(self):
        while True:
            ok, img = self.camera.read()
            if not ok:
                break

            self.hands_worker.submit(img.copy()) #wysyła klatkę do detekcji dłoni w osobnym wątku
            if self.object_system:
                self.objects_worker.submit(img.copy()) #wysyła klatkę do detekcji banknotów w osobnym wątku

            operations = self.hands_worker.result or [] #operacje z detekcji dłoni
            self.update_gestures(operations) #wykonuje komendy na podstawie operacji

            if self.object_system:
                visible_objects = self.counter.get_visible_objects()
                self.object_system.draw(img, visible_objects)

            model_name = None
            if self.object_system:
                model_name = self.object_system.model_name
                if self.recognizer_name != "YOLO":
                    suffix = "" if self.classifier else " (brak wag)"
                    model_name = f"{model_name} | {self.recognizer_name}{suffix}"
            self.ui.draw(img, self.ui_system, self.counter, operations, self.mode, model_name)

            self.ui.show(img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("\t"):
                self.next_mode()
            elif key == ord("y"):
                self.next_yolo_model()
            elif key == ord("n"):
                self.next_recognizer()

        self.camera.stop()
        self.cap.release()
        self.ui.close()


if __name__ == "__main__":
    App().run()