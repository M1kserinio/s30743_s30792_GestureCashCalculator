import numpy as np
import mediapipe as mp
import cv2, os, pickle

MIDDLE_MCP = 9

#A metoda stałe
PROG_ROZSTAW = 0.3
PROG_OK = 0.35

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B_MODEL_PATH = os.path.join(ROOT, "gest_results", "B_MLP_model.pkl")
C_MODEL_PATH = os.path.join(ROOT, "gest_results", "C_SVC_model.pkl")

class Gesture():
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
        self.gesture_labels = ("OK_HAND", "SPREAD_HAND", "TIGHT_HAND", "FIST", "POINT", "THUMB", "TWO")
        with open(B_MODEL_PATH, "rb") as f:
            self.modelB = pickle.load(f)
        with open(C_MODEL_PATH, "rb") as f:
            self.modelC = pickle.load(f)


    def detect_gesture(self, img, mode="B"):

        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        handLm = self.hands.process(img_rgb)

        detected_hands = [] #na obie rece lista
        gesture = None

        if handLm.multi_hand_landmarks:
            # hand-reka, handedness-sprawdza czy lewa, czy prawa
            for hand, handedness in zip(handLm.multi_hand_landmarks, handLm.multi_handedness):
                label = handedness.classification[0].label
                normalized_lm = self.normalize_landmarks(hand)
                if mode == "A":
                    gesture = self.A(normalized_lm)
                elif mode == "B":
                    gesture = self.B(normalized_lm)
                elif mode == "C":
                    gesture = self.C(normalized_lm)

                detected_hands.append({"side": label, "gesture": gesture, "landmarks": hand})
        return detected_hands

    def A(self, lm): #Geometryczne, matematyczne podejśćie do klasyfikacji
        finger_list = (self._finger_extended(lm[8], lm[6]),  # index
                       self._finger_extended(lm[12], lm[10]),  # middle
                       self._finger_extended(lm[16], lm[14]),  # ring
                       self._finger_extended(lm[20], lm[18]),  # pinky
                       bool(self._distance(lm[17], lm[4]) > self._distance(lm[17], lm[3])) #thumb (kciuck zgina sie w strone malego palca)
                       )

        #mediana z rozstawu palcow
        rozstaw = np.mean([self._distance(f1, f2) for f1, f2 in [(lm[8], lm[12]), (lm[12], lm[16]), (lm[16], lm[20])]])

        if finger_list[1] and finger_list[2] and finger_list[3]:
            if self._distance(lm[4], lm[8]) < PROG_OK:
                return "OK_HAND"

        match finger_list:
            case (True, True, True, True, True):
                if rozstaw > PROG_ROZSTAW:
                    return "SPREAD_HAND"
                return "TIGHT_HAND"
            case (False, False, False, False, False):
                return "FIST"
            case (True, False, False, False, False):
                return "POINT"
            case (True, True, False, False, False):
                return "TWO"
            case (False, False, False, False, True):
                return "THUMB"
            case _:
                return "NONE"

    def B(self, lm): #Uczenie maszynowe na surowych, znormalizowanych danych
        lm_flat = lm.flatten().reshape(1, -1)
        gesture = self.modelB.predict(lm_flat)[0]
        return gesture

    def C(self, lm): #Uczenie maszynowe na cechach inżynieryjnych
        lm_flat = lm.flatten().reshape(1, -1)
        gesture = self.modelC.predict(lm_flat)[0]
        return gesture

    def _distance(self, point1, point2):
        """Literalnie odleglosc punktow"""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def _finger_extended(self, tip, pip):
        """Sprawdza, czy dany palec jest wyprostowany."""
        # bool() bo bez niego nie rozpoznawal finger list nic xd
        return bool(self._distance(0, tip) > self._distance(0, pip))

    @staticmethod
    def normalize_landmarks(handLm):
        """Normalizacja landmarków, żeby wszystkie opierały sie na nadgarstkui były w przedziale 0-1.5."""
        landmarks = [[lm.x, lm.y, lm.z] for lm in handLm.landmark]  # pobranie współrzędnych
        landmarks = np.array(landmarks)  # konwersja do tablicy numpy
        landmarks -= landmarks[
            0]  # nadgarstek staje się bazowym punktem odniesienia, przesuwamy inne lmy względem niego

        middle_mip_scale = np.linalg.norm(
            landmarks[MIDDLE_MCP][:2])  # obliczenie odległości od nadgarstka do środkowego stawu palca
        if middle_mip_scale > 0:
            landmarks /= middle_mip_scale  # normalizacja skali względem middle mipa

        return landmarks