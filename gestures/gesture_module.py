import math

import cv2
import mediapipe as mp
import numpy as np
import pickle

class Gesture:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
        self.draw = mp.solutions.drawing_utils
        self.last_gesture = ""
        self.model = pickle.load(open("gesture_model.pkl", "rb"))

    def get_fingers(self, lm, label):
        if label == "Right":
            kciuk = 0 if lm[4][0] < lm[3][0] else 1
        else:
            kciuk = 0 if lm[4][0] > lm[3][0] else 1
        wskazujacy = 1 if lm[8][1] < lm[6][1] else 0
        srodkowy = 1 if lm[12][1] < lm[10][1] else 0
        serdeczny = 1 if lm[16][1] < lm[14][1] else 0
        maly = 1 if lm[20][1] < lm[18][1] else 0
        return [kciuk, wskazujacy, srodkowy, serdeczny, maly]

    def get_gesture(self, fingers):
        if fingers == [0, 1, 0, 0, 0]:
            return "POINT"
        if fingers == [0, 0, 1, 0, 0]:
            return "MIDDLE"
        if fingers == [1, 0, 0, 0, 0]:
            return "THUMB"
        if fingers == [0, 0, 0, 1, 0]:
            return "SERD"
        if fingers == [0, 0, 0, 0, 1]:
            return "SMALL"
        if fingers == [1,1,1,1,1]:
            return "FULL"
        if fingers == [0,0,0,0,0]:
            return "FIST"
        return ""

    def detection(self, img, middle_zone, manual=True):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        detected_hands = [] #Na dwie ręce lista

        if result.multi_hand_landmarks:
            #hand - reka, handednss - sprawdza czy lwea czy prawa
            for hand, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                self.draw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
                label = handedness.classification[0].label #lewa czy prawa

                lm = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]
                fingers = self.get_fingers(lm, label)
                gesture = self.get_gesture(fingers) if manual else self.predict_gesture(hand)

                #srodek lapy
                middle = int(hand.landmark[9].x * w)
                in_middle = True if middle_zone[0] < middle < middle_zone[1] else False

                detected_hands.append({
                    "label": label,
                    "fingers": fingers,
                    "in_middle": in_middle,
                    "lm": hand,
                    "gesture": gesture
                })

        return detected_hands

    def detect_info(self, detected_hands):
        # bierzemy dane z dwoch rak
        hands = {hand["label"]: hand for hand in detected_hands}
        left_hand = hands.get("Right") # Odwrócone w mediapipe
        right_hand = hands.get("Left") # Odwrócone w mediapipe
        l_gest, r_gest = None, None
        if left_hand:
            l_gest = left_hand["gesture"]
            if not left_hand["in_middle"]:
                left_hand = None
        if right_hand:
            r_gest = right_hand["gesture"]
            if not right_hand["in_middle"]:
                right_hand = None

        if len(detected_hands) == 1:
            # LEFT HAND
            if left_hand:
                if l_gest == "POINT":  # Wskazujący
                    return "L_ADD"
                elif l_gest == "THUMB":  # Kciuk
                    return "L_REMOVE"
                elif l_gest == "FULL_HAND":
                    return "L_FULL"
                elif l_gest == "OK_SIGN":
                    return "L_OK"
            # RIGHT HAND
            if right_hand:
                if r_gest == "POINT":  # Wskazujący
                    return "R_ADD"
                elif r_gest == "THUMB":  # Kciuk
                    return "R_REMOVE"
                elif r_gest == "FULL_HAND":
                    return "R_FULL"
                elif r_gest == "OK_SIGN":
                    return "R_OK"

        if left_hand and right_hand:
            if l_gest == "POINT" and r_gest == "POINT":  # Oba wskazujące
                return "SUM"

        return "NONE"

    def predict_gesture(self, hand):
        features = self.get_normalized_landmarks(hand)
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        return prediction

    def get_normalized_landmarks(self, hand):
        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y])

        landmarks = np.array(landmarks)

        #nadgarstek jest punktem 0,0
        base_x, base_y = landmarks[0]
        landmarks = landmarks - [base_x, base_y]

        #dystans od nadgarstka
        distances = np.linalg.norm(landmarks, axis=1)
        max_distance = np.max(distances)

        if max_distance > 0:
            landmarks = landmarks / max_distance

        return landmarks.flatten().tolist()