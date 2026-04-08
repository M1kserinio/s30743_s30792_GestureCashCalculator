import cv2
import mediapipe as mp

class Gesture:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
        self.draw = mp.solutions.drawing_utils
        self.last_gesture = "" # literalnuie bez tego mi kompa scielo pozdro pozdro
        self.xd = cv2.imread("RemedyOfAuthors/jamaks.png")
        self.goat = cv2.imread("RemedyOfAuthors/image.png")

    def get_fingers(self, lm):
        kciuk = 1 if lm[4][0] < lm[3][0] else 0
        wskazujacy = 1 if lm[8][1] < lm[6][1] else 0
        srodkowy = 1 if lm[12][1] < lm[10][1] else 0
        serdeczny = 1 if lm[16][1] < lm[14][1] else 0
        maly = 1 if lm[20][1] < lm[18][1] else 0
        return [kciuk, wskazujacy, srodkowy, serdeczny, maly]

    def get_gesture(self, fingers):
        if fingers == [0, 0, 0, 0, 0]:
            return "POTWIERDZ"
        if fingers == [1, 1, 1, 1, 1]:
            return "RESET"
        if fingers == [0, 1, 0, 0, 0]:
            return "DODAJ"
        if fingers == [0, 1, 1, 0, 0]:
            return "ODEJMIJ"
        if fingers == [1, 0, 0, 0, 0]:
            return "ZAPISZ" # to mozna pozminac alke zeby jakakolwiek logika byla
        if fingers == [0,1,0,0,1]:
            return "goat"
        if fingers == [0,0,1,0,0]:
            return "xd"
        return ""

    def detection(self, img):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        gesture = ""

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.draw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)

            lm = [(int(p.x * w), int(p.y * h)) for p in hand.landmark]
            fingers = self.get_fingers(lm)
            gesture = self.get_gesture(fingers)

        return gesture

    def detect_info(self, gesture):
        if gesture == "":
            return

        if gesture != self.last_gesture:
            print("WYKRYTO GEST:", gesture)
            if gesture == "xd":
                cv2.imshow("OVERWATCH", self.xd)
                cv2.waitKey(1200)
                cv2.destroyWindow("OVERWATCH")
            if gesture == "goat":
                cv2.imshow("OVERWATCH2", self.goat)
                cv2.waitKey(1200)
                cv2.destroyWindow("OVERWATCH2")
            self.last_gesture = gesture