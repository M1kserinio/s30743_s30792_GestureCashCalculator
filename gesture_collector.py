import cv2
import mediapipe as mp
import numpy as np
import os
import csv

GESTURE_NAME = "FULL_HAND"
DATA_PATH = "gesture_data"
FILE_NAME = "gestures.csv"

class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)
        self.draw = mp.solutions.drawing_utils

        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

    def normalize_landmarks(self, hand):
        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y])

        landmarks = np.array(landmarks)

        base_x, base_y = landmarks[0]
        landmarks = landmarks - [base_x, base_y]
        distances = np.linalg.norm(landmarks, axis=1)
        max_distance = np.max(distances)
        if max_distance > 0:
            landmarks = landmarks / max_distance
        return landmarks

    def save_to_csv(self, label, landmarks):
        file_path = os.path.join(DATA_PATH, FILE_NAME)
        flat_data = landmarks.flatten().tolist()

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([label] + flat_data)

    def run(self):
        cap = cv2.VideoCapture(0)
        print(f"Nagrywanie gestu: {GESTURE_NAME}")

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            current_landmarks = None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                self.draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                current_landmarks = self.normalize_landmarks(hand)

            cv2.putText(frame, f"Gest: {GESTURE_NAME}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("abc", frame)

            key = cv2.waitKey(1)
            if key == ord(' '):
                if current_landmarks is not None:
                    self.save_to_csv(GESTURE_NAME, current_landmarks)
                    mirrored = current_landmarks.copy()
                    mirrored[:, 0] = mirrored[:, 0] * -1
                    self.save_to_csv(GESTURE_NAME, mirrored)

                    print(f"Zapisano próbkę dla: {GESTURE_NAME}")
                else:
                    print("Nie wykryto dłoni!")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()