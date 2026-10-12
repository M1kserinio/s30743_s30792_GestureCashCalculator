import mediapipe as mp
import os, csv, cv2
from gestures.gesture_module import Gesture

GESTURES = ("OK_HAND", "SPREAD_HAND", "TIGHT_HAND", "FIST", "POINT", "THUMB", "TWO")
PERSONS = ("P01", "P02", "P03", "P04", "P05", "P06")

IMAGE_PATH = "gesture_images"
DATA_PATH = "gesture_data"
FILE_NAME = "gestures.csv"

def save_to_img(person, label, frame):
    gesture_dir = os.path.join(IMAGE_PATH, person, label)
    os.makedirs(gesture_dir, exist_ok=True)
    img_count = len(os.listdir(gesture_dir))
    img_path = os.path.join(gesture_dir, f"{img_count + 1}.jpg")
    print(f"Zapisano próbkę dla: {person} / {label} nr {img_count + 1}")
    cv2.imwrite(img_path, frame)


def draw(frame, hand):
    mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS) #rysuj landmarki

def generate_csv(hands_processor):
    """Przechodzi przez wszystkie zdjęcia w folderach i generuje CSV"""
    os.makedirs(DATA_PATH, exist_ok=True) #tworzenie folderu
    csv_path = os.path.join(DATA_PATH, FILE_NAME) #plik csv

    with open(csv_path, "w", newline="") as f: #otwiraniu pliku csv
        writer = csv.writer(f) #zapis do csv

        for person in PERSONS: #dla każdej osoby
            for gesture in GESTURES: #dla każdego gestu
                gesture_dir = os.path.join(IMAGE_PATH, person, gesture) #szukamy folderu
                if not os.path.exists(gesture_dir):
                    continue

                for img_name in os.listdir(gesture_dir): #dla każdego zdjęcia w folderze
                    img_path = os.path.join(gesture_dir, img_name)
                    frame = cv2.imread(img_path) #pobieramy zdjęcie
                    if frame is None:
                        continue

                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #zmieniamy kolor
                    results = hands_processor.process(img_rgb) #processing

                    if results.multi_hand_landmarks:
                        handLm = results.multi_hand_landmarks[0]
                        normalized_lm = Gesture.normalize_landmarks(handLm) #normalizacja landmarków
                        flat_data = normalized_lm.flatten().tolist() #zmieniamy w jedną linię do zapisu
                        writer.writerow([person, gesture] + flat_data)

                        mirrored_lm = normalized_lm.copy() #tworzymy kopię do lustrzanego odbicia
                        mirrored_lm[:, 0] *= -1 #odwracamy współrzędną x dla lustrzanego odbicia (wszystkie wiersze, pierwsza kolumna)
                        flat_mirrored_data = mirrored_lm.flatten().tolist() #zmieniamy w jedną linię do zapisu
                        writer.writerow([person, gesture] + flat_mirrored_data)


def collecting():
    """Metoda rozpoczynająca proces zbierania gestów."""
    cap = cv2.VideoCapture(0) #otwieramy okno kamery
    if not cap.isOpened():
        raise RuntimeError("Can't open camera") #błąd jeśli nie można

    gestureIndex = 0
    gestureCollected = GESTURES[(gestureIndex%len(GESTURES))] #obecnie pobierany gest
    personIndex = 0
    personCollected = PERSONS[personIndex % len(PERSONS)] #obecnie nagrywana osoba


    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.75) #inicjalizacja MediaPipe Hands dla jednej ręki

    current_landmarks = None #stworzenie zmiennej do landmarków

    while cap.isOpened(): #jeśli jest otwarta kamera
        ok, frame = cap.read() #pobierz stan i klatkę
        if not ok: break #jeśli nie udało się pobrać, przerwij
        frame = cv2.flip(frame, 1) #odwrócenie obrazu dla łątowśći użytkowania
        clean_frame = frame.copy() #skopiowanie obrazu do czystego obrazu
        h, w, _ = frame.shape #pobranie wymiarów obrazu

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #konwersja do RGB
        results = hands.process(img_rgb)  # przetwarzanie obrazu używając MediaPipe Hands

        if results.multi_hand_landmarks:  # jeśli wykryto landmarki
            handLm = results.multi_hand_landmarks[0]  # pobranie pierwszej ręki
            draw(frame, handLm)  # rysowanie landmarków

        cv2.putText(frame, f"Gest: {gestureCollected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Osoba: {personCollected}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Gesture Collector", frame)

        key = cv2.waitKey(1)
        if key == ord(" "):
            save_to_img(gestureCollected, clean_frame) # zapisz zdjęcie dłoni
        elif key == ord("c"):
            generate_csv(hands)
        elif key == ord("p"):
            personIndex += 1
            personCollected = PERSONS[(personIndex%len(PERSONS))]
        elif key == ord("\t"):
            gestureIndex += 1
            gestureCollected = GESTURES[(gestureIndex%len(GESTURES))]
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collecting()

