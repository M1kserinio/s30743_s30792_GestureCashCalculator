from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "gesture_images"
OUTPUT_DIR = ROOT / "gesture_images_cropped"

MARGIN = 0.4  # ile miejsca zostawiamy wokół dłoni
OUTPUT_SIZE = 128

hands = mp.solutions.hands.Hands(
    static_image_mode=True,  # każde zdjęcie jest niezależne
    max_num_hands=1,  # na zdjęciu jest jedna dłoń
    min_detection_confidence=0.5,  # niższy próg niż w aplikacji, żeby przycąć jak najwięcej zdjęć
)


def crop_around_hand(image):
    """Zwraca kwadratowy wycinek wokół dłoni albo None, jeśli nie znaleziono dłoni."""
    height, width = image.shape[:2]
    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # MediaPipe chce RGB
    if not result.multi_hand_landmarks:
        return None  # nie ma dłoni

    landmarks = result.multi_hand_landmarks[0].landmark  # 21 punktów dłoni
    xs = [point.x * width for point in landmarks]  # współrzędne są od 0 do 1, zamieniamy na piksele
    ys = [point.y * height for point in landmarks]

    hand_size = max(max(xs) - min(xs), max(ys) - min(ys))  # dłuższy bok prostokąta wokół dłoni
    side = max(int(hand_size * (1 + MARGIN)), 20)  # bok kwadratu = dłoń + margines
    center_x = int((min(xs) + max(xs)) / 2)  # środek dłoni
    center_y = int((min(ys) + max(ys)) / 2)

    # dodajemy ramkę dookoła zdjęcia, żeby wycinek nigdy nie wychodził poza zdjęcie
    padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
    left = center_x - side // 2 + side  # "+ side" bo zdjęcie jest teraz przesunięte o ramkę
    top = center_y - side // 2 + side
    crop = padded[top:top + side, left:left + side]  # wycinek kwadratowy

    return cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)


def crop_from_landmarks(image, hand_landmarks):
    height, width = image.shape[:2]
    landmarks = hand_landmarks.landmark  # 21 punktów dłoni
    xs = [point.x * width for point in landmarks]  # współrzędne są od 0 do 1, zamieniamy na piksele
    ys = [point.y * height for point in landmarks]

    hand_size = max(max(xs) - min(xs), max(ys) - min(ys))  # dłuższy bok prostokąta wokół dłoni
    side = max(int(hand_size * (1 + MARGIN)), 20)  # bok kwadratu = dłoń + margines
    center_x = int((min(xs) + max(xs)) / 2)  # środek dłoni
    center_y = int((min(ys) + max(ys)) / 2)

    # dodajemy ramkę dookoła zdjęcia, żeby wycinek nigdy nie wychodził poza zdjęcie
    padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
    left = center_x - side // 2 + side  # "+ side" bo zdjęcie jest teraz przesunięte o ramkę
    top = center_y - side // 2 + side
    crop = padded[top:top + side, left:left + side]  # wycinek kwadratowy

    return cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

saved_count = 0
skipped_files = []
for file in sorted(INPUT_DIR.glob("P*/*/*.jpg")):
    image = cv2.imread(str(file))
    crop = crop_around_hand(image)
    if crop is None:
        skipped_files.append(file)  # zapamiętujemy zdjęcia bez wykrytej dłoni
        continue
    save_path = OUTPUT_DIR / file.relative_to(INPUT_DIR)  # ta sama ścieżka, ale w nowym folderze
    save_path.parent.mkdir(parents=True, exist_ok=True)  # tworzy foldery, jeśli ich nie ma
    cv2.imwrite(str(save_path), crop)
    saved_count += 1

total = saved_count + len(skipped_files)
print(f"Cropped {saved_count} of {total} images -> {OUTPUT_DIR}")
print(f"No hand found in {len(skipped_files)} images (they were skipped).")
for file in skipped_files[:15]:
    print("  skipped:", file.relative_to(INPUT_DIR))