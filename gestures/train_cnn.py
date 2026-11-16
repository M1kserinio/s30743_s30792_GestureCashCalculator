import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "gesture_images"
RESULTS_DIR = ROOT / "gest_results"

SEED = 42
EPOCHS = 25
BATCH_SIZE = 64
IMAGE_SIZE = 64

SIDE_NAMES = ("Left", "Right")
PHOTO_SIDE = 1  # ZMIEŃ którą dłoń pokazują zdjęcia # 0 = Left, 1 = Right


def prepare_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # OpenCV daje BGR, zamieniamy na RGB
    small_image = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  # zmniejszamy
    return small_image


def load_images():
    image_files = sorted(IMAGES_DIR.glob("P*/*/*.jpg"))  # lista wszystkich plików .jpg
    images = []
    gesture_names = []
    person_names = []
    for file in image_files:
        photo = cv2.imread(str(file))
        images.append(prepare_image(photo))  # zmniejsz i dodaj do listy
        gesture_names.append(file.parent.name)  # nazwa folderu gestu, np. FIST
        person_names.append(file.parent.parent.name)  # nazwa folderu osoby, np. P01
    return np.array(images), np.array(gesture_names), np.array(person_names)


def add_both_hands(images, gesture_ids):
    """Dodaje lustrzane odbicia zdjęć"""
    flipped_images = images[:, :, ::-1]  # odbicie lewo-prawo
    all_images = np.concatenate([images, flipped_images])  # oryginały + odbicia
    all_gesture_ids = np.concatenate([gesture_ids, gesture_ids])  # gest się nie zmienia po odbiciu
    original_sides = np.full(len(images), PHOTO_SIDE)  # oryginały: strona ze zdjęć
    flipped_sides = np.full(len(images), 1 - PHOTO_SIDE)  # odbicia: druga strona
    all_side_ids = np.concatenate([original_sides, flipped_sides])
    return all_images, all_gesture_ids, all_side_ids


def build_model(number_of_gestures):

    shared_body = keras.Sequential([
        # losowe zmiany dla treningu
        layers.RandomRotation(0.04),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.3),
        layers.Rescaling(1 / 127.5, offset=-1),  # normalizacja pikseli

        layers.Conv2D(filters=16, kernel_size=3, padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),
        layers.MaxPool2D(),

        layers.Conv2D(filters=32, kernel_size=3, padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),
        layers.MaxPool2D(),

        layers.Conv2D(filters=64, kernel_size=3, padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),
        layers.MaxPool2D(),

        layers.Conv2D(filters=128, kernel_size=3, padding="same"),
        layers.BatchNormalization(momentum=0.9),
        layers.ReLU(),
        layers.MaxPool2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
    ])

    image_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    features = shared_body(image_input)  # zdjęcie przechodzi przez część wspólną
    gesture_output = layers.Dense(number_of_gestures, activation="softmax", name="gesture")(features)  # który gest
    side_output = layers.Dense(1, activation="sigmoid", name="side")(features)  # lewa (0) czy prawa (1)

    model = keras.Model(inputs=image_input, outputs=[gesture_output, side_output])
    model.compile(
        optimizer="adam",  # sposób uczenia
        loss={"gesture": "sparse_categorical_crossentropy",  # błąd dla gestu
              "side": "binary_crossentropy"},  # błąd dla strony
    )
    return model


def train_model(images, gesture_ids, side_ids, number_of_gestures):
    model = build_model(number_of_gestures)
    model.fit(
        images,
        {"gesture": gesture_ids, "side": side_ids},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,  # mieszamy zdjęcia w każdej epoce
        verbose=1,  # bez wypisywania postępu
    )
    return model


def predict_images(model, images):
    gesture_probabilities, side_probabilities = model.predict(images, verbose=0)
    predicted_gesture_ids = gesture_probabilities.argmax(axis=1)  # gest z najwyższym prawdopodobieństwem
    predicted_side_ids = (side_probabilities[:, 0] > 0.5).astype(int)  # powyżej 0.5 = prawa dłoń
    return predicted_gesture_ids, predicted_side_ids


def measure_latency(model, image):
    """Mierzy ilosc ms"""
    one_image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3).astype("float32")
    for _ in range(20):  # rozgrzewka
        model(one_image, training=False)
    times_ms = []
    for _ in range(300):  # 300 pomiarów
        start = time.perf_counter()
        model(one_image, training=False)
        times_ms.append((time.perf_counter() - start) * 1000)  # zamiana na milisekundy
    return np.median(times_ms)


def plot_confusion_matrix(confusion, gesture_list, save_path):
    """Macierz pomylel"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(gesture_list)), gesture_list, rotation=45)  # nazwy gestów na dole
    ax.set_yticks(range(len(gesture_list)), gesture_list)  # nazwy gestów z boku
    for row in range(len(gesture_list)):
        for column in range(len(gesture_list)):
            count = confusion[row, column]  # ile zdjęć w tej komórce
            text_color = "white" if count > confusion.max() / 2 else "black"
            ax.text(column, row, count, ha="center", va="center", color=text_color)
    ax.set_title("CNN confusion matrix")
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)


def run(name="D"):
    keras.utils.set_random_seed(SEED)

    images, gesture_names, person_names = load_images()
    gesture_list = sorted(set(gesture_names.tolist()))
    person_list = sorted(set(person_names.tolist()))
    gesture_ids = []
    for gesture in gesture_names:
        gesture_ids.append(gesture_list.index(gesture))
    gesture_ids = np.array(gesture_ids)
    print(f"{len(images)} images, {len(person_list)} people, gestures: {gesture_list}")

    true_gestures_list = []
    pred_gestures_list = []
    true_sides_list = []
    pred_sides_list = []
    persons_list = []
    for test_person in person_list:
        train_mask = person_names != test_person
        test_mask = person_names == test_person

        train_images, train_gestures, train_sides = add_both_hands(images[train_mask], gesture_ids[train_mask])
        test_images, test_gestures, test_sides = add_both_hands(images[test_mask], gesture_ids[test_mask])

        model = train_model(train_images, train_gestures, train_sides, len(gesture_list))
        pred_gestures, pred_sides = predict_images(model, test_images)

        true_gestures_list.append(test_gestures)
        pred_gestures_list.append(pred_gestures)
        true_sides_list.append(test_sides)
        pred_sides_list.append(pred_sides)
        persons_list.append(np.full(len(test_gestures), test_person))
        print(f"Test person {test_person}: gesture accuracy {accuracy_score(test_gestures, pred_gestures) * 100:.1f}%")

    true_gestures = np.concatenate(true_gestures_list)
    pred_gestures = np.concatenate(pred_gestures_list)
    true_sides = np.concatenate(true_sides_list)
    pred_sides = np.concatenate(pred_sides_list)
    persons = np.concatenate(persons_list)

    accuracy = accuracy_score(true_gestures, pred_gestures)
    f1 = f1_score(true_gestures, pred_gestures, average="macro")
    side_accuracy = accuracy_score(true_sides, pred_sides)
    accuracy_per_person = {}
    for person in person_list:
        person_mask = persons == person
        accuracy_per_person[person] = accuracy_score(true_gestures[person_mask], pred_gestures[person_mask])
    person_accuracies = np.array(list(accuracy_per_person.values()))

    print(f"CNN | Accuracy: {accuracy * 100:.2f}%  F1 macro: {f1 * 100:.2f}%  Side accuracy: {side_accuracy * 100:.2f}%  "
          f"people: mean {person_accuracies.mean() * 100:.2f}% / std {person_accuracies.std() * 100:.2f}% / "
          f"min {person_accuracies.min() * 100:.2f}%")

    all_images, all_gestures, all_sides = add_both_hands(images, gesture_ids)
    final_model = train_model(all_images, all_gestures, all_sides, len(gesture_list))
    latency_ms = measure_latency(final_model, images[0])

    RESULTS_DIR.mkdir(exist_ok=True)
    final_model.save(RESULTS_DIR / f"{name}_CNN_model.keras")
    with open(RESULTS_DIR / f"{name}_labels.json", "w") as file:
        json.dump(gesture_list, file)

    row = {"model": "CNN", "latency": latency_ms, "accuracy": accuracy, "f1": f1,
           "person_mean": person_accuracies.mean(), "person_std": person_accuracies.std(),
           "person_min": person_accuracies.min(), "side_acc": side_accuracy}
    for person, person_accuracy in accuracy_per_person.items():
        row[f"acc_{person}"] = person_accuracy
    pd.DataFrame([row]).to_csv(RESULTS_DIR / f"{name}_results_lopo.csv", index=False)

    confusion = confusion_matrix(true_gestures, pred_gestures)
    plot_confusion_matrix(confusion, gesture_list, RESULTS_DIR / f"{name}_confusion.png")


if __name__ == "__main__":
    run()