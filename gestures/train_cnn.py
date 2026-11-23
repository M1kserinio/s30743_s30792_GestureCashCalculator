import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch import nn
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "gesture_images_cropped"
RESULTS_DIR = ROOT / "gest_results"

SEED = 42
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMAGE_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Losowe zmiany zdjęć podczas treningu
augment = v2.Compose([
    v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),  # obrót, przesunięcie, zoom
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # jasność, kontrast, kolory
])

class HandCNN(nn.Module):

    def __init__(self, number_of_gestures):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, number_of_gestures),
        )

    def forward(self, images):
        images = (images.float() / 255 - 0.5) / 0.5  # piksele z 0...255 na -1...1
        return self.layers(images)


def prepare_image(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # OpenCV daje BGR, zamieniamy na RGB
    small_image = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  # zmniejszamy
    return torch.from_numpy(small_image).permute(2, 0, 1).contiguous()  # (64, 64, 3) -> (3, 64, 64)


def load_images():
    images = []
    gesture_names = []
    person_names = []
    for file in sorted(IMAGES_DIR.glob("P*/*/*.jpg")):
        photo = cv2.imread(str(file))  # wczytaj zdjęcie
        images.append(prepare_image(photo))  # zmniejsz i dodaj do listy
        gesture_names.append(file.parent.name)  # nazwa folderu gestu
        person_names.append(file.parent.parent.name)  # nazwa folderu osoby
    return torch.stack(images), gesture_names, person_names


def add_mirrored_copies(images, gesture_ids):
    mirrored_images = torch.flip(images, dims=[3])  # odbicie
    all_images = torch.cat([images, mirrored_images])  # oryginały + odbicia
    all_gesture_ids = torch.cat([gesture_ids, gesture_ids])  # id gestu się nie zmienia po odbiciu
    return all_images, all_gesture_ids


def train_model(images, gesture_ids, number_of_gestures):
    model = HandCNN(number_of_gestures).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()  # błąd dla klasyfikacji

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(images)).tolist()  # zdjęcia w losowej kolejności
        for start in range(0, len(order), BATCH_SIZE):
            batch_ids = order[start:start + BATCH_SIZE]  # numery zdjęć w tej paczce
            if len(batch_ids) < 2:
                continue
            batch_images = torch.stack([augment(images[i]) for i in batch_ids]).to(DEVICE)  # losowo zmienione zdjęcia
            batch_gestures = gesture_ids[batch_ids].to(DEVICE)  # poprawne odpowiedzi

            scores = model(batch_images)  # sieć zgaduje
            loss = loss_function(scores, batch_gestures)  # jak bardzo się myli
            optimizer.zero_grad()  # czyścimy poprzednie poprawki
            loss.backward()  # liczymy, co poprawić
            optimizer.step()  # poprawiamy sieć
    return model


def predict_images(model, images):
    model.eval()
    predicted_ids = []
    with torch.no_grad():  # bez liczenia poprawek
        for start in range(0, len(images), 256):
            batch_images = images[start:start + 256].to(DEVICE)
            scores = model(batch_images)
            predicted_ids.append(scores.argmax(dim=1).cpu())  # gest z najwyższym wynikiem
    return torch.cat(predicted_ids).numpy()


def measure_latency(model, image):
    model = model.cpu().eval()
    one_image = image.unsqueeze(0)
    times_ms = []
    with torch.no_grad():
        for _ in range(20):  # rozgrzewka
            model(one_image)
        for _ in range(300):  # 300 pomiarów
            start = time.perf_counter()
            model(one_image)
            times_ms.append((time.perf_counter() - start) * 1000)
    return np.median(times_ms)


def plot_confusion_matrix(confusion, gesture_list, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(gesture_list)), gesture_list, rotation=45)
    ax.set_yticks(range(len(gesture_list)), gesture_list)
    for row in range(len(gesture_list)):
        for column in range(len(gesture_list)):
            count = confusion[row, column]
            text_color = "white" if count > confusion.max() / 2 else "black"
            ax.text(column, row, count, ha="center", va="center", color=text_color)
    ax.set_title("CNN confusion matrix (leave-one-person-out)")
    ax.set_xlabel("Predicted gesture")
    ax.set_ylabel("True gesture")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)

def load_cnn(path):
    saved = torch.load(path, map_location="cpu")
    model = HandCNN(len(saved["gestures"]))
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved["gestures"]


def classify_crop(model, gesture_list, crop_bgr):
    image = prepare_image(crop_bgr).unsqueeze(0)
    with torch.no_grad():
        scores = model(image)
    probabilities = torch.softmax(scores, dim=1)[0]  # wyniki zamienione na prawdopodobieństwa
    best_id = int(probabilities.argmax())
    return gesture_list[best_id], float(probabilities[best_id])


def run(name="D"):
    torch.manual_seed(SEED)

    # Wczytanie zdjęć
    images, gesture_names, person_names = load_images()
    gesture_list = sorted(set(gesture_names))  # unikalne gesty
    person_list = sorted(set(person_names))  # unikalne osoby
    gesture_ids = []  # zamieniamy nazwy gestów na numery
    for gesture in gesture_names:
        gesture_ids.append(gesture_list.index(gesture))
    gesture_ids = torch.tensor(gesture_ids)
    print(f"{len(images)} images, {len(person_list)} people, gestures: {gesture_list}, device: {DEVICE}")

    # lopo
    true_gestures_list = []
    pred_gestures_list = []
    persons_list = []
    for test_person in person_list:
        train_ids = []  # numery zdjęć pozostałych osób
        test_ids = []  # numery zdjęć osoby testowej
        for index, person in enumerate(person_names):
            if person == test_person:
                test_ids.append(index)
            else:
                train_ids.append(index)

        # do treningu i testu dodajemy odbicia, więc sieć widzi obie dłonie
        train_images, train_gestures = add_mirrored_copies(images[train_ids], gesture_ids[train_ids])
        test_images, test_gestures = add_mirrored_copies(images[test_ids], gesture_ids[test_ids])

        model = train_model(train_images, train_gestures, len(gesture_list))
        pred_gestures = predict_images(model, test_images)

        # zapamiętujemy wyniki tej osoby
        true_gestures_list.append(test_gestures.numpy())
        pred_gestures_list.append(pred_gestures)
        persons_list.append(np.full(len(test_gestures), test_person))
        print(f"Test person {test_person}: gesture accuracy {accuracy_score(test_gestures.numpy(), pred_gestures) * 100:.1f}%")

    # łączymy wyniki wszystkich osób w jedną tablicę
    true_gestures = np.concatenate(true_gestures_list)
    pred_gestures = np.concatenate(pred_gestures_list)
    persons = np.concatenate(persons_list)

    # wyniki
    accuracy = accuracy_score(true_gestures, pred_gestures)  # ile gestów dobrze
    f1 = f1_score(true_gestures, pred_gestures, average="macro")  # F1
    accuracy_per_person = {}
    for person in person_list:
        person_mask = persons == person
        accuracy_per_person[person] = accuracy_score(true_gestures[person_mask], pred_gestures[person_mask])
    person_accuracies = np.array(list(accuracy_per_person.values()))

    print(f"CNN | Accuracy: {accuracy * 100:.2f}%  F1 macro: {f1 * 100:.2f}%  "
          f"people: mean {person_accuracies.mean() * 100:.2f}% / std {person_accuracies.std() * 100:.2f}% / "
          f"min {person_accuracies.min() * 100:.2f}%")

    all_images, all_gestures = add_mirrored_copies(images, gesture_ids)
    final_model = train_model(all_images, all_gestures, len(gesture_list))
    latency_ms = measure_latency(final_model, images[0])

    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": final_model.state_dict(), "gestures": gesture_list},
               RESULTS_DIR / f"{name}_CNN_model.pt")

    # zapis
    row = {"model": "CNN", "latency": latency_ms, "accuracy": accuracy, "f1": f1,
           "person_mean": person_accuracies.mean(), "person_std": person_accuracies.std(),
           "person_min": person_accuracies.min()}
    for person, person_accuracy in accuracy_per_person.items():
        row[f"acc_{person}"] = person_accuracy
    pd.DataFrame([row]).to_csv(RESULTS_DIR / f"{name}_results_lopo.csv", index=False)

    confusion = confusion_matrix(true_gestures, pred_gestures)
    plot_confusion_matrix(confusion, gesture_list, RESULTS_DIR / f"{name}_confusion.png")


if __name__ == "__main__":
    run()