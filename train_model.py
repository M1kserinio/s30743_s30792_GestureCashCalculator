import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

csv = pd.read_csv("gesture_data/gestures.csv", header=None)
X = csv.iloc[:, 1:]  # Landmarks
y = csv.iloc[:, 0]   # Etykiety

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = SVC(kernel='poly', degree=3, probability=True)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Dokładność: {accuracy * 100:.2f}%")

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)