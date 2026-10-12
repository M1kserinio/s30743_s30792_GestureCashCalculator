import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
import pickle

from gestures.features import extract_features

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SEED = 42  # seed dla modelu
RESULTS_DIR = os.path.join(ROOT / "gest_results")

def run(name, use_features, eval="lopo"):
    csv = pd.read_csv(ROOT / "gesture_data" / "gestures.csv", header=None)
    persons = csv.iloc[:, 0].astype(str).to_numpy() # osoby - to.numpy() zmienia na tablice
    y = csv.iloc[:, 1].to_numpy()  # etykiety
    X = csv.iloc[:, 2:].to_numpy()  # landmarki
    assert X.shape[1] == 63, "Wymagane 63 cechy"  # jesli jakis csv sie nie zgadza to tu sie wywali

    person_names = sorted(np.unique(persons))
    labels = sorted(np.unique(y))  # bez tego czasami sie kolejnosc gestow w tabelkach zmienia, po prostu sortowanie labeli
    groups = np.arange(len(X)) // 2  # csv maja tuz po sobie lustrzane odbicie, a to je grupuje (ten sam gest odbity to ta sama grupa)

    #tylko do ewaluacji wyniku
    if eval == "lopo":
        assert len(person_names) >= 2, "LOPO wymaga 2 osob"
        cv = LeaveOneGroupOut()  # kazda osoba raz jako zbior testowy
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED) # dzieli dane na 5 czesci, powatrza proces treningu i predyckji 5 razy, tak aby kzda probka miala swoja predykcje


    #modele
    modelLR = LogisticRegression(max_iter=1000, random_state=SEED)
    modelKNN = KNeighborsClassifier(n_neighbors=5)
    modelSVC = SVC(kernel='rbf', random_state=SEED)
    modelRF = RandomForestClassifier(n_estimators=300, random_state=SEED)
    modelMLP = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True) #tworzy katalog

    modelnames=("LR", "KNN", "SVC", "RF", "MLP")
    modelindex=0

    results = []

    for model in (modelLR, modelKNN, modelSVC, modelRF, modelMLP):
        if use_features: # albo transformuje punkty na cechy albo nie
            pipeline = Pipeline([("features", FunctionTransformer(extract_features)),('scaler', StandardScaler()), ('model', model)]) # pipeline, ktory skaluje dane i trenuje model, dzieki temu nie trzeba kazdego folda skalowac osobno
        else:
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=cv)  # musi zrobic predict z 5 modelu na podstawie 4 poprzenich foldów

        acc = accuracy_score(y, y_pred)  # accuracy liczone
        f1 = f1_score(y, y_pred, average="macro")  # F1
        cm = confusion_matrix(y, y_pred, labels=labels)  # macierz pomylek

        per_person = {p: accuracy_score(y[persons == p], y_pred[persons == p]) for p in person_names}
        pp = np.array(list(per_person.values()))

        print(f"{modelnames[modelindex]} | Accuracy: {acc * 100:.2f}%   F1 macro: {f1 * 100:.2f}%"
              f"osoby: srednia {pp.mean() * 100:.2f}% / std {pp.std() * 100:.2f}% / min {pp.min() * 100:.2f}%")
        print(pd.DataFrame(cm, index=labels, columns=labels)) #metryki i maciertz

        pipeline.fit(X, y)  # trening

        sample = X[:1] #bierze pierwsza probke do testowania predykcji
        for _ in range(20): #pierwsze 20 predykcji model sie rozgrzewa
            pipeline.predict(sample)
        times=[]
        for _ in range(300):
            start = time.perf_counter() #perf counter jest lepszy niz time
            pipeline.predict(sample)
            times.append((time.perf_counter() - start) * 1000) # w ms
        latency_ms = np.median(times) #mediana z 300 predykcji aby zobaczyc ile trwa predykcja dla modelu

        model_path = os.path.join(RESULTS_DIR, f"{name}_{modelnames[modelindex]}_model.pkl") # zapis modelu
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)

        row = {"model": modelnames[modelindex], "latency": latency_ms, "accuracy": acc, "f1": f1,
               "person_mean": pp.mean(), "person_std": pp.std(), "person_min": pp.min()}
        row.update({f"acc_{p}": a for p, a in per_person.items()})
        results.append(row)
        modelindex += 1

    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, f"{name}_results_{eval}.csv"), index=False) #rezultaty do csv
