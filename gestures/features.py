import numpy as np

FINGER_CHAINS = (
    (0, 1, 2, 3, 4),      # thumb
    (0, 5, 6, 7, 8),      # index
    (0, 9, 10, 11, 12),   # middle
    (0, 13, 14, 15, 16),  # ring
    (0, 17, 18, 19, 20),  # pinky
)
TIPS = (4, 8, 12, 16, 20)
MIDDLE_MCP = 9
ANTIZERO = 1e-9


def _angle(u, v):
    """Kat miedzy wektorami"""
    dot = np.sum(u * v, axis=-1) # mnozymy czyli licznik
    n = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) #mianownik czyli tez mnozymy ale dlugosci
    #arccos daje kat w radianach jak jest 0 to prosty im wieksze zgiecei tym wieksa liczna
    return np.arccos(np.clip(dot / (n + ANTIZERO), -1.0, 1.0)) #clip i antizero bo dziwnie sie dzieli i wyrzuca blad, w skrocie clip uzuwa .0001 z 1.0001 i dobrze dzieliw tedy


def extract_features(X):
    """63 punkty xyz w 30 sech zamienione, zeby nie bazowac sie pozycjami, tylko cechami dloni
    Cechy: 15 katow | 10 odleglosci tipow| 5 odleglosci tip-nadgarstek."""
    X = np.asarray(X, dtype=float) # punkty na tablice
    lm = X.reshape(len(X), 21, 3) #w skrocie, wiersz (ktora fota), ilosc punktow z mp i xyz

    # 15 katow zgiecia na palcach, 3 na palec
    angles = []
    for chain in FINGER_CHAINS: #dla kazdego palucha
        for i in range(3): #dla kazdego z 3 stawow
            u = lm[:, chain[i+1]] - lm[:, chain[i]] # wektor miedzy dwoma punktami to ich roznica (pierwsza kosc)
            v = lm[:, chain[i+2]] - lm[:, chain[i+1]] # drugi wektor (kosc)
            angles.append(_angle(u, v)) #kat miedzy nimi

    # skala lapy
    palm = np.linalg.norm(lm[:, MIDDLE_MCP] - lm[:, 0], axis=1) + ANTIZERO

    # 10 odleglosci miedzy tipami
    tip_pairs = []
    for i in range(5):# dla kazdego palca
        for j in range(i + 1, 5): # do pozostalcyh palcow po prawej
            length = np.linalg.norm(lm[:, TIPS[i]] - lm[:, TIPS[j]], axis=1) #dllugoisc
            tip_pairs.append(length / palm) #dodaj, ale znormalizowane o palm

    # 5 odleglosci koniuszek od nadgarstka
    tip_wrist = [np.linalg.norm(lm[:, t] - lm[:, 0], axis=1) / palm for t in TIPS]

    return np.column_stack(angles + tip_pairs + tip_wrist) # dodaj jako trzy kolumny na probke
