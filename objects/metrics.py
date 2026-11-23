from collections import defaultdict

import numpy as np

from objects.config import SEED


def bootstrap_interval(correct, groups, repeats=1000):
    #95% przedzial dla accuracy ,a Test ma ok. 100 zdjec, wiec 2-3 pomylki to juz kilka punktow procentowych roznicy
    #Losujemy cale zdjecia, a nie pojedyncze wycinki, bo wycinki z jednego zdjecia nie sa od siebie niezalezne
    hits = defaultdict(int)
    counts = defaultdict(int)
    for ok, group in zip(correct, groups):
        hits[group] += int(ok)
        counts[group] += 1
    names = list(counts)
    hits = np.array([hits[name] for name in names])
    counts = np.array([counts[name] for name in names])

    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(repeats):
        picked = rng.integers(0, len(names), len(names))
        scores.append(hits[picked].sum() / counts[picked].sum())
    low, high = np.percentile(scores, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def false_alarm_stats(found_per_image):
    #zdjecia bez zadnego banknotu: kazda ramka to falszywy alarm, a w kalkulatorze dodatkowe zlotowki
    if not found_per_image:
        return {"no_banknote_images": 0, "false_alarm_rate": "", "false_alarms_per_image": ""}
    return {
        "no_banknote_images": len(found_per_image),
        "false_alarm_rate": round(sum(1 for found in found_per_image if found) / len(found_per_image), 4),
        "false_alarms_per_image": round(sum(len(found) for found in found_per_image) / len(found_per_image), 3),
    }


def iou(a, b):
    #ramki (x1, y1, x2, y2) w pikselach
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    common = width * height
    return common / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - common)


def match_any_class(predictions, truths, threshold=0.5):
    #Para (prawdziwa klasa, przewidziana klasa) dla kazdej ramki, "tlo" gdy nie ma pary,
    #Ramke dopasowujemy po samym polozeniu, a klase sprawdzamy dopiero potem.
    #Dzieki temu widac, czy model zgubil banknot, czy znalazl go, ale pomylil nominal.

    pairs = []
    used = set()
    for prediction in sorted(predictions, key=lambda p: -p["score"]):
        best, best_iou = None, threshold
        for j, truth in enumerate(truths):
            overlap = iou(prediction["box"], truth["box"])
            if j not in used and overlap >= best_iou:
                best, best_iou = j, overlap
        if best is None:
            pairs.append(("tlo", prediction["label"]))
        else:
            used.add(best)
            pairs.append((truths[best]["label"], prediction["label"]))
    pairs += [(truth["label"], "tlo") for j, truth in enumerate(truths) if j not in used]
    return pairs


def mean_average_precision(all_predictions, all_truths, class_names):
    #mAP@0.5 i mAP@0.5:0.95 policzone tak samo dla kazdego potoku (interpolacja 101 punktow jak w COCO)
    thresholds = np.arange(0.5, 0.96, 0.05)
    per_class = []
    for name in class_names:
        positives = sum(1 for truths in all_truths for truth in truths if truth["label"] == name)
        if positives == 0:
            continue
        per_class.append([average_precision(*class_hits(all_predictions, all_truths, name, t), positives)
                          for t in thresholds])
    if not per_class:
        return 0.0, 0.0
    per_class = np.array(per_class)
    return float(per_class[:, 0].mean()), float(per_class.mean())


def class_hits(all_predictions, all_truths, name, threshold):
    scores, hits = [], []
    for predictions, truths in zip(all_predictions, all_truths):
        truths = [truth for truth in truths if truth["label"] == name]
        used = set()
        for prediction in sorted((p for p in predictions if p["label"] == name), key=lambda p: -p["score"]):
            best, best_iou = None, threshold
            for j, truth in enumerate(truths):
                overlap = iou(prediction["box"], truth["box"])
                if j not in used and overlap >= best_iou:
                    best, best_iou = j, overlap
            scores.append(prediction["score"])
            hits.append(best is not None)
            if best is not None:
                used.add(best)
    return scores, hits


def average_precision(scores, hits, positives):
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    hits = np.asarray(hits, dtype=float)[order]
    true_positives = np.cumsum(hits)
    recall = true_positives / positives
    precision = true_positives / np.arange(1, len(hits) + 1)
    #dla kazdego recall bierzemy najlepsza precision, jaka da sie osiagnac przy tym albo wiekszym recall
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in points]))
