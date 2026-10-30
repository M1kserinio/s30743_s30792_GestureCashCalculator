"""Tabela detektorow: YOLO11 n/s/m i RT-DETR na tym samym tescie i sprzecie.

Najlepszy po val trafia do models/pieniadze_best.pt, czyli do kamery.
"""

from objects.detection.evaluator import DetectorEvaluator

if __name__ == "__main__":
    DetectorEvaluator().evaluate_all()
