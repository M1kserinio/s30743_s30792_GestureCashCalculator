"""Evaluate trained YOLO models and select the best checkpoint."""

from objects.yolo_evaluator import YoloEvaluator

if __name__ == "__main__":
    YoloEvaluator().evaluate_all()
