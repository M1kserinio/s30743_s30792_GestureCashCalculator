from objects.classifiers.config import HOG_MODEL
from objects.classifiers.evaluator import ClassificationEvaluator


class HogEvaluator:
    def evaluate(self):
        ClassificationEvaluator().evaluate_models("hog", [("HOG+SVM", HOG_MODEL)])
