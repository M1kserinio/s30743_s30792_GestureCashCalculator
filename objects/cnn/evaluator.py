from objects.classifiers.config import CNN_MODEL
from objects.classifiers.evaluator import ClassificationEvaluator


class CnnEvaluator:
    def evaluate(self):
        ClassificationEvaluator().evaluate_models("cnn", [("Prosta CNN", CNN_MODEL)])
