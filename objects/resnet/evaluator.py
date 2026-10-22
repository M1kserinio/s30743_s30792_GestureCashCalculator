from objects.classifiers.config import RESNET_MODEL
from objects.classifiers.evaluator import ClassificationEvaluator


class ResNetEvaluator:
    def evaluate(self):
        ClassificationEvaluator().evaluate_models("resnet", [("ResNet-18", RESNET_MODEL)])
