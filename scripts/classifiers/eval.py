"""Porownanie HOG+SVM, prostej CNN i ResNet-18 w roznych warunkach."""

from objects.classifiers.evaluator import ClassificationEvaluator


if __name__ == "__main__":
    ClassificationEvaluator().evaluate_all()
