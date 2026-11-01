#Tabela klasyfikatorow: HOG+SVM, kolory+SVM, prosta CNN, MobileNetV3 i ResNet-18 na tych samych wycinkach

from objects.classifiers.evaluator import ClassificationEvaluator


if __name__ == "__main__":
    ClassificationEvaluator().evaluate_all()
