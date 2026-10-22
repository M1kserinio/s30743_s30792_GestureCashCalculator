"""Wytnij banknoty z ramek YOLO do zbioru dla klasyfikatorow."""

from objects.classifiers.dataset import ClassificationDataset


if __name__ == "__main__":
    ClassificationDataset().prepare()
