"""Tworzy datasets/odpornosc/<warunek>/ - zbior testowy pod roznymi zmianami warunkow."""

from objects.robustness.dataset import RobustnessDataset

if __name__ == "__main__":
    RobustnessDataset().prepare()
