import sys

from objects.roboflow_dataset import RoboflowDataset

#zip z Roboflow (format YOLOv8), mozna tez podac przy odpalaniu: python prepare_dataset.py ~/Downloads/plik.zip
ROBOFLOW_ZIP = "roboflow.zip"

if __name__ == "__main__":
    zip_path = sys.argv[1] if len(sys.argv) > 1 else ROBOFLOW_ZIP
    RoboflowDataset(zip_path).prepare()
