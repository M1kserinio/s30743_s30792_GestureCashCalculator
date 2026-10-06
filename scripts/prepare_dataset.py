"""Prepare a YOLO dataset from a Roboflow export."""

import sys

from objects.roboflow_dataset import RoboflowDataset

ROBOFLOW_ZIP = "roboflow.zip"

if __name__ == "__main__":
    zip_path = sys.argv[1] if len(sys.argv) > 1 else ROBOFLOW_ZIP
    RoboflowDataset(zip_path).prepare()
