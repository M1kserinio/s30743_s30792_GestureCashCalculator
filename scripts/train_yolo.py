"""Train the configured YOLO model variants."""

from objects.yolo_trainer import YoloTrainer

if __name__ == "__main__":
    YoloTrainer().train_all()
