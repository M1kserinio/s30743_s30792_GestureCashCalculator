from objects.yolo_trainer import YoloTrainer

#ustawienia treningu (modele, epoki, batch) sa w objects/config.py
if __name__ == "__main__":
    YoloTrainer().train_all()
