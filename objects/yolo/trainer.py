import shutil
from pathlib import Path

from ultralytics import YOLO

from objects.config import (BATCH, DATA_YAML, EPOCHS, IMGSZ, MODELS_DIR, RUNS_DIR, SEED, YOLO_MODELS, best_device,
                            model_path)


class YoloTrainer:
    def __init__(self):
        self.device = best_device()

    def train_all(self):
        if not DATA_YAML.exists():
            print("Brak data.yaml - najpierw uruchom python -m scripts.prepare_dataset")
            return

        MODELS_DIR.mkdir(exist_ok=True)
        for model_name in YOLO_MODELS:
            print(f"\n=== Trening {model_name} na {self.device} ===")
            self.train(model_name)

    def train(self, model_name):
        #model juz wytrenowany na zbiorze COCO, douczamy go tylko na pieniadzach (transfer learning)
        model = YOLO(model_name + ".pt")
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=self.device,
            seed=SEED,
            deterministic=True,
            patience=30,
            fliplr=0.0,
            project=str(RUNS_DIR),
            name=model_name,
            exist_ok=True,
        )
        self.save_best(model, model_name)

    def save_best(self, model, model_name):
        #YOLO zapisuje najlepsza epoke jako best.pt, kopiujemy ja do models/
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        shutil.copy(best, model_path(model_name))
        print(f"Zapisano {model_path(model_name)}")
