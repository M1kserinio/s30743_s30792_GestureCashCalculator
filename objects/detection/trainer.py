import shutil
from pathlib import Path

from ultralytics import RTDETR, YOLO

from objects.config import (BATCH, DATA_YAML, EPOCHS, IMGSZ, LOCATOR_BASE, LOCATOR_MODEL, MODELS_DIR, RTDETR_BATCH,
                            RTDETR_MODELS, RUNS_DIR, SEED, YOLO_MODELS, best_device, model_path)


class DetectorTrainer:
    """Wspolny trening YOLO i RT-DETR: ten sam zbior, te same epoki, ten sam seed."""

    def __init__(self):
        self.device = best_device()

    def train_yolo(self):
        for model_name in YOLO_MODELS:
            self.train(model_name, model_path(model_name))

    def train_rtdetr(self):
        for model_name in RTDETR_MODELS:
            self.train(model_name, model_path(model_name), batch=RTDETR_BATCH)

    def train_locator(self):
        #ten sam YOLO, ale wszystkie nominaly to jedna klasa "banknot". Mowi gdzie, a nie jaki -
        #nominal podaje potem klasyfikator (potok "znajdz banknot + klasyfikator")
        self.train(LOCATOR_BASE, LOCATOR_MODEL, single_cls=True)

    def train(self, model_name, save_as, batch=BATCH, **extra):
        if not DATA_YAML.exists():
            raise SystemExit("Brak data.yaml - najpierw uruchom python -m scripts.prepare_dataset")

        MODELS_DIR.mkdir(exist_ok=True)
        print(f"\n=== Trening {model_name} -> {save_as.name} na {self.device} ===")
        #model juz wytrenowany na zbiorze COCO, douczamy go tylko na pieniadzach (transfer learning)
        model_class = RTDETR if model_name.startswith("rtdetr") else YOLO
        model = model_class(model_name + ".pt")
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=batch,
            device=self.device,
            seed=SEED,
            deterministic=True,
            patience=30,
            fliplr=0.0,  #bez odbic: banknot w lustrze nie istnieje, a odbicie to tez przeciek miedzy zbiorami
            project=str(RUNS_DIR),
            name=save_as.stem,
            exist_ok=True,
            **extra,
        )
        #najlepsza epoka wg val jest w best.pt, kopiujemy ja do models/
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        shutil.copy(best, save_as)
        print(f"Zapisano {save_as}")
