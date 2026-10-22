import time

from ultralytics import YOLO

from objects.config import IMGSZ


class FpsMeter:
    def __init__(self, images, frames=100, warmup=10):
        self.images = images
        self.frames = frames  #na ilu klatkach mierzymy
        self.warmup = warmup

    def measure(self, weights, device):
        #nowy model dla kazdego urzadzenia, bo YOLO zapamietuje urzadzenie z pierwszego predict
        model = YOLO(str(weights))

        #pierwsze predykcje sa duzo wolniejsze , nie licze ich
        for i in range(self.warmup):
            self.predict(model, i, device)

        start = time.perf_counter()
        for i in range(self.frames):
            self.predict(model, i, device)
        seconds = time.perf_counter() - start

        ms_per_frame = seconds / self.frames * 1000
        fps = self.frames / seconds
        return ms_per_frame, fps

    def predict(self, model, i, device):
        #po jednym obrazku, tak jak w apce (klatka po klatce)
        image = self.images[i % len(self.images)]
        model.predict(image, imgsz=IMGSZ, device=device, verbose=False)
