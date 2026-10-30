import time


class FpsMeter:
    def __init__(self, images, frames=100, warmup=10):
        self.images = images
        self.frames = frames  #na ilu klatkach mierzymy
        self.warmup = warmup

    def measure(self, run_frame):
        #run_frame = cokolwiek, co dostaje jedna klatke: sam detektor albo caly potok z klasyfikatorem
        #pierwsze klatki sa duzo wolniejsze (model sie rozgrzewa), do wyniku ich nie liczymy
        for i in range(self.warmup):
            run_frame(self.images[i % len(self.images)])

        #po jednym obrazku, tak jak w apce (klatka po klatce)
        start = time.perf_counter()
        for i in range(self.frames):
            run_frame(self.images[i % len(self.images)])
        seconds = time.perf_counter() - start

        ms_per_frame = seconds / self.frames * 1000
        fps = self.frames / seconds
        return ms_per_frame, fps
