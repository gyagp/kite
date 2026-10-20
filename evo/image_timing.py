"""Sequential image-phase timers, including completion of asynchronous CUDA work."""

import time


class ImageTimer:
    def __init__(self, metrics=None):
        self.metrics = metrics
        self.started = time.perf_counter()

    def mark(self, key, *, cuda=False):
        if self.metrics is None:
            return
        if cuda:
            import torch

            torch.cuda.synchronize()
        finished = time.perf_counter()
        self.metrics[key] = self.metrics.get(key, 0.0) + (finished - self.started) * 1000
        self.started = finished
