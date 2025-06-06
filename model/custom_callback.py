from keras import callbacks
import time


class EpochTimer(callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        total_time = time.time() - self.start
        print(f"Epoch {epoch + 1}: took {total_time:.2f} seconds")


__all__ = [
    "EpochTimer",
]
