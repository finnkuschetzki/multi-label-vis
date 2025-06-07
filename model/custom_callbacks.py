import sys

from keras import callbacks
import time


class EpochTimer(callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        total_time = time.time() - self.start
        sys.stdout.write("\n")
        sys.stdout.write(f"Epoch {epoch + 1}: took {total_time:.2f} seconds")
        sys.stdout.flush()
        # print(f"Epoch {epoch + 1}: took {total_time:.2f} seconds")


epoch_timer = EpochTimer()

checkpoint_filepath = "output/checkpoints/{epoch:02d}-{val_loss:.2f}.keras"
model_checkpoint = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    save_best_only=True,
    mode="auto",
    verbose=1
)

tensor_board = callbacks.TensorBoard(log_dir="output/logs")

callbacks = [
    epoch_timer,
    model_checkpoint,
    tensor_board,
]


__all__ = [
    "callbacks",
]
