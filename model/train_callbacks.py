import sys
import time
from keras import callbacks

from _config_loader import config


class EpochTimer(callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        total_time = time.time() - self.start
        sys.stdout.write("\n")
        sys.stdout.write(f"Epoch {epoch + 1}: took {total_time:.2f} seconds")
        sys.stdout.flush()


# --- callback instances ---

epoch_timer = EpochTimer()

checkpoint_filepath = "output/checkpoints/head_only/{epoch:02d}-{val_loss:.2f}.keras"
model_checkpoint_head_only = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    save_best_only=True,
    mode="auto",
    verbose=1
)

checkpoint_filepath = "output/checkpoints/fine_tune/{epoch:02d}-{val_loss:.2f}.keras"
model_checkpoint_fine_tune = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    save_best_only=True,
    mode="auto",
    verbose=1
)

tensor_board_head_only = callbacks.TensorBoard(log_dir="output/logs/head_only", profile_batch=0)

tensor_board_fine_tune = callbacks.TensorBoard(log_dir="output/logs/fine_tune", profile_batch=0)

lr_scheduler_head_only = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=config["lr_scheduler"]["head_only"]["factor"],
    patience=config["lr_scheduler"]["head_only"]["patience"],
    min_delta=config["lr_scheduler"]["head_only"]["min_delta"],
    min_lr=config["lr_scheduler"]["head_only"]["min_lr"],
    verbose=1
)

lr_scheduler_fine_tune = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=config["lr_scheduler"]["fine_tune"]["factor"],
    patience=config["lr_scheduler"]["fine_tune"]["patience"],
    min_delta=config["lr_scheduler"]["fine_tune"]["min_delta"],
    min_lr=config["lr_scheduler"]["fine_tune"]["min_lr"],
    verbose=1,
)

early_stopping_head_only = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=config["early_stopping"]["head_only"]["patience"],
    restore_best_weights=True,
    min_delta=config["early_stopping"]["head_only"]["min_delta"],
    mode="min",
    verbose=1
)

early_stopping_fine_tune = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=config["early_stopping"]["fine_tune"]["patience"],
    restore_best_weights=True,
    min_delta=config["early_stopping"]["fine_tune"]["min_delta"],
    mode="min",
    verbose=1
)


# --- callback lists ---

callbacks_head_only = [
    epoch_timer,
    model_checkpoint_head_only,
    tensor_board_head_only,
    lr_scheduler_head_only,
    early_stopping_head_only,
]

callbacks_fine_tune = [
    epoch_timer,
    model_checkpoint_fine_tune,
    tensor_board_fine_tune,
    lr_scheduler_fine_tune,
    early_stopping_fine_tune,
]


__all__ = [
    "callbacks_head_only",
    "callbacks_fine_tune",
]
