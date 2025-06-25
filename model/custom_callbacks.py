from keras import callbacks
import time
import sys


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

# not used during head only training
lr_scheduler_fine_tune = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

early_stopping_head_only = callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, min_delta=0.001, mode="min", verbose=1)

early_stopping_fine_tune = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, min_delta=0.001, mode="min", verbose=1)


# --- callback lists ---

callbacks_head_only = [
    epoch_timer,
    model_checkpoint_head_only,
    tensor_board_head_only,
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
