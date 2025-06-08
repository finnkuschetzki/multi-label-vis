import os
import re
import sys
import shutil
from keras.applications import EfficientNetV2B0
from keras import layers, models, optimizers, metrics

from preprocess import *
from pipeline import *
from custom_callbacks import *


TRAIN_HEAD_EPOCHS = 5
MAX_FINETUNE_EPOCHS = 5  # todo small value for testing


# --- creating the model ---

def create_model():

    base_model = EfficientNetV2B0(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet",
        pooling="avg"
    )

    outputs_layer = layers.Dense(num_train_classes, activation="sigmoid")(base_model.output)

    model = models.Model(inputs=base_model.input, outputs=outputs_layer)

    return base_model, model


# --- train head only ---

def train_head_only(base_model, model):

    base_model.trainable = False

    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ]
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=TRAIN_HEAD_EPOCHS,
        callbacks=callbacks_head_only,
    )

    base_model.trainable = True

    model.save("output/head_only.keras")


# --- fine tuning ---

def fine_tune(model, initial_epoch=0):

    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ]
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_FINETUNE_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=[callbacks_fine_tune],
    )

    model.save("output/fine_tuned.keras")


# --- managing execution ---

if __name__ == "__main__":

    if len(sys.argv) < 2:

        # no command line arguments
        # start new training if override is confirmed

        if os.path.isdir("output"):
            print()
            input_ = input("Are you sure you want to override the output directory? (y/n)\n")
            if input_.lower() != "y":
                print("Aborting.", file=sys.stderr)
                sys.exit(1)

        # cleaning output directory
        if os.path.exists("output") and os.path.isdir("output"):
            shutil.rmtree("output")

        # training
        base_model, model = create_model()

        print()
        print("========== training head only ==========")
        train_head_only(base_model, model)

        print()
        print("========== fine-tuning ==========")
        fine_tune(model)

    elif sys.argv[1].lower() == "resume":

        # command line argument resume
        # resumes fine-tuning if previous fine-tuned checkpoints exist

        if not os.path.isdir("output/checkpoints/fine_tune"):
            print("There must be at least one checkpoint saved from fine-tuning to resume training.", file=sys.stderr)
            sys.exit(1)

        max_epochs, model_path = None, None

        # find model fine-tuned checkpoint with highest epochs
        pattern = re.compile(r'^(\d+)-(\d+\.\d+)\.keras$')
        for filename in os.listdir("output/checkpoints/fine_tune"):
            if match := pattern.match(filename):
                max_epochs = int(match.group(1))
                model_path = f"output/checkpoints/fine_tune/{filename}"

        if max_epochs is None or model_path is None:
            print("There must be at least one checkpoint saved from fine-tuning to resume training.", file=sys.stderr)
            sys.exit(1)

        # loading model
        model = models.load_model(model_path)

        # training
        print()
        print("========== fine-tuning ==========")
        fine_tune(model, max_epochs)

    else:

        # invalid command line arguments
        # exiting

        print("Incorrect arguments provided.", file=sys.stderr)
        sys.exit(2)
