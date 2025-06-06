import tensorflow as tf
from keras.applications import EfficientNetV2B0
from keras import layers, models, optimizers, callbacks

from preprocess import *
from pipeline import *
from custom_callback import *


EPOCHS = 1   # todo small value for testing


# creating the model
base_model = EfficientNetV2B0(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights="imagenet",
    pooling="avg"
)

outputs_layer = layers.Dense(num_train_classes, activation="sigmoid")(base_model.output)

model = models.Model(inputs=base_model.input, outputs=outputs_layer)

# compiling the model
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# training the model
checkpoint_filepath = "output/checkpoints/{epoch:02d}-{val_loss:.2f}.keras"
model_checkpoint = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    save_best_only=True,
    mode="auto",
    verbose=1
)

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        model_checkpoint,
        callbacks.TensorBoard(log_dir="output/logs"),
        EpochTimer()
    ]
)


def predict_image(image_path):
    img, _ = load_image(image_path, None)  # todo make this better
    img = tf.expand_dims(img, 0)

    probs = model.predict(img)[0]
    for i, prob in enumerate(probs):
        print(f"{multi_hot_index_to_category_id[i]}, {multi_hot_index_to_category_name[i]}: {prob:.3f}")


# todo only for testing
if __name__ == "__main__":
    image_path, multi_hot_encoding = val_image_infos[1]
    print([multi_hot_index_to_category_id[i] for i, val in enumerate(multi_hot_encoding) if val == 1])
    predict_image(image_path)
