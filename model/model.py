import tensorflow as tf
from keras.applications import EfficientNetV2B0
from keras import layers, models, optimizers
from preprocess import *
from pipeline import *


# creating the model
base_model = EfficientNetV2B0(
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    weights="imagenet",
    pooling="avg"
)

outputs_layer = layers.Dense(num_classes, activation="sigmoid")(base_model.output)

model = models.Model(inputs=base_model.input, outputs=outputs_layer)

# compiling the model
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


def predict_image(image_path):
    img = load_image(image_path)
    img = tf.expand_dims(img, 0)

    probs = model.predict(img)[0]
    for i, prob in enumerate(probs):
        print(f"{multi_hot_index_to_category_id[i]}, {multi_hot_index_to_category_name[i]}: {prob:.3f}")


# todo only for testing
if __name__ == "__main__":
    image_id = 100000
    print({ann["category_id"] for ann in coco_train.loadAnns(coco_train.getAnnIds(imgIds=[image_id]))})
    predict_image(TRAIN_IMG_DIR + f"000000{image_id}.jpg")
