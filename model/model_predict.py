import tensorflow as tf
from keras import models

from preprocess import *
from pipeline import *


def predict_image(model, image_path):
    img = load_image(image_path)
    img = tf.expand_dims(img, 0)

    probs = model.predict(img)[0]
    for i, prob in enumerate(probs):
        print(f"{multi_hot_index_to_category_id[i]}, {multi_hot_index_to_category_name[i]}: {prob:.3f}")


# todo only for testing
if __name__ == "__main__":
    model = models.load_model("output/fine_tuned.keras")

    image_path, multi_hot_encoding = val_image_infos[1]
    print([multi_hot_index_to_category_id[i] for i, val in enumerate(multi_hot_encoding) if val == 1])
    predict_image(model, image_path)
