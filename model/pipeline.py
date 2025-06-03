import tensorflow as tf
from preprocess import coco_train


IMG_SIZE = 224


def get_image_labels(image_id):
    labels = set()
    anns = coco_train.loadAnns(coco_train.getAnnIds(imgIds=image_id))
    for ann in anns:
        labels.add(ann["category_id"])
    return labels


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


# configure export
__all__ = [
    "IMG_SIZE",
    "get_image_labels",
    "load_image"
]
