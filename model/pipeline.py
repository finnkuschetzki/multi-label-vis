from preprocess import coco_train, train_image_infos, val_image_infos

import tensorflow as tf
import numpy as np


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


def get_dataset(image_infos, train=False):
    image_paths, label_vectors = zip(*image_infos)
    image_paths, label_vectors = np.array(image_paths), np.array(label_vectors)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_vectors))
    dataset = dataset.map(lambda path, label_vec: (load_image(path), label_vec), num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = get_dataset(train_image_infos, train=True)
val_dataset = get_dataset(val_image_infos)


# configure export
__all__ = [
    "IMG_SIZE",
    "get_image_labels",
    "load_image",
    "train_dataset",
    "val_dataset",
]
