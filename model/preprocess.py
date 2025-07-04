from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np


DATA_DIR = "../coco_dataset/data"
TRAIN_IMG_DIR = DATA_DIR + "/images/train2017"
VAL_IMG_DIR = DATA_DIR + "/images/val2017"

CATEGORY_IDS = [
    44, 46, 47, 48, 49, 50, 51,  # kitchen categories
    62, 63, 64, 65, 67, 70,  # furniture categories
    72, 73, 74, 75, 76, 77  # electronic furniture
]


# creating COCO instances (annotations for train2017 and val2017)
coco_train = COCO(DATA_DIR + "/annotations/instances_train2017.json")
print()
coco_val = COCO(DATA_DIR + "/annotations/instances_val2017.json")

categories = coco_train.loadCats(coco_train.getCatIds())  # this assumes that all classes are present in train and val data
train_categories = [cat for cat in categories if cat["id"] in CATEGORY_IDS]
num_coco_classes = len(categories)
num_train_classes = len(CATEGORY_IDS)

# map category_id to multi_hot index
category_id_to_multi_hot_index = {
    category["id"]: multi_hot_index
    for multi_hot_index, category in enumerate(train_categories)
}

multi_hot_index_to_category_id = {
    multi_hot_index: category_id
    for category_id, multi_hot_index in category_id_to_multi_hot_index.items()
}

multi_hot_index_to_category_name = dict([
    (category_id_to_multi_hot_index[category["id"]], category["name"])
    for category in train_categories
])

# map images to labels
image_labels = defaultdict(set)

train_anns = coco_train.loadAnns(coco_train.getAnnIds())
val_anns = coco_val.loadAnns(coco_val.getAnnIds())

for ann in train_anns + val_anns:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    if category_id in CATEGORY_IDS:
        multi_hot_index = category_id_to_multi_hot_index[category_id]
        image_labels[image_id].add(multi_hot_index)


# list images with labels
def get_image_infos(coco_instance, img_dir): # generates image_infos for given coco_instance
    image_infos = []

    for img in coco_instance.loadImgs(coco_instance.getImgIds()):
        img_path = img_dir + "/" + img["file_name"]
        img_labels = image_labels[img["id"]]
        multi_hot_encoding = np.zeros(num_train_classes, dtype=np.float32)
        multi_hot_encoding[list(img_labels)] = 1.0
        if not np.all(multi_hot_encoding == 0):
            image_infos.append((img_path, multi_hot_encoding))

    return image_infos


train_image_infos = get_image_infos(coco_train, TRAIN_IMG_DIR)
val_image_infos = get_image_infos(coco_val, VAL_IMG_DIR)


# configure export
__all__ = [
    "DATA_DIR",
    "TRAIN_IMG_DIR",
    "coco_train",
    "coco_val",
    "num_train_classes",
    "multi_hot_index_to_category_id",
    "multi_hot_index_to_category_name",
    "train_image_infos",
    "val_image_infos"
]
