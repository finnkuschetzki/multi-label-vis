from pycocotools.coco import COCO
import pandas as pd
import os


# creating output directory
directory = "output"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"created directory: {directory}")
    print()


# loading COCO
annotations_file = "data/annotations/instances_train2017.json"
coco = COCO(annotations_file)

cats = coco.loadCats(coco.getCatIds())


# --- categories file ---

# creating dataframe
df = pd.DataFrame(cats)[["id", "name", "supercategory"]]
print(f"\n{df.to_string(index=False)}\n")

# saving into csv
file_path = "output/categories.csv"
df.to_csv(file_path, index=False)
print(f"categories saved to {file_path}")
