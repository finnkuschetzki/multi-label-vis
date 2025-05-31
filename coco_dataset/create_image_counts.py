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

cat_ids = coco.getCatIds()
img_ids = coco.getImgIds()

cats = coco.loadCats(coco.getCatIds())


# --- image_counts per category ---

print()
print("calculating image_counts per category...")

category_stats = []

# calculate statistics
for cat in cats:
    category_stats.append(
        {
            "id": cat["id"],
            "name": cat["name"],
            "image_count": len(coco.getImgIds(catIds=[cat["id"]])),
            "supercategory": cat["supercategory"]
        }
    )

# creating dataframe
categories_df = pd.DataFrame(category_stats)[["id", "name", "image_count", "supercategory"]]
categories_df.sort_values(by=["image_count"], ascending=False, inplace=True)
print(f"\n{categories_df.to_string(index=False)}\n")

# saving into csv
file_path = 'output/image_counts_per_category.csv'
categories_df.to_csv(file_path, index=False)
print(f"image_counts per category saved to {file_path}")


# --- image counts per supercategory ---

print()
print("calculating image_counts per supercategory...")

supercategories_set = {cat["supercategory"] for cat in cats}
supercategories_stats = []

# calculate statistics
for sup_cat in supercategories_set:
    subcategories = [cat for cat in category_stats if cat["supercategory"] == sup_cat]
    supercategories_stats.append(
        {
            "name": sup_cat,
            "image_count": sum([cat["image_count"] for cat in subcategories]),
            "categories": [cat["name"] for cat in subcategories],
            "cat_count": len(subcategories)
        }
    )

# creating dataframe
supercategories_df = pd.DataFrame(supercategories_stats)[["name", "image_count", "categories", "cat_count"]]
supercategories_df.sort_values(by=["image_count"], ascending=False, inplace=True)
print(f"\n{supercategories_df.to_string(index=False)}\n")

# saving into csv
file_path = "output/image_counts_per_supercategory.csv"
supercategories_df.to_csv(file_path, index=False)
print(f"image_counts per supercategory saved to {file_path}")


# --- image counts per category set ---

print()
print("calculating image_counts per category set...")

category_sets_dict = dict()

for img_id in img_ids:
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    cat_ids_tuple = tuple(sorted(({ann["category_id"] for ann in anns})))

    if cat_ids_tuple in category_sets_dict.keys():
        category_sets_dict[cat_ids_tuple] += 1
    else:
        category_sets_dict[cat_ids_tuple] = 1

# creating dataframe
category_sets_df = pd.DataFrame(list(category_sets_dict.items()), columns=["cat_ids", "image_count"])

category_sets_df["cat_count"] = category_sets_df["cat_ids"].apply(lambda x: len(x))
category_sets_df["image_share"] = category_sets_df["image_count"] / category_sets_df["image_count"].sum()
category_sets_df["cat_names"] = category_sets_df["cat_ids"].apply(lambda x: [coco.loadCats(x)[0]["name"] for x in x])

category_sets_df = category_sets_df[["cat_count", "cat_ids", "cat_names", "image_count", "image_share"]]
category_sets_df.sort_values(by=["cat_count", "image_count"], ascending=[True, False], inplace=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(f"\n{category_sets_df}\n")

# check output
print(f"total image_count: {category_sets_df['image_count'].sum():,}")

# saving into csv
file_path = "output/image_counts_per_category_set.csv"
category_sets_df.to_csv(file_path, index=False)
print(f"image_count per category set saved to {file_path}")
