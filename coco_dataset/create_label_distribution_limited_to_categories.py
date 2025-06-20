import time

from _label_distribution_tools import *

from pycocotools.coco import COCO
from itertools import combinations
import pandas as pd
import ast
import os

# PREREQUISITE:
# run create_image_counts.py

CAT_COUNT = 15
MAX_CHOICE_CATS = 15


# --- loading COCO ---

annotations_file = "data/annotations/instances_train2017.json"
coco = COCO(annotations_file)

cats = coco.loadCats(coco.getCatIds())
super_cats = list({cat["supercategory"] for cat in cats})


# --- label distribution limited to categories ---

start = time.time()

category_df = pd.read_csv("output/image_counts_per_category.csv")
category_sets_df = pd.read_csv("output/image_counts_per_category_set.csv", converters={"cat_ids": ast.literal_eval})

cat_choices = category_df.sort_values("image_count", ascending=False)["id"].tolist()[:MAX_CHOICE_CATS]

category_subsets_df = pd.DataFrame({
    "cat_count": pd.Series(dtype="int64"),
    "cat_ids": pd.Series(dtype="object"),
    "entropy": pd.Series(dtype="float64"),
    **{f"count_min{i}": pd.Series(dtype="int64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_min{i}_over_all": pd.Series(dtype="float64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_min{i}_over_min1": pd.Series(dtype="float64") for i in range(2, MAX_MIN_LABEL_STATS + 1)},
})

directory = "output/image_counts_per_category_set/limited_to_categories"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"created directory: {directory}")
    print()

for cat_combination in combinations(cat_choices, CAT_COUNT):
    sorted_cat_combination_str = sorted(list(map(str, cat_combination)))
    print(f"calculating image_counts [limited to supercategories ({", ".join(sorted_cat_combination_str)})...")

    result_df = limit_to_categories(coco, category_sets_df, cat_combination)
    result_df.sort_values(by=["cat_count", "image_count"], ascending=[True, False], inplace=True)

    # save image_counts_per_category_set_limited_to_supercategories into csv
    file_path = f"{directory}/limited_to_({', '.join(sorted_cat_combination_str)}).csv"
    result_df.to_csv(file_path, index=False)
    print(f"saved to {file_path}")

    entropy, count_minX_, share_minX_over_all_, share_minX_over_min1_ = calc_category_statistics(result_df)

    new_row = {
        "cat_count": len(cat_combination),
        "cat_ids": cat_combination,
        "entropy": entropy,
        **count_minX_,
        **share_minX_over_all_,
        **share_minX_over_min1_
    }

    category_subsets_df = pd.concat([category_subsets_df, pd.DataFrame([new_row])], ignore_index=True)

category_subsets_df.sort_values(by=["cat_count", *[f"count_min{i}" for i in range(1, MAX_MIN_LABEL_STATS + 1)]], ascending=[True, *[False for _ in range(1, MAX_MIN_LABEL_STATS + 1)]], inplace=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(f"\n{category_subsets_df}\n")

# saving into csv
file_path = f"output/label_distribution_limited_to_categories-{CAT_COUNT}-{MAX_CHOICE_CATS}.csv"
category_subsets_df.to_csv(file_path, index=False)
print(f"label distribution limited to supercategories saved to {file_path}")

end = time.time()
print(f"total time: {end - start:.2f} seconds")
