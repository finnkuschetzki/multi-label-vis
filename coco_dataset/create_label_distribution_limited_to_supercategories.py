from _label_distribution_tools import *

from pycocotools.coco import COCO
from itertools import chain, combinations
import pandas as pd
import ast
import os

# PREREQUISITE:
# run create_image_counts.py

MAX_SUPER_CATS = 3


# loading COCO
annotations_file = "data/annotations/instances_train2017.json"
coco = COCO(annotations_file)

cats = coco.loadCats(coco.getCatIds())
super_cats = list({cat["supercategory"] for cat in cats})


# --- label distribution limited to supercategories ---

category_sets_df = pd.read_csv("output/image_counts_per_category_set.csv", converters={"cat_ids": ast.literal_eval})

directory = "output/image_counts_per_category_set/limited_to_supercategories"
if not os.path.exists(directory):
    os.makedirs(directory)
    print()
    print(f"created directory: {directory}")

print()

category_subsets_df = pd.DataFrame({
    "supercat_count": pd.Series(dtype="int64"),
    "supercat_names": pd.Series(dtype="object"),
    "cat_count": pd.Series(dtype="int64"),
    "entropy": pd.Series(dtype="float64"),
    **{f"count_min{i}": pd.Series(dtype="int64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_min{i}_over_all": pd.Series(dtype="float64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_min{i}_over_min1": pd.Series(dtype="float64") for i in range(2, MAX_MIN_LABEL_STATS + 1)},
})

for super_cat_combination in chain.from_iterable(combinations(super_cats, r) for r in range(MAX_SUPER_CATS + 1)):
    print(f"calculating image_counts [limited to supercategories ({", ".join(super_cat_combination)})]...")

    result_df = limit_to_supercategories(coco, category_sets_df, super_cat_combination)
    result_df.sort_values(by=["cat_count", "image_count"], ascending=[True, False], inplace=True)

    # result_total_image_count = result_df["image_count"].sum()

    # save image_counts_per_category_set_limited_to_supercategories into csv
    sorted_super_cats = sorted(list(super_cat_combination))
    file_path = f"{directory}/limited_to_({', '.join(sorted_super_cats)}).csv"
    result_df.to_csv(file_path, index=False)
    print(f"saved to {file_path}")

    # write into category_subsets_df
    entropy, count_minX_, share_minX_over_all_, share_minX_over_min1_ = calc_category_statistics(result_df)

    new_row = {
        "supercat_count": len(super_cat_combination),
        "supercat_names": super_cat_combination,
        "cat_count": len([cat for cat in cats if cat["supercategory"] in super_cat_combination]),
        "entropy": entropy,
        **count_minX_,
        **share_minX_over_all_,
        **share_minX_over_min1_
    }

    category_subsets_df = pd.concat([category_subsets_df, pd.DataFrame([new_row])], ignore_index=True)

category_subsets_df.sort_values(by=["supercat_count", *[f"count_min{i}" for i in range(1, MAX_MIN_LABEL_STATS + 1)]], ascending=[True, *[False for _ in range(1, MAX_MIN_LABEL_STATS + 1)]], inplace=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(f"\n{category_sets_df}\n")

# category_subsets_statistics into csv
file_path = "output/label_distribution_limited_to_supercategories.csv"
category_subsets_df.to_csv(file_path, index=False)
print(f"label distribution limited to supercategories saved to {file_path}")
