import time

from _label_distribution_tools import *

from pycocotools.coco import COCO
import pandas as pd
import ast


# --- loading COCO ---

# annotations_file = "data/annotations/instances_train2017.json"
annotations_file = "test/annotations/instances_train2014.json"
coco = COCO(annotations_file)

cats = coco.loadCats(coco.getCatIds())
super_cats = list({cat["supercategory"] for cat in cats})


# --- label distribution overall ---

start = time.time()

category_df = pd.read_csv("output/image_counts_per_category.csv")
category_sets_df = pd.read_csv("output/image_counts_per_category_set.csv", converters={"cat_ids": ast.literal_eval})

df = pd.DataFrame({
    "cat_count": pd.Series(dtype="int64"),
    "cat_ids": pd.Series(dtype="object"),
    "entropy": pd.Series(dtype="float64"),
    **{f"count_{i}": pd.Series(dtype="int64") for i in range(0, MAX_MIN_LABEL_STATS + 1)},
    **{f"count_min{i}": pd.Series(dtype="int64") for i in range(0, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_{i}_over_all": pd.Series(dtype="float64") for i in range(0, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_min{i}_over_all": pd.Series(dtype="float64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_{i}_over_min1": pd.Series(dtype="float64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"share_min{i}_over_min1": pd.Series(dtype="float64") for i in range(2, MAX_MIN_LABEL_STATS + 1)},
    "average_label_count": pd.Series(dtype="float64")
})

cat_ids = coco.getCatIds()

result_df = limit_to_categories(coco, category_sets_df, cat_ids)
result_df.sort_values(by=["cat_count", "image_count"], ascending=[True, False], inplace=True)

entropy, count_X_, count_minX_, share_X_over_all_, share_minX_over_all_, share_X_over_min1_, share_minX_over_min1_, average_label_count = calc_category_statistics(result_df)

new_row = {
    "cat_count": len(cat_ids),
    "cat_ids": cat_ids,
    "entropy": entropy,
    **count_X_,
    **count_minX_,
    **share_X_over_all_,
    **share_minX_over_all_,
    **share_X_over_min1_,
    **share_minX_over_min1_,
    "average_label_count": average_label_count
}

df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(f"\n{df}\n")

# saving into csv
file_path = f"output/label_distribution_overall.csv"
df.to_csv(file_path, index=False)
print(f"label distribution overall saved to {file_path}")

end = time.time()
print(f"total time: {end - start:.2f} seconds")
