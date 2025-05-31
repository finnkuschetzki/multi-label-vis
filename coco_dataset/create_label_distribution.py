from pycocotools.coco import COCO
from itertools import chain, combinations
import pandas as pd
import numpy as np
import ast
import os


# PREREQUISITE:
# run create_image_counts.py


# loading COCO
annotations_file = "data/annotations/instances_train2017.json"
coco = COCO(annotations_file)

cats = coco.loadCats(coco.getCatIds())
super_cats = list({cat["supercategory"] for cat in cats})


# --- limited to supercategories

def limited_to_supercategories(df, supercategories):
    category_ids = [cat["id"] for cat in cats if cat["supercategory"] in supercategories]

    print(f"calculating image_counts [limited to supercategories ({", ".join(supercategories)})]...")

    limited_dict = dict()

    for row in df.itertuples(index=False):
        limited_cat_ids_tuple = tuple(sorted(set(row.cat_ids).intersection(set(category_ids))))

        if limited_cat_ids_tuple in limited_dict.keys():
            limited_dict[limited_cat_ids_tuple] += row.image_count
        else:
            limited_dict[limited_cat_ids_tuple] = row.image_count

    limited_df = pd.DataFrame(list(limited_dict.items()), columns=["cat_ids", "image_count"])
    limited_df["cat_count"] = limited_df["cat_ids"].apply(lambda x: len(x))
    limited_df["image_share"] = limited_df["image_count"] / limited_df["image_count"].sum()
    limited_df["cat_names"] = limited_df["cat_ids"].apply(lambda x: [coco.loadCats(x)[0]["name"] for x in x])
    limited_df = limited_df[["cat_count", "cat_ids", "cat_names", "image_count", "image_share"]]

    return limited_df


# limit to supercategories
category_sets_df = pd.read_csv("output/image_counts_per_category_set.csv", converters={"cat_ids": ast.literal_eval})

MAX_SUPER_CATS = 3
MAX_MIN_LABEL_STATS = 5

category_subsets_df = pd.DataFrame({
    "supercat_count": pd.Series("int64"),
    "supercat_names": pd.Series("object"),
    "cat_count": pd.Series("int64"),
    "entropy": pd.Series("float64"),
    **{f"count_min{i}": pd.Series("int64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"count_min{i}_over_all": pd.Series("float64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
    **{f"count_min{i}_over_min1": pd.Series("float64") for i in range(1, MAX_MIN_LABEL_STATS + 1)},
})

directory = "output/image_counts_per_category_set"
if not os.path.exists(directory):
    os.makedirs(directory)
    print()
    print(f"created directory: {directory}")

print()

for super_cat_combination in chain.from_iterable(combinations(super_cats, r) for r in range(MAX_SUPER_CATS + 1)):
    result_df = limited_to_supercategories(category_sets_df, super_cat_combination)
    result_df.sort_values(by=["cat_count", "image_count"], ascending=[True, False], inplace=True)
    # print(f"\n{result_df}\n")

    result_total_image_count = result_df["image_count"].sum()

    # check output
    # print(f"total image_count: {result_total_image_count:,}")

    # save image_counts_per_category_set_limited_to_supercategories into csv
    sorted_super_cats = sorted(list(super_cat_combination))
    file_path = f"output/image_counts_per_category_set/limited_to_({', '.join(sorted_super_cats)}).csv"
    result_df.to_csv(file_path, index=False)
    print(f"saved to {file_path}")

    # write into category_subsets_df
    entropy = -np.sum(result_df["image_share"] * np.log2(result_df["image_share"] + 1e-10))

    count_minX_ = {
        f"count_min{i}": result_df.loc[result_df["cat_count"] >= i, "image_count"].sum()
        for i in range(1, MAX_MIN_LABEL_STATS + 1)
    }

    share_minX_over_all_ = {
        f"share_min{i}_over_all": np.divide(
            count_minX_[f"count_min{i}"],
            result_total_image_count if result_total_image_count > 0 else np.nan
        )
        for i in range(1, MAX_MIN_LABEL_STATS + 1)
    }

    share_minX_over_min1_ = {
        f"share_min{i}_over_min1": np.divide(
            count_minX_[f"count_min{i}"],
            count_minX_["count_min1"] if count_minX_["count_min1"] > 0 else np.nan
        )
        for i in range(2, MAX_MIN_LABEL_STATS + 1)
    }

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
