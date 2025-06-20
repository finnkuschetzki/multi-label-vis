import pandas as pd
import numpy as np


MAX_MIN_LABEL_STATS = 5


def limit_to_categories(coco, df, category_ids):
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


def limit_to_supercategories(coco, df, supercategories):
    category_ids = [cat["id"] for cat in coco.loadCats(coco.getCatIds()) if cat["supercategory"] in supercategories]

    return limit_to_categories(coco, df, category_ids)


def calc_category_statistics(df):
    result_total_image_count = df["image_count"].sum()

    entropy = -np.sum(df["image_share"] * np.log2(df["image_share"] + 1e-10))

    count_minX_ = {
        f"count_min{i}": df.loc[df["cat_count"] >= i, "image_count"].sum()
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

    return entropy, count_minX_, share_minX_over_all_, share_minX_over_min1_
