import ast

import pandas as pd
import os


# PREREQUISITE:
# run create_label_distribution.py


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# --- limited to supercategories ---

df = pd.read_csv("output/label_distribution_limited_to_supercategories.csv")

df.sort_values(by=["share_min2_over_min1"], ascending=False, inplace=True)

print(df.head(10))

# only export
# df.to_csv("output/export/limited_to_supercategories.csv", index=False)


# --- limited to supercategories (at least one outdoor supercategory) ----

print()

outdoor_super_cats = ["person", "vehicle", "outdoor", "animal", "accessory", "sports"]

df_contains_outdoor = df.copy()
df_contains_outdoor["supercat_names"] = df_contains_outdoor["supercat_names"].apply(ast.literal_eval)
df_contains_outdoor = df_contains_outdoor[df_contains_outdoor["supercat_names"].apply(lambda x: any(item in outdoor_super_cats for item in x))]

df_contains_outdoor.sort_values(by=["share_min2_over_min1"], ascending=False, inplace=True)

print(df_contains_outdoor.head(10))

# only export
# df_contains_outdoor.to_csv("output/export/limited_to_supercategories_contains_outdoor.csv", index=False)


# --- limited to categories ---

print()

directory = "output"

files_names = [file for file in os.listdir(directory) if file.startswith("label_distribution_limited_to_categories")]

cat_df = pd.DataFrame()

for file_name in files_names:
    file_df = pd.read_csv(os.path.join(directory, file_name))
    cat_df = pd.concat([cat_df, file_df], ignore_index=True)

cat_df.sort_values(by=["share_min2_over_min1"], ascending=False, inplace=True)

print(cat_df)

# only export
# cat_df.to_csv("output/export/limited_to_categories_with_highest_image_count.csv", index=False)
