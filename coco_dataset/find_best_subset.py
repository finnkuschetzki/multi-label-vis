import pandas as pd


# PREREQUISITE:
# run create_label_distribution.py


df = pd.read_csv("output/label_distribution_limited_to_supercategories.csv")

df.sort_values(by=["share_min2_over_min1"], ascending=False, inplace=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(df.head(10))
