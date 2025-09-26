import json
import pandas as pd


with open("../config.json", "r") as f:
    config = json.load(f)


avg_df = pd.DataFrame(columns=["model_name", "positions", "type", "intra_class_compactness", "inter_class_separation", "spread"])

for model in config["models"]:
    for positions in ["features", "pca", "umap", "tsne"]:
        for data_type in ["binarized_predictions", "ground_truth"]:

            model_df = pd.read_csv(f"evaluation_data/{model['name']}/{positions}/{data_type}.csv")

            avg_intra_class_compactness = model_df["intra_class_compactness"].mean()
            avg_inter_class_separation = model_df["inter_class_separation"].mean()
            avg_spread = model_df["spread"].mean()

            avg_df.loc[len(avg_df)] = [model["name"], positions, data_type, avg_intra_class_compactness, avg_inter_class_separation, avg_spread]

avg_df.to_csv(f"evaluation_data/averages.csv", index=False)
