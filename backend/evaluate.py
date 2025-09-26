import json
import pandas as pd
from scipy.stats import spearmanr


EXCLUDE_PERSON_CLASS = False


if __name__ == "__main__":
    with open("../config.json", "r") as f:
        config = json.load(f)

    correlation_df = pd.DataFrame(columns=["model_name", "dr_technique", "correlation", "p_value"])

    # for every model
    for model in config["models"]:

        classification_metrics = pd.read_csv(f"../{model["path"]}/val_stats/metrics_per_classes.csv")
        # filtering
        if EXCLUDE_PERSON_CLASS:
            classification_metrics = classification_metrics[classification_metrics["cat_name"] != "person"]

        f1 = classification_metrics["f1"]

        # for every dimensionality reduction technique (and original features)
        for dimensionality_reduction in ["features", "pca", "umap", "tsne"]:

            visual_metrics = pd.read_csv(f"evaluation_data/{model["name"]}/{dimensionality_reduction}/binarized_predictions.csv")
            if EXCLUDE_PERSON_CLASS:
                # filtering
                visual_metrics = visual_metrics[visual_metrics["name"] != "person"]

            spread = visual_metrics["spread"]

            print()
            print(model["name"], dimensionality_reduction)

            # spearman correlation
            corr, p_value = spearmanr(f1, spread)
            print("Spearman correlation:", corr)
            print("P-value:", p_value)

            correlation_df.loc[len(correlation_df)] = [model["name"], dimensionality_reduction, corr, p_value]

    correlation_df.to_csv("evaluation_data/spearman_correlation.csv", index=False)
