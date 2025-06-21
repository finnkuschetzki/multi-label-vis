import os
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --- read and process data ---

in_df = pd.read_csv("../model/output/embedding_data.csv")
in_df["ground_truth"] = in_df["ground_truth"].apply(literal_eval)
in_df["features"] = in_df["features"].apply(literal_eval)
in_df["predictions"] = in_df["predictions"].apply(literal_eval)
in_df["binarized_predictions"] = in_df["binarized_predictions"].apply(literal_eval)
features = np.array(in_df["features"].tolist())

sc = StandardScaler()
standardized_features = sc.fit_transform(features)


# --- prepare data saving ---

directory = "data"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"created directory: {directory}")

out_df = in_df.filter(["ground_truth", "predictions", "binarized_predictions"])


# --- PCA ---

pca = PCA(n_components=2)
pca_features = pca.fit_transform(standardized_features)
out_df["pca_features"] = pca_features.tolist()


# --- saving data ---

out_df.to_csv('data/dimensionality_reduction.csv', index=False)
