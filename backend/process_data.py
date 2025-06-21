import os
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from dimensionality_reduction.dgrid.dgrid import DGrid  # see https://github.com/fpaulovich/dimensionality-reduction/


WIDTH, HEIGHT = 0.01, 0.01  # results in 1 / (0.01 * 0.01) = 10,000 cells in visualization
DELTA = 1.0


# --- read and process data ---

in_df = pd.read_csv("../model/output/embedding_data.csv")
in_df["ground_truth"] = in_df["ground_truth"].apply(literal_eval)
in_df["features"] = in_df["features"].apply(literal_eval)
in_df["predictions"] = in_df["predictions"].apply(literal_eval)
in_df["binarized_predictions"] = in_df["binarized_predictions"].apply(literal_eval)
features = np.array(in_df["features"].tolist())

standard_scaler = StandardScaler()
standardized_features = standard_scaler.fit_transform(features)

min_max_scaler = MinMaxScaler()


# --- prepare data saving ---

directory = "data"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"created directory: {directory}")

out_df = in_df.filter(["ground_truth", "predictions", "binarized_predictions"])


# --- PCA ---

pca = PCA(n_components=2)
pca_features = pca.fit_transform(standardized_features)
scaled_pca_features = min_max_scaler.fit_transform(pca_features)
out_df["pca_features"] = scaled_pca_features.tolist()

# overlap removal
scaled_pca_features_or = DGrid(WIDTH, HEIGHT, DELTA).fit_transform(scaled_pca_features)
out_df["pca_features_or"] = scaled_pca_features_or.tolist()


# --- saving data ---

out_df.to_csv('data/dimensionality_reduction.csv', index=False)
