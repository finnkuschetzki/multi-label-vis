import time
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from dimensionality_reduction.dgrid.dgrid import DGrid  # see https://github.com/fpaulovich/dimensionality-reduction/


WIDTH, HEIGHT = 0.01, 0.01  # results in 1 / (0.01 * 0.01) = 10,000 cells in visualization
DELTA = 1.0


def read_csv_with_list_attributes(path, list_attributes):
    df = pd.read_csv(path)
    for list_attribute in list_attributes:
        df[list_attribute] = df[list_attribute].apply(literal_eval)
    return df


def apply_dimensionality_reduction(in_df):
    out_df = pd.DataFrame()
    out_df["ground_truth"] = in_df["ground_truth"].copy()
    out_df["predictions"] = in_df["predictions"].copy()
    out_df["binarized_predictions"] = in_df["binarized_predictions"].copy()

    features = np.array(in_df["features"].tolist())
    standard_scaler = StandardScaler()
    standardized_features = standard_scaler.fit_transform(features)

    min_max_scaler = MinMaxScaler()

    # --- PCA ---

    print()
    start = time.time()

    print("PCA...")
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(standardized_features)
    scaled_pca_features = min_max_scaler.fit_transform(pca_features)
    out_df["pca_features"] = scaled_pca_features.tolist()

    end = time.time()
    print(f"Done (t={end - start:.2f}s)")

    # --- UMAP ---

    print()
    start = time.time()

    print("UMAP...")
    umap = UMAP()
    umap_features = umap.fit_transform(standardized_features)
    scaled_umap_features = min_max_scaler.fit_transform(umap_features)
    out_df["umap_features"] = scaled_umap_features.tolist()

    end = time.time()
    print(f"Done (t={end - start:.2f}s)")

    # --- t-SNE ---

    print()
    start = time.time()

    print("t-SNE...")
    tsne = TSNE(n_components=2)
    tsne_features = tsne.fit_transform(standardized_features)
    scaled_tsne_features = min_max_scaler.fit_transform(tsne_features)
    out_df["tsne_features"] = scaled_tsne_features.tolist()

    end = time.time()
    print(f"Done (t={end - start:.2f}s)")

    return out_df


def apply_overlap_removal(in_df: pd.DataFrame):
    out_df = in_df.copy()

    pca_features = np.array(in_df["pca_features"].tolist())
    pca_feature_or = DGrid(WIDTH, HEIGHT, DELTA).fit_transform(pca_features)
    out_df["pca_features_or"] = pca_feature_or.tolist()

    umap_features = np.array(in_df["umap_features"].tolist())
    umap_feature_or = DGrid(WIDTH, HEIGHT, DELTA).fit_transform(umap_features)
    out_df["umap_features_or"] = umap_feature_or.tolist()

    tsne_features = np.array(in_df["tsne_features"].tolist())
    tsne_feature_or = DGrid(WIDTH, HEIGHT, DELTA).fit_transform(tsne_features)
    out_df["tsne_features_or"] = tsne_feature_or.tolist()

    return out_df
