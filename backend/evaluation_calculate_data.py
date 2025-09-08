import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from process_data import read_csv_with_list_attributes, apply_overlap_removal
from ast import literal_eval


def class_centroids(positions, labels, list_of_labels):
    centroids = []
    for i, _ in enumerate(list_of_labels):
        class_centroid = positions[labels[:, i] == 1].mean(axis=0)
        centroids.append(class_centroid)
    return centroids


def intra_class_compactness(positions, labels, list_of_labels, centroids):
    compactness = []
    for i, _ in enumerate(list_of_labels):
        distances = np.linalg.norm(positions[labels[:, i] == 1] - centroids[i], axis=1)
        avg_distance = distances.mean()
        compactness.append(avg_distance)
    compactness = np.array(compactness)
    return compactness


def inter_class_separation(centroids):
    dist_matrix = pairwise_distances(centroids, metric="euclidean")
    avg_distances = ((dist_matrix.sum(axis=1) - np.diag(dist_matrix)) / (len(centroids) - 1))
    separation = avg_distances
    return separation


def spread(compactness, separation):
    return separation / compactness


# --- evaluate for every model (val data) and every dimensionality reduction technique --- todo might add train data

if __name__ == "__main__":

    with open("../config.json", "r") as f:
        config = json.load(f)

    if not os.path.isdir(f"evaluation_data"):
        os.mkdir(f"evaluation_data")

    # for every model
    for model in config["models"]:

        if not os.path.isdir(f"evaluation_data/{model["name"]}"):
            os.mkdir(f"evaluation_data/{model["name"]}")

        class_info = pd.read_csv(f"../{model["path"]}/class_info.csv")

        model_name = model["name"]
        data_type = "val"

        # read computed_model_data
        df = read_csv_with_list_attributes(
            f"computed_model_data/{model_name}/{data_type}_data.csv",
            ["ground_truth", "predictions", "binarized_predictions", "pca_features", "umap_features", "tsne_features"]
        )

        # apply overlap removal
        or_df = apply_overlap_removal(df, model["glyphSize"][data_type])  # factor_x: 1, factor_y: 1

        # for every dimensionality reduction technique
        for dimensionality_reduction in ["pca", "umap", "tsne"]:

            if not os.path.isdir(f"evaluation_data/{model["name"]}/{dimensionality_reduction}"):
                os.mkdir(f"evaluation_data/{model["name"]}/{dimensionality_reduction}")

            features_or = np.vstack(or_df[f"{dimensionality_reduction}_features_or"])
            label_names = class_info["name"].to_numpy()

            # for binarizes_predictions and ground_truth
            for data_type in ["binarized_predictions", "ground_truth"]:

                labels = np.vstack(or_df[data_type])

                # calculate metrics
                class_centroids_ = class_centroids(features_or, labels, label_names)
                intra_class_compactness_ = intra_class_compactness(features_or, labels, label_names, class_centroids_)
                inter_class_separation_ = inter_class_separation(class_centroids_)
                spread_ = spread(intra_class_compactness_, inter_class_separation_)

                # store metrics in df
                metrics_df = pd.DataFrame()
                metrics_df["id"] = class_info["id"]
                metrics_df["name"] = class_info["name"]
                metrics_df["intra_class_compactness"] = intra_class_compactness_
                metrics_df["inter_class_separation"] = inter_class_separation_
                metrics_df["spread"] = spread_

                # save to csv
                metrics_df.to_csv(f"evaluation_data/{model['name']}/{dimensionality_reduction}/{data_type}.csv")

        # original features
        feat_df = pd.read_csv(f"../{model['path']}/embedding_data_val.csv")
        # reading list attributes
        feat_df["features"] = feat_df["features"].apply(literal_eval)
        feat_df["ground_truth"] = feat_df["ground_truth"].apply(literal_eval)
        feat_df["predictions"] = feat_df["predictions"].apply(literal_eval)
        feat_df["binarized_predictions"] = feat_df["binarized_predictions"].apply(literal_eval)

        if not os.path.isdir(f"evaluation_data/{model["name"]}/features"):
            os.mkdir(f"evaluation_data/{model["name"]}/features")

        features = np.vstack(feat_df["features"])
        label_names = class_info["name"].to_numpy()

        # for binarizes_predictions and ground_truth
        for data_type in ["binarized_predictions", "ground_truth"]:
            labels = np.vstack(feat_df[data_type])

            # calculate metrics
            class_centroids_ = class_centroids(features, labels, label_names)
            intra_class_compactness_ = intra_class_compactness(features, labels, label_names, class_centroids_)
            inter_class_separation_ = inter_class_separation(class_centroids_)
            spread_ = spread(intra_class_compactness_, inter_class_separation_)

            # store metrics in df
            metrics_df = pd.DataFrame()
            metrics_df["id"] = class_info["id"]
            metrics_df["name"] = class_info["name"]
            metrics_df["intra_class_compactness"] = intra_class_compactness_
            metrics_df["inter_class_separation"] = inter_class_separation_
            metrics_df["spread"] = spread_

            # save to csv
            metrics_df.to_csv(f"evaluation_data/{model['name']}/features/{data_type}.csv")
