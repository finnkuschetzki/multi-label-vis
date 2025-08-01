import time
import numpy as np
from itertools import product
from scipy.spatial import ConvexHull


feature_names = ["pca_features", "pca_features_or", "umap_features", "umap_features_or", "tsne_features", "tsne_features_or"]
labels = ["ground_truth", "predictions"]


def calculate_convex_hulls(in_df):
    start = time.time()

    num_classes = len(in_df.iloc[0]["ground_truth"])
    convex_hull_dict = dict()

    # for each combination of feature_name, label and class_index
    for feature_name, label, class_index in product(feature_names, labels, range(num_classes)):

        if label == "ground_truth":
            filter_func = lambda l: l[class_index] == 1
        elif label == "predictions":
            filter_func = lambda l: l[class_index] >= 0.5
        else:
            raise ValueError("Unknown label")

        # filter for values according to label and class_index
        filtered_df = in_df[in_df[label].apply(filter_func)]

        # calculate convex hull
        positions = np.array(filtered_df[feature_name].tolist())
        convex_hull = ConvexHull(positions)

        # save into convex_ull_dict
        if not feature_name in convex_hull_dict.keys():
            convex_hull_dict[feature_name] = dict()
        if not label in convex_hull_dict[feature_name].keys():
            convex_hull_dict[feature_name][label] = dict()
        convex_hull_dict[feature_name][label][class_index] = positions[convex_hull.vertices].tolist()

    end = time.time()
    print(f"--- Convex hulls calculated in {end - start} seconds ---")

    return convex_hull_dict
