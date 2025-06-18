import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from sklearn.metrics import accuracy_score, precision_score, recall_score

from preprocess import *
from pipeline import *


model = models.load_model("output/fine_tuned.keras")


# --- per class metrics ---

print()

all_y_true = []
all_y_pred = []

# calculate y_true and y_pred per batch and add to list
print("predicting...")
for batch_images, batch_labels in val_dataset:
    y_probs = model.predict(batch_images, verbose=0)
    y_pred = (y_probs >= 0.5).astype(int)
    all_y_true.extend(batch_labels)
    all_y_pred.extend(y_pred)
print("finished!")
print()

# flatten list into numpy array
all_y_true = np.vstack(all_y_true)
all_y_pred = np.vstack(all_y_pred)

metrics_per_classes = []

for i in range(num_train_classes):
    y_true_class = all_y_true[:, i]
    y_pred_class = all_y_pred[:, i]

    # calculating metrics
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)

    metrics_per_classes.append({
        "cat_id": multi_hot_index_to_category_id[i],
        "cat_name": multi_hot_index_to_category_name[i],
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "image_count": np.sum(y_true_class),
    })

    # printing metrics
    print(f"{multi_hot_index_to_category_name[i]}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")


# --- overall metrics ---

print()

y_true_flat = all_y_true.flatten()
y_pred_flat = all_y_pred.flatten()

overall_accuracy = accuracy_score(y_true_flat, y_pred_flat)  # hamming accuracy
overall_micro_precision = precision_score(all_y_true, all_y_pred, zero_division=0, average="micro")
overall_macro_precision = precision_score(all_y_true, all_y_pred, zero_division=0, average="macro")
overall_weighted_precision = precision_score(all_y_true, all_y_pred, zero_division=0, average="weighted")
overall_micro_recall = recall_score(all_y_true, all_y_pred, zero_division=0, average="micro")
overall_macro_recall = recall_score(all_y_true, all_y_pred, zero_division=0, average="macro")
overall_weighted_recall = recall_score(all_y_true, all_y_pred, zero_division=0, average="weighted")

print("Overall Metrics:")
print(f"  Accuracy: {overall_accuracy:.4f}")
print(f"  Micro Precision: {overall_micro_precision:.4f}")
print(f"  Macro Precision: {overall_macro_precision:.4f}")
print(f"  Weighted Precision: {overall_weighted_precision:.4f}")
print(f"  Micro Recall: {overall_micro_recall:.4f}")
print(f"  Macro Recall: {overall_macro_recall:.4f}")
print(f"  Weighted Recall: {overall_weighted_recall:.4f}")


# --- saving metrics ---

directory = "output/val_stats"
if not os.path.exists(directory):
    os.makedirs(directory)

df = pd.DataFrame(metrics_per_classes, columns=["cat_id", "cat_name", "accuracy", "precision", "recall", "image_count"])
df.to_csv(f"{directory}/metrics_per_classes.csv", index=False)


# --- creating plots ---

print()

metrics = ["accuracy", "precision", "recall"]
class_names = [class_metrics["cat_name"] for class_metrics in metrics_per_classes]
image_counts = [class_metrics["image_count"] for class_metrics in metrics_per_classes]

for metric in metrics:
    values = [class_metrics[metric] for class_metrics in metrics_per_classes]

    plt.figure()
    plt.scatter(image_counts, values)

    # Annotate each point with the class name
    for i, cls in enumerate(class_names):
        plt.annotate(cls, (image_counts[i], values[i]), textcoords="offset points", xytext=(5, 5), ha='left')

    plt.title(f"{metric.capitalize()} vs Image Count")
    plt.xlabel("Image Count")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{directory}/{metric}_vs_image_count.svg")
    plt.savefig(f"{directory}/{metric}_vs_image_count.png")
    plt.show()

    print(f"saved {metric} plot as svg and png")
