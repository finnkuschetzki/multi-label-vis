import numpy as np
from keras import models
from sklearn.metrics import accuracy_score, precision_score, recall_score

from preprocess import *
from pipeline import *


model = models.load_model("output/fine_tuned.keras")

all_y_true = []
all_y_pred = []

# calculate y_true and y_pred per batch and add to list
for batch_images, batch_labels in val_dataset:
    y_probs = model.predict(batch_images, verbose=0)
    y_pred = (y_probs >= 0.5).astype(int)
    all_y_true.extend(batch_labels)
    all_y_pred.extend(y_pred)

# flatten list into numpy array
all_y_true = np.vstack(all_y_true)
all_y_pred = np.vstack(all_y_pred)

print("========== Metrics per classes ==========")

metrics_per_classes = dict()

for i in range(num_train_classes):
    y_true_class = all_y_true[:, i]
    y_pred_class = all_y_pred[:, i]

    # calculating metrics
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)

    metrics_per_classes[multi_hot_index_to_category_name[i]] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    # printing metrics
    print(f"{multi_hot_index_to_category_name[i]}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
