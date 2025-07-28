import time
import pandas as pd
from keras import models

from _command_line_tools import *
from _preprocess import *
from _pipeline import *


# --- embedding data ---

MODEL_DIR = get_model_dir()

print()
print(f"using model from directory: {MODEL_DIR}")

model = models.load_model(f"{MODEL_DIR}/fine_tuned.keras")

base_model = models.Model(
    inputs = model.input,
    outputs = model.layers[-2].output
)

def generate_predictions(dataset, file_name):

    print()
    print("predicting...")

    start = time.time()

    rows = []
    for batch_paths, batch_images, batch_labels in dataset:
        features = base_model.predict(batch_images, verbose=0)
        predictions = model.predict(batch_images, verbose=0)
        binarized_predictions = [(ps >= 0.5).astype(int) for ps in predictions]
        for i in range(len(predictions)):
            rows.append({
                "image_path": batch_paths[i].numpy().decode("utf-8"),  # relative path will also work from backend
                "ground_truth": batch_labels[i].numpy().tolist(),
                "features": features[i].tolist(),
                "predictions": predictions[i].tolist(),
                "binarized_predictions": binarized_predictions[i].tolist(),
            })

    end = time.time()

    print(f"Done (t={end-start:.2f}s)")

    # saving into csv
    df = pd.DataFrame(rows)
    df.to_csv(f"{MODEL_DIR}/{file_name}.csv", index=False)

    print()
    print(f"saved ground_truth, features, predictions, binarized_predictions as {MODEL_DIR}/{file_name}.csv")

generate_predictions(train_dataset_for_prediction, "embedding_data_train")
generate_predictions(val_dataset_for_prediction, "embedding_data_val")


# --- class info ---

rows = []
for id_, name in category_id_to_category_name.items():
    rows.append({
        "id": id_,
        "name": name,
    })

# saving into csv
class_df = pd.DataFrame(rows)
class_df.to_csv(f"{MODEL_DIR}/class_info.csv", index=False)
