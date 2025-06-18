import pandas as pd
from keras import models

from pipeline import *


model = models.load_model("output/fine_tuned.keras")

base_model = models.Model(
    inputs = model.input,
    outputs = model.layers[-2].output
)

print()
print("predicting...")

rows = []
for batch_images, batch_labels in val_dataset:
    features = base_model.predict(batch_images, verbose=0)
    predictions = model.predict(batch_images, verbose=0)
    binarized_predictions = [(ps >= 0.5).astype(int) for ps in predictions]
    for i in range(len(predictions)):
        rows.append({
            "ground_truth": batch_labels[i].numpy().tolist(),
            "features": features[i].tolist(),
            "predictions": predictions[i].tolist(),
            "binarized_predictions": binarized_predictions[i].tolist(),
        })

print("finished!")


# saving into csv
df = pd.DataFrame(rows)
df.to_csv("output/embedding_data.csv", index=False)

print()
print("saved ground_truth, features, predictions, binarized_predictions as output/embedding_data.csv")
