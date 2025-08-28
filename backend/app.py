import os
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from process_data import *
from convex_hull import *


# frontend expects Flask App on port 5001


# only for development
_INCLUDE_TRAIN_DATA = True


# --- load config ---

with open("../config.json", "r") as f:
    config = json.load(f)


# --- get dimensionality reduction data for every model (for train and val data) ---

models_dict = dict()

def load_or_apply_and_save(model_name, model_path, data_type):
    print()
    print(f"--- {model_name}, {data_type} ---")
    if os.path.exists(f"computed_model_data/{model_name}/{data_type}_data.csv"):
        start = time.time()
        print()
        print("Loading data...")
        data_ = read_csv_with_list_attributes(
            f"computed_model_data/{model_name}/{data_type}_data.csv",
            ["ground_truth", "pca_features", "umap_features", "tsne_features", "predictions", "binarized_predictions"]
        )
        end = time.time()
        print(f"Done (t={end-start:.2f}s)")
    else:
        start = time.time()
        print()
        print("Reading embedding data...")
        embedding_data = read_csv_with_list_attributes(
            f"../{model_path}/embedding_data_{data_type}.csv",
            ["ground_truth", "features", "predictions", "binarized_predictions"]
        )
        end = time.time()
        print(f"Done (t={end - start:.2f}s)")
        data_ = apply_dimensionality_reduction(embedding_data)
        data_.to_csv(f"computed_model_data/{model_name}/{data_type}_data.csv", index=False)
    return data_

if not os.path.isdir(f"computed_model_data"):
    os.mkdir(f"computed_model_data")

for model in config["models"]:
    if not os.path.isdir(f"computed_model_data/{model["name"]}"):
        os.mkdir(f"computed_model_data/{model["name"]}")
    if _INCLUDE_TRAIN_DATA:
        train_data = load_or_apply_and_save(model["name"], model["path"], "train")
        val_data = load_or_apply_and_save(model["name"], model["path"], "val")
        models_dict[model["name"]] = {
            "data": { "train": train_data, "val": val_data },
            "glyph_size": model["glyphSize"],
        }
    else:  # only for development
        val_data = load_or_apply_and_save(model["name"], model["path"], "val")
        models_dict[model["name"]] = {
            "data": {"val": val_data },
            "glyph_size": model["glyphSize"],
        }


# --- load class info ---

class_infos_dict = dict()

for model in config["models"]:
    class_infos_dict[model["name"]] = pd.read_csv(f"../{model["path"]}/class_info.csv")


# --- Flask App ---

print()
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/class-info/")
def class_info():
    model_name = str(request.args.get("modelName"))
    return class_infos_dict[model_name].to_json(orient="records")


@app.route("/data/")
def data():
    model_name = str(request.args.get("modelName"))
    data_type = str(request.args.get("dataType"))
    x_factor = float(request.args.get("factorX"))
    y_factor = float(request.args.get("factorY"))

    if data_type == "train":
        df = models_dict[model_name]["data"]["train"]
        glyph_size = models_dict[model_name]["glyph_size"]["train"]
    elif data_type == "val":
        df = models_dict[model_name]["data"]["val"]
        glyph_size = models_dict[model_name]["glyph_size"]["val"]
    else:
        raise ValueError("Type must be 'train' or 'val'")

    # overlap removal
    if x_factor is None or y_factor is None:
        or_df = apply_overlap_removal(df, glyph_size)
    else:
        or_df = apply_overlap_removal(df, glyph_size, x_factor, y_factor)
    data_points = or_df.to_dict(orient="records")

    # convex hulls
    convex_hulls = calculate_convex_hulls(or_df)

    response_dict = {
        "data_points": data_points,
        "convex_hulls": convex_hulls,
    }

    return jsonify(response_dict)


@app.route("/image/")
def image():
    image_path = request.args.get("imagePath")

    # todo not safe for public use, use send_from_directory instead
    return send_file(image_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run()
