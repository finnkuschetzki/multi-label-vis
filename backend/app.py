from flask import Flask, request, send_file
from flask_cors import CORS

from process_data import *


df = read_csv_with_list_attributes("../model/output/embedding_data.csv", ["ground_truth", "features", "predictions", "binarized_predictions"])
df = apply_dimensionality_reduction(df)

class_info_df = pd.read_csv("../model/output/class_info.csv")


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/class-info/")
def class_info():
    return class_info_df.to_json(orient="records")


@app.route("/data/")
def data():
    x_factor = float(request.args.get("factorX"))
    y_factor = float(request.args.get("factorY"))

    if x_factor is None or y_factor is None:
        or_df = apply_overlap_removal(df)
    else:
        or_df = apply_overlap_removal(df, x_factor, y_factor)

    return or_df.to_json(orient="records")


@app.route("/image/")
def image():
    image_path = request.args.get("imagePath")

    # todo not safe for public use, use send_from_directory instead
    return send_file(image_path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run()
