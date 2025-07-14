from flask import Flask, request, jsonify
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


@app.route("/data/")
def data():
    x_factor = float(request.args.get("factorX"))
    y_factor = float(request.args.get("factorY"))

    if x_factor is None or y_factor is None:
        or_df = apply_overlap_removal(df)
    else:
        or_df = apply_overlap_removal(df, x_factor, y_factor)

    data_points = or_df.to_dict(orient="records")
    class_info = class_info_df.to_dict(orient="records")

    response = {
        "data_points": data_points,
        "class_info": class_info
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run()
