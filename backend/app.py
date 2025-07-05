from flask import Flask
from flask_cors import CORS

from process_data import *


df = read_csv_with_list_attributes("../model/output/embedding_data.csv", ["ground_truth", "features", "predictions", "binarized_predictions"])
df = apply_dimensionality_reduction(df)


app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/data/")
def data():
    or_df = apply_overlap_removal(df)
    return or_df.to_json(orient="records")


if __name__ == "__main__":
    app.run()
