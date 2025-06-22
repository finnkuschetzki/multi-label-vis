import os
from flask import Flask

from process_data import *


# checking for dimensionality reduction data
if not os.path.exists("data/dimensionality_reduction.csv"):
    print()
    process_data()  # generates and saves dimensionality reduction data
    print()

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route("/data/")
def data():
    return get_data_as_json()


if __name__ == '__main__':
    app.run()
