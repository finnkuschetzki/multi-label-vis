import os
import sys


def get_model_dir():

    # checking command line arguments
    if len(sys.argv) < 2:
        MODEL_DIR = "output"
    else:
        MODEL_DIR = sys.argv[1]

    # checking model directory
    if not os.path.exists(MODEL_DIR):
        print("Model directory doesn't exist", file=sys.stderr)
        sys.exit(1)
    elif not os.path.isfile(f"{MODEL_DIR}/fine_tuned.keras"):
        print("Fine tuned model doesn't exist", file=sys.stderr)
        sys.exit(1)

    return MODEL_DIR


__all__ = [
    "get_model_dir",
]
