import json


with open("_config.json") as f:
    config = json.load(f)


__all__ = [
    "config"
]
