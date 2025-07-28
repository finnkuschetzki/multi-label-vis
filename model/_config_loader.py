import json


with open("_config.json") as f:
    config = json.load(f)
    print(config)


__all__ = [
    "config"
]
