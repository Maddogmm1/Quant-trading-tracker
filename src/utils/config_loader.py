"""Loads config/config.yaml. Nothing in the pipeline should hardcode a
threshold that belongs here."""
import yaml
import os


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)
