import json
import os
import pickle

#add some functions to save and load files

def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path: str, payload) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_pickle(path: str, payload) -> None:
    ensure_parent(path)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)