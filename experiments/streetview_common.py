"""Shared paths and JSON helpers for the street-view annotation tooling."""

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AREAS = ("soho", "hudson_yards")
DATASET = "nyc_streetview_v1"


def dataset_root():
    return ROOT / "data" / "annotations" / DATASET


def read_features(inputs, area, name):
    path = Path(inputs) / area / f"{name}.geojson"
    return json.loads(path.read_bytes())["features"]


def write_json(path, payload):
    """Atomic write so interrupted runs never leave a half-written manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     encoding="utf-8", suffix=".tmp") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(handle.name, path)
