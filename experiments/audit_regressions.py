"""Offline audit checks; no training, network calls, or writes to source data."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "archive/2026-11-11/experiments"
sys.path.insert(0, str(LEGACY / "step3"))

import numpy as np
import torch
import rgscl


def view(area, building_ids):
    return SimpleNamespace(
        block=SimpleNamespace(area=area, key="b001", buildings=dict.fromkeys(building_ids)),
        visible=[("shop", building_ids[0])], visible_coordinates=[(0.0, 0.0)],
    )


def main():
    train = view("train_district", [1])
    test = view("test_district", [2, 3])
    geometry = {i: (np.array([float(i), 0.0]), np.full(8, float(i))) for i in (1, 2, 3)}
    pairs, _ = rgscl.make_graphs([train], [test, test], geometry)
    assert pairs[0][1].num_nodes == 2
    assert pairs[1][1].num_nodes == 3, "Cross-district cache collision"
    assert pairs[1][1] is pairs[2][1], "Same-district repeated masks should share static graph"
    assert torch.equal(pairs[1][1].x[:2], torch.tensor([[1.0] * 8, [2.0] * 8]))
    # Evaluation geometry must not affect the training scaler/graph.
    changed = {**geometry, 2: (geometry[2][0], np.full(8, 10000.0))}
    again, _ = rgscl.make_graphs([train], [test], changed)
    assert torch.equal(pairs[0][1].x, again[0][1].x)
    real_checks = {}
    for training_area, test_area in (("hudson_yards", "soho"), ("soho", "hudson_yards")):
        training, train_geometry = rgscl.load_area(training_area)
        testing, test_geometry = rgscl.load_area(test_area)
        train_views = [rgscl.make_view(b, 0) for b in training]
        test_views = [v for b in testing for seed in rgscl.TEST_MASK_SEEDS
                      if (v := rgscl.make_view(b, seed)).target is not None]
        graphs, _ = rgscl.make_graphs(train_views, test_views, train_geometry | test_geometry)
        training_ids = sorted({bid for v in train_views for bid in v.block.buildings})
        shapes = np.array([train_geometry[bid][1] for bid in training_ids])
        center, scale = shapes.mean(axis=0), shapes.std(axis=0)
        scale[scale == 0] = 1
        for v, (_, graph) in zip(test_views, graphs[len(train_views):], strict=True):
            assert graph.num_nodes == len(v.block.buildings) + 1
            footprints = [test_geometry[bid] for bid in v.block.buildings]
            features = np.array([shape for _, shape in footprints]).reshape(-1, len(center))
            expected = rgscl.region_graph((features - center) / scale, [point for point, _ in footprints])
            assert torch.equal(graph.x, expected.x), "Incorrect building features"
            assert torch.equal(graph.edge_index, expected.edge_index), "Incorrect building edges"
        real_checks[test_area] = {"views": len(test_views), "blocks": len({v.block.key for v in test_views})}
    raw = ROOT / "data/raw/nyc/2026-11-09"
    manifest = json.loads((raw / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == expected
    result = subprocess.run([sys.executable, str(LEGACY / "step1/fetch.py")], capture_output=True, text=True)
    assert result.returncode == 2 and "Source snapshot already exists" in result.stderr
    import_code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import run; "
        "assert str(run.__file__).endswith('step3/run.py')"
    )
    subprocess.run([sys.executable, "-c", import_code, str(LEGACY / "step3")], check=True)
    for script in ("step2/embedding.py", "step3/run.py", "step3/rgscl.py", "step3/building_auxiliary.py"):
        subprocess.run([sys.executable, str(LEGACY / script), "--help"], check=True, capture_output=True)
    # Check historical metrics and predictions have not been overwritten.
    migration = json.loads((ROOT / "archive/2026-11-11/migration_manifest.json").read_text())
    checked = 0
    for item in migration["files"]:
        if "/outputs/" in item["current"] and not item["current"].endswith(".DS_Store"):
            assert hashlib.sha256((ROOT / item["current"]).read_bytes()).hexdigest() == item["sha256"]
            checked += 1
    print(json.dumps({"status": "passed", "synthetic_cache_and_scaler": "passed",
                      "real_test_graphs": real_checks, "raw_files_verified": len(manifest["files"]),
                      "historical_outputs_verified": checked, "fetch_overwrite_guard": "passed",
                      "step3_run_import_and_four_cli_entries": "passed"}, indent=2))


if __name__ == "__main__":
    main()
