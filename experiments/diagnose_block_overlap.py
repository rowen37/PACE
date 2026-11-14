"""Score the existing BID matches and attribute IoU loss without selecting again."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean

import networkx as nx
from shapely import union_all
from shapely.geometry import shape
from shapely.ops import transform

from block_boundary_benchmark import AREAS, REFERENCE_NAMES, ROOT, TO_METRES, map_inputs
from select_block_partition import write


def overlap_metrics(predicted, reference, overlap):
    return {"predicted_m2": predicted, "reference_m2": reference, "overlap_m2": overlap,
            "extra_m2": predicted - overlap, "missed_m2": reference - overlap,
            "precision": overlap / predicted, "recall": overlap / reference,
            "dice": 2 * overlap / (predicted + reference),
            "iou": overlap / (predicted + reference - overlap)}


def block_ceiling(sizes, overlaps, reference):
    """Exact best whole-block union: an optimum is a prefix by overlap fraction."""
    order = sorted(sizes, key=lambda k: (-overlaps[k] / sizes[k], k))
    chosen = []
    best = None
    for key in order:
        chosen.append(key)
        scores = overlap_metrics(sum(sizes[k] for k in chosen), reference, sum(overlaps[k] for k in chosen))
        if best is None or scores["iou"] > best["iou"]:
            best = {**scores, "blocks": sorted(chosen)}
    return best


def diagnose(source, selected, output):
    original_raw = (source / "metrics.json").read_bytes()
    cut_metrics_raw = (selected / "metrics.json").read_bytes()
    cut_raw = (selected / "selection.json").read_bytes()
    original, scores, selection = map(json.loads, (original_raw, cut_metrics_raw, cut_raw))
    if sha256(cut_raw).hexdigest() != scores["selection_sha256"]:
        raise ValueError("Selected partitions differ from the scored partitions")
    bid_raw = (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes()
    bid_hash = sha256(bid_raw).hexdigest()
    if not bid_hash == original["reference_sha256"] == scores["reference_sha256"]:
        raise ValueError("BID reference geometry changed")
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(bid_raw)["features"]}
    result = {"original_metrics_sha256": sha256(original_raw).hexdigest(),
              "selected_metrics_sha256": sha256(cut_metrics_raw).hexdigest(),
              "selection_sha256": scores["selection_sha256"], "reference_sha256": bid_hash, "areas": {}}
    for area in AREAS:
        frozen_raw = (source / f"{area}_predictions.json").read_bytes()
        before, cuts = original["areas"][area], selection["areas"][area]
        if not sha256(frozen_raw).hexdigest() == before["predictions_sha256"] == cuts["predictions_sha256"]:
            raise ValueError(f"Prior predictions changed: {area}")
        frozen = json.loads(frozen_raw)
        records, adjacency, _ = map_inputs(area)
        if list(map(tuple, frozen["adjacency"])) != adjacency or set(records) != set(frozen["keys"]):
            raise ValueError(f"Blocks or walk adjacency changed: {area}")
        blocks = {k: transform(TO_METRES, shape(f["geometry"])) for k, f in records.items()}
        sizes = {k: b.area for k, b in blocks.items()}
        extent = union_all(list(blocks.values()))
        if not math.isclose(sum(sizes.values()), extent.area, rel_tol=1e-9):
            raise ValueError(f"Overlapping atomic block interiors: {area}")
        reference = fences[REFERENCE_NAMES[area]].intersection(extent).area
        overlaps = {k: b.intersection(fences[REFERENCE_NAMES[area]]).area for k, b in blocks.items()}
        ceiling = block_ceiling(sizes, overlaps, reference)
        graph = nx.Graph(adjacency)
        graph.add_nodes_from(blocks)
        ceiling["walk_connected"] = nx.is_connected(graph.subgraph(ceiling["blocks"]))
        ceiling["overlapping_blocks"] = sum(v > 0 for v in overlaps.values())
        ceiling["partly_overlapping_blocks"] = sum(0 < v < sizes[k] - 1e-8 for k, v in overlaps.items())

        def evaluate(group):
            return overlap_metrics(sum(sizes[k] for k in group), reference, sum(overlaps[k] for k in group))

        methods = {}
        for name, method in scores["areas"][area]["methods"].items():
            rows = {}
            for seed, run in method["runs"].items():
                cut = cuts["methods"][name][seed]
                if run["matched_group"] not in cut["partition"]:
                    raise ValueError(f"Matched group is not in the frozen partition: {area}/{name}/{seed}")
                actual = evaluate(run["matched_group"])
                tree = evaluate(before["methods"][name]["runs"][seed]["best_candidate"])
                if not all(math.isclose(v, expected, abs_tol=1e-12) for v, expected in (
                    (actual["iou"], run["selected_cut_iou"]), (tree["iou"], run["tree_oracle_iou"])
                )):
                    raise ValueError(f"IoU no longer matches the previous run: {area}/{name}/{seed}")
                rows[seed] = {**actual, "ap": run["ap"], "tree_best": tree,
                              "block_granularity_loss": 1 - ceiling["iou"],
                              "tree_loss": ceiling["iou"] - tree["iou"],
                              "cut_loss": tree["iou"] - actual["iou"]}
            fields = ("ap", "precision", "recall", "dice", "iou", "block_granularity_loss", "tree_loss", "cut_loss")
            methods[name] = {**{k: mean(r[k] for r in rows.values()) for k in fields}, "runs": rows}
        result["areas"][area] = {"block_ceiling": ceiling, "methods": methods}
    result["area_macro_average"] = {
        name: {k: mean(a["methods"][name][k] for a in result["areas"].values()) for k in fields}
        for name in result["areas"][AREAS[0]]["methods"]}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "metrics.json", result)
    print(json.dumps(result["area_macro_average"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source", "selected", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args()
    diagnose(args.source, args.selected, args.output)


if __name__ == "__main__":
    main()
