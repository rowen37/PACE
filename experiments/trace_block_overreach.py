"""Replay frozen equal-concat partitions and diagnose extra area against the BID."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

import networkx as nx
import numpy as np
from shapely import union_all
from shapely.geometry import shape
from shapely.ops import transform

from block_boundary_benchmark import REFERENCE_NAMES, ROOT, TO_METRES, predict, unit_vectors, write
from diagnose_block_overlap import block_ceiling
from fusion_partition_benchmark import DEFAULT_INPUT, block_geometry, canonical, graph_for, partition


def replay(data, run, keys):
    views = np.asarray(run["view_embeddings"], dtype=np.float32)
    weights = np.asarray(run["fusion"]["equal_concat"]["view_weights"], dtype=np.float32)
    fused = (views * np.sqrt(weights)[..., None]).reshape(len(keys), -1)
    model = run["fusion"]["equal_concat"]
    prediction = predict(keys, fused, data["adjacency"])
    if not np.array_equal(prediction["edge_scores"], model["edge_scores"]):
        raise ValueError("Replay did not reproduce the frozen edge scores")
    graph = graph_for(keys, data["adjacency"], model["edge_scores"])
    trace = {}
    current = partition(graph, prediction["candidates"], trace=trace)
    if json.loads(json.dumps(current)) != model["partitions"]:
        raise ValueError("Replay did not reproduce the frozen partitions")
    histories = {key: [(0, frozenset({key}))] for key in keys}
    events = [None]
    groups = {(key,) for key in keys}
    last_q = trace["singleton_modularity"]
    for phase in ("tree_steps", "move_steps"):
        for step in trace[phase]:
            events.append({**step, "phase": phase, "delta_q": step["modularity"] - last_q})
            last_q = step["modularity"]
            groups.difference_update(step["before"])
            groups.update(step["after"])
            for group in step["after"]:
                for k in group:
                    histories[k].append((len(events) - 1, frozenset(group)))
    if canonical(groups) != current["block_moves"]["groups"]:
        raise ValueError("Event trace did not reconstruct the final partition")
    return graph, views, weights, fused, histories, events, current["block_moves"]["start"]


def diagnose(source, inputs, output, area):
    protocol = json.loads((source / "protocol.json").read_bytes())
    prediction_raw = (source / f"{area}_predictions.json").read_bytes()
    folder = inputs / area
    for filename, digest in protocol["input_sha256"][area].items():
        if sha256((folder / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen input changed: {filename}")

    data = json.loads(prediction_raw)
    keys = data["keys"]
    index = {key: i for i, key in enumerate(keys)}
    replays = {seed: replay(data, run, keys) for seed, run in data["runs"].items()}
    # Only annotate the replay after reproducing every frozen result.
    metrics = json.loads((source / "metrics.json").read_bytes())
    reference_raw = (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes()
    reference_hash = sha256(reference_raw).hexdigest()
    area_metrics = metrics["areas"][area]
    if (sha256(prediction_raw).hexdigest() != area_metrics["predictions_sha256"]
            or reference_hash != metrics["reference_sha256"]):
        raise ValueError("Frozen prediction or BID geometry changed")
    blocks = block_geometry(folder / "blocks.geojson")
    if blocks.keys() != index.keys():
        raise ValueError("Frozen block IDs differ from the geometry")
    bid = next(f for f in json.loads(reference_raw)["features"]
               if f["properties"]["f_all_bi_2"] == REFERENCE_NAMES[area])
    fence = transform(TO_METRES, shape(bid["geometry"]))
    sizes = {k: g.area for k, g in blocks.items()}
    overlap = {k: g.intersection(fence).area for k, g in blocks.items()}
    reference = union_all(list(blocks.values())).intersection(fence).area
    if not math.isclose(sum(overlap.values()), reference):
        raise ValueError("Block intersection areas do not sum to the reference area")
    ceiling = block_ceiling(sizes, overlap, reference)
    poi = {key: Counter() for key in keys}
    for feature in json.loads((folder / "places.geojson").read_bytes())["features"]:
        p = feature["properties"]
        poi[p["block_id"]][p["category"].split("/", 1)[0]] += 1
    coords = {f["properties"]["block_id"]: list(shape(f["geometry"]).representative_point().coords[0])
              for f in json.loads((folder / "blocks.geojson").read_bytes())["features"]}
    fields = ("poi", "building", "walk")
    runs = {}
    for seed, (graph, views, weights, fused, histories, events, start) in replays.items():
        metric = area_metrics["runs"]["equal_concat/block_moves"][seed]
        matched = metric["matched_group"]
        anchors = {k for k in matched if overlap[k] > 0}
        outside = set(matched) - anchors
        extra = sum(sizes[k] - overlap[k] for k in matched)
        outside_area = sum(sizes[k] for k in outside)
        if not math.isclose(extra, metric["extra_ha"] * 10_000):
            raise ValueError("Extra area does not match the prior benchmark")
        sims = np.einsum("ivd,jvd->ijv", views, views, dtype=np.float64)
        unit = unit_vectors(fused)
        fused_sims = unit @ unit.T
        reference_edges = [(a, b) for a, b in graph.edges if a in anchors and b in anchors]

        def values(a, b):
            i, j = index[a], index[b]
            return [float(sims[i, j, k]) if weights[i, k] and weights[j, k] else None
                    for k in range(len(fields))] + [float(fused_sims[i, j])]

        edge_vals = [values(a, b) for a, b in reference_edges]
        internal = [sorted(row[k] for row in edge_vals if row[k] is not None) for k in range(len(fields) + 1)]

        def compare_edge(a, b):
            scores = values(a, b)
            return {"blocks": [a, b], "cosine": dict(zip((*fields, "fused"), scores, strict=True)),
                    "above_internal_median": {name: (v > median(base) if v is not None and base else None)
                                              for name, v, base in zip((*fields, "fused"), scores, internal, strict=True)}}

        entries = []
        for key in sorted(outside):
            history = histories[key]
            # Earliest association with any eventual BID-overlapping member
            # after which the block never loses contact with all such members.
            detached = max(i for i, (_, group) in enumerate(history) if not group & anchors)
            event_id, group = history[detached + 1]
            event = events[event_id]
            previous = history[detached][1]
            new_peers = group - previous
            # A tree join attaches all edges across its two child groups; a
            # local move may instead be made by the BID-overlapping block.
            joining = [(a, b) for a in sorted(previous & group) for b in sorted(new_peers) if graph.has_edge(a, b)]
            links = [compare_edge(key, k) for k in sorted(group & anchors & set(graph[key]))]
            entries.append({"block": key, "area_m2": sizes[key], "poi_counts": dict(poi[key].most_common()),
                            "lon_lat": coords[key], "event_id": event_id, "event": event,
                            "overlapping_members_at_entry": sorted(group & anchors),
                            "joining_edges": [compare_edge(a, b) for a, b in joining],
                            "direct_edges_to_overlapping_members_at_entry": links,
                            "direct_edges_to_overlapping_members_final": sorted(anchors & set(graph[key]))})
        entries.sort(key=lambda entry: (entry["event_id"], entry["block"]))
        runs[seed] = {"iou": metric["iou"], "start": start, "matched_group": matched,
                      "fully_outside_area_m2": outside_area, "outside_portions_of_intersecting_blocks_m2": extra - outside_area,
                      "internal_edges": len(reference_edges),
                      "internal_edge_medians": dict(zip((*fields, "fused"),
                                                        (median(x) if x else None for x in internal), strict=True)),
                      "entries": entries}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "diagnosis.json", {"area": area, "reference": REFERENCE_NAMES[area], "reference_sha256": reference_hash,
                                     "predictions_sha256": sha256(prediction_raw).hexdigest(),
                                     "whole_block_oracle": {**ceiling, "walk_connected": nx.is_connected(
                                         replays[next(iter(replays))][0].subgraph(ceiling["blocks"]))}, "runs": runs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--inputs", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--area", default="soho", choices=REFERENCE_NAMES)
    args = parser.parse_args()
    diagnose(args.source, args.inputs, args.output, args.area)


if __name__ == "__main__":
    main()
