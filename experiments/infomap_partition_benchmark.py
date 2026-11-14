"""Compare standard two-level Infomap with modularity on frozen equal-concat graphs."""

import argparse
from collections import defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
from statistics import mean

from infomap import Network, Options
import networkx as nx
from shapely import union_all
from shapely.geometry import shape
from shapely.ops import transform

from block_boundary_benchmark import AREAS, REFERENCE_NAMES, ROOT, TO_METRES, write
from fusion_partition_benchmark import DEFAULT_INPUT, block_geometry, canonical, graph_for


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    previous = json.loads((source / "protocol.json").read_bytes())
    protocol = {"algorithm": "standard two-level undirected Infomap", "version": version("infomap"),
                "flow": "P(u,v) = w(u,v) / sum-neighbor weights; Markov time 1 (default)",
                "edge_weights": previous["adjacency_weight"], "seeds": previous["seeds"],
                "input_sha256": previous["input_sha256"], "source_sha256": {}}
    for area in AREAS:
        raw = (source / f"{area}_predictions.json").read_bytes()
        protocol["source_sha256"][area] = sha256(raw).hexdigest()
        data = json.loads(raw)
        keys = data["keys"]
        node_ids = {k: i for i, k in enumerate(keys)}
        rows = {}
        for seed, run in data["runs"].items():
            f = run["fusion"]["equal_concat"]
            graph = graph_for(keys, data["adjacency"], f["edge_scores"])
            network = Network()
            for i in range(len(keys)):
                network.add_node(i)
            for a, b, w in graph.edges(data="weight"):
                network.add_link(node_ids[a], node_ids[b], w)
            result = network.run(options=Options(flow_model="undirected", two_level=True, seed=int(seed)))
            modules = defaultdict(list)
            labels = result.modules()
            for i, module in labels.items():
                modules[module].append(keys[i])
            missing = set(range(len(keys))) - labels.keys()
            if any(graph.degree(keys[i], weight="weight") > 0 for i in missing):
                raise ValueError("Infomap omitted a non-isolated node")
            groups = canonical([*modules.values(), *((keys[i],) for i in missing)])
            if sorted(k for g in groups for k in g) != sorted(keys) or not all(
                    nx.is_connected(graph.subgraph(g)) for g in groups):
                raise ValueError("Partition must cover each block once and groups must be walk-connected")
            transitions = {}
            for k in ("b014", "b023") if area == "soho" else ():
                total = graph.degree(k, weight="weight")
                if total:
                    transitions[k] = {v: {"weight": attrs["weight"], "probability": attrs["weight"] / total}
                                      for v, attrs in graph[k].items()}
            rows[seed] = {"infomap": groups, "modularity": f["partitions"]["block_moves"]["groups"],
                          "infomap_codelength": result.codelength,
                          "infomap_one_module_codelength": result.one_level_codelength,
                          "example_transitions": transitions}
        write(output / f"{area}_partitions.json", {"runs": rows})
    write(output / "protocol.json", protocol)


def score(source, inputs, output):
    protocol = json.loads((output / "protocol.json").read_bytes())
    frozen = {a: (output / f"{a}_partitions.json").read_bytes() for a in AREAS}
    prior = json.loads((source / "metrics.json").read_bytes())
    bid_raw = (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes()
    digest = sha256(bid_raw).hexdigest()
    if prior["reference_sha256"] != digest:
        raise ValueError("BID reference changed")
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(bid_raw)["features"]}
    summary = {"reference_sha256": digest, "areas": {}}
    for area in AREAS:
        old = prior["areas"][area]
        path = inputs / area / "blocks.geojson"
        if (old["predictions_sha256"] != protocol["source_sha256"][area]
                or sha256(path.read_bytes()).hexdigest() != protocol["input_sha256"][area][path.name]):
            raise ValueError("Frozen blocks or baseline predictions changed")
        blocks = block_geometry(path)
        reference = fences[REFERENCE_NAMES[area]].intersection(union_all(list(blocks.values())))
        sizes = {k: g.area for k, g in blocks.items()}
        overlaps = {k: g.intersection(reference).area for k, g in blocks.items()}
        reference_area = reference.area

        def evaluate(group):
            hit = sum(overlaps[k] for k in group)
            size = sum(sizes[k] for k in group)
            return {"iou": hit / (size + reference_area - hit),
                    "extra_ha": (size - hit) / 10000, "missed_ha": (reference_area - hit) / 10000,
                    "wholly_outside_ha": sum(sizes[k] for k in group if not overlaps[k]) / 10000}

        stats = defaultdict(dict)
        for seed, row in json.loads(frozen[area])["runs"].items():
            for method in ("modularity", "infomap"):
                group = max(row[method], key=lambda g: evaluate(g)["iou"])
                result = evaluate(group)
                if method == "modularity":
                    previous = old["runs"]["equal_concat/block_moves"][seed]
                    if group != previous["matched_group"] or not math.isclose(result["iou"], previous["iou"]):
                        raise ValueError("Baseline does not reproduce the earlier BID match")
                stats[method][seed] = {**result, "matched_group": group, "groups": len(row[method])}
        fields = ("iou", "extra_ha", "missed_ha", "wholly_outside_ha", "groups")
        summary["areas"][area] = {"seeds": stats, "mean": {
            method: {field: mean(r[field] for r in runs.values()) for field in fields} for method, runs in stats.items()}}
    write(output / "metrics.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--inputs", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.source, args.output)
    else:
        score(args.source, args.inputs, args.output)
