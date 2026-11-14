"""Compare whole-block representations against held-back commercial boundaries."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from pyproj import Transformer
from shapely import union_all
from shapely.geometry import shape
from shapely.ops import transform
from torch import nn
from torch_geometric.data import Batch

from pedestrian_rgscl import (AREAS, HEADS, ROOT, WIDTH, ThreeViewRGSCL, baseline,
                             batch, observed_views, walk_graph)

EPOCHS = 50
SEEDS = (13, 29, 53)
REFERENCE_NAMES = {"hudson_yards": "Hudson Yards Hells Kitchen Alliance", "soho": "SoHo Broadway"}
STEP2_OUTPUT = baseline.STEP1.parent / "step2" / "outputs"
TO_METRES = Transformer.from_crs(4326, 6538, always_xy=True).transform
WALK_STATS = ("log_walk_m_per_hectare", "log_junctions_per_hectare", "dead_end_ratio", "main_component_share")


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def map_inputs(area):
    folder = STEP2_OUTPUT / area
    blocks = {f["properties"]["block_id"]: f for f in baseline.read(folder / "blocks.geojson")["features"]
              if f["properties"]["eligible"]}
    adjacency = [tuple(sorted(f["properties"]["blocks"]))
                 for f in baseline.read(folder / "adjacency.geojson")["features"]
                 if f["properties"]["walk_connected"]]
    edges = [f["properties"] for f in baseline.read(folder / "network.geojson")["features"]
             if f["properties"]["kind"] == "walk"]
    degree = Counter(node for e in edges for node in (e["u"], e["v"]))
    junctions = defaultdict(set)
    for e in edges:
        junctions[e.get("block")].update(node for node in (e["u"], e["v"]) if degree[node] >= 3)
    return blocks, sorted(adjacency), junctions


def statistics(blocks, geometry, vocab, maps):
    rows = []
    for b in blocks:
        records, _, junctions = maps[b.area]
        counts = np.zeros(len(vocab))
        for category, _ in b.places:
            counts[vocab.get(category, vocab[None])] += 1
        poi = [np.log1p(len(b.places) / b.hectares), *np.sqrt(counts / max(1, len(b.places)))]
        morphology = np.asarray([geometry[k][1] for k in b.buildings], dtype=float).reshape(-1, 8)
        buildings = [np.log1p(len(morphology) / b.hectares),
                     *(morphology.mean(axis=0) if len(morphology) else np.zeros(8))]
        line_graph = nx.Graph(b.walk_pairs)
        line_graph.add_nodes_from(range(len(b.walk_features)))
        lengths = np.expm1(5 * np.array([f[0] for f in b.walk_features]))
        largest = max((lengths[list(c)].sum() for c in nx.connected_components(line_graph)), default=0)
        walk = [np.log1p(b.walk_length / b.hectares), np.log1p(len(junctions[b.key]) / b.hectares),
                records[b.key]["properties"]["dead_end_ratio"], largest / b.walk_length if b.walk_length else 0]
        rows.append((poi, buildings, walk))
    return tuple(np.asarray(col, dtype=float) for col in zip(*rows, strict=True))


def unit_vectors(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1)


def standardized_groups(training, testing):
    for a, b in zip(training, testing, strict=True):
        std = a.std(axis=0)
        yield unit_vectors((b - a.mean(axis=0)) / np.where(std > 0, std, 1))


class EqualWeights(nn.Module):
    def forward(self, views, available):
        weights = available.to(views[0].dtype)
        weights = weights / weights.sum(dim=1, keepdim=True)
        return (weights[..., None] * torch.stack(views, dim=1)).sum(dim=1), weights


def no_walk(inputs):
    poi, building, walk = inputs
    return poi, building, Batch.from_data_list([walk_graph([], []) for _ in range(walk.num_graphs)])


def fit_embeddings(training, testing, vocab_dim, encoder, seed, method):
    torch.manual_seed(seed)
    model = ThreeViewRGSCL(encoder, vocab_dim, width=WIDTH, heads=HEADS)
    test_count = testing[0].num_graphs
    supported = None
    if method == "rgscl_two_view":
        poi, buildings, _ = testing
        supported = ((poi.ptr.diff() > 1) | (buildings.ptr.diff() > 1)).nonzero().flatten().tolist()
        testing = tuple(Batch.from_data_list(g.index_select(supported)) for g in testing)
        training, testing = no_walk(training), no_walk(testing)
    if method == "three_view_equal":
        model.fusion = EqualWeights()
    optim = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=2e-5)
    model.train()
    for _ in range(EPOCHS):
        optim.zero_grad()
        loss = model.loss(*training)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in {method} seed {seed}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optim.step()
    model.eval()
    with torch.no_grad():
        encoded = model(*testing)[0].numpy()
    if not np.isfinite(encoded).all():
        raise RuntimeError(f"Non-finite embeddings in {method} seed {seed}")
    if supported is not None:
        complete = np.zeros((test_count, WIDTH), dtype=encoded.dtype)
        complete[supported] = encoded
        encoded = complete
    return unit_vectors(encoded)


def predict(keys, vectors, adjacency):
    """Produce edge scores and the full adjacency-constrained complete-link tree."""
    lookup = {k: i for i, k in enumerate(keys)}
    vectors = unit_vectors(vectors)
    similarities = vectors @ vectors.T
    graph = nx.Graph(adjacency)
    graph.add_nodes_from(keys)
    active = {(k,): set(graph[k]) for k in keys}
    candidates = sorted(active)
    while True:
        pair = min(((-min(similarities[lookup[u], lookup[v]] for u in left for v in right), left, right)
                    for left, right in combinations(sorted(active), 2) if active[left].intersection(right)),
                   default=None)
        if pair is None:
            break
        _, left, right = pair
        merged = tuple(sorted(left + right))
        active[merged] = (active.pop(left) | active.pop(right)) - set(merged)
        candidates.append(merged)
    return {"edge_scores": [float(similarities[lookup[u], lookup[v]]) for u, v in adjacency],
            "candidates": candidates}


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    encoder = baseline.load_author_model().Encoder
    maps = {area: map_inputs(area) for area in AREAS}
    data = {area: baseline.load_area(area) for area in AREAS}
    protocol = {"epochs": EPOCHS, "seeds": SEEDS, "width": WIDTH, "heads": HEADS, "learning_rate": 0.0002,
                "weight_decay": 2e-5, "walk_statistics": WALK_STATS, "merge": "complete link, full tree"}
    write(output / "protocol.json", protocol)
    for training_area, test_area in (AREAS, AREAS[::-1]):
        training, train_geometry = data[training_area]
        testing, test_geometry = data[test_area]
        if any(baseline.shared_entities(training, testing)):
            raise ValueError("Training and holdout share source entities")
        # Keep the same training blocks in every arm. A block with neither
        # base view has no two-view embedding; retain it at test with zero score.
        training = [b for b in training if b.places or b.buildings]
        train_views, test_views = observed_views(training), observed_views(testing)
        geometry = train_geometry | test_geometry
        pairs, dim = baseline.make_graphs(train_views, test_views, geometry)
        training_graphs = batch(pairs[:len(training)], training)
        testing_graphs = batch(pairs[len(training):], testing)
        vocab = baseline.training_vocabulary(train_views)
        features = tuple(standardized_groups(statistics(training, geometry, vocab, maps),
                                              statistics(testing, geometry, vocab, maps)))
        keys = [b.key for b in testing]
        records, adjacency, _ = maps[test_area]
        assert set(keys) == records.keys()
        predictions = {"map_statistics": {"fixed": predict(keys, np.column_stack(features), adjacency)}}
        for method in ("rgscl_two_view", "rgscl_walk_concat", "three_view_equal", "three_view_attention"):
            predictions[method] = {}
        for seed in SEEDS:
            for method in ("rgscl_two_view", "three_view_equal", "three_view_attention"):
                vectors = fit_embeddings(training_graphs, testing_graphs, dim, encoder, seed, method)
                predictions[method][str(seed)] = predict(keys, vectors, adjacency)
                if method == "rgscl_two_view":
                    predictions["rgscl_walk_concat"][str(seed)] = predict(
                        keys, np.column_stack((vectors, features[2])), adjacency)
            print(f"{training_area} -> {test_area}, seed {seed}: predictions ready", flush=True)
        write(output / f"{test_area}_predictions.json", {"training_area": training_area,
                                                         "training_blocks": len(training), "keys": keys,
                                                         "missing_both_base_views": [b.key for b in testing
                                                                                     if not (b.places or b.buildings)],
                                                         "adjacency": adjacency, "predictions": predictions})


def average_precision(labels, scores):
    """Step-wise AP, grouping equal scores before computing precision."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    y, s = np.asarray(labels)[order], np.asarray(scores)[order]
    if not 0 < y.sum() < len(y):
        raise ValueError("Need both internal and crossing edges to score")
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    hits = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, hits]) * hits / (ends + 1)) / hits[-1])


def score(output):
    # Only this phase opens the BID labels, after both prediction files exist.
    files = {a: output / f"{a}_predictions.json" for a in AREAS}
    frozen = {a: f.read_bytes() for a, f in files.items()}
    boundary = ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson"
    source = boundary.read_bytes()
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(source)["features"]}
    result = {"reference_sha256": sha256(source).hexdigest(), "areas": {}}
    for area in AREAS:
        payload = json.loads(frozen[area])
        records, adjacency, _ = map_inputs(area)
        assert adjacency == list(map(tuple, payload["adjacency"]))
        blocks = {k: transform(TO_METRES, shape(f["geometry"])) for k, f in records.items()}
        fence = fences[REFERENCE_NAMES[area]]
        reference = fence.intersection(union_all(list(blocks.values())))
        inside = {k for k, block in blocks.items() if fence.covers(block.centroid)}
        support = [i for i, (u, v) in enumerate(adjacency) if u in inside or v in inside]
        labels = [int(adjacency[i][0] in inside and adjacency[i][1] in inside) for i in support]
        intersect = {k: block.intersection(reference).area for k, block in blocks.items()}

        def iou(candidate):
            inter = sum(intersect[k] for k in candidate)
            # Atomic blocks have disjoint interiors.
            return inter / (sum(blocks[k].area for k in candidate) + reference.area - inter)

        methods = {}
        for method, reps in payload["predictions"].items():
            rows = {}
            for seed, predictions in reps.items():
                ap = average_precision(labels, [predictions["edge_scores"][i] for i in support])
                best = max(predictions["candidates"], key=iou)
                rows[seed] = {"ap": ap, "best_candidate_iou": iou(best), "best_candidate": best}
            methods[method] = {"mean_ap": float(np.mean([r["ap"] for r in rows.values()])),
                               "mean_best_candidate_iou": float(np.mean([r["best_candidate_iou"] for r in rows.values()])),
                               "runs": rows}
        result["areas"][area] = {"training_area": payload["training_area"],
                                 "training_blocks": payload["training_blocks"], "bid": REFERENCE_NAMES[area],
                                 "missing_both_base_views": payload["missing_both_base_views"],
                                 "positive_edges": sum(labels), "negative_edges": len(labels) - sum(labels),
                                 "predictions_sha256": sha256(frozen[area]).hexdigest(), "methods": methods}
    method_names = result["areas"][AREAS[0]]["methods"].keys()
    result["area_macro_average"] = {
        method: {metric: float(np.mean([row["methods"][method][metric] for row in result["areas"].values()]))
                 for metric in ("mean_ap", "mean_best_candidate_iou")} for method in method_names}
    write(output / "metrics.json", result)
    print(json.dumps(result["area_macro_average"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else score)(args.output)


if __name__ == "__main__":
    main()
