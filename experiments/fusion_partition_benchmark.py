"""Freeze independent views, compare fusion and connected partitions, then score."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import networkx as nx
import torch
from shapely import union_all
from shapely.geometry import shape
from shapely.ops import transform
from torch import nn
from torch.nn import functional as F

from block_boundary_benchmark import (EPOCHS, REFERENCE_NAMES, SEEDS, TO_METRES,
                                      average_precision, predict, write)
from pedestrian_rgscl import AREAS, HEADS, ROOT, WIDTH, baseline, batch, observed_views
from pace.pedestrian_rgscl import VIEW_NAMES, WALK_FEATURES
from torch_geometric.nn import global_mean_pool

DEFAULT_INPUT = ROOT / "data/processed/nyc-2026-11-09-block-inputs"
INPUT_FILES = ("blocks.geojson", "places.geojson", "buildings.geojson", "walk.geojson", "adjacency.json")
LR, WEIGHT_DECAY = 0.0002, 2e-5


def block_geometry(path):
    return {f["properties"]["block_id"]: transform(TO_METRES, shape(f["geometry"]))
            for f in json.loads(path.read_bytes())["features"]}


def read_area(folder, area):
    files = {name: (folder / area / name).read_bytes() for name in INPUT_FILES}
    layers = {name: json.loads(raw) for name, raw in files.items()}
    keys = [f["properties"]["block_id"] for f in layers["blocks.geojson"]["features"]]
    blocks = {k: SimpleNamespace(area=area, key=k, places=[], place_coordinates=[], buildings=[],
                                 walk_features=[], walk_pairs=[], source_walk_edges=set(), poi_ids=set())
              for k in keys}
    geometry = {}
    for f in layers["buildings.geojson"]["features"]:
        p = f["properties"]
        key = p["building_id"]
        geometry[key] = baseline.morphology(transform(TO_METRES, shape(f["geometry"])))
        for block in p["blocks"]:
            blocks[block].buildings.append(key)
    for f in layers["places.geojson"]["features"]:
        p = f["properties"]
        block = blocks[p["block_id"]]
        block.places.append((p["category"], None))
        block.poi_ids.add(p["place_id"])
        block.place_coordinates.append(transform(TO_METRES, shape(f["geometry"])).coords[0])
    incident = defaultdict(lambda: defaultdict(list))
    for f in layers["walk.geojson"]["features"]:
        p = f["properties"]
        block = blocks[p["block_id"]]
        line = transform(TO_METRES, shape(f["geometry"]))
        a, b = line.coords[0], line.coords[-1]
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        i = len(block.walk_features)
        block.walk_features.append((math.log1p(p["length_m"]) / 5, math.cos(2 * angle), math.sin(2 * angle)))
        block.source_walk_edges.add(p["source_id"].split(":", 1)[0])
        for node in {p["u"], p["v"]}:
            incident[block.key][node].append(i)
    for k, block in blocks.items():
        block.walk_pairs = sorted({pair for incident_edges in incident[k].values()
                                   for pair in combinations(incident_edges, 2)})
    edges = [tuple(r["blocks"]) for r in layers["adjacency.json"]]
    hashes = {name: sha256(raw).hexdigest() for name, raw in files.items()}
    return list(blocks.values()), geometry, edges, hashes


def normalize(x):
    length = x.norm(dim=-1, keepdim=True)
    return x / torch.where(length == 0, 1, length)


def observed(graph, kind):
    return graph.ptr.diff() > int(kind != 2)


def readout(encoder, graph, kind, augmented=False):
    h = encoder(graph.x, graph.edge_index, auger=augmented)
    return h[graph.dummy] if kind != 2 else global_mean_pool(h, graph.batch, size=graph.num_graphs)


def paired_loss(left, right):
    labels = torch.arange(len(left))
    logits = normalize(left) @ normalize(right).T
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def fit_views(train, test, poi_dim, encoder_class, seed):
    normal, perturbed, holdout = [], [], []
    for kind, (g, h, dim) in enumerate(zip(train, test, (poi_dim, 8, len(WALK_FEATURES)), strict=True)):
        torch.manual_seed(seed + kind)
        encoder = encoder_class(dim, WIDTH, 1, "gat", HEADS, "prelu")
        optimizer = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        present = observed(g, kind)
        if present.sum() <= 1:
            raise ValueError("A view needs at least two observed training blocks for contrastive learning")
        encoder.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = paired_loss(readout(encoder, g, kind)[present], readout(encoder, g, kind, True)[present])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite view loss")
            loss.backward()
            optimizer.step()
        encoder.eval()
        with torch.no_grad():
            normal.append(normalize(readout(encoder, g, kind)))
            perturbed.append(normalize(readout(encoder, g, kind, True)))
            holdout.append(normalize(readout(encoder, h, kind)))
    return tuple(torch.stack(v, dim=1) for v in (normal, perturbed, holdout))


def masked_weights(scores, present):
    available = present.any(dim=1, keepdim=True)
    return torch.softmax(scores.masked_fill(~present & available, -torch.inf), dim=1) * present


def weighted_concat(views, weights):
    # Weight represents each view's share of the squared norm. Mask empty
    # views without taking sqrt(0) along the differentiable weight path.
    nonzero = weights > 0
    safe = torch.where(nonzero, weights, 1).sqrt() * nonzero
    return (views * safe[..., None]).flatten(1)


class FusionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Linear(3 * WIDTH, 3)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, views, present):
        clean = views * present[..., None]
        weights = masked_weights(self.score(clean.flatten(1)), present)
        return weighted_concat(clean, weights), weights


def fuse(train, augmented, test, train_present, test_present):
    model = FusionAttention()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    active = train_present.any(dim=1)
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        left, _ = model(train[active], train_present[active])
        right, _ = model(augmented[active], train_present[active])
        loss = paired_loss(left, right)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite attention loss")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        equal = masked_weights(torch.zeros_like(test_present, dtype=test.dtype), test_present)
        learned, weights = model(test, test_present)
        return {"equal_concat": (weighted_concat(test, equal), equal), "attention_concat": (learned, weights)}


def canonical(groups):
    return tuple(sorted(tuple(sorted(group)) for group in groups if group))


def graph_for(keys, edges, similarities):
    graph = nx.Graph()
    graph.add_nodes_from(keys)
    for (a, b), similarity in zip(edges, similarities, strict=True):
        if not math.isfinite(similarity):
            raise ValueError("Non-finite similarity")
        graph.add_edge(a, b, weight=(1 + max(-1, min(1, similarity))) / 2)
    return graph


def partition(graph, candidates, *, trace=None):
    has_weight = bool(graph.size(weight="weight"))

    def q(groups):
        return nx.community.modularity(graph, groups, weight="weight") if has_weight else 0.0

    active = {(k,) for k in graph}
    lookup = {k: (k,) for k in graph}
    singletons = canonical(active)
    tree, best_q = singletons, q(singletons)
    tree_steps = [] if trace is not None else None
    best_step = 0
    for merged in candidates[len(graph):]:
        children = {lookup[k] for k in merged}
        if len(children) != 2 or set().union(*map(set, children)) != set(merged):
            raise ValueError("Expected a binary merge of active groups")
        active.difference_update(children)
        active.add(merged)
        lookup.update(dict.fromkeys(merged, merged))
        groups = canonical(active)
        score = q(groups)
        if tree_steps is not None:
            tree_steps.append({"before": canonical(children), "after": (merged,), "modularity": score})
        if score > best_q:
            tree, best_q = groups, score
            if tree_steps is not None:
                best_step = len(tree_steps)

    # Allow moving a block to a neighboring group or back to a singleton.
    # Accept only a strict objective improvement, without a tunable tolerance.
    def improve(initial, steps):
        current, current_q = initial, q(initial)
        moves = 0
        while True:
            groups = tuple(map(frozenset, current))
            membership = {node: group for group in groups for node in group}
            next_groups, next_q = current, current_q
            for node in sorted(graph):
                source = membership[node]
                remainder = source - {node}
                if remainder and not nx.is_connected(graph.subgraph(remainder)):
                    continue
                targets = {membership[v] for v in graph[node]} - {source}
                if remainder:
                    targets.add(frozenset())
                for target in sorted(targets, key=lambda g: tuple(sorted(g))):
                    proposal = canonical([*(g for g in groups if g not in (source, target)),
                                          remainder, target | {node}])
                    score = q(proposal)
                    if score > next_q:
                        next_groups, next_q = proposal, score
                        moved_node = node
            if next_groups == current:
                return current, current_q, moves
            if steps is not None:
                steps.append({"before": canonical(set(current) - set(next_groups)),
                              "after": canonical(set(next_groups) - set(current)),
                              "block": moved_node, "modularity": next_q})
            current, current_q = next_groups, next_q
            moves += 1

    # Single-block moves may get stuck if an early merge creates a group
    # that can only be improved by splitting several blocks at once.
    move_steps = {start: [] if trace is not None else None for start in ("tree_cut", "singletons")}
    start, (current, current_q, moves) = max(
        (("tree_cut", improve(tree, move_steps["tree_cut"])),
         ("singletons", improve(singletons, move_steps["singletons"]))), key=lambda result: result[1][1])
    if trace is not None:
        trace.update(singleton_modularity=q(singletons),
                     tree_steps=tree_steps[:best_step] if start == "tree_cut" else [],
                     move_steps=move_steps[start])
    for groups in (tree, current):
        if sorted(node for group in groups for node in group) != sorted(graph):
            raise ValueError("Partition must cover each input block exactly once")
        if not all(nx.is_connected(graph.subgraph(group)) for group in groups):
            raise ValueError("Partition breaks walking connectivity")
    return {"tree_cut": {"groups": tree, "modularity": best_q},
            "block_moves": {"groups": current, "modularity": current_q, "moves": moves, "start": start}}


def prepare(inputs, output):
    torch.set_num_threads(1)
    encoder = baseline.load_author_model().Encoder
    data = {area: read_area(inputs, area) for area in AREAS}
    output.mkdir(parents=True)
    protocol = {"seeds": SEEDS, "epochs_per_view_and_fusion": EPOCHS, "views": VIEW_NAMES, "width": WIDTH, "heads": HEADS,
                "learning_rate": LR, "weight_decay": WEIGHT_DECAY, "fitting": "held-out area; independent view losses",
                "fusion": ["equal normalized concat", "attention-weighted normalized concat"],
                "partition": ["best modularity on complete-link tree",
                              "connected single-block moves from both tree cut and singletons; best modularity"],
                "adjacency_weight": "(1 + cosine) / 2", "resolution": "standard modularity",
                "input_sha256": {area: data[area][3] for area in AREAS}}
    write(output / "protocol.json", protocol)
    for train_area, test_area in (AREAS, AREAS[::-1]):
        train, train_geometry, _, _ = data[train_area]
        test, test_geometry, edges, _ = data[test_area]
        if (any(baseline.shared_entities(train, test)) or
                set.union(*(b.poi_ids for b in train)) & set.union(*(b.poi_ids for b in test))):
            raise ValueError("Training and holdout share source entities")
        pairs, dim = baseline.make_graphs(observed_views(train), observed_views(test), train_geometry | test_geometry)
        train_graph = batch(pairs[:len(train)], train)
        test_graph = batch(pairs[len(train):], test)
        train_present, test_present = (torch.stack([observed(g, k) for k, g in enumerate(b)], dim=1)
                                       for b in (train_graph, test_graph))
        keys = [b.key for b in test]
        unobserved = {k for k, row in zip(keys, test_present, strict=True) if not row.any()}
        if any(set(e) & unobserved for e in edges):
            raise ValueError("An unobserved block cannot have a mapped walking connection")
        runs = {}
        for seed in SEEDS:
            z, z_aug, z_test = fit_views(train_graph, test_graph, dim, encoder, seed)
            fused = fuse(z, z_aug, z_test, train_present, test_present)
            runs[str(seed)] = {"view_embeddings": (z_test * test_present[..., None]).tolist(), "fusion": {}}
            for method, (vectors, weights) in fused.items():
                run = predict(keys, vectors.numpy(), edges)
                graph = graph_for(keys, edges, run["edge_scores"])
                runs[str(seed)]["fusion"][method] = {"edge_scores": run["edge_scores"],
                                                     "view_weights": weights.tolist(),
                                                     "partitions": partition(graph, run["candidates"])}
            print(f"{train_area} -> {test_area}, seed {seed}: frozen", flush=True)
        write(output / f"{test_area}_predictions.json", {"keys": keys, "adjacency": edges,
                                                       "available_views": test_present.tolist(), "runs": runs})


def score(inputs, output):
    protocol = json.loads((output / "protocol.json").read_bytes())
    frozen = {a: (output / f"{a}_predictions.json").read_bytes() for a in AREAS}
    boundary = (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes()
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(boundary)["features"]}
    summary = {"reference_sha256": sha256(boundary).hexdigest(), "areas": {}}
    for area, raw in frozen.items():
        geometry = inputs / area / "blocks.geojson"
        if sha256(geometry.read_bytes()).hexdigest() != protocol["input_sha256"][area]["blocks.geojson"]:
            raise ValueError(f"Block geometry changed: {area}")
        data = json.loads(raw)
        blocks = block_geometry(geometry)
        if blocks.keys() != set(data["keys"]):
            raise ValueError(f"Frozen partition keys differ from the geometry: {area}")
        fence = fences[REFERENCE_NAMES[area]]
        reference = fence.intersection(union_all(list(blocks.values())))
        inside = {k for k, g in blocks.items() if fence.covers(g.centroid)}
        support = [i for i, (a, b) in enumerate(data["adjacency"]) if a in inside or b in inside]
        labels = [int(set(data["adjacency"][i]) <= inside) for i in support]
        overlap = {k: g.intersection(reference).area for k, g in blocks.items()}

        def compare(group):
            hit = sum(overlap[k] for k in group)
            total = sum(blocks[k].area for k in group)
            return {"iou": hit / (total + reference.area - hit), "extra_ha": (total - hit) / 10_000,
                    "missed_ha": (reference.area - hit) / 10_000, "blocks": len(group),
                    "outside_blocks": sum(overlap[k] == 0 for k in group)}

        results = defaultdict(dict)
        for seed, run in data["runs"].items():
            for method, f in run["fusion"].items():
                ap = (average_precision(labels, [f["edge_scores"][i] for i in support])
                      if 0 < sum(labels) < len(labels) else None)
                for name, p in f["partitions"].items():
                    group = max(p["groups"], key=lambda g: compare(g)["iou"])
                    results[f"{method}/{name}"][seed] = {**compare(group), "ap": ap,
                                                         "modularity": p["modularity"], "matched_group": group}
        mean_scores = {name: {k: mean(row[k] for row in runs.values()) for k in
                              ("iou", "extra_ha", "missed_ha", "blocks", "outside_blocks", "modularity")}
                       for name, runs in results.items()}
        for name, runs in results.items():
            aps = [r["ap"] for r in runs.values() if r["ap"] is not None]
            mean_scores[name]["ap"] = mean(aps) if aps else None
        summary["areas"][area] = {"predictions_sha256": sha256(raw).hexdigest(), "runs": results,
                                  "mean": mean_scores, "boundary_edges": dict(Counter(labels))}
        print(area, json.dumps(mean_scores, indent=2))
    write(output / "metrics.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else score)(args.inputs, args.output)


if __name__ == "__main__":
    main()
