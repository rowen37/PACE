"""E4: reliability-weighted fusion, profile-guided selection, budget-pruned candidates."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import statistics

import numpy as np

from block_boundary_benchmark import ROOT, predict
from fusion_partition_benchmark import canonical
from streetview_common import AREAS, dataset_root, write_json
from streetview_evaluate import consensus_ratings, knn_loo, load_responses
from streetview_profile_levels import (FIELDS, SEEDS, annotator_profiles, edge_split,
                                       purity, welch_t)
from view_ablation import select_layer

SUBSET = ("poi", "walk")
NEIGHBOR_KS = (3, 5, 7)  # Jaccard averaged over a fixed grid instead of picking one k
SPEED_KMH = 4.5  # literature walking speed; the only physical constant in the tour model
BUDGETS_MIN = (15, 30, 45, 90, 180)  # reporting grid only; budgets are user input
ORDER = ("poi", "building", "walk")


def block_stats(area):
    import orjson
    folder = ROOT / f"data/processed/nyc-2026-11-09-block-inputs/{area}"
    walk = Counter()
    for f in orjson.loads((folder / "walk.geojson").read_bytes())["features"]:
        walk[f["properties"]["block_id"]] += f["properties"]["length_m"]
    pois = Counter()
    for f in orjson.loads((folder / "places.geojson").read_bytes())["features"]:
        pois[f["properties"]["block_id"]] += 1
    return dict(walk), dict(pois)


def reliability(data):
    """r_v(block) = observed × mean pairwise Jaccard of seed-neighbour sets, averaged over a k grid."""
    out = {}
    order = list(data["keys"])
    avail = {k: np.array(a) for k, a in zip(order, data["available_views"])}
    for view in SUBSET:
        j = ORDER.index(view)
        per_seed = {k: {} for k in NEIGHBOR_KS}
        for seed in SEEDS:
            embs = np.array(data["runs"][seed]["view_embeddings"])
            blocks = [k for k in order if avail[k][j]]
            if len(blocks) < max(NEIGHBOR_KS) + 1:
                continue
            matrix = np.stack([embs[order.index(k), j] for k in blocks])
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
            sim = matrix @ matrix.T
            np.fill_diagonal(sim, -np.inf)
            for r, k in enumerate(blocks):
                ranked = sim[r].argsort()
                for k_n in NEIGHBOR_KS:
                    per_seed[k_n].setdefault(k, {})[seed] = {blocks[i] for i in ranked[-k_n:]}
        for k in blocks:
            scores = []
            for k_n in NEIGHBOR_KS:
                for i, a in enumerate(SEEDS):
                    for b in SEEDS[i + 1:]:
                        if a in per_seed[k_n].get(k, {}) and b in per_seed[k_n].get(k, {}):
                            x, y = per_seed[k_n][k][a], per_seed[k_n][k][b]
                            scores.append(len(x & y) / len(x | y))
            out.setdefault(k, {})[view] = statistics.mean(scores) if scores else 0.0
    return {k: v for k, v in out.items() if any(v.values())}


def fused(data, seed, mode, rel):
    order = list(data["keys"])
    avail = {k: np.array(a) for k, a in zip(order, data["available_views"])}
    embs = np.array(data["runs"][seed]["view_embeddings"])
    width = embs.shape[2]
    vectors = np.zeros((len(order), width * 3))
    for i, key in enumerate(order):
        if mode == "equal":
            active = [(j, 1.0 / len(SUBSET)) for j in (ORDER.index(v) for v in SUBSET) if avail[key][j]]
        else:
            r = rel.get(key, {})
            total = sum(r.get(v, 0.0) for v in SUBSET if avail[key][ORDER.index(v)])
            if total <= 0:
                continue
            active = [(j, r.get(v, 0.0) / total) for v in SUBSET
                      for j in (ORDER.index(v),) if avail[key][j]]
        for j, w in active:
            vectors[i, j * width:(j + 1) * width] = embs[i, j] * np.sqrt(w)
    return vectors


def merge_nodes(keys, adjacency, vectors):
    """Complete-link merge tree as {node: (left, right)} in merge order."""
    tree = predict(keys, vectors, adjacency)
    lookup = {k: (k,) for k in keys}
    nodes = []
    for merged in tree["candidates"][len(keys):]:
        children = {lookup[k] for k in merged}
        if len(children) != 2:
            raise ValueError("expected binary merge")
        nodes.append((merged, *sorted(children)))
        lookup.update(dict.fromkeys(merged, merged))
    return nodes


def tour_minutes(group, walk_len, pois):
    """Pure walking time on internal segments; POI counts ride along, unstopped."""
    return sum(walk_len.get(b, 0.0) for b in group) / 1000 / SPEED_KMH * 60


def budget_groups(nodes, keys, budget, walk_len, pois):
    """Finest connected candidates whose tour fits the budget, per branch."""
    children = {node: (a, b) for node, a, b in nodes}
    time = {node: tour_minutes(node, walk_len, pois) for node, _, _ in nodes}
    for k in keys:
        time[(k,)] = tour_minutes((k,), walk_len, pois)
    lookup = {k: (k,) for k in keys}
    for node, _, _ in nodes:
        for k in node:
            lookup[k] = node
    roots = sorted({lookup[k] for k in keys})
    out = []

    def descend(node):
        kids = children.get(node)
        if time[node] <= budget or kids is None:
            out.append(node)
        else:
            for kid in kids:
                descend(kid)

    for root in roots:
        descend(root)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/fusion_partition_topology_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e4_connected_candidates")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    responses = load_responses(args.dataset)
    consensus = consensus_ratings(responses)
    selector = annotator_profiles(args.dataset, "001")
    auditor = annotator_profiles(args.dataset, "002")

    report = {"protocol": {"views": list(SUBSET), "neighbor_ks": list(NEIGHBOR_KS),
                           "speed_kmh": SPEED_KMH,
                           "tour_model": "internal walk length / literature walking speed; no per-POI stop allowance",
                           "budgets_min": "reporting grid only; budgets are user input",
                           "no_hand_set_thresholds": "no similarity cutoffs, no target group counts, "
                                                     "no stopping tolerances anywhere in this pipeline",
                           "selection": "001 (smoothed Welch t)", "audit": "002"},
              "areas": {}}
    global_model = {mode: {} for mode in ("equal", "reliable")}
    for area in AREAS:
        data = json.loads((args.source / f"{area}_predictions.json").read_bytes())
        keys, adjacency = data["keys"], [tuple(e) for e in data["adjacency"]]
        walk_len, pois = block_stats(area)
        rel = reliability(data)
        rel_rows = {k: {v: round(r, 3) for v, r in v_.items()} for k, v_ in rel.items()}
        variants = {}
        for mode in ("equal", "reliable"):
            audits = []
            for seed in SEEDS:
                vectors = fused(data, seed, mode, rel)
                candidates = predict(keys, vectors, adjacency)["candidates"]
                index = select_layer_from(candidates, keys, adjacency, selector, area)
                chosen = layer_at(candidates, keys, index)
                within, across = edge_split(chosen, adjacency, auditor, area)
                audits.append({"contrast": (sum(within) / len(within) - sum(across) / len(across))
                               if within and across else None, "t": welch_t(within, across),
                               "groups": len(chosen), "layer": index})
                for i, k in enumerate(keys):
                    if np.linalg.norm(vectors[i]) > 0:
                        global_model[mode].setdefault((area, k), {})[seed] = vectors[i]
            variants[mode] = {"audit_by_seed": audits,
                              "mean_contrast": statistics.mean(a["contrast"] for a in audits
                                                               if a["contrast"] is not None)}
        better = max(variants, key=lambda m: variants[m]["mean_contrast"])
        vectors = fused(data, "13", better, rel)
        nodes = merge_nodes(keys, adjacency, vectors)
        budget_rows = []
        for budget in BUDGETS_MIN:
            groups = budget_groups(nodes, keys, budget, walk_len, pois)
            member = {b: i for i, g in enumerate(groups) for b in g}
            within, across = edge_split(canonical(tuple(tuple(g) for g in groups)),
                                        adjacency, auditor, area)
            budget_rows.append({
                "budget_min": budget, "candidates": len(groups),
                "sizes": dict(sorted(Counter(len(g) for g in groups).items())),
                "tour_minutes_p50": statistics.median(tour_minutes(g, walk_len, pois) for g in groups),
                "audit_contrast": (sum(within) / len(within) - sum(across) / len(across))
                if within and across else None,
                "b014_group_size": len(groups[member["b014"]]) if "b014" in member else None,
            })
        report["areas"][area] = {"reliability_sample": dict(list(rel_rows.items())[:5]),
                                 "variants": variants, "better": better,
                                 "budgets": budget_rows}
        print(f"\n[{area}] variant comparison (audit contrast, 002): " +
              " | ".join(f"{m} {v['mean_contrast']:+.3f}" for m, v in variants.items()))
        for row in budget_rows:
            print(f"  budget {row['budget_min']:>3}min -> {row['candidates']} candidates, "
                  f"中位游时 {row['tour_minutes_p50']:.0f}min, contrast "
                  f"{row['audit_contrast'] if row['audit_contrast'] is None else round(row['audit_contrast'], 3)}, "
                  f"b014组大小 {row['b014_group_size']}")
    for mode, model in global_model.items():
        for field in ("door_density", "continuity"):
            per = knn_loo(model, consensus[field])
            valid = [v for v in per.values() if v]
            if valid:
                print(f"LOO {mode:9s} {field:12s} {statistics.mean(v['accuracy'] for v in valid):.0%} "
                      f"(基线 {statistics.mean(v['majority_baseline'] for v in valid):.0%})")
                report.setdefault("loo", {}).setdefault(mode, {})[field] = {
                    "accuracy": statistics.mean(v["accuracy"] for v in valid),
                    "baseline": statistics.mean(v["majority_baseline"] for v in valid)}
    report["inputs"] = {"source_sha256": {a: sha256((args.source / f"{a}_predictions.json").read_bytes()).hexdigest() for a in AREAS},
                        "responses_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                             for p in sorted((args.dataset / "responses").glob("*.jsonl"))}}
    write_json(args.output / "e4.json", report)
    print(f"\nwritten: {args.output / 'e4.json'}")


def select_layer_from(candidates, keys, adjacency, selector, area):
    from fusion_partition_benchmark import canonical as canon
    active = {(k,) for k in keys}
    lookup = {k: (k,) for k in keys}
    layers = [canon(active)]
    for merged in candidates[len(keys):]:
        children = {lookup[k] for k in merged}
        active.difference_update(children)
        active.add(merged)
        lookup.update(dict.fromkeys(merged, merged))
        layers.append(canon(active))
    return select_layer(layers, adjacency, selector, area)


def layer_at(candidates, keys, index):
    from fusion_partition_benchmark import canonical as canon
    active = {(k,) for k in keys}
    lookup = {k: (k,) for k in keys}
    for step, merged in enumerate(candidates[len(keys):], start=1):
        children = {lookup[k] for k in merged}
        active.difference_update(children)
        active.add(merged)
        lookup.update(dict.fromkeys(merged, merged))
        if step == index:
            return canon(active)
    return canon(active)


if __name__ == "__main__":
    main()
