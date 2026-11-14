"""Select merge-tree layers by human-profile alignment; 001 selects, 002 audits."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import tempfile
import os

import numpy as np
from shapely.geometry import shape
from shapely.ops import transform
from shapely import union_all

from block_boundary_benchmark import REFERENCE_NAMES, ROOT, TO_METRES
from fusion_partition_benchmark import block_geometry, canonical
from streetview_common import AREAS, dataset_root, write_json

FIELDS = ("frontage", "door_density", "continuity")
SEEDS = ("13", "29", "53")
PERMUTATIONS = 100
NULL_SEED = 20261117


def annotator_profiles(dataset, annotator):
    profiles = {}
    for line in (dataset / "responses" / f"{annotator}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") == "rating" and r["value"] != "X":
            profiles.setdefault((r["area"], r["block"]), {})[r["field"]] = r["value"]
    return profiles


ALL_VIEWS = ("poi", "building", "walk")


def fused_for(data, seed, subset):
    """Equal-weight concat over the views in the subset; missing views stay zero."""
    avail = {k: np.array(a) for k, a in zip(data["keys"], data["available_views"])}
    embs = np.array(data["runs"][seed]["view_embeddings"])
    width = embs.shape[2]
    vectors = np.zeros((len(data["keys"]), width * 3))
    for i, key in enumerate(data["keys"]):
        active = [j for j, name in enumerate(ALL_VIEWS) if name in subset and avail[key][j]]
        for j in active:
            vectors[i, j * width:(j + 1) * width] = embs[i, j] * np.sqrt(1.0 / len(active))
    return vectors


def rebuild_layers(keys, adjacency, vectors):
    """Replay the complete-link merge order and yield every intermediate layer."""
    from block_boundary_benchmark import predict
    tree = predict(keys, vectors, adjacency)
    groups = {(k,) for k in keys}
    lookup = {k: (k,) for k in keys}
    layers = [canonical(groups)]
    for merged in tree["candidates"][len(keys):]:
        children = {lookup[k] for k in merged}
        if len(children) != 2 or set().union(*map(set, children)) != set(merged):
            raise ValueError("Expected a binary merge of active groups")
        groups.difference_update(children)
        groups.add(merged)
        lookup.update(dict.fromkeys(merged, merged))
        layers.append(canonical(groups))
    return layers


def purity(groups, profiles, area):
    tuples = {}
    for group in groups:
        rows = [tuple(sorted(profiles[(area, b)].items())) for b in group
                if (area, b) in profiles and len(profiles[(area, b)]) == len(FIELDS)]
        if not rows:
            continue
        tuples[group] = rows
    if not tuples:
        return None
    total = sum(len(r) for r in tuples.values())
    agree = sum(Counter(r).most_common(1)[0][1] for r in tuples.values())
    return agree / total


def excess_purity(groups, profiles, area, rng):
    """Purity above size-matched random label assignments (kills the finer-is-purer cheat)."""
    observed = purity(groups, profiles, area)
    if observed is None:
        return None, None
    blocks = [b for g in groups for b in g if (area, b) in profiles
              and len(profiles[(area, b)]) == len(FIELDS)]
    labels = [tuple(sorted(profiles[(area, b)].items())) for b in blocks]
    if len(set(labels)) < 2:
        return observed, 0.0
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = labels[:]
        rng.shuffle(shuffled)
        fake = dict(zip(blocks, shuffled))
        value = purity(groups, fake, area)
        if value is not None:
            null.append(value)
    return observed, (observed - (sum(null) / len(null))) if null else 0.0


def edge_split(groups, adjacency, profiles, area):
    """Profile agreement per walking edge, split into within-group and cross-group."""
    member = {b: i for i, group in enumerate(groups) for b in group}

    def agreement(a, b):
        pa, pb = profiles.get((area, a), {}), profiles.get((area, b), {})
        shared = [f for f in FIELDS if pa.get(f) and pb.get(f)]
        if not shared:
            return None
        return sum(pa[f] == pb[f] for f in shared) / len(shared)

    within, across = [], []
    for a, b in adjacency:
        score = agreement(a, b)
        if score is None:
            continue
        (within if member.get(a) == member.get(b) else across).append(score)
    return within, across


def welch_t(within, across):
    """Contrast z-scored by its own standard error; few edges on either side means noise."""
    if len(within) < 2 or len(across) < 2:
        return 0.0
    mw, ma = sum(within) / len(within), sum(across) / len(across)
    vw = sum((x - mw) ** 2 for x in within) / (len(within) - 1) or 1e-12
    va = sum((x - ma) ** 2 for x in across) / (len(across) - 1) or 1e-12
    return (mw - ma) / (vw / len(within) + va / len(across)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/fusion_partition_topology_v2")
    parser.add_argument("--infomap", type=Path, default=ROOT / "outputs/infomap_partition_pilot")
    parser.add_argument("--views", default="poi,walk",
                        help="fused view subset; the frozen tree check only covers all three")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    subset = tuple(args.views.split(","))
    if args.output is None:
        args.output = ROOT / f"outputs/profile_level_selection_{args.views.replace(',', '+')}"
    args.output.mkdir(parents=True, exist_ok=False)

    selector = annotator_profiles(args.dataset, "001")
    auditor = annotator_profiles(args.dataset, "002")
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(
                  (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes())["features"]}

    report = {"protocol": {"selection_labels": "001", "audit_labels": "002", "permutations": PERMUTATIONS,
                           "score": "smoothed Welch t on within-vs-cross edge profile agreement",
                           "views": list(subset), "null_seed": NULL_SEED},
              "areas": {}}
    for area in AREAS:
        data = json.loads((args.source / f"{area}_predictions.json").read_bytes())
        info = json.loads((args.infomap / f"{area}_partitions.json").read_bytes())["runs"]
        blocks = block_geometry(ROOT / f"data/processed/nyc-2026-11-09-block-inputs/{area}/blocks.geojson")
        reference = fences[REFERENCE_NAMES[area]].intersection(union_all(list(blocks.values())))
        keys, adjacency = data["keys"], [tuple(e) for e in data["adjacency"]]

        def bid_iou(groups):
            best = 0.0
            for group in groups:
                size = sum(blocks[b].area for b in group)
                hit = sum(blocks[b].intersection(reference).area for b in group)
                best = max(best, hit / (size + reference.area - hit))
            return best

        seeds_out = {}
        for seed in SEEDS:
            run = data["runs"][seed]["fusion"]["equal_concat"]
            vectors = fused_for(data, seed, subset)
            layers = rebuild_layers(keys, adjacency, vectors)

            frozen = canonical(tuple(tuple(g) for g in run["partitions"]["tree_cut"]["groups"]))
            modularity_layer = next((i for i, g in enumerate(layers) if g == frozen), None)
            if modularity_layer is None and set(subset) == set(ALL_VIEWS):
                raise ValueError(f"Tree rebuild does not reproduce the frozen tree cut: {area}/{seed}")

            rng = random.Random(NULL_SEED)
            raw = []
            for index, layer in enumerate(layers):
                observed, excess = excess_purity(layer, selector, area, rng)
                within, across = edge_split(layer, adjacency, selector, area)
                raw.append({"layer": index, "groups": len(layer), "excess_purity": excess,
                            "t": welch_t(within, across), "n_within": len(within),
                            "n_across": len(across)})
            # 3-layer moving average damps single-layer noise; the t-score already
            # downweights layers whose contrast rests on a handful of edges.
            curve = [{**row, "score": sum(raw[j]["t"] for j in (index - 1, index, index + 1)
                                          if 0 <= j < len(raw)) /
                      len([j for j in (index - 1, index, index + 1) if 0 <= j < len(raw)])}
                     for index, row in enumerate(raw)]
            best = max(curve, key=lambda row: row["score"])
            chosen = layers[best["layer"]]

            def audit(groups, name):
                groups = canonical(tuple(tuple(g) for g in groups))
                _, excess = excess_purity(groups, auditor, area, random.Random(NULL_SEED))
                within, across = edge_split(groups, adjacency, auditor, area)
                contrast = (sum(within) / len(within) - sum(across) / len(across)) \
                    if within and across else None
                return {"method": name, "groups": len(groups),
                        "audit_excess_purity": excess, "audit_edge_contrast": contrast,
                        "audit_t": welch_t(within, across),
                        "audit_purity": purity(groups, auditor, area), "bid_iou": bid_iou(groups)}

            comparisons = [audit(run["partitions"]["block_moves"]["groups"], "modularity_block_moves"),
                           audit(info[seed]["infomap"], "infomap"),
                           audit(layers[0], "singletons_reference")]
            if modularity_layer is not None:
                comparisons.insert(0, audit(layers[modularity_layer], "modularity_tree_cut"))
            seeds_out[seed] = {
                "selected": {**best, "audit": audit(chosen, "profile_selected")},
                "comparisons": comparisons,
                "curve": curve}
            note = (f"vs modularity layer {modularity_layer} ({len(frozen)} groups)"
                    if modularity_layer is not None else f"(views={'+'.join(subset)}, no frozen check)")
            print(f"[{area} seed {seed}] profile layer {best['layer']} "
                  f"({best['groups']} groups, score {best['score']:.3f}) {note}")
            for row in [seeds_out[seed]["selected"]["audit"], *seeds_out[seed]["comparisons"]]:
                ep = row["audit_excess_purity"] if row["audit_excess_purity"] is not None else 0.0
                ec = row["audit_edge_contrast"] if row["audit_edge_contrast"] is not None else 0.0
                pu = f"{row['audit_purity']:.0%}" if row["audit_purity"] is not None else "n/a"
                print(f"    {row['method']:24s} groups {row['groups']:2d} | "
                      f"excess {ep:.3f} | contrast {ec:+.3f} | purity {pu} "
                      f"| BID IoU {row['bid_iou']:.2f}")
        report["areas"][area] = seeds_out
    report["inputs"] = {"source_sha256": {a: sha256((args.source / f"{a}_predictions.json").read_bytes()).hexdigest() for a in AREAS},
                        "responses_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                                             for p in sorted((args.dataset / "responses").glob("*.jsonl"))}}
    write_json(args.output / "selection.json", report)
    print(f"\nwritten: {args.output / 'selection.json'}")


if __name__ == "__main__":
    main()
