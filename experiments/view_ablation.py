"""Ablate view subsets on rating prediction and profile-selected partitions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics

import numpy as np
from shapely.geometry import shape
from shapely.ops import transform
from shapely import union_all

from block_boundary_benchmark import REFERENCE_NAMES, ROOT, TO_METRES
from fusion_partition_benchmark import block_geometry
from streetview_common import AREAS, dataset_root, write_json
from streetview_evaluate import consensus_ratings, knn_loo, load_responses
from streetview_profile_levels import (FIELDS, SEEDS, annotator_profiles, edge_split, fused_for,
                                       rebuild_layers, welch_t)

SUBSETS = {"poi+building+walk": ("poi", "building", "walk"),
           "poi+walk": ("poi", "walk"),
           "building+walk": ("building", "walk"),
           "poi+building": ("poi", "building")}
SMOOTH = 3


def select_layer(layers, adjacency, selector, area):
    raw = []
    for layer in layers:
        within, across = edge_split(layer, adjacency, selector, area)
        raw.append(welch_t(within, across))
    scores = [statistics.mean(raw[max(0, i - SMOOTH // 2):i - SMOOTH // 2 + SMOOTH])
              for i in range(len(raw))]
    best = max(range(len(layers)), key=lambda i: scores[i])
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/fusion_partition_topology_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/view_ablation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    responses = load_responses(args.dataset)
    consensus = consensus_ratings(responses)
    selector = annotator_profiles(args.dataset, "001")
    auditor = annotator_profiles(args.dataset, "002")
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(
                  (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes())["features"]}

    rating_rows = {name: {} for name in SUBSETS}
    partition_rows = {name: {} for name in SUBSETS}
    for area in AREAS:
        data = json.loads((args.source / f"{area}_predictions.json").read_bytes())
        keys, adjacency = data["keys"], [tuple(e) for e in data["adjacency"]]
        blocks = block_geometry(ROOT / f"data/processed/nyc-2026-11-09-block-inputs/{area}/blocks.geojson")
        reference = fences[REFERENCE_NAMES[area]].intersection(union_all(list(blocks.values())))

        def bid_iou(groups):
            best = 0.0
            for group in groups:
                size = sum(blocks[b].area for b in group)
                hit = sum(blocks[b].intersection(reference).area for b in group)
                best = max(best, hit / (size + reference.area - hit))
            return best

        for seed in SEEDS:
            for name, subset in SUBSETS.items():
                vectors = fused_for(data, seed, subset)
                model = {(area, k): {seed: vectors[i]} for i, k in enumerate(keys)
                         if np.linalg.norm(vectors[i]) > 0}
                for field, labels in consensus.items():
                    hit = knn_loo(model, labels).get(seed)
                    if hit:
                        rating_rows[name].setdefault(field, []).append(
                            (hit["accuracy"], hit["majority_baseline"], hit["n"]))
                layers = rebuild_layers(keys, adjacency, vectors)
                index = select_layer(layers, adjacency, selector, area)
                chosen = layers[index]
                within, across = edge_split(chosen, adjacency, auditor, area)
                contrast = (sum(within) / len(within) - sum(across) / len(across)) \
                    if within and across else None
                partition_rows[name].setdefault(area, []).append(
                    {"seed": seed, "layer": index, "groups": len(chosen),
                     "audit_contrast": contrast, "audit_t": welch_t(within, across),
                     "bid_iou": bid_iou(chosen)})

    print("== 画像预测（共识标签，留一 1NN，三种子均值；括号=多数类基线） ==")
    for name in SUBSETS:
        cells = []
        for field in FIELDS:
            rows = rating_rows[name].get(field, [])
            if rows:
                cells.append(f"{field[:4]} {statistics.mean(r[0] for r in rows):.0%}"
                             f"({statistics.mean(r[1] for r in rows):.0%})")
        print(f"  {name:20s} " + " ".join(cells))

    print("\n== 画像选层 + 002 审计（contrast / 组数 / BID IoU，三种子均值） ==")
    for name in SUBSETS:
        for area in AREAS:
            rows = partition_rows[name].get(area, [])
            if not rows:
                continue
            contrasts = [r["audit_contrast"] for r in rows if r["audit_contrast"] is not None]
            print(f"  {name:20s} {area:13s} contrast "
                  f"{statistics.mean(contrasts):+.3f} | groups "
                  f"{statistics.mean(r['groups'] for r in rows):.0f} | BID "
                  f"{statistics.mean(r['bid_iou'] for r in rows):.2f}")

    write_json(args.output / "ablation.json",
               {"rating": {name: {field: rows for field, rows in fields.items()}
                           for name, fields in rating_rows.items()},
                "partition": partition_rows,
                "source_sha256": {a: sha256((args.source / f"{a}_predictions.json").read_bytes()).hexdigest()
                                  for a in AREAS}})
    print(f"\nwritten: {args.output / 'ablation.json'}")


if __name__ == "__main__":
    main()
