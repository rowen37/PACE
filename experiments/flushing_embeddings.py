"""Train view encoders on SoHo (frozen protocol) and embed Flushing as a held-out area."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

from block_boundary_benchmark import ROOT, predict
from fusion_partition_benchmark import (DEFAULT_INPUT, graph_for, partition, read_area,
                                        observed, fuse, fit_views)
from pedestrian_rgscl import baseline, batch, observed_views
from streetview_common import write_json

TRAIN_AREA, TEST_AREA = "soho", "flushing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=TRAIN_AREA)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/flushing_embeddings")
    args = parser.parse_args()
    train_area = args.train
    if train_area != TRAIN_AREA:
        args.output = args.output.parent / f"flushing_embeddings_from_{train_area}"
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)

    encoder = baseline.load_author_model().Encoder
    train, train_geometry, _, _ = read_area(args.inputs, train_area)
    test, test_geometry, edges, hashes = read_area(args.inputs, TEST_AREA)
    if (any(baseline.shared_entities(train, test))
            or set.union(*(b.poi_ids for b in train)) & set.union(*(b.poi_ids for b in test))):
        raise ValueError("Training and holdout share source entities")
    pairs, dim = baseline.make_graphs(observed_views(train), observed_views(test),
                                      train_geometry | test_geometry)
    train_graph = batch(pairs[:len(train)], train)
    test_graph = batch(pairs[len(train):], test)
    from torch import stack
    train_present = stack([observed(g, k) for k, g in enumerate(train_graph)], dim=1)
    test_present = stack([observed(g, k) for k, g in enumerate(test_graph)], dim=1)
    keys = [b.key for b in test]
    unobserved = {k for k, row in zip(keys, test_present) if not row.any()}
    if any(set(e) & unobserved for e in edges):
        raise ValueError("An unobserved block cannot have a mapped walking connection")

    from fusion_partition_benchmark import SEEDS
    runs = {}
    for seed in SEEDS:
        z, z_aug, z_test = fit_views(train_graph, test_graph, dim, encoder, seed)
        fused = fuse(z, z_aug, z_test, train_present, test_present)
        runs[str(seed)] = {"view_embeddings": (z_test * test_present[..., None]).tolist(),
                           "fusion": {}}
        for method, (vectors, weights) in fused.items():
            run = predict(keys, vectors.numpy(), edges)
            graph = graph_for(keys, edges, run["edge_scores"])
            runs[str(seed)]["fusion"][method] = {
                "edge_scores": run["edge_scores"], "view_weights": weights.tolist(),
                "partitions": partition(graph, run["candidates"])}
        print(f"{train_area} -> {TEST_AREA}, seed {seed}: frozen", flush=True)
    write_json(args.output / f"{TEST_AREA}_predictions.json",
               {"keys": keys, "adjacency": edges, "available_views": test_present.tolist(),
                "runs": runs})
    write_json(args.output / "protocol.json", {
        "train_area": train_area, "test_area": TEST_AREA,
        "note": "held-out embedding for the frozen-then-annotated Flushing window; "
                "protocol identical to fusion_partition_topology_v2",
        "input_sha256": {"flushing": hashes}})
    print(f"written: {args.output / f'{TEST_AREA}_predictions.json'}")


if __name__ == "__main__":
    main()
