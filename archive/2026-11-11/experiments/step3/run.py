"""Compare attention operators on held-out blocks and hidden POI organization."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GATv2Conv

from data import (HERE, STEP1, TEST_MASK_SEEDS, TRAIN_MASK_SEEDS, make_blocks, make_graph,
                  pair_share, place_counts, summary, training_vocabulary, views_for)
from model import EMBEDDING_DIM, MODEL_SEEDS, fit_graph
from cluster import write

MODELS = ("visible_pair_share", "pca", "gat", "gatv2")


def fit_pca(train, test, vocab):
    train_x = np.array([summary(view, vocab) for view in train])
    test_x = np.array([summary(view, vocab) for view in test])
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale == 0] = 1
    train_x, test_x = (train_x - center) / scale, (test_x - center) / scale
    _, _, components = np.linalg.svd(train_x, full_matrices=False)
    scores = torch.tensor(train_x @ components[:EMBEDDING_DIM].T, dtype=torch.float32)
    test_scores = torch.tensor(test_x @ components[:EMBEDDING_DIM].T, dtype=torch.float32)
    target = torch.tensor([view.target for view in train], dtype=torch.float32)
    torch.manual_seed(MODEL_SEEDS[0])
    head = nn.Linear(EMBEDDING_DIM, 1)
    optimizer = torch.optim.LBFGS(head.parameters(), line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(scores).flatten(), target)
        loss = loss + 0.005 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return head(test_scores).flatten().sigmoid().tolist()


def score(rows):
    # Repeated masks on one block are repeated measurements, not independent blocks.
    errors = {name: defaultdict(list) for name in MODELS}
    for row in rows:
        for name in MODELS:
            errors[name][row["block_id"]].append(row[name] - row["target"])
    return {name: {"macro_mae": mean(mean(abs(x) for x in values) for values in by_block.values()),
                   "macro_rmse": mean(mean(x * x for x in values) for values in by_block.values())**0.5}
            for name, by_block in errors.items()}


def run_fold(blocks, held_out, epochs):
    train = views_for([b for b in blocks if b.area != held_out], TRAIN_MASK_SEEDS)
    test = views_for([b for b in blocks if b.area == held_out], TEST_MASK_SEEDS)
    if not train or not test:
        raise ValueError(f"Missing scorable train or test views for {held_out}")
    vocab = training_vocabulary(train)
    prior_by_block = defaultdict(list)
    for view in train:
        prior_by_block[view.block.key].append(view.target)
    prior = mean(mean(values) for values in prior_by_block.values())
    visible = [pair_share(place_counts(view)) for view in test]
    predictions = {"visible_pair_share": [prior if value is None else value for value in visible],
                   "pca": fit_pca(train, test, vocab)}
    train_graphs = [make_graph(view, vocab) for view in train]
    test_graphs = [make_graph(view, vocab) for view in test]
    parameters = {}
    for name, operator in [("gat", GATConv), ("gatv2", GATv2Conv)]:
        samples = []
        for seed in MODEL_SEEDS:
            predicted, parameters[name] = fit_graph(train_graphs, test_graphs, len(vocab), operator, epochs, seed)
            samples.append(predicted)
        predictions[name] = np.mean(samples, axis=0).tolist()
        for seed, prediction in zip(MODEL_SEEDS, samples, strict=True):
            predictions[f"{name}_seed_{seed}"] = prediction.tolist()
    rows = [{"block_id": view.block.key, "mask_seed": view.mask_seed,
             "visible_pois": len(view.visible), "hidden_matched_pois": view.hidden_matches,
             "target": view.target, **{name: values[i] for name, values in predictions.items()}}
            for i, view in enumerate(test)]
    stats = {"train_blocks": len({view.block.key for view in train}),
             "test_blocks": len({view.block.key for view in test}),
             "train_views": len(train), "test_views": len(test),
             "parameters": parameters, "scores": score(rows)}
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-data", type=Path, default=STEP1 / "data")
    parser.add_argument("--step1-outputs", type=Path, default=STEP1 / "outputs")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()
    torch.set_num_threads(1)
    blocks = make_blocks(args.step1_data, args.step1_outputs)
    areas = sorted({b.area for b in blocks})
    args.output.mkdir(parents=True, exist_ok=True)
    folds = {}
    for area in areas:
        rows, stats = run_fold(blocks, area, args.epochs)
        folds[area] = stats
        write(args.output / f"{area}_predictions.json", rows)
        print(area, stats, flush=True)
    write(args.output / "metrics.json", {
        "target": "share of held-out uniquely matched commercial POI pairs in the same footprint",
        "epochs": args.epochs, "model_seeds": MODEL_SEEDS, "embedding_dimensions": EMBEDDING_DIM,
        "walk_entrance_edges": {area: sum(len(b.access_pairs) for b in blocks if b.area == area)
                                for area in areas},
        "available_blocks": {area: sum(b.area == area for b in blocks) for area in areas},
        "folds": folds,
    })


if __name__ == "__main__":
    main()
