"""Check GATv2 tuning with validation restricted to the outer training district."""

import argparse
from collections import defaultdict
from itertools import chain
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch_geometric.nn import GATv2Conv

from data import (HERE, STEP1, TEST_MASK_SEEDS, TRAIN_MASK_SEEDS, make_blocks, make_graph,
                  training_vocabulary, views_for)
from model import MODEL_SEEDS, graph_checkpoints
from cluster import write

SETTINGS = {
    "original": {"width": 16, "lr": 0.004, "weight_decay": 0.01},
    "lower_lr": {"width": 16, "lr": 0.001, "weight_decay": 0.01},
    "small_regularized": {"width": 8, "lr": 0.004, "weight_decay": 0.05},
}
CHECKPOINTS = (5, 15, 30, 45)
VALIDATION_MASK_SEEDS = range(200, 205)


def split_bands(blocks):
    centers = np.array([block.center for block in blocks])
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    ordered = sorted(blocks, key=lambda block: (block.center[axis], block.key))
    bands = [list(band) for band in np.array_split(np.array(ordered, dtype=object), 3)]
    return bands


def rows_for(views, predictions):
    return [{"block_id": view.block.key, "mask_seed": view.mask_seed, "target": view.target,
             "prediction": float(prediction)}
            for view, prediction in zip(views, predictions, strict=True)]


def block_average(rows, metric):
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["block_id"]].append(metric(row))
    return mean(mean(values) for values in blocks.values())


def mae(rows):
    return block_average(rows, lambda row: abs(row["prediction"] - row["target"]))


def graphs_for(train, evaluation):
    vocab = training_vocabulary(train)
    return ([make_graph(view, vocab) for view in train],
            [make_graph(view, vocab) for view in evaluation], len(vocab))


def select(training_blocks):
    bands = split_bands(training_blocks)
    scores = defaultdict(list)
    details = []
    for i, validation in enumerate(bands):
        train = views_for(chain.from_iterable(band for j, band in enumerate(bands) if i != j),
                          TRAIN_MASK_SEEDS)
        valid = views_for(validation, VALIDATION_MASK_SEEDS)
        if not train or not valid:
            raise ValueError(f"Insufficient data in spatial validation band {i}")
        train_graphs, valid_graphs, categories = graphs_for(train, valid)
        for setting, kwargs in SETTINGS.items():
            checkpoints, _, _ = graph_checkpoints(train_graphs, valid_graphs, categories, GATv2Conv,
                                                CHECKPOINTS, MODEL_SEEDS[0], **kwargs)
            for epoch, prediction in checkpoints.items():
                scores[setting, epoch].extend(rows_for(valid, prediction))
        details.append({"training_blocks": sorted({v.block.key for v in train}),
                        "validation_blocks": sorted({v.block.key for v in valid})})
    ranked = sorted(({"setting": setting, "epoch": epoch, "validation_mae": mae(rows)}
                     for (setting, epoch), rows in scores.items()),
                    key=lambda result: (result["validation_mae"], result["epoch"], result["setting"]))
    return ranked, details


def evaluate(training_blocks, test_blocks, selected):
    train = views_for(training_blocks, TRAIN_MASK_SEEDS)
    test = views_for(test_blocks, TEST_MASK_SEEDS)
    train_graphs, test_graphs, categories = graphs_for(train, test)
    # Reuse the masks from the original held-out comparison.
    epoch = selected["epoch"]
    samples = []
    for seed in MODEL_SEEDS:
        checkpoint, _, _ = graph_checkpoints(train_graphs, test_graphs, categories, GATv2Conv,
                                           (epoch,), seed, **SETTINGS[selected["setting"]])
        samples.append(checkpoint[epoch])
    rows = rows_for(test, np.mean(samples, axis=0))
    return rows, {
        "test_mae": mae(rows), "test_blocks": len({v.block.key for v in test}),
        "mean_prediction": block_average(rows, lambda row: row["prediction"]),
        "mean_target": block_average(rows, lambda row: row["target"]),
        "per_seed_mae": {str(seed): mae(rows_for(test, values))
                         for seed, values in zip(MODEL_SEEDS, samples, strict=True)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-data", type=Path, default=STEP1 / "data")
    parser.add_argument("--step1-outputs", type=Path, default=STEP1 / "outputs")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "tuning")
    args = parser.parse_args()
    torch.set_num_threads(1)
    blocks = make_blocks(args.step1_data, args.step1_outputs)
    args.output.mkdir(parents=True, exist_ok=True)
    folds = {}
    for area in sorted({block.area for block in blocks}):
        training = [block for block in blocks if block.area != area]
        testing = [block for block in blocks if block.area == area]
        ranking, bands = select(training)
        selected = ranking[0]
        print(area, "selected from training district", selected, flush=True)
        rows, test = evaluate(training, testing, selected)
        folds[area] = {"selected": selected, "validation_bands": bands,
                       "validation_ranking": ranking, "outer_test": test}
        write(args.output / f"{area}_predictions.json", rows)
        print(area, "outer test", test, flush=True)
    write(args.output / "metrics.json", {"settings": SETTINGS, "checkpoints": CHECKPOINTS,
                                         "validation_seed": MODEL_SEEDS[0], "model_seeds": MODEL_SEEDS,
                                         "folds": folds})


if __name__ == "__main__":
    main()
