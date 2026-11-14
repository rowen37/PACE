"""Tune GATv2 in isolated training districts, then score the sealed selections."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import StatisticsError, correlation, mean

import numpy as np
import torch
from torch_geometric.nn import GATv2Conv

from data import (HERE, STEP1, TEST_MASK_SEEDS, TRAIN_MASK_SEEDS, make_blocks, make_graph,
                  training_vocabulary, views_for)
from model import MODEL_SEEDS, graph_checkpoints
from cluster import read, write
from tune import VALIDATION_MASK_SEEDS, block_average, mae, rows_for, split_bands

SETTINGS = {
    "original": {},
    "lower_lr": {"lr": 0.001},
    "narrower": {"width": 8},
    "stronger_decay": {"weight_decay": 0.10},
    "feature_dropout": {"dropout": 0.20},
    "mae_objective": {"loss_name": "mae"},
    "small_regularized": {"width": 8, "weight_decay": 0.05},
}
CHECKPOINTS = [1, 5, 15, 30, 45]
INNER_SEEDS = MODEL_SEEDS[:2]
PROTOCOL = {
    "settings": SETTINGS, "checkpoints": CHECKPOINTS, "inner_seeds": list(INNER_SEEDS),
    "outer_seeds": list(MODEL_SEEDS), "train_masks": list(TRAIN_MASK_SEEDS),
    "validation_masks": list(VALIDATION_MASK_SEEDS), "test_masks": list(TEST_MASK_SEEDS),
    "purged_entities": ["building_footprint", "original_nycwalks_segment"],
}


def shared_entities(left, right):
    left_buildings = {key for block in left for key in block.buildings}
    left_walks = {key for block in left for key in block.source_walk_edges}
    right_buildings = {key for block in right for key in block.buildings}
    right_walks = {key for block in right for key in block.source_walk_edges}
    return left_buildings & right_buildings, left_walks & right_walks


def graph_inputs(train, other):
    vocab = training_vocabulary(train)
    return ([make_graph(view, vocab) for view in train],
            [make_graph(view, vocab, include_target=False) for view in other], len(vocab))


def mean_predictions(traces, epoch):
    return np.mean([trace[epoch] for trace in traces], axis=0)


def select(training_blocks):
    bands = split_bands(training_blocks)
    trials = defaultdict(list)
    details = []
    for i, validation in enumerate(bands):
        candidates = [block for j, band in enumerate(bands) if i != j for block in band]
        train_blocks = [b for b in candidates if not any(shared_entities([b], validation))]
        assert not any(shared_entities(train_blocks, validation))
        train = views_for(train_blocks, TRAIN_MASK_SEEDS)
        valid = views_for(validation, VALIDATION_MASK_SEEDS)
        if not train or not valid:
            raise ValueError(f"Insufficient independent labels in validation band {i + 1}")
        train_graphs, valid_graphs, vocabulary_size = graph_inputs(train, valid)
        details.append({
            "training_blocks": sorted({v.block.key for v in train}),
            "validation_blocks": sorted({v.block.key for v in valid}),
            "purged_training_blocks": sorted({b.key for b in candidates} - {b.key for b in train_blocks}),
        })
        for setting, kwargs in SETTINGS.items():
            traces = [graph_checkpoints(train_graphs, valid_graphs, vocabulary_size, GATv2Conv,
                                        CHECKPOINTS, seed, track_training=True, **kwargs)
                      for seed in INNER_SEEDS]
            for epoch in CHECKPOINTS:
                train_rows = rows_for(train, mean_predictions([trace[2] for trace in traces], epoch))
                valid_rows = rows_for(valid, mean_predictions([trace[0] for trace in traces], epoch))
                trials[setting, epoch].append((train_rows, valid_rows))
        print(training_blocks[0].area, "finished validation band", i + 1, flush=True)

    ranking = []
    for (setting, epoch), folds in trials.items():
        val = [row for _, validation in folds for row in validation]
        ranking.append({"setting": setting, "epoch": epoch, "validation_mae": mae(val),
                        "train_mae": mean(mae(train) for train, _ in folds),
                        "validation_mae_by_band": [mae(valid) for _, valid in folds]})
    ranking.sort(key=lambda row: (row["validation_mae"], row["epoch"], row["setting"]))
    return {"protocol": PROTOCOL, "training_area": training_blocks[0].area,
            "selected": ranking[0], "validation_bands": details, "ranking": ranking}


def summarize(rows):
    by_block = defaultdict(list)
    for row in rows:
        by_block[row["block_id"]].append(row)
    actual = [mean(row["target"] for row in block) for block in by_block.values()]
    predicted = [mean(row["prediction"] for row in block) for block in by_block.values()]
    try:
        rank = correlation(actual, predicted, method="ranked")
    except StatisticsError:
        rank = None
    return {"mae": mae(rows), "mean_actual": mean(actual), "mean_prediction": mean(predicted),
            "bias": mean(predicted) - mean(actual), "spearman_by_block": rank}


def external_test(train_blocks, test_blocks, selected):
    assert not any(shared_entities(train_blocks, test_blocks))
    train = views_for(train_blocks, TRAIN_MASK_SEEDS)
    test = views_for(test_blocks, TEST_MASK_SEEDS)
    train_graphs, test_graphs, vocabulary_size = graph_inputs(train, test)
    assert all("y" not in graph for graph in test_graphs)
    epoch = selected["epoch"]
    traces = [graph_checkpoints(train_graphs, test_graphs, vocabulary_size, GATv2Conv,
                                (epoch,), seed, track_training=True, **SETTINGS[selected["setting"]])
              for seed in MODEL_SEEDS]
    train_rows = rows_for(train, mean_predictions([trace[2] for trace in traces], epoch))
    test_rows = rows_for(test, mean_predictions([trace[0] for trace in traces], epoch))
    area = test_blocks[0].area
    reference = read(HERE / "outputs" / f"{area}_predictions.json")
    for left, right in zip(test_rows, reference, strict=True):
        assert (left["block_id"], left["mask_seed"], left["target"]) == (
            right["block_id"], right["mask_seed"], right["target"])
    return test_rows, {
        "train": summarize(train_rows), "test": summarize(test_rows),
        "test_blocks": len({v.block.key for v in test}),
        "by_seed_test_mae": [mae(rows_for(test, trace[0][epoch])) for trace in traces],
        "reference": {name: summarize([{**row, "prediction": row[name]} for row in reference])
                      for name in ("gatv2", "visible_pair_share", "pca")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--training-area")
    parser.add_argument("--step1-data", type=Path, default=STEP1 / "data")
    parser.add_argument("--step1-outputs", type=Path, default=STEP1 / "outputs")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "tuning_audit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    areas = sorted(read(STEP1 / "areas.json"))
    if args.phase == "select":
        if args.training_area not in areas:
            parser.error("--training-area must name a study district for selection")
        path = args.output / f"{args.training_area}_selection.json"
        if path.exists():
            parser.error("Selection already sealed in this output directory")
        blocks = make_blocks(args.step1_data, args.step1_outputs, areas={args.training_area})
        selection = select(blocks)
        write(path, selection)
        print(args.training_area, "sealed", selection["selected"], flush=True)
        return

    selections = {area: read(args.output / f"{area}_selection.json") for area in areas}
    for area, selection in selections.items():
        assert selection["protocol"] == PROTOCOL and selection["training_area"] == area
        assert selection["selected"] == selection["ranking"][0]
    if (args.output / "metrics.json").exists():
        parser.error("Outer evaluation already recorded in this output directory")
    blocks = {area: make_blocks(args.step1_data, args.step1_outputs, areas={area}) for area in areas}
    results = {}
    for train_area, test_area in (areas, list(reversed(areas))):
        selected = selections[train_area]["selected"]
        rows, report = external_test(blocks[train_area], blocks[test_area], selected)
        results[test_area] = {"training_area": train_area, "selected": selected, **report}
        write(args.output / f"{test_area}_predictions.json", rows)
        print(test_area, results[test_area], flush=True)
    write(args.output / "metrics.json", {"protocol": PROTOCOL, "folds": results})


if __name__ == "__main__":
    main()
