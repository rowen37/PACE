"""Measure the incremental value of a building assignment training objective."""

import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

from data import (HERE, STEP1, TEST_MASK_SEEDS, TRAIN_MASK_SEEDS, make_blocks, make_graph,
                  pair_share, place_counts, summary, training_vocabulary, views_for)
from audit_tuning import mean_predictions, shared_entities, summarize
from cluster import read, write
from model import MODEL_SEEDS, graph_checkpoints
from tune import VALIDATION_MASK_SEEDS, block_average, mae, rows_for, split_bands

CHECKPOINTS = [5, 15, 30, 45]
RIDGE_PENALTIES = [0.01, 0.1, 1.0, 10.0]
AUX_WEIGHTS = [0.0, 0.03, 0.1, 0.3]
GRAPH_SETTINGS = {"width": 16, "lr": 0.004, "weight_decay": 0.01, "loss_name": "mae"}
PROTOCOL = {
    "train_masks": list(TRAIN_MASK_SEEDS), "validation_masks": list(VALIDATION_MASK_SEEDS),
    "test_masks": list(TEST_MASK_SEEDS), "inner_seeds": list(MODEL_SEEDS[:2]),
    "outer_seeds": list(MODEL_SEEDS), "checkpoints": CHECKPOINTS,
    "ridge_penalties": RIDGE_PENALTIES, "auxiliary_weights": AUX_WEIGHTS,
    "graph_settings": GRAPH_SETTINGS,
    "auxiliary_target": "visible POI building assignment among all footprints in its block",
    "auxiliary_inputs": ["POI category", "building footprint area", "building compactness"],
    "purged_entities": ["building_footprint", "original_nycwalks_segment"],
}
METHODS = ("visible_pair_share", "ridge", "gatv2", "gatv2_building_auxiliary")


def prepare_inputs(train, evaluation):
    vocab = training_vocabulary(train)
    train_counts, evaluation_counts = ([place_counts(view) for view in views]
                                       for views in (train, evaluation))
    train_x, eval_x = (
        np.array([summary(view, vocab) + [np.log1p(count.total())]
                  for view, count in zip(views, counts, strict=True)])
        for views, counts in ((train, train_counts), (evaluation, evaluation_counts))
    )
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale == 0] = 1
    train_x, eval_x = (train_x - center) / scale, (eval_x - center) / scale
    graphs = []
    for views, features, include_target in ((train, train_x, True), (evaluation, eval_x, False)):
        batch = []
        for view, row in zip(views, features, strict=True):
            graph = make_graph(view, vocab, include_target=include_target)
            graph["block"].x = torch.tensor(row[None, :], dtype=torch.float32)
            batch.append(graph)
        graphs.append(batch)
    assert all("y" not in graph for graph in graphs[1])
    return train_x, eval_x, graphs[0], graphs[1], len(vocab)


def direct_predictions(train, evaluation):
    prior = block_average(rows_for(train, [v.target for v in train]), lambda row: row["target"])
    shares = (pair_share(place_counts(view)) for view in evaluation)
    return [prior if share is None else share for share in shares]


def ridge_predictions(train, train_x, eval_x, penalty):
    y = np.array([view.target for view in train])
    y_mean = y.mean()
    gram = train_x.T @ train_x / len(y)
    beta = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]),
                           train_x.T @ (y - y_mean) / len(y))
    return np.clip(y_mean + eval_x @ beta, 0, 1)


def building_assignment_loss(model, batch, weight):
    # The edge supplies the training label only. Score morphology without the visible
    # occupancy count and category without any message from the building or its neighbors.
    place_index, building_index = batch["place", "in_building", "building"].edge_index
    buildings = batch["building"].x
    block_index = batch["place"].batch[place_index]
    building_blocks = batch["building"].batch
    eligible = torch.bincount(building_blocks, minlength=batch.num_graphs)[block_index] > 1
    if not eligible.any():
        return buildings.new_zeros(())
    places = model.inputs["place"](batch["place"].x[place_index[eligible]])
    morphology = torch.cat((buildings[:, :2], torch.zeros_like(buildings[:, 2:])), dim=1)
    footprint_embeddings = model.inputs["building"](morphology)
    logits = places @ footprint_embeddings.T / places.shape[1] ** 0.5
    same_block = block_index[eligible, None] == building_blocks[None, :]
    logits = logits.masked_fill(~same_block, -torch.inf)
    return weight * F.cross_entropy(logits, building_index[eligible])


def graph_traces(train_graphs, eval_graphs, vocabulary_size, epochs, seeds, weight):
    auxiliary = partial(building_assignment_loss, weight=weight) if weight else None
    return [
        graph_checkpoints(train_graphs, eval_graphs, vocabulary_size, GATv2Conv,
                          epochs, seed, auxiliary_loss=auxiliary, **GRAPH_SETTINGS)[0]
        for seed in seeds
    ]


def select(training_blocks):
    bands = split_bands(training_blocks)
    ridge_rows, graph_rows = defaultdict(list), defaultdict(list)
    direct_rows, details = [], []
    area = training_blocks[0].area
    for i, validation in enumerate(bands):
        candidates = [block for j, band in enumerate(bands) if i != j for block in band]
        train_blocks = [block for block in candidates if not any(shared_entities([block], validation))]
        assert not any(shared_entities(train_blocks, validation))
        train = views_for(train_blocks, TRAIN_MASK_SEEDS)
        valid = views_for(validation, VALIDATION_MASK_SEEDS)
        train_x, valid_x, train_graphs, valid_graphs, categories = prepare_inputs(train, valid)
        direct_rows.extend(rows_for(valid, direct_predictions(train, valid)))
        for penalty in RIDGE_PENALTIES:
            ridge_rows[penalty].extend(rows_for(valid, ridge_predictions(train, train_x, valid_x, penalty)))
        for weight in AUX_WEIGHTS:
            traces = graph_traces(train_graphs, valid_graphs, categories, CHECKPOINTS, MODEL_SEEDS[:2], weight)
            for epoch in CHECKPOINTS:
                graph_rows[weight, epoch].extend(rows_for(valid, mean_predictions(traces, epoch)))
        details.append({
            "training_blocks": sorted({v.block.key for v in train}),
            "validation_blocks": sorted({v.block.key for v in valid}),
            "purged_training_blocks": sorted({b.key for b in candidates} - {b.key for b in train_blocks}),
        })
        print(area, "finished validation band", i + 1, flush=True)

    ridge = sorted(({"penalty": p, "mae": mae(rows)} for p, rows in ridge_rows.items()),
                   key=lambda row: (row["mae"], row["penalty"]))
    baseline = sorted(({"epoch": e, "mae": mae(graph_rows[0, e])} for e in CHECKPOINTS),
                      key=lambda row: (row["mae"], row["epoch"]))
    # Match the stopping epoch, chosen on the no-auxiliary baseline, in the two graph arms.
    epoch = baseline[0]["epoch"]
    auxiliary = sorted(({"weight": w, "mae": mae(graph_rows[w, epoch])} for w in AUX_WEIGHTS),
                       key=lambda row: (row["mae"], row["weight"]))
    return {
        "protocol": PROTOCOL, "training_area": area, "validation_bands": details,
        "direct_validation_mae": mae(direct_rows), "ridge_ranking": ridge, "epoch_ranking": baseline,
        "auxiliary_ranking": auxiliary, "selected_weight": auxiliary[0]["weight"],
        "positive_weight_for_comparison": next(row["weight"] for row in auxiliary if row["weight"] > 0),
    }


def method_scores(rows):
    return {method: summarize([{**row, "prediction": row[method]} for row in rows]) for method in METHODS}


def paired_difference(rows, candidate, baseline):
    deltas = defaultdict(list)
    for row in rows:
        actual = row["target"]
        deltas[row["block_id"]].append(abs(row[candidate] - actual) - abs(row[baseline] - actual))
    by_block = {block: mean(values) for block, values in deltas.items()}
    return {
        "mae_difference": mean(by_block.values()),
        "blocks_improved": sum(value < 0 for value in by_block.values()),
        "blocks_worsened": sum(value > 0 for value in by_block.values()),
        "blocks_tied": sum(value == 0 for value in by_block.values()),
        "by_block": by_block,
    }


def evaluate(train_blocks, test_blocks, selection):
    assert not any(shared_entities(train_blocks, test_blocks))
    train = views_for(train_blocks, TRAIN_MASK_SEEDS)
    test = views_for(test_blocks, TEST_MASK_SEEDS)
    train_x, test_x, train_graphs, test_graphs, categories = prepare_inputs(train, test)
    epoch = selection["epoch_ranking"][0]["epoch"]
    weight = selection["positive_weight_for_comparison"]
    penalty = selection["ridge_ranking"][0]["penalty"]
    predictions = {"visible_pair_share": direct_predictions(train, test),
                   "ridge": ridge_predictions(train, train_x, test_x, penalty)}
    per_seed = {}
    for method, aux_weight in (("gatv2", 0), ("gatv2_building_auxiliary", weight)):
        traces = graph_traces(train_graphs, test_graphs, categories, [epoch], MODEL_SEEDS, aux_weight)
        predictions[method] = mean_predictions(traces, epoch)
        per_seed[method] = {str(seed): mae(rows_for(test, trace[epoch]))
                            for seed, trace in zip(MODEL_SEEDS, traces, strict=True)}
    rows = [{"block_id": view.block.key, "mask_seed": view.mask_seed, "target": view.target,
             "visible_matched_pois": place_counts(view).total(),
             **{method: float(predictions[method][i]) for method in METHODS}}
            for i, view in enumerate(test)]
    area = test_blocks[0].area
    reference = read(HERE / "outputs" / f"{area}_predictions.json")
    for row, old in zip(rows, reference, strict=True):
        assert all(row[key] == old[key] for key in ("block_id", "mask_seed", "target"))
        assert np.isclose(row["visible_pair_share"], old["visible_pair_share"], rtol=0, atol=1e-12)
    by_count = {}
    for label, lower, upper in (("0–1", 0, 2), ("2–3", 2, 4), ("4–7", 4, 8), ("8+", 8, np.inf)):
        group = [row for row in rows if lower <= row["visible_matched_pois"] < upper]
        if group:
            by_count[label] = {"blocks": len({row["block_id"] for row in group}), "masks": len(group),
                               "mae": {method: result["mae"] for method, result in method_scores(group).items()}}
    return rows, {
        "training_area": train_blocks[0].area, "training_blocks": len({v.block.key for v in train}),
        "test_blocks": len({v.block.key for v in test}), "test_masks": len(test), "epoch": epoch,
        "ridge_penalty": penalty, "tested_positive_weight": weight,
        "selected_weight": selection["selected_weight"], "scores": method_scores(rows),
        "paired_aux_vs_gatv2": paired_difference(rows, "gatv2_building_auxiliary", "gatv2"),
        "mae_by_seed": per_seed, "by_visible_matched_pois": by_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--training-area")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "building_auxiliary")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    areas = sorted(read(STEP1 / "areas.json"))
    if args.phase == "select":
        if args.training_area not in areas:
            parser.error("--training-area must name a study district")
        path = args.output / f"{args.training_area}_selection.json"
        if path.exists():
            parser.error("Selection already sealed in this output directory")
        selection = select(make_blocks(STEP1 / "data", STEP1 / "outputs", areas={args.training_area}))
        write(path, selection)
        print(args.training_area, "sealed", selection["epoch_ranking"][0], selection["auxiliary_ranking"], flush=True)
        return

    selections = {area: read(args.output / f"{area}_selection.json") for area in areas}
    for area, selection in selections.items():
        assert selection["protocol"] == PROTOCOL and selection["training_area"] == area
        assert selection["selected_weight"] == selection["auxiliary_ranking"][0]["weight"]
    path = args.output / "metrics.json"
    if path.exists():
        parser.error("Outer evaluation already recorded in this output directory")
    blocks = {area: make_blocks(STEP1 / "data", STEP1 / "outputs", areas={area}) for area in areas}
    results = {}
    for train_area, test_area in (areas, list(reversed(areas))):
        rows, results[test_area] = evaluate(blocks[train_area], blocks[test_area], selections[train_area])
        write(args.output / f"{test_area}_predictions.json", rows)
        print(test_area, {method: row["mae"] for method, row in results[test_area]["scores"].items()}, flush=True)
    write(path, {"protocol": PROTOCOL, "folds": results})


if __name__ == "__main__":
    main()
