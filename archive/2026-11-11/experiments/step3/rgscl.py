"""Adapt the released RGSCL encoder to PACE's district-held-out POI task."""

import argparse
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from shapely import delaunay_triangles, get_coordinates
from shapely.geometry import MultiPoint
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from audit_tuning import shared_entities, summarize
from building_auxiliary import direct_predictions
from cluster import read, write
from data import (HERE, STEP1, TEST_MASK_SEEDS, TRAIN_MASK_SEEDS, load_buildings, make_blocks,
                  make_view, pair_share, place_counts, training_vocabulary)
from model import MODEL_SEEDS
from tune import VALIDATION_MASK_SEEDS, mae, rows_for, split_bands

AUTHOR_MODEL = HERE.parents[1] / "references" / "RGSCL" / "code" / "model.py"
WIDTH = 64
ENCODER_EPOCHS = (50, 100, 200)
HEAD_EPOCHS = (15, 45, 100)
METHODS = ("rgscl", "rgscl_with_visible_share")
PROTOCOL = {
    "encoder_epochs": list(ENCODER_EPOCHS), "encoder_width": WIDTH, "attention_heads": 4,
    "intra_weight": 0.7, "l2_weight": 1e-5, "encoder_lr": 0.0002,
    "head_epochs": list(HEAD_EPOCHS), "head_width": 16, "head_lr": 0.003,
    "inner_seeds": list(MODEL_SEEDS[:2]), "outer_seeds": list(MODEL_SEEDS),
    "train_masks": list(TRAIN_MASK_SEEDS), "validation_masks": list(VALIDATION_MASK_SEEDS),
    "test_masks": list(TEST_MASK_SEEDS),
}


def load_author_model():
    spec = spec_from_file_location("rgscl_author_model", AUTHOR_MODEL)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def morphology(polygon):
    area, perimeter = polygon.area, polygon.length
    centroid = np.array((polygon.centroid.x, polygon.centroid.y))
    radius = np.linalg.norm(get_coordinates(polygon) - centroid, axis=1).mean()
    rectangle = get_coordinates(polygon.minimum_rotated_rectangle)
    sides = rectangle[1:3] - rectangle[:2]
    lengths = np.linalg.norm(sides, axis=1)
    long_side = sides[lengths.argmax()]
    orientation = np.arctan2(long_side[1], long_side[0]) % np.pi
    compactness = 4 * np.pi * area / perimeter**2
    fractality = 1 - np.log(area) / (2 * np.log(perimeter)) if perimeter != 1 else 0.0
    concavity = area / polygon.convex_hull.area
    elongation = lengths.max() / lengths.min()
    return centroid, [area, perimeter, radius, orientation, compactness, fractality, concavity, elongation]


def region_edges(coordinates):
    by_point = defaultdict(list)
    for i, point in enumerate(coordinates):
        by_point[tuple(point)].append(i)
    edges = set()
    triangles = delaunay_triangles(MultiPoint(list(by_point)), only_edges=True)
    for edge in triangles.geoms:
        a, b = map(tuple, edge.coords)
        edges.add(tuple(sorted((by_point[a][0], by_point[b][0]))))
    if not edges:
        # Two points or a collinear set have no triangles. Connect consecutive unique points.
        ordered = [by_point[point][0] for point in sorted(by_point)]
        edges.update(zip(ordered, ordered[1:]))
    edges.update((indices[0], i) for indices in by_point.values() for i in indices[1:])
    dummy = len(coordinates)
    edges.update((i, dummy) for i in range(dummy))
    undirected = sorted(edges | {(b, a) for a, b in edges})
    return torch.tensor(undirected, dtype=torch.long).reshape(-1, 2).T.contiguous()


def region_graph(features, coordinates):
    dummy = features.mean(axis=0) if len(features) else np.zeros(features.shape[1])
    return Data(
        x=torch.tensor(np.vstack([features, dummy]), dtype=torch.float32),
        edge_index=region_edges(coordinates),
        dummy=torch.arange(len(features) + 1) == len(features),
    )


def batch_views(graphs):
    poi, buildings = zip(*graphs, strict=True)
    return Batch.from_data_list(poi), Batch.from_data_list(buildings)


def make_graphs(training, evaluation, geometry):
    vocab = training_vocabulary(training)
    building_ids = sorted({key for view in training for key in view.block.buildings})
    shape_features = np.array([geometry[key][1] for key in building_ids])
    center, scale = shape_features.mean(axis=0), shape_features.std(axis=0)
    scale[scale == 0] = 1
    eyes = np.eye(len(vocab))
    building_graphs = {}
    pairs = []
    for view in [*training, *evaluation]:
        # Block numbers restart in each district. Cache only within a district.
        key = (view.block.area, view.block.key)
        if key not in building_graphs:
            footprints = [geometry[bid] for bid in view.block.buildings]
            positions = [point for point, _ in footprints]
            features = np.array([shape for _, shape in footprints]).reshape(-1, len(center))
            building_graphs[key] = region_graph((features - center) / scale, positions)
        categories = [vocab.get(category, vocab[None]) for category, _ in view.visible]
        poi = region_graph(eyes[categories], view.visible_coordinates)
        pairs.append((poi, building_graphs[key]))
    return pairs, len(vocab)


def class_balanced_loss(logits, positives):
    if not positives.any() or positives.all():
        return logits.sum() * 0
    return F.softplus(-logits[positives]).mean() + F.softplus(logits[~positives]).mean()


def cross_view_loss(nodes, other_view_regions, node_graph, discriminator):
    logits = nodes @ discriminator.score_weight @ other_view_regions.T
    matches = node_graph[:, None] == torch.arange(len(other_view_regions))[None, :]
    return class_balanced_loss(logits, matches)


def fused_regions(model, poi, buildings, *, augmented=False):
    p = model.ecd_poi(poi.x, poi.edge_index, auger=augmented)
    b = model.ecd_building(buildings.x, buildings.edge_index, auger=augmented)
    fused = model.attn_fusion(p[poi.dummy], b[buildings.dummy])
    norm = model.bn2 if augmented else model.bn1
    return p, b, norm(fused)


def pretraining_loss(model, poi, buildings):
    p, b, region = fused_regions(model, poi, buildings)
    _, _, augmented = fused_regions(model, poi, buildings, augmented=True)
    p, b = model.proj_poi(p), model.proj_building(b)
    intra = cross_view_loss(p[~poi.dummy], b[buildings.dummy], poi.batch[~poi.dummy],
                            model.disc_intra_region)
    intra = intra + cross_view_loss(b[~buildings.dummy], p[poi.dummy], buildings.batch[~buildings.dummy],
                                    model.disc_intra_region)
    logits = model.proj_region(augmented) @ model.disc_inter_region.score_weight @ model.proj_region(region).T
    inter = class_balanced_loss(logits, torch.eye(len(region), dtype=torch.bool))
    return PROTOCOL["intra_weight"] * intra + (1 - PROTOCOL["intra_weight"]) * inter


def visible_features(views):
    result = []
    for view in views:
        counts = place_counts(view)
        share = pair_share(counts)
        result.append([0 if share is None else share, float(share is None), np.log1p(counts.total())])
    return np.array(result)


def head_predictions(train_x, eval_x, train, epochs, seed):
    torch.manual_seed(seed)
    center, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale == 0] = 1
    train_x, eval_x = [torch.tensor((x - center) / scale, dtype=torch.float32) for x in (train_x, eval_x)]
    target = torch.tensor([v.target for v in train], dtype=torch.float32)
    head = nn.Sequential(nn.Linear(train_x.shape[1], PROTOCOL["head_width"]), nn.ReLU(),
                         nn.Linear(PROTOCOL["head_width"], 1))
    optimizer = torch.optim.Adam(head.parameters(), lr=PROTOCOL["head_lr"], weight_decay=0.01)
    predictions = {}
    for epoch in range(1, max(epochs) + 1):
        optimizer.zero_grad()
        loss = F.mse_loss(head(train_x).flatten().sigmoid(), target)
        loss.backward()
        optimizer.step()
        if epoch in epochs:
            with torch.no_grad():
                predictions[epoch] = head(eval_x).flatten().sigmoid().numpy()
    return predictions


def make_fold(train_blocks, eval_blocks, eval_masks, geometry):
    train_views = [make_view(block, seed) for seed in TRAIN_MASK_SEEDS for block in train_blocks]
    evaluation = [view for block in eval_blocks for seed in eval_masks
                  if (view := make_view(block, seed)).target is not None]
    graphs, vocabulary_size = make_graphs(train_views, evaluation, geometry)
    n_blocks = len(train_blocks)
    pretraining = [batch_views(graphs[start:start + n_blocks])
                   for start in range(0, len(train_views), n_blocks)]
    labeled = [(view, graph) for view, graph in zip(train_views, graphs[:len(train_views)], strict=True)
               if view.target is not None]
    train = [view for view, _ in labeled]
    scored_graphs = [graph for _, graph in labeled] + graphs[len(train_views):]
    batches = [batch_views(scored_graphs[start:start + 32]) for start in range(0, len(scored_graphs), 32)]
    stats = visible_features(train + evaluation)
    return train, evaluation, pretraining, batches, stats, vocabulary_size


def fit(author, fold, seed, checkpoints, head_epochs):
    train, _, pretraining, scoring, stats, vocabulary_size = fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = author.RGSCLNet(vocabulary_size, 8, WIDTH, len(pretraining[0][0].ptr) - 1, 1, "gat",
                            PROTOCOL["attention_heads"], "prelu", PROTOCOL["intra_weight"],
                            PROTOCOL["l2_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL["encoder_lr"],
                                 weight_decay=2 * PROTOCOL["l2_weight"])
    predictions = {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        optimizer.zero_grad()
        loss = pretraining_loss(model, *pretraining[rng.integers(len(pretraining))])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if epoch not in checkpoints:
            continue
        model.eval()
        with torch.no_grad():
            embeddings = torch.cat([fused_regions(model, *batch)[2] for batch in scoring]).numpy()
        for method, features in zip(METHODS, (embeddings, np.column_stack((embeddings, stats))), strict=True):
            for head_epoch, values in head_predictions(features[:len(train)], features[len(train):],
                                                       train, head_epochs, seed).items():
                predictions[method, epoch, head_epoch] = values
    return predictions


def select(author, blocks, geometry):
    candidates = defaultdict(list)
    details = []
    bands = split_bands(blocks)
    for i, validation in enumerate(bands):
        train = [block for j, band in enumerate(bands) if i != j for block in band
                 if not any(shared_entities([block], validation))]
        assert not any(shared_entities(train, validation))
        fold = make_fold(train, validation, VALIDATION_MASK_SEEDS, geometry)
        traces = [fit(author, fold, seed, ENCODER_EPOCHS, HEAD_EPOCHS) for seed in MODEL_SEEDS[:2]]
        for key in traces[0]:
            values = np.mean([trace[key] for trace in traces], axis=0)
            candidates[key].extend(rows_for(fold[1], values))
        details.append({"training_blocks": sorted(b.key for b in train),
                        "validation_blocks": sorted(b.key for b in validation)})
        print(blocks[0].area, "validated band", i + 1, flush=True)

    ranking = {method: sorted([
        {"encoder_epochs": enc, "head_epochs": head, "mae": mae(rows)}
        for (name, enc, head), rows in candidates.items() if name == method
    ], key=lambda row: (row["mae"], row["encoder_epochs"], row["head_epochs"])) for method in METHODS}
    return {"protocol": PROTOCOL, "training_area": blocks[0].area, "ranking": ranking, "bands": details}


def evaluate(author, training, test, geometry, selection):
    assert not any(shared_entities(training, test))
    fold = make_fold(training, test, TEST_MASK_SEEDS, geometry)
    train_views, test_views = fold[:2]
    chosen = {name: (name, rows[0]["encoder_epochs"], rows[0]["head_epochs"])
              for name, rows in selection["ranking"].items()}
    traces = [fit(author, fold, seed, sorted({key[1] for key in chosen.values()}),
                  sorted({key[2] for key in chosen.values()})) for seed in MODEL_SEEDS]
    area = test[0].area
    prior = read(HERE / "outputs" / "building_auxiliary" / f"{area}_predictions.json")
    direct = direct_predictions(train_views, test_views)
    rows = []
    predictions = {name: np.mean([trace[key] for trace in traces], axis=0) for name, key in chosen.items()}
    for i, (view, reference) in enumerate(zip(test_views, prior, strict=True)):
        assert (view.block.key, view.mask_seed, view.target) == (
            reference["block_id"], reference["mask_seed"], reference["target"])
        assert np.isclose(direct[i], reference["visible_pair_share"], rtol=0, atol=1e-12)
        rows.append({**reference, **{name: float(values[i]) for name, values in predictions.items()}})

    def scores(scored):
        return {name: summarize([{**row, "prediction": row[name]} for row in scored])
                for name in ("visible_pair_share", "ridge", "gatv2", *METHODS)}

    sparse = [row for row in rows if row["visible_matched_pois"] < 2]
    return rows, {
        "training_area": training[0].area,
        "train_blocks": len({v.block.key for v in train_views}),
        "pretrain_blocks": len(training),
        "test_blocks": len({v.block.key for v in test_views}),
        "test_masks": len(test_views),
        "selected": {name: candidates[0] for name, candidates in selection["ranking"].items()},
        "scores": scores(rows),
        "sparse_blocks": len({row["block_id"] for row in sparse}),
        "sparse_scores": scores(sparse) if sparse else {},
        "mae_by_seed": {name: {str(seed): mae(rows_for(test_views, trace[key]))
                              for seed, trace in zip(MODEL_SEEDS, traces, strict=True)}
                        for name, key in chosen.items()},
    }


def load_area(area):
    blocks = make_blocks(STEP1 / "data", STEP1 / "outputs", areas={area}, include_coordinates=True)
    polygons = load_buildings(STEP1 / "data" / f"{area}_buildings.geojson")
    geometry = {key: morphology(polygons[key]) for key in {key for b in blocks for key in b.buildings}}
    return blocks, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "test"))
    parser.add_argument("--training-area")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "rgscl")
    args = parser.parse_args()
    torch.set_num_threads(1)
    author = load_author_model()
    areas = sorted(read(STEP1 / "areas.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "select":
        if args.training_area not in areas:
            parser.error("--training-area must name a study district")
        path = args.output / f"{args.training_area}_selection.json"
        if path.exists():
            parser.error("Selection already sealed in this output directory")
        selection = select(author, *load_area(args.training_area))
        write(path, selection)
        print(args.training_area, {method: ranking[0] for method, ranking in selection["ranking"].items()}, flush=True)
        return

    selections = {area: read(args.output / f"{area}_selection.json") for area in areas}
    for area, selection in selections.items():
        assert selection["protocol"] == PROTOCOL and selection["training_area"] == area
    path = args.output / "metrics.json"
    if path.exists():
        parser.error("Outer evaluation already recorded in this output directory")
    loaded = {area: load_area(area) for area in areas}
    results = {}
    for train_area, test_area in (areas, list(reversed(areas))):
        training, train_geometry = loaded[train_area]
        test, test_geometry = loaded[test_area]
        rows, results[test_area] = evaluate(author, training, test, train_geometry | test_geometry,
                                             selections[train_area])
        write(args.output / f"{test_area}_predictions.json", rows)
        print(test_area, {name: item["mae"] for name, item in results[test_area]["scores"].items()}, flush=True)
    write(path, {"protocol": PROTOCOL, "folds": results})


if __name__ == "__main__":
    main()
