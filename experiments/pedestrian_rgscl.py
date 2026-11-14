"""Train and export the three-view RGSCL prototype on the two development folds."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "archive/2026-11-11/experiments/step3"
sys.path.insert(0, str(ROOT / "src"))
from pace.pedestrian_rgscl import ThreeViewRGSCL, VIEW_NAMES, WALK_FEATURES, walk_graph

sys.path.insert(0, str(LEGACY))
import rgscl as baseline

SNAPSHOT = "nyc-2026-11-09"
AREAS = ("hudson_yards", "soho")
WIDTH, HEADS = 64, 4


def observed_views(blocks):
    return [SimpleNamespace(block=b, visible=b.places, visible_coordinates=b.place_coordinates)
            for b in blocks]


def batch(graph_pairs, blocks):
    poi, buildings = baseline.batch_views(graph_pairs)
    walk = Batch.from_data_list([walk_graph(b.walk_features, b.walk_pairs) for b in blocks])
    return poi, buildings, walk


def gradient_norm(module):
    return sum(float(p.grad.detach().square().sum())
               for p in module.parameters() if p.grad is not None) ** 0.5


def train_fold(train, test, geometry, encoder, epochs, seed):
    if any(baseline.shared_entities(train, test)):
        raise ValueError("Train and test districts share source entities")
    pairs, poi_dim = baseline.make_graphs(observed_views(train), observed_views(test), geometry)
    training = batch(pairs[:len(train)], train)
    evaluation = batch(pairs[len(train):], test)
    torch.manual_seed(seed)
    model = ThreeViewRGSCL(encoder, poi_dim, width=WIDTH, heads=HEADS)
    optim = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=2e-5)
    losses = []
    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        loss = model.loss(*training)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        if epoch == 0:
            gradients = {name: gradient_norm(m) for name, m in zip(VIEW_NAMES, model.encoders, strict=True)}
            gradients["attention"] = gradient_norm(model.fusion)
            gradients["walk_attention"] = gradient_norm(model.fusion.scorers[2])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optim.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        regions, weights, views, _ = model(*evaluation)
        torch.testing.assert_close(weights.sum(-1), torch.ones(len(test)))
        if not torch.isfinite(regions).all():
            raise RuntimeError("Non-finite test embeddings")
        # Counterfactual check: hold segment features fixed and remove walk connections.
        walk = evaluation[2].clone()
        walk.edge_index = walk.edge_index[:, :0]
        _, _, edited_views, _ = model(evaluation[0], evaluation[1], walk)
        topology_delta = (views[2] - edited_views[2]).abs().max().item()
    rows = [{"snapshot_id": SNAPSHOT, "area_id": block.area, "block_id": block.key,
             "embedding": vector.tolist(), "attention": dict(zip(VIEW_NAMES, alpha.tolist(), strict=True))}
            for block, vector, alpha in zip(test, regions, weights, strict=True)]
    checks = {"train_blocks": len(train), "test_blocks": len(test), "embedding_dim": WIDTH,
              "epochs": epochs, "seed": seed, "training_losses": losses,
              "first_step_grad_norm": gradients, "walk_topology_max_abs_delta": topology_delta,
              "test_walk_nodes": evaluation[2].num_nodes, "test_walk_graph_edges": evaluation[2].num_edges,
              "test_walk_weight_mean": weights[:, 2].mean().item()}
    return rows, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    author = baseline.load_author_model()
    loaded = {area: baseline.load_area(area) for area in AREAS}
    summary = {"snapshot_id": SNAPSHOT, "walk_features": WALK_FEATURES, "views": VIEW_NAMES, "folds": {}}
    for train_area, test_area in (AREAS, AREAS[::-1]):
        train, train_geometry = loaded[train_area]
        test, test_geometry = loaded[test_area]
        rows, checks = train_fold(train, test, train_geometry | test_geometry, author.Encoder,
                                  args.epochs, args.seed)
        summary["folds"][f"{train_area}_to_{test_area}"] = checks
        (args.output / f"{test_area}_embeddings.json").write_text(json.dumps(rows, allow_nan=False) + "\n")
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
