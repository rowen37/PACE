"""Make district-disjoint graphs with hidden commercial POIs."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
import sys

import numpy as np
from shapely import STRtree
import torch
from torch_geometric.data import HeteroData

HERE = Path(__file__).resolve().parent
STEP2 = HERE.parent / "step2"
sys.path.insert(0, str(STEP2))

from cluster import STEP1, prepare, read
from embedding import block_places
from run import load_buildings

TRAIN_MASK_SEEDS = range(8)
TEST_MASK_SEEDS = range(100, 105)

RELATIONS = (
    ("place", "in_building", "building"), ("building", "contains", "place"),
    ("walk", "touches", "walk"),
    ("building", "public_entrance", "walk"), ("walk", "access_to", "building"),
    ("place", "in_block", "block"), ("building", "in_block", "block"),
    ("walk", "in_block", "block"),
)


@dataclass
class SampleBlock:
    area: str
    key: str
    hectares: float
    places: list
    buildings: dict
    walk_features: list
    walk_pairs: list
    access_pairs: list
    walk_length: float
    center: tuple[float, float]
    source_walk_edges: set[str]
    place_coordinates: list[tuple[float, float]] | None = None


@dataclass
class View:
    block: SampleBlock
    visible: list
    hidden_matches: int
    target: float | None
    mask_seed: int
    visible_coordinates: list[tuple[float, float]] | None = None


def make_blocks(source, step1_output, *, areas=None, include_coordinates=False):
    configs = read(STEP1 / "areas.json")
    rows = []
    for area, config in configs.items():
        if areas is not None and area not in areas:
            continue
        blocks, _, _, entrances, _, _, entrance_edges, _ = prepare(area, config, source, step1_output)
        buildings = load_buildings(source / f"{area}_buildings.geojson")
        building_ids = list(buildings)
        building_index = STRtree(list(buildings.values()))
        places = block_places(source / f"{area}_places.geojson", blocks, buildings,
                              include_coordinates=include_coordinates)
        doors = {f["properties"]["graph_node"]: f["properties"]["building_id"] for f in entrances
                 if f["properties"]["in_largest_component"]}
        anchors = defaultdict(set)
        for edge in entrance_edges:
            p = edge["properties"]
            for door, anchor in ((p["u"], p["v"]), (p["v"], p["u"])):
                if door in doors:
                    anchors[anchor].add(doors[door])

        for key, records in places.items():
            block = blocks[key]
            ids = {building_ids[i] for i in building_index.query(block.geometry, predicate="intersects")
                   if buildings[building_ids[i]].intersection(block.geometry).area > 0}
            footprints = {}
            for bid in sorted(ids):
                polygon = buildings[bid]
                footprints[bid] = (math.log1p(polygon.area) / 10,
                                   4 * math.pi * polygon.area / polygon.length**2)

            incident = defaultdict(list)
            access = set()
            walk_features = []
            walk_length = 0
            edges = sorted(block.walk.edges(keys=True, data=True), key=lambda edge: edge[2])
            for i, (u, v, _, p) in enumerate(edges):
                start, end = p["geometry"].coords[0], p["geometry"].coords[-1]
                dx, dy = end[0] - start[0], end[1] - start[1]
                angle = math.atan2(dy, dx)
                length = p["length_m"]
                walk_length += length
                walk_features.append((math.log1p(length) / 5, math.cos(2 * angle), math.sin(2 * angle)))
                for node in {u, v}:
                    incident[node].append(i)
                    access.update((bid, i) for bid in anchors[node] & ids)
            pairs = {pair for indexes in incident.values() for pair in combinations(indexes, 2)}
            center = block.geometry.centroid.coords[0]
            source_walk_edges = {attrs["source_edge"].split(".", 1)[0] for *_, attrs in edges}
            coordinates = [record[2] for record in records] if include_coordinates else None
            rows.append(SampleBlock(area, key, block.geometry.area / 10_000,
                                    [record[:2] for record in records] if include_coordinates else records,
                                    footprints, walk_features, sorted(pairs), sorted(access),
                                    walk_length, center, source_walk_edges, coordinates))
    return rows


def pair_share(counts):
    total = counts.total()
    return sum(n * (n - 1) for n in counts.values()) / (total * (total - 1)) if total >= 2 else None


def make_view(block, seed):
    # Set a fixed observation fraction, with exactly the same inputs for every model.
    seed_sequence = np.random.SeedSequence([seed, int(block.key[1:]), *block.area.encode()])
    order = np.random.default_rng(seed_sequence).permutation(len(block.places))
    cut = len(order) // 2
    hidden = Counter(block.places[i][1] for i in order[cut:] if block.places[i][1] is not None)
    coordinates = None if block.place_coordinates is None else [block.place_coordinates[i] for i in order[:cut]]
    return View(block, [block.places[i] for i in order[:cut]], hidden.total(), pair_share(hidden), seed, coordinates)


def place_counts(view):
    return Counter(building for _, building in view.visible if building is not None)


def summary(view, vocab):
    counts = place_counts(view)
    types = Counter(category if category in vocab else None for category, _ in view.visible)
    n = len(view.visible)
    matched = counts.total()
    known_pair_share = pair_share(counts)
    return [
        *(math.sqrt(types[key] / max(n, 1)) for key in vocab),
        math.log1p(n / view.block.hectares),
        max(counts.values(), default=0) / max(matched, 1),
        known_pair_share if known_pair_share is not None else 0,
        float(known_pair_share is None),
        matched / max(n, 1),
        math.log1p(view.block.walk_length / view.block.hectares),
        math.log1p(len(view.block.buildings) / view.block.hectares),
        sum(x for x, _ in view.block.buildings.values()) / max(len(view.block.buildings), 1),
    ]


def make_graph(view, vocab, *, include_target=True):
    block = view.block
    graph = HeteroData()
    n_places = len(view.visible)
    counts = place_counts(view)
    category_ids = [vocab.get(label, vocab[None]) for label, _ in view.visible]
    graph["place"].x = torch.eye(len(vocab), dtype=torch.float32)[category_ids]
    building_ids = {bid: i for i, bid in enumerate(block.buildings)}
    graph["building"].x = torch.tensor([
        [area, compactness, math.log1p(counts[bid])]
        for bid, (area, compactness) in block.buildings.items()
    ], dtype=torch.float32).reshape(-1, 3)
    graph["walk"].x = torch.tensor(block.walk_features, dtype=torch.float32).reshape(-1, 3)
    graph["block"].x = torch.tensor(
        [[math.log1p(n_places / block.hectares),
          math.log1p(block.walk_length / block.hectares) / 8,
          counts.total() / max(n_places, 1)]], dtype=torch.float32)

    matches = [(i, building_ids[bid]) for i, (_, bid) in enumerate(view.visible) if bid is not None]
    walk = block.walk_pairs
    doors = [(building_ids[bid], segment) for bid, segment in block.access_pairs]
    pairs = [
        matches, [(b, a) for a, b in matches],
        walk + [(b, a) for a, b in walk],
        doors, [(b, a) for a, b in doors],
        [(i, 0) for i in range(n_places)],
        [(i, 0) for i in range(len(building_ids))],
        [(i, 0) for i in range(len(block.walk_features))],
    ]
    for relation, edges in zip(RELATIONS, pairs, strict=True):
        graph[relation].edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
    if include_target:
        graph.y = torch.tensor([view.target], dtype=torch.float32)
    return graph


def views_for(blocks, seeds):
    views = [make_view(block, seed) for block in blocks for seed in seeds]
    return [view for view in views if view.target is not None]


def training_vocabulary(train):
    vocab = {key: i for i, key in enumerate(sorted({category for view in train
                                                    for category, _ in view.visible}))}
    vocab[None] = len(vocab)
    return vocab
