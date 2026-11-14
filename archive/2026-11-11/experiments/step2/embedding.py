"""Learn a block embedding and merge only walk-adjacent, similar blocks."""

import argparse
from collections import Counter
from itertools import combinations
import math
from pathlib import Path

import networkx as nx
import numpy as np
from shapely import STRtree

from cluster import HERE, STEP1, adjacent, candidate, feature, layer, prepare, projected, read, write
from cluster import CONFIDENCE, load_buildings

COMMERCIAL = {"shopping", "food_and_drink", "lifestyle_services"}
MIN_POIS = 5
MIN_BUILDING_MATCHES = 3
DIMENSIONS = 6
THRESHOLDS = (0.55, 0.65, 0.80)
SHAPE_FEATURES = ("log_poi_density", "largest_building_share", "log_walk_density")


def block_places(path, blocks, buildings, *, include_coordinates=False):
    keys = sorted(key for key, block in blocks.items() if block.eligible)
    block_index = STRtree([blocks[key].geometry for key in keys])
    building_ids = list(buildings)
    building_index = STRtree(list(buildings.values()))
    places = {key: [] for key in keys}
    for item in read(path)["features"]:
        p = item["properties"]
        hierarchy = (p.get("taxonomy") or {}).get("hierarchy") or []
        if (p.get("confidence") or 0) < CONFIDENCE or not hierarchy or hierarchy[0] not in COMMERCIAL:
            continue
        point = projected(item)
        hits = block_index.query(point, predicate="within")
        if len(hits) != 1:
            continue
        matches = building_index.query(point, predicate="covered_by")
        building = building_ids[matches[0]] if len(matches) == 1 else None
        record = ("/".join(hierarchy[:2]), building)
        places[keys[hits[0]]].append((*record, (point.x, point.y)) if include_coordinates else record)
    return places


def describe_blocks(area, data, blocks):
    buildings = load_buildings(data / f"{area}_buildings.geojson")
    rows = {}
    for key, places in block_places(data / f"{area}_places.geojson", blocks, buildings).items():
        categories = Counter(category for category, _ in places)
        occupied = Counter(building for _, building in places if building is not None)
        n = len(places)
        hectares = blocks[key].geometry.area / 10_000
        matched = occupied.total()
        rows[(area, key)] = {
            "commercial_pois": n,
            "poi_categories": dict(sorted(categories.items())),
            "poi_density_per_hectare": n / hectares,
            "unique_building_matches": matched,
            "largest_building_share": max(occupied.values()) / matched
            if matched >= MIN_BUILDING_MATCHES else None,
            "walk_metres_per_hectare": blocks[key].walk.size(weight="length_m") / hectares,
        }
    return rows


def fit_embedding(rows):
    keys = sorted(rows)
    fit_mask = np.array([rows[key]["commercial_pois"] >= MIN_POIS for key in keys])
    fit_count = int(fit_mask.sum())
    if fit_count <= DIMENSIONS:
        raise ValueError("Too few blocks with commercial POIs for the embedding")
    labels = sorted({cat for row in rows.values() for cat in row["poi_categories"]})
    counts = np.array([[rows[key]["poi_categories"].get(label, 0) for label in labels] for key in keys])
    totals = counts.sum(axis=1)
    prior = counts[fit_mask].sum(axis=0) / totals[fit_mask].sum()
    padding = np.maximum(MIN_POIS - totals, 0)
    composition = np.sqrt((counts + padding[:, None] * prior) / (totals + padding)[:, None])
    structure = np.array([
        [math.log1p(rows[key]["poi_density_per_hectare"]),
         rows[key]["largest_building_share"],
         math.log1p(rows[key]["walk_metres_per_hectare"])] for key in keys
    ], dtype=float)
    medians = np.nanmedian(structure[fit_mask], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("A morphology feature has no observations")
    structure = np.where(np.isnan(structure), medians, structure)

    category_center = composition[fit_mask].mean(axis=0)
    category_residuals = composition - category_center
    category_scale = float(np.sqrt(np.mean(np.sum(category_residuals[fit_mask] ** 2, axis=1))))
    structure_center = structure[fit_mask].mean(axis=0)
    structure_scale = structure[fit_mask].std(axis=0)
    structure_scale[structure_scale == 0] = 1.0
    standardized = (structure - structure_center) / structure_scale
    # Give the category profile and spatial measurements equal total variance.
    matrix = np.column_stack([category_residuals / (category_scale or 1),
                              standardized / math.sqrt(len(SHAPE_FEATURES))])
    _, values, directions = np.linalg.svd(matrix[fit_mask], full_matrices=False)
    components = directions[:DIMENSIONS]
    pivots = np.argmax(np.abs(components), axis=1)
    components = components * np.sign(components[np.arange(DIMENSIONS), pivots])[:, None]
    scores = matrix @ components.T
    norms = np.linalg.norm(scores, axis=1)
    vectors = {key: vector / norm for key, vector, norm in zip(keys, scores, norms)
               if norm > 1e-10}
    return vectors, {
        "method": "PCA (centered, six-dimensional, L2 normalized)",
        "blocks_used_for_fit": fit_count,
        "variance_retained": float(np.sum(values[:DIMENSIONS] ** 2) / np.sum(values ** 2)),
        "category_order": labels,
        "category_prior_for_sparse_blocks": prior.tolist(),
        "category_center": category_center.tolist(),
        "category_scale": category_scale or 1,
        "structure_order": SHAPE_FEATURES,
        "structure_center": structure_center.tolist(),
        "structure_scale": structure_scale.tolist(),
        "structure_medians": medians.tolist(),
        "components": components.tolist(),
    }


def cosine_lookup(vectors):
    return {(a, b): float(np.clip(vectors[a] @ vectors[b], -1, 1))
            for a, b in combinations(sorted(vectors), 2)}


def merge_similar(adjacency, similarities, threshold):
    """Complete linkage prevents dissimilar blocks from chaining via a third block."""
    neighbors = nx.relabel_nodes(adjacency, {key: (key,) for key in adjacency})
    while neighbors.number_of_edges():
        options = []
        for left, right in neighbors.edges():
            a, b = sorted((left, right))
            score = min(similarities.get(tuple(sorted((x, y))), -math.inf) for x in a for y in b)
            if score >= threshold:
                options.append((-score, a, b))
        if not options:
            break
        _, a, b = min(options)
        merged = tuple(sorted(a + b))
        contacts = (set(neighbors[a]) | set(neighbors[b])) - {a, b}
        neighbors.remove_nodes_from([a, b])
        neighbors.add_node(merged)
        neighbors.add_edges_from((merged, key) for key in sorted(contacts))
    return sorted(neighbors)


def run_area(area, prepared, rows, all_vectors, output):
    blocks, pieces, shops, *_ = prepared
    adjacency, borders = adjacent(blocks, pieces)
    vectors = {key: all_vectors[(area, key)] for key in adjacency if (area, key) in all_vectors}
    all_similarities = cosine_lookup(vectors)
    supported = {key for key in vectors if rows[(area, key)]["commercial_pois"] >= MIN_POIS}
    similarities = {pair: score for pair, score in all_similarities.items()
                    if all(key in supported for key in pair)}
    for border in borders:
        p = border["properties"]
        pair = tuple(sorted(p["blocks"]))
        p["embedding_cosine"] = all_similarities.get(pair)
        p["enough_pois_to_compare"] = pair in similarities

    regions, gateways = [], []
    stats = {}
    eligible_edges = [similarities.get(tuple(sorted(edge))) for edge in adjacency.edges()]
    for threshold in THRESHOLDS:
        groups = merge_similar(adjacency, similarities, threshold)
        merged = [g for g in groups if len(g) > 1]
        sizes = [len(g) for g in merged]
        for group in groups:
            region, entries = candidate(area, f"cos-{threshold:.2f}", group, blocks, pieces, shops)
            for item in [region, *entries]:
                p = item["properties"]
                p.pop("scale_max_blocks")
                p["embedding_threshold"] = threshold
            region["properties"]["min_block_cosine"] = min(
                (similarities[pair] for pair in combinations(group, 2)), default=None)
            regions.append(region)
            gateways.extend(entries)
        stats[f"{threshold:.2f}"] = {
            "multi_block_regions": len(merged),
            "blocks_in_multi_block_regions": sum(sizes),
            "multi_block_sizes": sorted(sizes),
            "singleton_blocks": len(groups) - len(merged),
            "walk_neighbor_pairs_above_threshold": sum(
                s is not None and s >= threshold for s in eligible_edges),
        }
    folder = output / area
    folder.mkdir(parents=True, exist_ok=True)
    layer(folder / "block_embeddings.geojson", [
        feature(blocks[key].geometry, block_id=key, **rows[(area, key)],
                embedding=vectors[key].tolist() if key in vectors else None,
                enough_pois_for_merging=key in supported)
        for key in sorted(adjacency)
    ])
    layer(folder / "neighbor_similarity.geojson", borders)
    layer(folder / "candidates.geojson", regions)
    layer(folder / "gateways.geojson", gateways)
    return {
        "blocks": len(adjacency), "embedded_blocks": len(vectors),
        "blocks_with_enough_pois_for_merging": len(supported),
        "commercial_pois": sum(rows[(area, key)]["commercial_pois"] for key in adjacency),
        "walk_neighbor_pairs": adjacency.number_of_edges(),
        "walk_neighbor_pairs_with_enough_pois": sum(score is not None for score in eligible_edges),
        "thresholds": stats,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-data", type=Path, default=STEP1 / "data")
    parser.add_argument("--step1-outputs", type=Path, default=STEP1 / "outputs")
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "embedding")
    args = parser.parse_args()
    configs = read(STEP1 / "areas.json")
    inputs = {area: prepare(area, config, args.step1_data, args.step1_outputs)
              for area, config in configs.items()}
    rows = {}
    for area, result in inputs.items():
        rows.update(describe_blocks(area, args.step1_data, result[0]))
    vectors, model = fit_embedding(rows)
    areas = {area: run_area(area, result, rows, vectors, args.output)
             for area, result in inputs.items()}
    write(args.output / "model.json", model)
    write(args.output / "metrics.json", {
        "dimensions": DIMENSIONS, "variance_retained": model["variance_retained"],
        "min_commercial_pois": MIN_POIS, "min_building_matches": MIN_BUILDING_MATCHES,
        "min_poi_confidence": CONFIDENCE, "areas": areas,
    })


if __name__ == "__main__":
    main()
