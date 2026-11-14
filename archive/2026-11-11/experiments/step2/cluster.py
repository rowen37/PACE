"""Build walk-connected, multi-scale candidate areas using the step 1 snapshots."""

import argparse
import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
from statistics import mean, median, pstdev
import sys

import networkx as nx
from shapely import STRtree, make_valid
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import polygonize, substring, transform, unary_union

HERE = Path(__file__).resolve().parent
STEP1 = HERE.parent / "step1"
sys.path.insert(0, str(STEP1))
# The step 3 experiment is also named run.py. Load this dependency by its
# explicit path so importing that experiment cannot resolve to itself.
_spec = importlib.util.spec_from_file_location("_pace_step1_core", STEP1 / "run.py")
_step1_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_step1_core)
ROAD_CLASSES = _step1_core.ROAD_CLASSES
TO_METRES = _step1_core.TO_METRES
CONFIDENCE = _step1_core.CONFIDENCE
load_buildings = _step1_core.load_buildings
feature, layer, read, write = (_step1_core.feature, _step1_core.layer,
                              _step1_core.read, _step1_core.write)

SCALES = (2, 5, 10)
DIVIDER_CLASSES = ROAD_CLASSES - {"service"}
MIN_BLOCK_M2 = 500
MIN_SHARED_M = 5
FEATURES = ("edge_betweenness", "mean_degree", "dead_end_ratio", "orientation_cos2", "orientation_sin2")


@dataclass
class Block:
    key: str
    geometry: Polygon
    at_window_edge: bool
    walk: nx.MultiGraph
    core: set
    stats: dict

    @property
    def eligible(self):
        return not self.at_window_edge and self.geometry.area >= MIN_BLOCK_M2 and bool(self.core)


def projected(record):
    return transform(TO_METRES, shape(record["geometry"]))


def osm_coords(nodes, way):
    return [(nodes[key]["lon"], nodes[key]["lat"]) for key in way["nodes"]]


def street_blocks(nodes, ways, window):
    # Use named public streets as planar dividers, including bridges carrying streets.
    named = defaultdict(list)
    for way in ways.values():
        tags = way.get("tags", {})
        if (not tags.get("name") or tags.get("highway") not in DIVIDER_CLASSES
                or tags.get("tunnel") not in {None, "no"} or tags.get("covered") == "yes"
                or tags.get("access") in {"private", "no"}
                or not all(key in nodes for key in way["nodes"])):
            continue
        line = transform(TO_METRES, LineString(osm_coords(nodes, way))).intersection(window)
        if line.length:
            named[tags["name"]].append(line)
    lines = [window.boundary, *(line for segments in named.values() for line in segments)]
    polygons = sorted(polygonize(unary_union(lines)),
                      key=lambda g: (-g.centroid.y, g.centroid.x))
    blocks = {f"b{i:03d}": polygon for i, polygon in enumerate(polygons, 1)}
    return absorb_road_strips(blocks, named, window)


def absorb_road_strips(blocks, named, window):
    """Fold narrow faces between two mapped carriageways into bordering blocks."""
    streets = [(name, unary_union(lines).buffer(0.15)) for name, lines in sorted(named.items())]
    tree = STRtree([g for _, g in streets])
    strips = {}
    for key, polygon in blocks.items():
        a, b, c, *_ = polygon.minimum_rotated_rectangle.exterior.coords
        width, length = sorted([math.dist(a, b), math.dist(b, c)])
        if width > 25 or length < 2 * width:
            continue
        matches = [(polygon.boundary.intersection(streets[i][1]).length / polygon.length,
                    streets[i][0]) for i in tree.query(polygon)]
        share, name = max(matches, default=(0, ""))
        if share >= 0.65:
            strips[key] = name

    edge = window.boundary.buffer(0.05)
    exterior = {key for key, polygon in blocks.items()
                if polygon.boundary.intersection(edge).length > 1}
    assignments = defaultdict(list)
    records = []
    for key, name in strips.items():
        strip = blocks[key]
        choices = []
        for other, polygon in blocks.items():
            if other in strips:
                continue
            shared = strip.boundary.intersection(polygon.boundary).length
            if shared >= MIN_SHARED_M:
                choices.append((key in exterior and other not in exterior, -shared, other))
        if choices:
            _, _, target = min(choices)
            assignments[target].append(strip)
        else:
            target = None
        records.append(feature(strip, original_block_id=key, street=name,
                               absorbed_into=target, area_m2=strip.area))
    return ({key: unary_union([g, *assignments[key]]) if key in assignments else g
             for key, g in blocks.items() if key not in strips}, records)


def source_graph(records):
    graph = nx.MultiGraph()
    for record in records:
        p = record["properties"]
        if p["kind"] == "walk":
            graph.add_edge(p["u"], p["v"], key=p["edge_id"], length_m=p["length_m"],
                           geometry=projected(record), u=p["u"], v=p["v"])
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def split_points(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Point":
        yield geometry
    elif geometry.geom_type == "LineString":
        yield Point(geometry.coords[0])
        yield Point(geometry.coords[-1])
    else:
        for part in geometry.geoms:
            yield from split_points(part)


def split_walk(source, polygons):
    borders = unary_union([g.boundary for g in polygons.values()])
    tree = STRtree(list(polygons.values()))
    block_ids = list(polygons)
    betweenness = nx.edge_betweenness_centrality(source, weight="length_m")
    pieces = nx.MultiGraph()
    for u, v, key, p in source.edges(keys=True, data=True):
        line = p["geometry"]
        crossings = [line.project(point) for point in split_points(line.intersection(borders))]
        cuts = sorted({round(offset, 6) for offset in crossings
                       if 1e-5 < offset < line.length - 1e-5})
        offsets = [0.0, *cuts, line.length]
        vertices = [p["u"], *(f"{key}@{offset:.6f}" for offset in cuts), p["v"]]
        for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
            part = substring(line, start, end)
            midpoint = part.interpolate(0.5, normalized=True)
            matches = tree.query(midpoint, predicate="within")
            block = block_ids[matches[0]] if len(matches) == 1 else None
            a, b = vertices[i:i + 2]
            pieces.add_node(a, point=Point(part.coords[0]))
            pieces.add_node(b, point=Point(part.coords[-1]))
            pieces.add_edge(a, b, key=f"{key}/{i}", u=a, v=b, block=block, length_m=part.length,
                            geometry=part, source_edge=key, betweenness=betweenness[u, v, key])
    return pieces


def block_features(core, source):
    edges = [p for _, _, _, p in sorted(core.edges(keys=True, data=True), key=lambda e: e[2])]
    total = math.fsum(edge["length_m"] for edge in edges)
    degree = [source.degree(node) for node in core if node in source]
    cos2 = sin2 = 0.0
    for edge in edges:
        coords = list(edge["geometry"].coords)
        for (ax, ay), (bx, by) in zip(coords, coords[1:]):
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            if length:
                cos2 += (dx * dx - dy * dy) / length
                sin2 += 2 * dx * dy / length
    return {
        "edge_betweenness": math.fsum(e["betweenness"] * e["length_m"] for e in edges) / total,
        "mean_degree": mean(degree) if degree else 2,
        "dead_end_ratio": mean(d == 1 for d in degree) if degree else 0,
        "orientation_cos2": cos2 / total,
        "orientation_sin2": sin2 / total,
        "source_nodes": len(degree),
    }


def prepare(area, config, data, inputs):
    folder = inputs / area
    elements = read(data / f"{area}_osm.json")["elements"]
    nodes = {e["id"]: e for e in elements if e["type"] == "node"}
    ways = {e["id"]: e for e in elements if e["type"] == "way"}
    window = transform(TO_METRES, box(*config["bounds"]))
    polygons, strips = street_blocks(nodes, ways, window)
    source_records = read(folder / "graph.geojson")["features"]
    source = source_graph(source_records)
    pieces = split_walk(source, polygons)
    edge_groups = defaultdict(list)
    for a, b, key, attrs in pieces.edges(keys=True, data=True):
        edge_groups[attrs["block"]].append((a, b, key))
    edge_buffer = window.boundary.buffer(0.05)
    blocks = {}
    for key, polygon in polygons.items():
        walk = pieces.edge_subgraph(edge_groups[key])
        components = list(nx.connected_components(walk))
        core = max(components, key=lambda c: (walk.subgraph(c).size(weight="length_m"),
                                             min(c)), default=set())
        at_edge = polygon.boundary.intersection(edge_buffer).length > 1
        stats = block_features(walk.subgraph(core), source) if core else {}
        blocks[key] = Block(key, polygon, at_edge, walk, core, stats)
    shops = [projected(f) for f in read(folder / "shops.geojson")["features"]]
    entrances = read(folder / "entrances.geojson")["features"]
    entrance_edges = [f for f in source_records
                      if f["properties"]["kind"] == "entrance"]
    return blocks, pieces, shops, entrances, nodes, ways, entrance_edges, strips


def adjacent(blocks, pieces):
    eligible = [block for block in blocks.values() if block.eligible]
    graph = nx.Graph()
    graph.add_nodes_from(block.key for block in eligible)
    records = []
    for a, b in combinations(eligible, 2):
        boundary = a.geometry.boundary.intersection(b.geometry.boundary)
        if boundary.length < MIN_SHARED_M:
            continue
        connections = sorted(node for node in a.core & b.core
                             if boundary.distance(pieces.nodes[node]["point"]) < 1e-4)
        if connections:
            graph.add_edge(a.key, b.key)
        records.append(feature(boundary, blocks=[a.key, b.key], shared_border_m=boundary.length,
                               walk_connected=bool(connections), connection_nodes=connections))
    return graph, records


def feature_vectors(prepared):
    blocks = [b for area in prepared.values() for b in area[0].values() if b.eligible]
    stats = {}
    for key in FEATURES:
        values = [b.stats[key] for b in blocks]
        stats[key] = (mean(values), pstdev(values) or 1)
    return {area: {b.key: [(b.stats[key] - m) / sd for key, (m, sd) in stats.items()]
                   for b in values[0].values() if b.eligible}
            for area, values in prepared.items()}


def merge_levels(graph, vectors):
    pair_dist = {frozenset((a, b)): math.dist(vectors[a], vectors[b])
                 for a, b in combinations(sorted(graph), 2)}
    neighbor = nx.relabel_nodes(graph, {node: (node,) for node in graph})
    levels = {}
    merges = []

    def linkage(pair):
        a, b = sorted(pair)
        return mean(pair_dist[frozenset((i, j))] for i in a for j in b), a, b

    for cap in SCALES:
        while True:
            feasible = ((a, b) for a, b in neighbor.edges() if len(a) + len(b) <= cap)
            choice = min(map(linkage, feasible), default=None)
            if choice is None:
                break
            score, a, b = choice
            merged = tuple(sorted(a + b))
            contacts = (set(neighbor[a]) | set(neighbor[b])) - {a, b}
            neighbor.remove_nodes_from([a, b])
            neighbor.add_node(merged)
            neighbor.add_edges_from((merged, node) for node in sorted(contacts))
            merges.append({"blocks": merged, "left": a, "right": b, "distance": score,
                           "scale_max_blocks": cap})
        levels[cap] = sorted(neighbor)
    return levels, merges


def candidate(area, cap, group, blocks, pieces, shops):
    key = f"{area}-{cap}-{'-'.join(group)}"
    geometry = unary_union([blocks[b].geometry for b in group])
    core_nodes = set().union(*(blocks[b].core for b in group))
    contained_edges = [(a, b, k) for block in group
                       for a, b, k in blocks[block].walk.edges(keys=True)]
    internal = pieces.edge_subgraph(contained_edges)
    anchor = min(blocks[group[0]].core, key=lambda n: (-internal.degree(n), n))
    reachable = nx.node_connected_component(internal, anchor)
    core_connected = core_nodes <= reachable
    gates = []
    for node in sorted(reachable):
        point = pieces.nodes[node]["point"]
        outside = any(p["block"] is not None and p["block"] not in group
                      for *_, p in pieces.edges(node, data=True))
        if outside and geometry.boundary.distance(point) < 1e-4:
            gates.append(feature(point, candidate_id=key, kind="network_boundary",
                                 graph_node=node, scale_max_blocks=cap))
    reachable_m = internal.subgraph(reachable).size(weight="length_m")
    return feature(geometry, candidate_id=key, kind="street", scale_max_blocks=cap,
                   block_ids=list(group), block_count=len(group), area_m2=geometry.area,
                   contiguous_polygon=geometry.geom_type == "Polygon" and geometry.is_valid,
                   block_cores_reachable=core_connected,
                   reachable_walk_fraction=reachable_m / internal.size(weight="length_m"),
                   gateway_count=len(gates), shopping_pois=sum(geometry.covers(p) for p in shops)), gates


def mall_candidate(area, entrances, nodes, ways, shops):
    if area != "hudson_yards":
        return [], []
    door = next(f for f in entrances if f["properties"]["osm_venue_way"] == 398440363)
    p = door["properties"]
    if not p["in_largest_component"]:
        raise ValueError("The mall entrance is missing from the main step 1 graph")
    way = ways[p["osm_venue_way"]]
    coords = osm_coords(nodes, way)
    if coords[0] != coords[-1] or way.get("tags", {}).get("shop") != "mall":
        raise ValueError("Review the mapped mall outline")
    mall = make_valid(transform(TO_METRES, Polygon(coords)))
    key = f"{area}-mall"
    return [feature(mall, candidate_id=key, kind="mall", name="The Shops at Hudson Yards",
                    area_m2=mall.area, osm_way=way["id"], gateway_count=1,
                    shopping_pois=sum(mall.covers(point) for point in shops))], [
        feature(projected(door), candidate_id=key, kind="reviewed_entrance",
                graph_node=p["graph_node"], osm_node=p["osm_node"], source=p["source"])
    ]


def measure(features):
    props = [f["properties"] for f in features]
    sizes = [p["block_count"] for p in props]
    return {
        "candidates": len(sizes),
        "block_count_range": [min(sizes), max(sizes)],
        "median_blocks": median(sizes),
        "single_block_candidates": sizes.count(1),
        "valid_contiguous_polygons": sum(p["contiguous_polygon"] for p in props),
        "all_block_cores_reachable": sum(p["block_cores_reachable"] for p in props),
        "with_boundary_gateways": sum(p["gateway_count"] > 0 for p in props),
        "median_reachable_walk_fraction": median(p["reachable_walk_fraction"] for p in props),
        "minimum_reachable_walk_fraction": min(p["reachable_walk_fraction"] for p in props),
    }


def run_area(area, prepared, vectors, output):
    blocks, pieces, shops, entrances, nodes, ways, entrance_edges, strips = prepared
    adjacency, records = adjacent(blocks, pieces)
    if not adjacency:
        raise ValueError(f"No eligible street blocks for {area}")
    levels, merges = merge_levels(adjacency, vectors)
    candidates, gates = mall_candidate(area, entrances, nodes, ways, shops)
    stats = {
        "polygonized_faces": len(blocks) + len(strips),
        "road_strips_detected": len(strips),
        "road_strips_absorbed": sum(f["properties"]["absorbed_into"] is not None for f in strips),
        "blocks_at_window_edge": sum(b.at_window_edge for b in blocks.values()),
        "eligible_blocks": len(adjacency),
        "interior_blocks_below_min_area": sum(not b.at_window_edge and b.geometry.area < MIN_BLOCK_M2
                                              for b in blocks.values()),
        "interior_large_blocks_without_main_walk": sum(not b.at_window_edge and b.geometry.area >= MIN_BLOCK_M2
                                                       and not b.core for b in blocks.values()),
        "geometric_adjacencies": len(records),
        "walk_adjacencies": adjacency.number_of_edges(),
        "walk_components": sorted(map(len, nx.connected_components(adjacency)), reverse=True),
        "mall_candidates": [f["properties"] for f in candidates],
        "scales": {},
    }
    for cap, groups in levels.items():
        level = []
        for group in groups:
            region, gateways = candidate(area, cap, group, blocks, pieces, shops)
            level.append(region)
            gates.extend(gateways)
        stats["scales"][str(cap)] = measure(level)
        candidates.extend(level)
    folder = output / area
    folder.mkdir(parents=True, exist_ok=True)
    layer(folder / "blocks.geojson", [
        feature(b.geometry, block_id=b.key, eligible=b.eligible, at_window_edge=b.at_window_edge,
                area_m2=b.geometry.area, **b.stats) for b in blocks.values()
    ])
    layer(folder / "adjacency.geojson", records)
    layer(folder / "road_strips.geojson", strips)
    layer(folder / "candidates.geojson", candidates)
    layer(folder / "gateways.geojson", gates)
    layer(folder / "network.geojson", [
        feature(p["geometry"], edge_id=key, kind="walk",
                **{k: v for k, v in p.items() if k != "geometry"})
        for _, _, key, p in pieces.edges(keys=True, data=True)
    ] + entrance_edges)
    write(folder / "merges.json", merges)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1-data", type=Path, default=STEP1 / "data")
    parser.add_argument("--step1-outputs", type=Path, default=STEP1 / "outputs")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    configs = read(STEP1 / "areas.json")
    prepared = {area: prepare(area, config, args.step1_data, args.step1_outputs)
                for area, config in configs.items()}
    vectors = feature_vectors(prepared)
    areas = {area: run_area(area, result, vectors[area], args.output)
             for area, result in prepared.items()}
    write(args.output / "metrics.json", {"scale_max_blocks": SCALES, "feature_names": FEATURES,
                                        "min_block_m2": MIN_BLOCK_M2, "min_shared_border_m": MIN_SHARED_M,
                                        "areas": areas})


if __name__ == "__main__":
    main()
