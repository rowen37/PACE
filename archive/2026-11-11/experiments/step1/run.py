"""Build and measure the PACE pilot walking graphs from local source snapshots."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import networkx as nx
import orjson
from pyproj import Transformer
from shapely import STRtree, make_valid
from shapely.geometry import LineString, Point, box, mapping, shape
from shapely.ops import substring, transform

from fetch import HERE, RELEASE

TO_METRES = Transformer.from_crs(4326, 6538, always_xy=True).transform
TO_WGS84 = Transformer.from_crs(6538, 4326, always_xy=True).transform
ROAD_CLASSES = {
    "primary", "secondary", "tertiary", "residential", "unclassified",
    "living_street", "pedestrian", "service",
}
CONFIDENCE = 0.8
MAX_SNAP_M = 15


def read(path):
    return orjson.loads(path.read_bytes())


def write(path, payload):
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")


def feature(geom, **properties):
    return {"type": "Feature", "geometry": mapping(transform(TO_WGS84, geom)),
            "properties": properties}


def layer(path, features):
    write(path, {"type": "FeatureCollection", "features": features})


def line_parts(geom):
    if geom.geom_type == "LineString" and geom.length > 0:
        yield geom
    elif geom.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geom.geoms:
            yield from line_parts(part)


def walking_lines(data, windows):
    source = read(data / "nycwalks.geojson")
    if source["crs"]["properties"]["name"] != "EPSG:6538":
        raise ValueError("Unexpected NYCWalks CRS")
    edges = {area: {} for area in windows}
    for record in source["features"]:
        geom = shape(record["geometry"])
        for area, window in windows.items():
            if geom.intersects(window):
                for i, line in enumerate(line_parts(geom.intersection(window))):
                    edges[area][f'{record["id"]}:{i}'] = line
    return edges


def load_osm(path):
    elements = read(path)["elements"]
    nodes = {item["id"]: item for item in elements if item["type"] == "node"}
    ways = {item["id"]: item for item in elements if item["type"] == "way"}
    roads = []
    for way in ways.values():
        tags = way.get("tags", {})
        if not tags.get("name") or tags.get("highway") not in ROAD_CLASSES:
            continue
        if not all(node in nodes for node in way["nodes"]):
            continue
        points = [(nodes[node]["lon"], nodes[node]["lat"]) for node in way["nodes"]]
        line = transform(TO_METRES, LineString(points))
        if line.length > 0:
            roads.append((line, {"osm_way": way["id"], "street": tags["name"],
                                "street_class": tags["highway"]}))
    return nodes, ways, roads


def direction(line, offset):
    left = line.interpolate(max(0, offset - 3))
    right = line.interpolate(min(line.length, offset + 3))
    return right.x - left.x, right.y - left.y


def street_match(line, roads, tree):
    """Label lines parallel to a nearby named road at three sample points."""
    offsets = [line.length * fraction for fraction in (0.2, 0.5, 0.8)]
    samples = [line.interpolate(offset) for offset in offsets]
    candidates = []
    for idx in tree.query(samples[1], predicate="dwithin", distance=18):
        road, tags = roads[idx]
        distances = [road.distance(point) for point in samples]
        if max(distances) > 18:
            continue
        a, b = direction(line, offsets[1])
        c, d = direction(road, road.project(samples[1]))
        norm = math.hypot(a, b) * math.hypot(c, d)
        if norm == 0 or abs(a * c + b * d) < 0.9 * norm:
            continue
        candidates.append((sum(distances), tags["osm_way"], tags))
    return min(candidates)[2] if candidates else {}


def load_buildings(path):
    records = read(path)["features"]
    if len(records) >= 10000:
        raise ValueError(f"Possible NYC API truncation in {path}")
    return {
        record["properties"]["doitt_id"]: make_valid(transform(TO_METRES, shape(record["geometry"])))
        for record in records
    }


def match_shops(path, bounds, buildings, tree):
    window = box(*bounds)
    building_ids = list(buildings)
    shops = []
    for record in read(path)["features"]:
        props = record["properties"]
        taxonomy = props.get("taxonomy") or {}
        hierarchy = taxonomy.get("hierarchy") or []
        if hierarchy[:1] != ["shopping"] or (props.get("confidence") or 0) < CONFIDENCE:
            continue
        lonlat = shape(record["geometry"])
        if not window.covers(lonlat):
            continue
        point = transform(TO_METRES, lonlat)
        matches = sorted(building_ids[i] for i in tree.query(point, predicate="covered_by"))
        distance = min((buildings[building_ids[i]].distance(point)
                        for i in tree.query(point, predicate="dwithin", distance=5)),
                       default=None)
        shops.append((point, {
            "place_id": record["id"], "name": (props.get("names") or {}).get("primary"),
            "category": taxonomy["primary"], "confidence": props["confidence"],
            "building_id": matches[0] if len(matches) == 1 else None,
            "building_candidates": matches, "distance_to_footprint_within_5m": distance,
        }))
    return shops


def osm_point(node):
    return transform(TO_METRES, Point(node["lon"], node["lat"]))


def approach_path(reviewed, nodes, ways):
    node_ids = [reviewed["osm_node"], *reviewed.get("approach_osm_nodes", [])]
    for (start, end), way_id in zip(zip(node_ids, node_ids[1:]),
                                    reviewed.get("approach_osm_ways", []), strict=True):
        way = ways[way_id]
        pairs = zip(way["nodes"], way["nodes"][1:])
        if (way.get("tags", {}).get("highway") != "footway"
                or not any({a, b} == {start, end} for a, b in pairs)):
            raise ValueError(f"Review changed entrance approach along OSM way {way_id}")
    return [osm_point(nodes[node]) for node in node_ids]


def snap_entrances(config, nodes, ways, buildings, building_tree, lines, street_tags):
    edge_ids = list(lines)
    tree = STRtree(list(lines.values()))
    entrances = []
    for reviewed in config["entrances"]:
        node = nodes[reviewed["osm_node"]]
        tags = node["tags"]
        venue = ways[reviewed["osm_venue_way"]]
        if (node["id"] not in venue["nodes"] or tags.get("entrance") != "main"
                or tags.get("access") in {"no", "private"}):
            raise ValueError(f"Review changed OSM entrance {node['id']}")
        approach = approach_path(reviewed, nodes, ways)
        point = approach[0]
        boundary_distance = buildings[reviewed["building_id"]].boundary.distance(point)
        if boundary_distance > 3:
            raise ValueError(f"Review footprint for entrance {node['id']}")
        choices = []
        for i in tree.query(approach[-1], predicate="dwithin", distance=MAX_SNAP_M):
            edge_id = edge_ids[i]
            line = lines[edge_id]
            if reviewed.get("street") and street_tags[edge_id].get("street") != reviewed["street"]:
                continue
            offset = line.project(approach[-1])
            target = line.interpolate(offset)
            distance = approach[-1].distance(target)
            connector = LineString([*approach, target])
            # Allow up to 2 m inside the entrance's own footprint at the door.
            own = buildings[reviewed["building_id"]]
            if any(building_tree.geometries[j].intersection(connector).length >
                   (2 if building_tree.geometries[j].equals(own) else 0.05)
                   for j in building_tree.query(connector, predicate="intersects")):
                continue
            choices.append((distance, edge_id, offset, connector))
        snap = min(choices, key=lambda choice: choice[:3], default=(None, None, None, None))
        entrances.append((point, {**reviewed, "osm_version": node["version"],
                                  "footprint_boundary_m": boundary_distance,
                                  "snap_m": snap[0], "nycwalks_edge": snap[1],
                                  "offset_m": snap[2],
                                  "connector_m": snap[3].length if snap[3] is not None else None},
                          snap[3]))
    return entrances


def build_graph(lines, street_tags, entrances):
    graph = nx.MultiGraph()
    splits = defaultdict(set)
    for _, entry, _ in entrances:
        if entry["nycwalks_edge"] is not None:
            splits[entry["nycwalks_edge"]].add(entry["offset_m"])

    def add_edge(start, end, line, edge_id, **props):
        graph.add_edge(start, end, key=edge_id, u=start, v=end, geometry=line,
                       length_m=line.length, **props)

    def vertex(point):
        return f"{point[0]:.4f},{point[1]:.4f}"

    for edge_id, line in lines.items():
        cuts = sorted({0, line.length, *splits[edge_id]})
        for i, (start, end) in enumerate(zip(cuts, cuts[1:])):
            piece = substring(line, start, end)
            add_edge(vertex(piece.coords[0]), vertex(piece.coords[-1]), piece,
                     f"{edge_id}.{i}", kind="walk", nycwalks_edge=edge_id, **street_tags[edge_id])

    for _, entry, connector in entrances:
        edge_id = entry["nycwalks_edge"]
        if edge_id is None:
            continue
        target = lines[edge_id].interpolate(entry["offset_m"])
        entry["graph_node"] = f'entrance:{entry["osm_node"]}'
        add_edge(entry["graph_node"], vertex(target.coords[0]), connector,
                 entry["graph_node"], kind="entrance",
                 building_id=entry["building_id"])
    return graph


def run_area(area, config, data, output, lines):
    nodes, ways, roads = load_osm(data / f"{area}_osm.json")
    road_tree = STRtree([line for line, _ in roads])
    street_tags = {edge_id: street_match(line, roads, road_tree) for edge_id, line in lines.items()}
    buildings = load_buildings(data / f"{area}_buildings.geojson")
    building_tree = STRtree(list(buildings.values()))
    shops = match_shops(data / f"{area}_places.geojson", config["bounds"], buildings, building_tree)
    entrances = snap_entrances(config, nodes, ways, buildings, building_tree, lines, street_tags)
    graph = build_graph(lines, street_tags, entrances)
    if not graph:
        raise ValueError(f"No walking graph for {area}")
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = components[0]
    connected_buildings = defaultdict(list)
    for _, entry, _ in entrances:
        entry["in_largest_component"] = entry.get("graph_node") in largest
        if entry["in_largest_component"]:
            connected_buildings[entry["building_id"]].append(entry["graph_node"])
    for _, shop in shops:
        shop["building_entrance_nodes"] = connected_buildings.get(shop["building_id"], [])

    folder = output / area
    folder.mkdir(parents=True, exist_ok=True)
    layer(folder / "graph.geojson", [
        feature(props["geometry"], edge_id=key,
                **{k: v for k, v in props.items() if k != "geometry"})
        for _, _, key, props in graph.edges(keys=True, data=True)
    ])
    layer(folder / "buildings.geojson", [feature(g, building_id=k) for k, g in buildings.items()])
    layer(folder / "shops.geojson", [feature(g, **p) for g, p in shops])
    layer(folder / "entrances.geojson", [feature(g, **p) for g, p, _ in entrances])

    counts = Counter(p["building_id"] for _, p in shops if p["building_id"] is not None)
    return {
        "bounds": config["bounds"], "walk_segments": len(lines),
        "walk_km": round(sum(g.length for g in lines.values()) / 1000, 3),
        "named_street_matches": sum(bool(tags) for tags in street_tags.values()),
        "graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
        "components": len(components), "largest_component_nodes": len(largest),
        "buildings": len(buildings), "shopping_pois": len(shops),
        "unique_footprint_matches": counts.total(),
        "ambiguous_footprint_matches": sum(len(p["building_candidates"]) > 1 for _, p in shops),
        "outside_footprints_within_5m": sum(not p["building_candidates"]
                                           and p["distance_to_footprint_within_5m"] is not None
                                           for _, p in shops),
        "most_pois_in_one_footprint": counts.most_common(1),
        "pois_in_buildings_with_connected_entrance": sum(bool(p["building_entrance_nodes"])
                                                         for _, p in shops),
        "reviewed_entrances": [entry for _, entry, _ in entrances],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--output", type=Path, default=HERE / "outputs")
    args = parser.parse_args()
    areas = read(HERE / "areas.json")
    windows = {key: transform(TO_METRES, box(*config["bounds"])) for key, config in areas.items()}
    lines = walking_lines(args.data, windows)
    metrics = {key: run_area(key, config, args.data, args.output, lines[key])
               for key, config in areas.items()}
    write(args.output / "metrics.json", {
        "overture_release": RELEASE, "poi_confidence_min": CONFIDENCE,
        "entrance_snap_max_m": MAX_SNAP_M, "areas": metrics,
    })


if __name__ == "__main__":
    main()
