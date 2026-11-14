"""Build street blocks and their three source views without numeric cutoffs."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import orjson
from pyproj import Transformer
from shapely import STRtree, make_valid, union_all
from shapely.geometry import LineString, Point, box, mapping, shape
from shapely.ops import polygonize, substring, transform

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/raw/nyc/2026-11-09"
AREAS = ROOT / "archive/2026-11-11/experiments/step1/areas.json"
TO_METRES = Transformer.from_crs(4326, 6538, always_xy=True).transform
TO_WGS84 = Transformer.from_crs(6538, 4326, always_xy=True).transform
DIVIDERS = {"primary", "secondary", "tertiary", "residential", "unclassified", "living_street", "pedestrian"}
COMMERCIAL = {"shopping", "food_and_drink", "lifestyle_services"}


def read(path):
    return orjson.loads(path.read_bytes())


def write(path, payload):
    with path.open("xb") as file:
        file.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")


def feature(geometry, **properties):
    return {"type": "Feature", "geometry": mapping(transform(TO_WGS84, geometry)), "properties": properties}


def layer(path, features):
    write(path, {"type": "FeatureCollection", "features": features})


def line_parts(geometry):
    if geometry.geom_type == "LineString" and not geometry.is_empty:
        yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from line_parts(part)


def boundary_points(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Point":
        yield geometry
    elif geometry.geom_type == "LineString":
        yield Point(geometry.coords[0])
        yield Point(geometry.coords[-1])
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from boundary_points(part)


def street_blocks(elements, window):
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
    lines = [window.boundary]
    for way in elements:
        tags = way.get("tags", {})
        if (way["type"] != "way" or tags.get("highway") not in DIVIDERS or not tags.get("name")
                or tags.get("tunnel") not in {None, "no"} or tags.get("covered") == "yes"
                or tags.get("access") in {"private", "no"} or not all(n in nodes for n in way["nodes"])):
            continue
        line = transform(TO_METRES, LineString([nodes[n] for n in way["nodes"]]))
        lines.extend(line_parts(line.intersection(window)))
    polygons = sorted(polygonize(union_all(lines)), key=lambda g: (-g.centroid.y, g.centroid.x, g.wkb))
    # Drop only faces cut by the observation window. Retain every closed face,
    # including small faces and ones with no mapped POI, building or walk edge.
    return {f"b{i:03d}": g for i, g in enumerate(polygons, 1)
            if g.boundary.intersection(window.boundary).length == 0}


def walking_views(lines, blocks, keys, index):
    borders = union_all([g.boundary for g in blocks.values()])
    edges = []
    blocks_at_node = defaultdict(set)
    for source_id, line in lines:
        cuts = {line.project(p) for p in boundary_points(line.intersection(borders))}
        offsets = sorted({0.0, line.length, *cuts})
        # Exact source endpoints connect lines. Interior cuts are local to this
        # source line, so intersecting lines never acquire an invented junction.
        a, b = line.coords[0], line.coords[-1]
        nodes = [f"endpoint:{a[0]},{a[1]}",
                 *(f"cut:{source_id}:{i}" for i in range(1, len(offsets) - 1)),
                 f"endpoint:{b[0]},{b[1]}"]
        for i, (start, end) in enumerate(zip(offsets, offsets[1:])):
            hits = index.query(line.interpolate((start + end) / 2), predicate="within")
            if len(hits) != 1:
                continue
            block = keys[hits[0]]
            u, v = nodes[i:i + 2]
            part = substring(line, start, end)
            edges.append(feature(part, edge_id=f"{source_id}/{i}", source_id=source_id,
                                 block_id=block, u=u, v=v, length_m=part.length))
            blocks_at_node[u].add(block)
            blocks_at_node[v].add(block)
    connections = defaultdict(list)
    for node, members in blocks_at_node.items():
        for pair in combinations(sorted(members), 2):
            connections[pair].append(node)
    adjacency = [{"blocks": pair, "connection_nodes": sorted(nodes)}
                 for pair, nodes in sorted(connections.items())]
    return edges, adjacency


def prepare_area(area, folder, window, lines):
    osm = read(SNAPSHOT / f"{area}_osm.json")["elements"]
    blocks = street_blocks(osm, window)
    keys = list(blocks)
    index = STRtree(list(blocks.values()))
    places, buildings = [], []
    for item in read(SNAPSHOT / f"{area}_places.geojson")["features"]:
        properties = item["properties"]
        hierarchy = (properties.get("taxonomy") or {}).get("hierarchy") or []
        if not hierarchy or hierarchy[0] not in COMMERCIAL or not item["geometry"]:
            continue
        point = transform(TO_METRES, shape(item["geometry"]))
        hits = index.query(point, predicate="covered_by")
        if len(hits) == 1:
            places.append(feature(point, place_id=item["id"], block_id=keys[hits[0]],
                                  category="/".join(hierarchy[:2]), confidence=properties.get("confidence")))
    for item in read(SNAPSHOT / f"{area}_buildings.geojson")["features"]:
        if not item["geometry"]:
            continue
        footprint = make_valid(transform(TO_METRES, shape(item["geometry"])))
        if footprint.geom_type not in {"Polygon", "MultiPolygon"} or footprint.is_empty:
            continue
        # One row per source building; a footprint can cover more than one block.
        hits = [keys[i] for i in index.query(footprint, predicate="intersects")
                if footprint.relate_pattern(blocks[keys[i]], "2********")]
        if hits:
            buildings.append(feature(footprint, building_id=item["properties"]["doitt_id"], blocks=sorted(hits),
                                     area_m2=footprint.area))
    walk, adjacency = walking_views(lines, blocks, keys, index)
    folder.mkdir()
    layer(folder / "blocks.geojson", [feature(g, block_id=k, area_m2=g.area) for k, g in blocks.items()])
    layer(folder / "places.geojson", places)
    layer(folder / "buildings.geojson", buildings)
    layer(folder / "walk.geojson", walk)
    write(folder / "adjacency.json", adjacency)
    walk_by_block = Counter(f["properties"]["block_id"] for f in walk)
    return {"blocks": len(blocks), "commercial_pois": len(places), "buildings": len(buildings),
            "walk_segments": len(walk), "walk_adjacencies": len(adjacency),
            "blocks_without_mapped_walk_segments": sum(k not in walk_by_block for k in keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    windows = {area: transform(TO_METRES, box(*config["bounds"])) for area, config in read(AREAS).items()}
    source = read(SNAPSHOT / "nycwalks.geojson")
    if source["crs"]["properties"]["name"] != "EPSG:6538":
        raise ValueError("Unexpected NYCWalks CRS")
    lines = defaultdict(list)
    for record in source["features"]:
        geometry = shape(record["geometry"])
        for area, window in windows.items():
            if geometry.intersects(window):
                lines[area].extend((f'{record["id"]}:{i}', line) for i, line in enumerate(
                    line_parts(geometry.intersection(window))))
    output.mkdir(parents=True)
    summary = {area: prepare_area(area, output / area, window, lines[area]) for area, window in windows.items()}
    write(output / "summary.json", summary)
    print(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
