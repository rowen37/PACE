"""Sample street-view camera points along each block's walk segments."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import shape

from streetview_common import DATA_AREAS, dataset_root, read_features, write_json

TO_METRES = Transformer.from_crs(4326, 6538, always_xy=True)
SAMPLE_INTERVAL_M = 35.0
DEDUPE_M = 25.0


def azimuth_deg(dx, dy):
    return (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0


def farthest_points(points, keep):
    """Greedy spread: start near the centroid, then repeatedly add the farthest point."""
    if len(points) <= keep:
        return points
    cx = sum(p["x"] for p in points) / len(points)
    cy = sum(p["y"] for p in points) / len(points)
    chosen = [min(points, key=lambda p: (p["x"] - cx) ** 2 + (p["y"] - cy) ** 2)]
    while len(chosen) < keep:
        best, best_gap = None, -1.0
        for p in points:
            if p in chosen:
                continue
            gap = min((p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2 for q in chosen)
            if gap > best_gap:
                best, best_gap = p, gap
        chosen.append(best)
    return chosen


def sample_block(segments, max_points):
    points = []
    for seg in segments:
        length = seg.length
        marks = [i * SAMPLE_INTERVAL_M + SAMPLE_INTERVAL_M / 2 for i in range(int(length // SAMPLE_INTERVAL_M))]
        if not marks:
            marks = [length / 2]
        for d in marks:
            pt = seg.interpolate(d)
            lon, lat = pt.x, pt.y
            x, y = TO_METRES.transform(lon, lat)
            if any((x - q["x"]) ** 2 + (y - q["y"]) ** 2 < DEDUPE_M ** 2 for q in points):
                continue
            start = seg.interpolate(max(0.0, d - 5.0))
            end = seg.interpolate(min(length, d + 5.0))
            sx, sy = TO_METRES.transform(start.x, start.y)
            ex, ey = TO_METRES.transform(end.x, end.y)
            points.append({"lon": lon, "lat": lat, "x": x, "y": y,
                           "heading": azimuth_deg(ex - sx, ey - sy)})
    return farthest_points(points, max_points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=Path("data/processed/nyc-2026-11-09-block-inputs"))
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--max-points", type=int, default=8,
                        help="camera points kept per block before fetching")
    args = parser.parse_args()

    output = args.dataset / "points"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"sample_interval_m": SAMPLE_INTERVAL_M, "dedupe_m": DEDUPE_M,
               "max_points_per_block": args.max_points, "areas": {}}
    for area in DATA_AREAS:
        blocks = read_features(args.inputs, area, "blocks")
        walk = read_features(args.inputs, area, "walk")
        per_block = {f["properties"]["block_id"]: [] for f in blocks}
        for f in walk:
            per_block[f["properties"]["block_id"]].append(shape(f["geometry"]))
        features, counts = [], {}
        for block_id, segments in per_block.items():
            points = sample_block(segments, args.max_points)
            counts[block_id] = len(points)
            for i, p in enumerate(points):
                features.append({"type": "Feature",
                                 "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                                 "properties": {"block_id": block_id, "point_index": i,
                                                "heading": round(p["heading"], 1)}})
        path = output / f"{area}_points.geojson"
        write_json(path, {"type": "FeatureCollection", "features": features})
        summary["areas"][area] = {"blocks": len(blocks),
                                  "blocks_with_points": sum(1 for c in counts.values() if c),
                                  "total_points": len(features),
                                  "points_sha256": sha256(path.read_bytes()).hexdigest()}
        print(f"{area}: {len(features)} points over {len(blocks)} blocks")
    write_json(args.dataset / "points_summary.json", summary)


if __name__ == "__main__":
    main()
