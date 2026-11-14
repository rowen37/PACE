"""Build stratified block-similarity triplets for street-view annotation."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random

from pyproj import Transformer
from shapely.geometry import shape

from streetview_common import AREAS, ROOT, dataset_root, read_features, write_json

TO_METRES = Transformer.from_crs(4326, 6538, always_xy=True)
SEED = 20261114
RULES = {
    "poi_similar": "same dominant category and min/max total count >= 0.5",
    "poi_different": "different dominant category or min/max total count <= 0.25",
    "walk_similar": "length-weighted orientation cosine >= 0.8, length and segment ratios >= 0.5",
    "walk_different": "orientation cosine <= 0.2 or length ratio <= 0.25",
    "S1": "poi_similar(A,B) and poi_similar(A,C) and walk_different(B,C)",
    "S2": "walk_similar(A,B) and walk_similar(A,C) and poi_different(B,C)",
    "S3": "random distinct blocks",
}


def block_stats(inputs):
    stats = {}
    for area in AREAS:
        for f in read_features(inputs, area, "blocks"):
            key = f["properties"]["block_id"]
            stats[(area, key)] = {"key": (area, key), "area": area, "poi": {}, "poi_total": 0,
                                  "walk_segments": 0, "walk_length": 0.0, "orient": (0.0, 0.0)}
        for f in read_features(inputs, area, "places"):
            row = stats[(area, f["properties"]["block_id"])]
            category = f["properties"]["category"]
            row["poi"][category] = row["poi"].get(category, 0) + 1
            row["poi_total"] += 1
        for f in read_features(inputs, area, "walk"):
            row = stats[(area, f["properties"]["block_id"])]
            length = f["properties"]["length_m"]
            line = shape(f["geometry"])
            (x1, y1), (x2, y2) = [TO_METRES.transform(*c) for c in (line.coords[0], line.coords[-1])]
            angle = math.atan2(y2 - y1, x2 - x1)
            row["walk_segments"] += 1
            row["walk_length"] += length
            row["orient"] = (row["orient"][0] + length * math.cos(2 * angle),
                             row["orient"][1] + length * math.sin(2 * angle))
    for row in stats.values():
        row["dominant"] = max(row["poi"], key=lambda c: (row["poi"][c], c)) if row["poi"] else None
        x, y = row["orient"]
        norm = math.hypot(x, y)
        row["orient_unit"] = (x / norm, y / norm) if norm else (0.0, 0.0)
    return stats


def poi_similar(a, b):
    return a["dominant"] is not None and a["dominant"] == b["dominant"] and \
        min(a["poi_total"], b["poi_total"]) / max(a["poi_total"], b["poi_total"], 1) >= 0.5


def poi_different(a, b):
    return a["dominant"] != b["dominant"] or \
        min(a["poi_total"], b["poi_total"]) / max(a["poi_total"], b["poi_total"], 1) <= 0.25


def walk_similar(a, b):
    cos = a["orient_unit"][0] * b["orient_unit"][0] + a["orient_unit"][1] * b["orient_unit"][1]
    ratio = min(a["walk_length"], b["walk_length"]) / max(a["walk_length"], b["walk_length"], 1e-9)
    count = min(a["walk_segments"], b["walk_segments"]) / max(a["walk_segments"], b["walk_segments"], 1)
    return cos >= 0.8 and ratio >= 0.5 and count >= 0.5


def walk_different(a, b):
    cos = a["orient_unit"][0] * b["orient_unit"][0] + a["orient_unit"][1] * b["orient_unit"][1]
    ratio = min(a["walk_length"], b["walk_length"]) / max(a["walk_length"], b["walk_length"], 1e-9)
    return cos <= 0.2 or ratio <= 0.25


def eligible_pairs(pool, predicate):
    return [(a, b) for i, a in enumerate(pool) for b in pool[i + 1:] if predicate(a, b)]


def build_stratum(stratum, anchors, stats, usage, rng, cross_area):
    """Pick (A, B, C) with B, C from the other area when cross_area else A's own area."""
    if stratum == "S1":
        def edge(a, b):
            return poi_similar(a, b)
        def between(b, c):
            return walk_different(b, c)
    elif stratum == "S2":
        def edge(a, b):
            return walk_similar(a, b)
        def between(b, c):
            return poi_different(b, c)
    else:
        def edge(a, b):
            return True
        def between(b, c):
            return True
    for anchor in anchors:
        if usage.get(anchor["key"], 0) >= 3:
            continue
        pool = [s for s in stats.values()
                if (s["area"] != anchor["area"]) == cross_area and s is not anchor]
        candidates = [s for s in pool if edge(anchor, s) and usage.get(s["key"], 0) < 5]
        pairs = eligible_pairs(candidates, between)
        rng.shuffle(pairs)
        for b, c in pairs:
            if usage.get(b["key"], 0) < 5 and usage.get(c["key"], 0) < 5:
                for row in (anchor, b, c):
                    usage[row["key"]] = usage.get(row["key"], 0) + 1
                return anchor["key"], b["key"], c["key"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=ROOT / "data/processed/nyc-2026-11-09-block-inputs")
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--per-stratum", type=int, default=30)
    parser.add_argument("--min-images", type=int, default=1,
                        help="blocks below this image count are excluded")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_bytes())
    covered = {(area, block): len(rows) for area, blocks in manifest["blocks"].items()
               for block, rows in blocks.items()}
    stats = {k: v for k, v in block_stats(args.inputs).items()
             if covered.get(k, 0) >= args.min_images}
    rng = random.Random(SEED)
    usage, triplets = {}, []
    plan = [("S1", True), ("S1", False), ("S2", True), ("S2", False),
            ("S3", True), ("S3", False)]
    quotas = {("S1", True): args.per_stratum // 2, ("S1", False): args.per_stratum - args.per_stratum // 2,
              ("S2", True): args.per_stratum // 2, ("S2", False): args.per_stratum - args.per_stratum // 2,
              ("S3", True): args.per_stratum // 2, ("S3", False): args.per_stratum - args.per_stratum // 2}
    made = {}
    for stratum, cross in plan:
        for _ in range(quotas[(stratum, cross)]):
            anchors = list(stats.values())
            rng.shuffle(anchors)
            picked = build_stratum(stratum, anchors, stats, usage, rng, cross)
            if picked:
                anchor, b, c = picked
                triplets.append({"anchor": anchor, "b": b, "c": c, "stratum": stratum,
                                 "cross_area": cross, "fallback": False})
                made[(stratum, cross)] = made.get((stratum, cross), 0) + 1
    shortfall = args.per_stratum * 3 - len(triplets)
    for _ in range(max(0, shortfall)):
        anchors = list(stats.values())
        rng.shuffle(anchors)
        picked = build_stratum("S3", anchors, stats, usage, rng, rng.random() < 0.5)
        if not picked:
            break
        anchor, b, c = picked
        triplets.append({"anchor": anchor, "b": b, "c": c, "stratum": "S3",
                         "cross_area": anchor[0] != b[0], "fallback": True})

    for i, t in enumerate(triplets, start=1):
        t["triplet_id"] = f"t{i:04d}"
    rng2 = random.Random(SEED + 1)
    repeats = rng2.sample(triplets, max(1, len(triplets) // 10))
    for j, t in enumerate(repeats, start=1):
        triplets.append({"triplet_id": f"t{len(triplets) + j:04d}", "anchor": t["anchor"],
                         "b": t["c"], "c": t["b"], "stratum": t["stratum"],
                         "cross_area": t["cross_area"], "fallback": t.get("fallback", False),
                         "repeat_of": t["triplet_id"]})
    rng2.shuffle(triplets)

    def key(row):
        return {"area": row[0], "block": row[1]}

    payload = [{"triplet_id": t["triplet_id"], "anchor": key(t["anchor"]),
                "b": key(t["b"]), "c": key(t["c"]), "stratum": t["stratum"],
                "cross_area": t["cross_area"], "fallback": t["fallback"],
                **({"repeat_of": t["repeat_of"]} if "repeat_of" in t else {})}
               for t in triplets]
    write_json(args.dataset / "triplets.json", payload)
    write_json(args.dataset / "triplets_protocol.json", {
        "seed": SEED, "rules": RULES, "per_stratum_target": args.per_stratum,
        "min_images_per_block": args.min_images,
        "usage_caps": {"anchor": 3, "candidate": 5}, "repeat_fraction": "10% with b/c swapped",
        "manifest_provider": manifest["provider"],
        "counts": {"total": len(payload),
                   "by_stratum": {s: sum(1 for t in payload if t["stratum"] == s) for s in ("S1", "S2", "S3")},
                   "fallbacks": sum(1 for t in payload if t["fallback"]),
                   "repeats": sum(1 for t in payload if "repeat_of" in t),
                   "cross_area": sum(1 for t in payload if t["cross_area"])},
        "triplets_sha256": sha256((args.dataset / "triplets.json").read_bytes()).hexdigest()})
    print(json.dumps(json.loads((args.dataset / "triplets_protocol.json").read_bytes())["counts"], indent=2))


if __name__ == "__main__":
    main()
