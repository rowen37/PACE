"""Measure parcel coverage and post-hoc IoU ceilings in the same block extent."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

from shapely import union_all
from shapely.geometry import shape
from shapely.ops import transform

from block_boundary_benchmark import AREAS, REFERENCE_NAMES, ROOT, TO_METRES, write
from diagnose_block_overlap import block_ceiling
from fusion_partition_benchmark import DEFAULT_INPUT, block_geometry


def ratio(part, total):
    return part / total if total else None


def ceiling(units, reference):
    sizes = {key: g.area for key, g in units.items()}
    hits = {key: g.intersection(reference).area for key, g in units.items()}
    result = block_ceiling(sizes, hits, reference.area)
    # Recompute the selected union exactly in case source parcels have tiny slivers of overlap.
    selected = union_all([units[key] for key in result["blocks"]])
    hit = selected.intersection(reference).area
    return {"iou": hit / selected.union(reference).area, "selected_units": len(result["blocks"]),
            "coverage_of_reference": hit / reference.area,
            "extra_ha": selected.difference(reference).area / 10_000,
            "missed_ha": reference.difference(selected).area / 10_000, "selected_ids": result["blocks"]}


def measure(source, inputs, parcels, output):
    frozen = json.loads((source / "protocol.json").read_bytes())
    old = json.loads((source / "metrics.json").read_bytes())
    manifest = json.loads((parcels / "manifest.json").read_bytes())
    bid_raw = (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes()
    bid_hash = sha256(bid_raw).hexdigest()
    if old["reference_sha256"] != bid_hash:
        raise ValueError("BID reference differs from the original benchmark")
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(bid_raw)["features"]}
    report = {"reference_sha256": bid_hash, "parcel_manifest": manifest,
              "scope": "all original blocks; geometries clipped to study extent only, prior to BID scoring", "areas": {}}
    for area in AREAS:
        block_path = inputs / area / "blocks.geojson"
        lot_raw = (parcels / f"{area}.geojson").read_bytes()
        m = manifest["areas"][area]
        if (sha256(block_path.read_bytes()).hexdigest() != frozen["input_sha256"][area][block_path.name]
                or sha256(lot_raw).hexdigest() != m["sha256"]):
            raise ValueError(f"Block or parcel input changed: {area}")
        blocks = block_geometry(block_path)
        extent = union_all(list(blocks.values()))
        bbox = tuple(map(float, m["bbox_4326"].split(",")))
        features = json.loads((inputs / area / block_path.name).read_bytes())["features"]
        ll_extent = union_all([shape(f["geometry"]) for f in features])
        if not all(sign * (a - b) >= 0 for a, b, sign in zip(ll_extent.bounds, bbox, (1, 1, -1, -1), strict=True)):
            raise ValueError(f"The API bounding box does not contain the complete block extent: {area}")
        records = json.loads(lot_raw)["features"]
        if len(records) != m["feature_count"] or len({f["properties"]["BBL"] for f in records}) != len(records):
            raise ValueError(f"Parcel count or BBL IDs invalid: {area}")
        lot_geometries = {str(int(f["properties"]["BBL"])): transform(TO_METRES, shape(f["geometry"])) for f in records}
        if not all(g.is_valid for g in lot_geometries.values()):
            raise ValueError(f"Invalid parcel polygon: {area}")
        clipped = {k: g.intersection(extent) for k, g in lot_geometries.items()}
        lots = {k: g for k, g in clipped.items() if g.area > 0}
        lot_union = union_all(list(lots.values()))
        # Quantify polygon overlaps so that the disjoint-unit ceiling's numerical precision is explicit.
        duplicated = max(0.0, math.fsum(g.area for g in lots.values()) - lot_union.area)
        fence = fences[REFERENCE_NAMES[area]]
        reference = fence.intersection(extent)
        covered_reference = reference.intersection(lot_union)
        places = json.loads((inputs / area / "places.geojson").read_bytes())["features"]
        query = [transform(TO_METRES, shape(f["geometry"])) for f in places]
        block_coverage = {k: ratio(g.intersection(lot_union).area, g.area) for k, g in blocks.items()}
        report["areas"][area] = {
            "blocks": len(blocks), "blocks_with_any_parcel_area": sum(v > 0 for v in block_coverage.values()),
            "lots_requested": len(records), "lots_in_extent": len(lots),
            "lot_overlap_area_m2": duplicated, "study_ha": extent.area / 10_000,
            "study_area_with_lot": lot_union.area / extent.area,
            "reference_ha": reference.area / 10_000, "full_reference_ha": fence.area / 10_000,
            "reference_area_with_lot": covered_reference.area / reference.area,
            "reference_ha_without_lot": reference.difference(lot_union).area / 10_000,
            "poi_count": len(query), "pois_in_lots": sum(lot_union.covers(p) for p in query),
            "block_coverage": block_coverage,
            "same_full_reference_ceiling": {"whole_blocks": ceiling(blocks, reference),
                                             "tax_lots": ceiling(lots, reference)},
            "lot_covered_reference_only_ceiling": ceiling(lots, covered_reference)}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "geometry.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parcels", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--inputs", default=DEFAULT_INPUT, type=Path)
    args = parser.parse_args()
    measure(args.source, args.inputs, args.parcels, args.output)
