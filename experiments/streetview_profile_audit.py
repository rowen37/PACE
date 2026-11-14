"""Audit frozen partitions against consensus human street-profile labels."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from shapely.geometry import shape
from shapely.ops import transform

from block_boundary_benchmark import REFERENCE_NAMES, ROOT, TO_METRES
from streetview_common import AREAS, dataset_root, write_json

FIELDS = ("frontage", "door_density", "continuity")
NAMED = ("b014", "b054", "b040")
SEEDS = ("13", "29", "53")


def consensus_profiles(dataset):
    votes = {}
    for path in sorted((dataset / "responses").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") == "rating" and r["value"] != "X":
                votes.setdefault((r["area"], r["block"], r["field"]), []).append(r["value"])
    profiles = {}
    for (area, block, field), vals in votes.items():
        top, count = Counter(vals).most_common(1)[0]
        if count * 2 > len(vals):  # strict majority; ties stay unknown
            profiles.setdefault((area, block), {})[field] = top
    return profiles


def purity(groups, profiles, area, field):
    """Share of blocks whose label matches their group's modal label."""
    agree, total = 0, 0
    for group in groups:
        labels = [profiles[(area, b)][field] for b in group
                  if (area, b) in profiles and field in profiles[(area, b)]]
        if not labels:
            continue
        agree += Counter(labels).most_common(1)[0][1]
        total += len(labels)
    return (agree / total, total) if total else (None, 0)


def full_profile_purity(groups, profiles, area):
    agree, total = 0, 0
    for group in groups:
        tuples = [tuple(sorted(profiles[(area, b)].items())) for b in group
                  if (area, b) in profiles and len(profiles[(area, b)]) == len(FIELDS)]
        if not tuples:
            continue
        agree += Counter(tuples).most_common(1)[0][1]
        total += len(tuples)
    return (agree / total, total) if total else (None, 0)


def edge_contrast(groups, adjacency, profiles, area):
    """Mean profile agreement on walking edges within groups vs across groups."""
    member = {b: i for i, group in enumerate(groups) for b in group}

    def agreement(a, b):
        pa, pb = profiles.get((area, a), {}), profiles.get((area, b), {})
        shared = [f for f in FIELDS if pa.get(f) and pb.get(f)]
        if not shared:
            return None
        return sum(pa[f] == pb[f] for f in shared) / len(shared)

    within, across = [], []
    for a, b in adjacency:
        score = agreement(a, b)
        if score is None:
            continue
        (within if member.get(a) == member.get(b) else across).append(score)
    return (mean(within) if within else None, len(within),
            mean(across) if across else None, len(across))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/fusion_partition_topology_v2")
    parser.add_argument("--infomap", type=Path, default=ROOT / "outputs/infomap_partition_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/streetview_profile_audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    profiles = consensus_profiles(args.dataset)
    fence_raw = (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes()
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(fence_raw)["features"]}

    report = {"profiles": {"blocks_with_consensus": len(profiles)}, "areas": {}}
    for area in AREAS:
        data = json.loads((args.source / f"{area}_predictions.json").read_bytes())
        info = json.loads((args.infomap / f"{area}_partitions.json").read_bytes())["runs"]
        blocks_geo = {f["properties"]["block_id"]: transform(TO_METRES, shape(f["geometry"]))
                      for f in json.loads((ROOT / f"data/processed/nyc-2026-11-09-block-inputs/"
                                                f"{area}/blocks.geojson").read_bytes())["features"]}
        fence = fences[REFERENCE_NAMES[area]]
        bid_groups = [[b for b in blocks_geo if fence.covers(blocks_geo[b].centroid)],
                      [b for b in blocks_geo if not fence.covers(blocks_geo[b].centroid)]]
        methods = {f"seed{seed}/modularity_moves":
                   data["runs"][seed]["fusion"]["equal_concat"]["partitions"]["block_moves"]["groups"]
                   for seed in SEEDS}
        methods.update({f"seed{seed}/tree_cut":
                        data["runs"][seed]["fusion"]["equal_concat"]["partitions"]["tree_cut"]["groups"]
                        for seed in SEEDS})
        methods.update({f"seed{seed}/infomap": info[seed]["infomap"] for seed in SEEDS})
        methods["bid_reference"] = bid_groups

        rows = {}
        for name, groups in methods.items():
            purities = {f: purity(groups, profiles, area, f) for f in FIELDS}
            full = full_profile_purity(groups, profiles, area)
            within, nw, across, na = edge_contrast(groups, data["adjacency"], profiles, area)
            rows[name] = {"groups": len(groups),
                          "purity": {f: {"value": purities[f][0], "n": purities[f][1]} for f in FIELDS},
                          "full_profile_purity": {"value": full[0], "n": full[1]},
                          "edge_agreement": {"within": within, "n_within": nw,
                                             "across": across, "n_across": na}}
        base = {f: purity([list(blocks_geo)], profiles, area, f)[0] for f in FIELDS}
        report["areas"][area] = {"trivial_purity": base, "methods": rows,
                                 "named_blocks": {}}
        for block in NAMED:
            profile = profiles.get((area, block), {})
            report["areas"][area]["named_blocks"][block] = {
                "consensus_profile": profile,
                "groups": {name: (lambda hit: None if hit is None else {
                    "index": hit[0], "size": len(hit[1]), "members": hit[1],
                    "modal_profile": {f: Counter(
                        profiles[(area, b)][f] for b in hit[1]
                        if (area, b) in profiles and f in profiles[(area, b)]).most_common(1)[0][0]
                        for f in FIELDS}})(
                    next(((i, list(g)) for i, g in enumerate(groups) if block in g), None))
                    for name, groups in list(methods.items())[:2]}}
        print(f"\n[{area}] trivial purity {base}")
        for name, row in rows.items():
            p = row["purity"]
            print(f"  {name:26s} groups {row['groups']:2d} | "
                  + " ".join(f"{f[:4]} {p[f]['value']:.0%}" for f in FIELDS if p[f]["value"] is not None)
                  + f" | full {row['full_profile_purity']['value']:.0%}"
                  + f" | edge in {row['edge_agreement']['within']:.0%} vs cross "
                    f"{row['edge_agreement']['across']:.0%}")
        for block in NAMED:
            row = report["areas"][area]["named_blocks"][block]
            if not row["consensus_profile"]:
                print(f"  {block}: no consensus")
                continue
            detail = " | ".join(
                f"{name.split('/', 1)[1]}: 组{g['index']}(n={g['size']}) 模态={g['modal_profile']}"
                for name, g in row["groups"].items() if g) or "not in this area"
            print(f"  {block}: {row['consensus_profile']}  {detail}")

    report["inputs"] = {
        "predictions_sha256": {a: sha256((args.source / f"{a}_predictions.json").read_bytes()).hexdigest()
                               for a in AREAS},
        "infomap_sha256": {a: sha256((args.infomap / f"{a}_partitions.json").read_bytes()).hexdigest()
                           for a in AREAS},
        "responses_sha256": {p.name: sha256(p.read_bytes()).hexdigest()
                             for p in sorted((args.dataset / "responses").glob("*.jsonl"))}}
    write_json(args.output / "audit.json", report)
    print(f"\nwritten: {args.output / 'audit.json'}")


if __name__ == "__main__":
    main()
