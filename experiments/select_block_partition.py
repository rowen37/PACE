"""Select a cut of each frozen whole-block tree, then evaluate that cut."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import networkx as nx

AREAS = ("hudson_yards", "soho")


def write(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def select_cut(keys, edges, similarities, candidates):
    """Maximize standard weighted modularity over existing tree levels only."""
    graph = nx.Graph()
    graph.add_nodes_from(keys)
    for (a, b), cosine in zip(edges, similarities, strict=True):
        if not math.isfinite(cosine) or not -1.000001 <= cosine <= 1.000001:
            raise ValueError(f"Invalid cosine: {cosine}")
        graph.add_edge(a, b, weight=(1 + min(1.0, max(-1.0, cosine))) / 2)
    n = len(keys)
    if {tuple(c) for c in candidates[:n]} != {(k,) for k in keys}:
        raise ValueError("Tree does not start with exactly the original block leaves")
    active = {frozenset((k,)) for k in keys}
    by_key = {k: frozenset((k,)) for k in keys}
    levels = []
    best = None

    def evaluate_level():
        nonlocal best
        ordered = sorted(tuple(sorted(group)) for group in active)
        modularity = nx.community.modularity(graph, ordered, weight="weight", resolution=1)
        levels.append({"groups": len(active), "modularity": modularity})
        if best is None or modularity > best["modularity"]:
            best = {"partition": ordered, **levels[-1]}

    evaluate_level()
    for nodes in candidates[n:]:
        merged = frozenset(nodes)
        children = {by_key[k] for k in merged}
        if len(children) != 2 or frozenset.union(*children) != merged:
            raise ValueError("Not a binary merge of active groups")
        if not nx.is_connected(graph.subgraph(merged)):
            raise ValueError("Frozen candidate breaks the walk adjacency constraint")
        active.difference_update(children)
        active.add(merged)
        by_key.update(dict.fromkeys(merged, merged))
        evaluate_level()
    return {**best, "levels": levels}


def select(source, output):
    result = {"selector": "max weighted modularity on the original complete-link tree",
              "edge_weight": "(1 + cosine) / 2", "resolution": 1,
              "tie_break": "retain the finer cut on an exact tie",
              "source_protocol_sha256": sha256((source / "protocol.json").read_bytes()).hexdigest(),
              "areas": {}}
    for area in AREAS:
        raw = (source / f"{area}_predictions.json").read_bytes()
        payload = json.loads(raw)
        cuts = {method: {seed: select_cut(payload["keys"], payload["adjacency"], run["edge_scores"], run["candidates"])
                         for seed, run in reps.items()}
                for method, reps in payload["predictions"].items()}
        result["areas"][area] = {"predictions_sha256": sha256(raw).hexdigest(), "methods": cuts}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "selection.json", result)
    print("Frozen partitions saved without opening BID labels.")


def score(source, output):
    # These imports and label reads belong to the scoring phase only.
    from statistics import mean
    from shapely import union_all
    from shapely.geometry import shape
    from shapely.ops import transform
    from block_boundary_benchmark import REFERENCE_NAMES, ROOT, TO_METRES, average_precision, map_inputs

    selected = (output / "selection.json").read_bytes()
    cuts = json.loads(selected)
    previous = json.loads((source / "metrics.json").read_bytes())
    source_labels = (ROOT / "data/annotations/nyc_boundary_pilot/commercial_bids.geojson").read_bytes()
    label_hash = sha256(source_labels).hexdigest()
    if label_hash != previous["reference_sha256"]:
        raise ValueError("BID geometry has changed since the prior benchmark")
    fences = {f["properties"]["f_all_bi_2"]: transform(TO_METRES, shape(f["geometry"]))
              for f in json.loads(source_labels)["features"]}
    result = {"selection_sha256": sha256(selected).hexdigest(), "reference_sha256": label_hash, "areas": {}}
    for area in AREAS:
        raw = (source / f"{area}_predictions.json").read_bytes()
        payload = json.loads(raw)
        before, chosen = previous["areas"][area], cuts["areas"][area]
        if sha256(raw).hexdigest() != chosen["predictions_sha256"] or chosen["predictions_sha256"] != before["predictions_sha256"]:
            raise ValueError(f"Predictions differ from the earlier benchmark: {area}")
        records, adjacency, _ = map_inputs(area)
        if adjacency != list(map(tuple, payload["adjacency"])):
            raise ValueError(f"Walk adjacency changed: {area}")
        blocks = {k: transform(TO_METRES, shape(f["geometry"])) for k, f in records.items()}
        fence = fences[REFERENCE_NAMES[area]]
        reference = fence.intersection(union_all(list(blocks.values())))
        inside = {k for k, b in blocks.items() if fence.covers(b.centroid)}
        support = [i for i, (a, b) in enumerate(adjacency) if a in inside or b in inside]
        labels = [int(adjacency[i][0] in inside and adjacency[i][1] in inside) for i in support]
        if (sum(labels), len(labels) - sum(labels)) != (before["positive_edges"], before["negative_edges"]):
            raise ValueError(f"BID evaluation edges changed: {area}")
        intersect = {k: b.intersection(reference).area for k, b in blocks.items()}

        def iou(group):
            overlap = sum(intersect[k] for k in group)
            return overlap / (sum(blocks[k].area for k in group) + reference.area - overlap)

        methods = {}
        for method, reps in chosen["methods"].items():
            runs = {}
            for seed, cut in reps.items():
                frozen = payload["predictions"][method][seed]
                old = before["methods"][method]["runs"][seed]
                ap = average_precision(labels, [frozen["edge_scores"][i] for i in support])
                oracle = max(map(iou, frozen["candidates"]))
                if ap != old["ap"] or not math.isclose(oracle, old["best_candidate_iou"], abs_tol=1e-12):
                    raise ValueError(f"Prior scores no longer reproduce: {area}/{method}/{seed}")
                match = max(cut["partition"], key=iou)
                score_iou = iou(match)
                if score_iou > oracle + 1e-12:
                    raise ValueError("Selected group must be an original tree candidate")
                runs[seed] = {"ap": ap, "tree_oracle_iou": oracle, "selected_cut_iou": score_iou,
                              "iou_gap": oracle - score_iou, "matched_group": match,
                              "groups": cut["groups"], "modularity": cut["modularity"]}
            methods[method] = {**{k: mean(r[k] for r in runs.values())
                                  for k in ("ap", "tree_oracle_iou", "selected_cut_iou", "iou_gap", "groups")},
                               "runs": runs}
        result["areas"][area] = {"methods": methods}
    result["area_macro_average"] = {
        method: {k: mean(r["methods"][method][k] for r in result["areas"].values())
                 for k in ("ap", "tree_oracle_iou", "selected_cut_iou", "iou_gap")}
        for method in result["areas"][AREAS[0]]["methods"]}
    write(output / "metrics.json", result)
    print(json.dumps(result["area_macro_average"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "score"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (select if args.phase == "select" else score)(args.source, args.output)


if __name__ == "__main__":
    main()
