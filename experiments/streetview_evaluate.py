"""QC and frozen-embedding comparison for street-view annotations."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics

import numpy as np

from streetview_common import ROOT, dataset_root

VIEWS = ("poi", "building", "walk")
AREAS = ("soho", "hudson_yards")
STRATA = ("S1", "S2", "S3")
QUESTION_DIMS = ("entry", "connect", "chain", "overall")
BOOTSTRAP_RESAMPLES = 1000
SEEDS = ("13", "29", "53")


def row_key(row):
    return row.get("question_id") or row["triplet_id"]


def value_of(row):
    return row["value"] if "value" in row else row["choice"]


def is_rating(row):
    return "value" in row or row.get("kind") == "rating"


def load_responses(dataset):
    out = {}
    for path in sorted((dataset / "responses").glob("*.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        out[path.stem] = {row_key(r): r for r in rows}
    return out


def cohen_kappa(left, right):
    if len(left) < 2:
        return None
    cats = sorted(set(left) | set(right))
    po = sum(x == y for x, y in zip(left, right)) / len(left)
    pe = sum(left.count(c) * right.count(c) for c in cats) / len(left) ** 2
    return None if pe >= 1 else (po - pe) / (1 - pe)


def per_annotator(name, rows, triplets, ratings):
    ratings_rows = [r for r in rows.values() if is_rating(r)]
    pairwise_rows = [r for r in rows.values() if not is_rating(r)]
    ms = sorted(r["elapsed_ms"] for r in rows.values())
    by_qid = {row_key(r): r for r in rows.values()}
    repeat_pairs = [(q["repeat_of"], q["question_id"]) for q in ratings.values() if "repeat_of" in q]
    repeats_ok, repeats_conflict, repeats_skip = 0, 0, 0
    for source, rep in repeat_pairs:
        if source not in by_qid or rep not in by_qid:
            continue
        a, b = value_of(by_qid[source]), value_of(by_qid[rep])
        if a not in ("b", "c") and a in ("tie", "insufficient", "X"):
            repeats_skip += 1
        elif a == b:
            repeats_ok += 1
        else:
            repeats_conflict += 1
    report = {"answered": len(rows), "rating_questions": len(ratings_rows),
              "pairwise_questions": len(pairwise_rows),
              "elapsed_ms": {"median": statistics.median(ms), "min": ms[0], "max": ms[-1],
                             "under_2s": sum(1 for m in ms if m < 2000)},
              "repeats": {"consistent": repeats_ok, "conflict": repeats_conflict,
                          "skipped": repeats_skip}}
    if ratings_rows:
        report["rating_values_by_field"] = {
            field: dict(Counter(r["value"] for r in ratings_rows if r["field"] == field))
            for field in sorted({r["field"] for r in ratings_rows})}
    if pairwise_rows:
        decided = [r for r in pairwise_rows if r["choice"] in ("b", "c")]
        left = [r for r in decided if r["left"] == "b"]
        right = [r for r in decided if r["left"] == "c"]
        report.update({"pairwise_choices": dict(Counter(r["choice"] for r in pairwise_rows)),
                       "chose_b_when_b_left": (sum(r["choice"] == "b" for r in left) / len(left)) if left else None,
                       "chose_b_when_b_right": (sum(r["choice"] == "c" for r in right) / len(right)) if right else None})
    return report


def consensus_triplets(responses, triplets):
    votes_by = {}
    for rows in responses.values():
        for r in rows.values():
            if not is_rating(r) and r["choice"] != "insufficient":
                votes_by.setdefault((r["triplet_id"], r.get("dimension")), []).append(r["choice"])
    out = {}
    for (tid, dim), votes in votes_by.items():
        if tid not in triplets or len(votes) < 2:
            continue
        top, count = Counter(votes).most_common(1)[0]
        if count * 2 > len(votes):
            out.setdefault(dim, {})[tid] = top
    return out


def consensus_ratings(responses):
    votes_by = {}
    for rows in responses.values():
        for r in rows.values():
            if is_rating(r):
                votes_by.setdefault((r["area"], r["block"], r["field"]), []).append(r["value"])
    out = {}
    for key, votes in votes_by.items():
        top, count = Counter(votes).most_common(1)[0]
        if count * 2 > len(votes):
            out.setdefault(key[2], {})[key[:2]] = top
    return out


def build_vectors(source):
    """Rebuild equal-concat fused vectors and single views from frozen predictions."""
    fused, views = {}, {name: {} for name in VIEWS}
    for area in AREAS:
        data = json.loads((source / f"{area}_predictions.json").read_bytes())
        avail = {k: np.array(a) for k, a in zip(data["keys"], data["available_views"])}
        for seed, run in data["runs"].items():
            embs = np.array(run["view_embeddings"])
            for i, k in enumerate(data["keys"]):
                mask = avail[k]
                observed = int(mask.sum())
                if not observed:
                    continue
                fused.setdefault((area, k), {})[seed] = np.concatenate(
                    [embs[i, j] * (np.sqrt(1 / observed) if mask[j] else 0.0) for j in range(3)])
                for j, name in enumerate(VIEWS):
                    if mask[j]:
                        views[name].setdefault((area, k), {})[seed] = embs[i, j]
    return fused, views


def model_agreement(model, labels, triplets, seed):
    items = []
    for tid, choice in labels.items():
        if choice not in ("b", "c") or tid not in triplets:
            continue
        t = triplets[tid]
        pick, drop = (t["b"], t["c"]) if choice == "b" else (t["c"], t["b"])
        try:
            a, w, l = (model[(x["area"], x["block"])][seed]
                       for x in (t["anchor"], pick, drop))
        except KeyError:
            continue
        items.append((t["anchor"]["area"] + t["anchor"]["block"], float(a @ w) > float(a @ l),
                      t["stratum"]))
    if not items:
        return None

    def rate(rows):
        return sum(r[1] for r in rows) / len(rows)

    anchors = sorted({r[0] for r in items})
    rng = random.Random(20261114)
    boots = sorted(rate([r for a in [rng.choice(anchors) for _ in anchors]
                         for r in items if r[0] == a]) for _ in range(BOOTSTRAP_RESAMPLES))
    return {"overall": {"n": len(items), "agreement": rate(items)},
            "by_stratum": {s: {"n": sum(r[2] == s for r in items),
                               "agreement": rate([r for r in items if r[2] == s])} for s in STRATA},
            "anchor_bootstrap_ci95": [boots[int(0.025 * len(boots))],
                                      boots[int(0.975 * len(boots)) - 1]]}


def knn_loo(model, labels):
    """Leave-one-out 1-NN accuracy of predicting each rating from the frozen vector."""
    per_seed = {}
    for seed in SEEDS:
        entries = []
        for (area, block), label in labels.items():
            vecs = model.get((area, block))
            if vecs and seed in vecs and label != "X":
                entries.append((vecs[seed], label))
        if len(entries) < 5:
            per_seed[seed] = None
            continue
        vectors = np.stack([e[0] for e in entries])
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        labels_arr = np.array([e[1] for e in entries])
        similarity = vectors @ vectors.T
        np.fill_diagonal(similarity, -np.inf)
        predicted = labels_arr[similarity.argmax(axis=1)]
        majority = Counter(labels_arr).most_common(1)[0][1] / len(entries)
        per_seed[seed] = {"n": len(entries), "accuracy": float((predicted == labels_arr).mean()),
                          "majority_baseline": float(majority)}
    return per_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--source", type=Path,
                        default=ROOT / "outputs/fusion_partition_topology_v2")
    args = parser.parse_args()

    triplets = {t["triplet_id"]: t for t in json.loads((args.dataset / "triplets.json").read_bytes())}
    ratings_path = args.dataset / "ratings.json"
    ratings = ({q["question_id"]: q for q in json.loads(ratings_path.read_bytes())}
               if ratings_path.exists() else {})
    responses = load_responses(args.dataset)
    report = {"triplets_sha256": sha256((args.dataset / "triplets.json").read_bytes()).hexdigest(),
              "ratings_sha256": (sha256(ratings_path.read_bytes()).hexdigest()
                                 if ratings_path.exists() else None),
              "source_sha256": {a: sha256((args.source / f"{a}_predictions.json").read_bytes()).hexdigest()
                                for a in AREAS},
              "annotators": {}, "agreement_between_annotators": {},
              "triplet_model_vs_labels": {}, "rating_prediction": {}}
    print(f"annotators: {sorted(responses)} ({len(triplets)} triplets, {len(ratings)} rating questions)")

    for name, rows in responses.items():
        qc = per_annotator(name, rows, triplets, ratings)
        report["annotators"][name] = qc
        print(f"[{name}] {qc['answered']} answers, median {qc['elapsed_ms']['median'] / 1000:.1f}s, "
              f"repeats {qc['repeats']}")
        for field, dist in qc.get("rating_values_by_field", {}).items():
            print(f"    {field}: {dist}")

    names = sorted(responses)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            common = sorted(set(responses[left]) & set(responses[right]))
            lv = [value_of(responses[left][k]) for k in common]
            rv = [value_of(responses[right][k]) for k in common]
            kappa = cohen_kappa(lv, rv)
            raw = sum(x == y for x, y in zip(lv, rv)) / len(common) if common else None
            report["agreement_between_annotators"][f"{left}:{right}"] = {
                "common": len(common), "cohen_kappa": kappa, "raw_agreement": raw}
            print(f"{left} vs {right}: raw {raw:.0%}, kappa {kappa:.2f} on {len(common)}")

    consensus = consensus_triplets(responses, triplets)
    if consensus:
        report["triplet_consensus"] = {dim: len(rows) for dim, rows in consensus.items()}
    fused, views = build_vectors(args.source)
    label_sets = {}
    for name, rows in responses.items():
        for dim in QUESTION_DIMS:
            labels = {r["triplet_id"]: r["choice"] for r in rows.values()
                      if not is_rating(r) and r.get("dimension") == dim}
            if labels:
                label_sets[f"{name}/{dim}"] = labels
    for dim, rows in consensus.items():
        if len(rows) >= 10:
            label_sets[f"consensus/{dim}"] = rows
    for label_name, labels in label_sets.items():
        report["triplet_model_vs_labels"][label_name] = {}
        for model_name, model in [("equal_concat_fused", fused), *views.items()]:
            per_seed = {seed: model_agreement(model, labels, triplets, seed) for seed in SEEDS}
            report["triplet_model_vs_labels"][label_name][model_name] = per_seed
            rows = [v for v in per_seed.values() if v]
            if rows:
                overall = statistics.mean(r["overall"]["agreement"] for r in rows)
                s1 = [r["by_stratum"]["S1"]["agreement"] for r in rows if r["by_stratum"]["S1"]["n"]]
                print(f"  {label_name:16s} {model_name:18s} overall {overall:.0%}"
                      + (f"  S1 {statistics.mean(s1):.0%}" if s1 else ""))

    rating_consensus = consensus_ratings(responses)
    for field, labels in sorted(rating_consensus.items()):
        report["rating_prediction"][field] = {}
        for model_name, model in [("equal_concat_fused", fused), *views.items()]:
            per_seed = knn_loo(model, labels)
            report["rating_prediction"][field][model_name] = per_seed
            valid = [v for v in per_seed.values() if v]
            if valid:
                print(f"  rating {field:14s} {model_name:18s} "
                      f"LOO-1NN {statistics.mean(v['accuracy'] for v in valid):.0%} "
                      f"(majority {statistics.mean(v['majority_baseline'] for v in valid):.0%}, "
                      f"n≈{valid[0]['n']})")

    out = args.dataset / "evaluation.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
