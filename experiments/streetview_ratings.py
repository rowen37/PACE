"""Generate per-block categorical rating questions from visible street evidence."""

import argparse
from hashlib import sha256
import json
import random

from streetview_common import dataset_root, write_json

SEED = 20261116
FIELDS = {
    "frontage": {
        "question": "【街面类型】这个街区一层的商业街面主要是哪种？",
        "options": [
            {"key": "A", "label": "临街小店为主：店门一间挨一间，直接开在人行道上"},
            {"key": "B", "label": "混合：既有临街店门，也有明显的大入口或非商业面"},
            {"key": "C", "label": "大入口为主：广场、商场或大楼门厅，店面不直接临街"},
            {"key": "D", "label": "几乎无商业街面：住宅、仓库、空墙为主"},
        ],
        "hint": "以这组照片能看到的街面为准（Gehl 活跃临街面）。判断不了选「看不出」。",
    },
    "door_density": {
        "question": "【店门密度】沿街的店门有多密？",
        "options": [
            {"key": "A", "label": "密：店门几乎一个挨一个"},
            {"key": "B", "label": "中：店门之间有明显间隔"},
            {"key": "C", "label": "稀：整片街面几乎见不到店门"},
        ],
        "hint": "只数面向街道的商业店门，不数大楼门厅和住宅门。判断不了选「看不出」。",
    },
    "continuity": {
        "question": "【沿街连续性】店面沿街墙的连续程度？",
        "options": [
            {"key": "A", "label": "连续：店面沿街墙贯穿视野，基本不断"},
            {"key": "B", "label": "有断口：主体连续，但有空墙、停车场或闲置段"},
            {"key": "C", "label": "破碎：频繁断开，或没有连续商业街面"},
        ],
        "hint": "只看这组照片里的街墙（street wall）。判断不了选「看不出」。",
    },
}
UNSURE = {"key": "X", "label": "看不出 / 证据不足"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_root().__class__, default=dataset_root())
    parser.add_argument("--repeat-count", type=int, default=15)
    parser.add_argument("--only", help="generate questions for one area only; "
                                       "ids continue after the frozen set")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_bytes())
    areas = [args.only] if args.only else list(manifest["blocks"])
    blocks = [(area, block) for area in areas for block in sorted(manifest["blocks"].get(area, {}))]
    path = args.dataset / "ratings.json"
    existing = json.loads(path.read_bytes()) if path.exists() else []
    if any(q["area"] in areas for q in existing):
        raise SystemExit(f"questions for {areas} already exist; refusing to regenerate")
    next_id = max((int(q["question_id"][1:]) for q in existing), default=0) + 1
    rng = random.Random(SEED)
    questions = []
    for i, (area, block) in enumerate(blocks):
        for field, spec in FIELDS.items():
            questions.append({"question_id": f"r{next_id + i * len(FIELDS) + list(FIELDS).index(field):04d}",
                              "kind": "rating", "area": area, "block": block,
                              "field": field, "question": spec["question"],
                              "options": spec["options"] + [UNSURE], "hint": spec["hint"]})
    repeats = rng.sample(questions, min(args.repeat_count, len(questions)))
    for j, q in enumerate(repeats, start=1):
        questions.append({**q, "question_id": f"r{next_id + len(questions) + j - 1:04d}",
                          "repeat_of": q["question_id"]})
    rng.shuffle(questions)

    added = len(questions)
    if existing:
        questions = existing + questions
    write_json(path, questions)
    write_json(args.dataset / "ratings_protocol.json", {
        "seed": SEED, "fields": FIELDS, "unsure": UNSURE, "repeat_count": args.repeat_count,
        "blocks": len(blocks), "areas": areas,
        "counts": {"total": len(questions), "added": added,
                   **{f: sum(1 for q in questions if q["field"] == f) for f in FIELDS}},
        "ratings_sha256": sha256(path.read_bytes()).hexdigest()})
    print(f"added {added} questions ({areas}); file now holds {len(questions)}")


if __name__ == "__main__":
    main()
