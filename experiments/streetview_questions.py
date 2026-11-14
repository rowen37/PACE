"""Expand triplets into one-question-per-dimension annotation items."""

import argparse
from hashlib import sha256
import json
import random

from streetview_common import dataset_root, write_json

SEED = 20261115
DIMENSIONS = {
    "entry": {
        "question": "【进入衔接】哪个候选街区「店面与街道的衔接方式」更像 A？",
        "hint": "只看一层门面：店门是否直接开向人行道、门口密度如何、要不要经过广场或商场中庭过渡。"
                "两边都是密集临街小门，或都是大入口过渡，算像。（构念：Gehl 活跃临街面）",
    },
    "connect": {
        "question": "【公共连通】哪个候选街区「沿公共步行道的连续性」更像 A？",
        "hint": "只看逛的时候会不会断：沿街店面连续还是被空墙、停车场、住宅面打断，需不需要绕行或过街。"
                "断开的多少与位置相近，算像。（构念：空间句法 permeability、Lynch 路径连续性）",
    },
    "chain": {
        "question": "【串联能力】哪个候选街区「一次连续步行能串起的目的地」更像 A？",
        "hint": "只看能顺路逛到的目的地：数量、集中在一条街还是散在四周、走线是否顺路。"
                "密度和分布相近，算像。（构念：destination accessibility）",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_root().__class__, default=dataset_root())
    parser.add_argument("--repeat-count", type=int, default=10)
    args = parser.parse_args()

    triplets = json.loads((args.dataset / "triplets.json").read_bytes())
    rng = random.Random(SEED)
    questions = [{"question_id": f"q{i + 1:04d}", "triplet_id": t["triplet_id"],
                  "dimension": dim, **DIMENSIONS[dim]}
                 for i, t in enumerate(triplets) for dim in DIMENSIONS]
    repeats = rng.sample(questions, min(args.repeat_count, len(questions)))
    for j, q in enumerate(repeats, start=1):
        questions.append({"question_id": f"q{len(questions) + j:04d}",
                          "triplet_id": q["triplet_id"], "dimension": q["dimension"],
                          "question": q["question"], "hint": q["hint"],
                          "repeat_of": q["question_id"]})
    rng.shuffle(questions)

    path = args.dataset / "questions.json"
    write_json(path, questions)
    counts = {}
    for q in questions:
        counts[q["dimension"]] = counts.get(q["dimension"], 0) + 1
    write_json(args.dataset / "questions_protocol.json", {
        "seed": SEED, "dimensions": DIMENSIONS, "repeat_count": args.repeat_count,
        "counts": {"total": len(questions), **counts},
        "questions_sha256": sha256(path.read_bytes()).hexdigest()})
    print(f"{len(questions)} questions:", counts)


if __name__ == "__main__":
    main()
