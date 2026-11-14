"""Local annotation server for street-view block annotation (ratings or triplets)."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import urllib.parse

from streetview_common import dataset_root

CHOICES = ("b", "c", "tie", "insufficient")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml"}


def load_items(dataset, manifest):
    """Ratings take precedence over per-dimension triplets, then v1 holistic triplets."""
    def views(area, block):
        return [f"/img/{area}/{block}/{r['file']}"
                for r in manifest["blocks"].get(area, {}).get(block, [])]

    rating_path = dataset / "ratings.json"
    if rating_path.exists():
        return [{"kind": "rating", "images": views(q["area"], q["block"]), **q}
                for q in json.loads(rating_path.read_bytes())]
    questions_path = dataset / "questions.json"
    if questions_path.exists():
        triplets = {t["triplet_id"]: t
                    for t in json.loads((dataset / "triplets.json").read_bytes())}
        items = []
        for q in json.loads(questions_path.read_bytes()):
            t = triplets[q["triplet_id"]]
            items.append({"kind": "pairwise", **q,
                          "anchor_images": views(t["anchor"]["area"], t["anchor"]["block"]),
                          "b_images": views(t["b"]["area"], t["b"]["block"]),
                          "c_images": views(t["c"]["area"], t["c"]["block"])})
        return items
    triplets = json.loads((dataset / "triplets.json").read_bytes())
    return [{"kind": "pairwise", "question_id": t["triplet_id"], "triplet_id": t["triplet_id"],
             "dimension": "overall",
             "question": "哪个候选街区逛起来的方式更像 A —— 怎么进入、公共步行如何连通、能否一路串联多个目的地？",
             "hint": "基于街景判断街区的步行探索结构。只判断相似，不判断好坏。",
             "anchor_images": views(t["anchor"]["area"], t["anchor"]["block"]),
             "b_images": views(t["b"]["area"], t["b"]["block"]),
             "c_images": views(t["c"]["area"], t["c"]["block"])}
            for t in triplets]


class Handler(BaseHTTPRequestHandler):
    dataset = None
    html = (Path(__file__).resolve().parent / "streetview_annotation.html").read_bytes()

    def log_message(self, *args):
        pass

    def reply(self, code, body, content_type="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type if isinstance(body, bytes) else
                         "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            return self.reply(200, self.html, "text/html; charset=utf-8")
        if parsed.path == "/api/config":
            annotator = (query.get("annotator") or ["anonymous"])[0]
            manifest = json.loads((self.dataset / "manifest.json").read_bytes())
            items = load_items(self.dataset, manifest)
            answered = set()
            path = self.dataset / "responses" / f"{SAFE_NAME.fullmatch(annotator) and annotator}.jsonl"
            if SAFE_NAME.fullmatch(annotator) and path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        answered.add(row.get("question_id") or row["triplet_id"])
            return self.reply(200, {"provider": manifest["provider"], "total": len(items),
                                    "answered": sorted(answered), "items": items})
        match = re.fullmatch(r"/img/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", parsed.path)
        if match:
            target = self.dataset / "images" / match.group(1) / match.group(2) / match.group(3)
            if not target.is_file():
                return self.reply(404, {"error": "not found"})
            return self.reply(200, target.read_bytes(), TYPES.get(target.suffix, "application/octet-stream"))
        return self.reply(404, {"error": "unknown route"})

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/response":
            return self.reply(404, {"error": "unknown route"})
        length = int(self.headers.get("Content-Length", 0))
        row = json.loads(self.rfile.read(length))
        annotator = row.get("annotator", "")
        if not SAFE_NAME.fullmatch(annotator or ""):
            return self.reply(400, {"error": "invalid annotator"})
        manifest = json.loads((self.dataset / "manifest.json").read_bytes())
        items = {q["question_id"]: q for q in load_items(self.dataset, manifest)}
        question = items.get(row.get("question_id"))
        if question is None:
            return self.reply(400, {"error": "unknown question"})
        value = row.get("value")
        if question["kind"] == "rating":
            valid = {o["key"] for o in question["options"]}
        else:
            valid = set(CHOICES)
        if value not in valid:
            return self.reply(400, {"error": "invalid value"})
        record = {"question_id": question["question_id"], "annotator": annotator,
                  "kind": question["kind"], "value": value,
                  "elapsed_ms": row.get("elapsed_ms"),
                  "rated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if question["kind"] == "rating":
            record.update({"area": question["area"], "block": question["block"],
                           "field": question["field"]})
        else:
            record.update({"triplet_id": question["triplet_id"],
                           "dimension": question["dimension"], "left": row.get("left")})
        folder = self.dataset / "responses"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{annotator}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.reply(200, {"saved": record["question_id"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    Handler.dataset = args.dataset.resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving {args.dataset} at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
