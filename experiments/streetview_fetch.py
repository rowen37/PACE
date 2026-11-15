"""Fetch street-view images for sampled camera points (Google, Mapillary, or demo)."""

import argparse
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

from pyproj import Transformer

from streetview_common import DATA_AREAS as AREAS, dataset_root, read_features, write_json

TO_METRES = Transformer.from_crs(4326, 6538, always_xy=True)
IMAGE_SIZE, FOV, PITCH, RADIUS_M, SNAP_M = "640x640", 100, 5, 25, 25
REQUEST_GAP_S, RETRIES = 0.15, 3


def http_get(url, binary=False):
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read() if binary else json.loads(response.read())
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def spread(points, keep):
    if len(points) <= keep:
        return points
    cx = sum(p["lon"] for p in points) / len(points)
    cy = sum(p["lat"] for p in points) / len(points)
    chosen = [min(points, key=lambda p: (p["lon"] - cx) ** 2 + (p["lat"] - cy) ** 2)]
    while len(chosen) < keep:
        best, best_gap = None, -1.0
        for p in points:
            if p in chosen:
                continue
            gap = min((p["lat"] - q["lat"]) ** 2 + (p["lon"] - q["lon"]) ** 2 for q in chosen)
            if gap > best_gap:
                best, best_gap = p, gap
        chosen.append(best)
    return chosen


def load_points(dataset):
    points = {}
    for area in AREAS:
        raw = json.loads((dataset / "points" / f"{area}_points.geojson").read_bytes())
        per_block = {}
        for f in raw["features"]:
            lon, lat = f["geometry"]["coordinates"]
            per_block.setdefault(f["properties"]["block_id"], []).append(
                {"lon": lon, "lat": lat, "heading": f["properties"]["heading"]})
        points[area] = per_block
    return points


def save_image(dataset, area, block, index, payload, record, manifest_entry):
    folder = dataset / "images" / area / block
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "svg" if isinstance(payload, str) else "jpg"
    name = f"{index}.{suffix}"
    (folder / name).write_bytes(payload if isinstance(payload, bytes) else payload.encode("utf-8"))
    record.append({"block_id": block, "file": name, **manifest_entry,
                   "sha256": sha256((folder / name).read_bytes()).hexdigest()})


def fetch_google(dataset, points, images_per_block, key):
    def metadata(p):
        query = urllib.parse.urlencode({"location": f"{p['lat']:.6f},{p['lon']:.6f}",
                                        "radius": RADIUS_M, "source": "outdoor", "key": key})
        row = http_get(f"https://maps.googleapis.com/maps/api/streetview/metadata?{query}")
        return {**p, "pano_id": row.get("pano_id"), "date": row.get("date"),
                "lat": (row.get("location") or {}).get("lat", p["lat"]),
                "lon": (row.get("location") or {}).get("lon", p["lon"])} if row.get("status") == "OK" else None

    images = {}
    for area, blocks in points.items():
        images[area] = {}
        for block, candidates in blocks.items():
            found = []
            for p in candidates:
                hit = metadata(p)
                time.sleep(REQUEST_GAP_S)
                if hit and not any(h["pano_id"] == hit["pano_id"] for h in found):
                    found.append(hit)
            chosen = spread(found, images_per_block)
            rows = []
            for i, p in enumerate(chosen):
                query = urllib.parse.urlencode({"pano": p["pano_id"], "size": IMAGE_SIZE,
                                                "heading": round(p["heading"]), "pitch": PITCH,
                                                "fov": FOV, "key": key})
                data = http_get(f"https://maps.googleapis.com/maps/api/streetview?{query}", binary=True)
                time.sleep(REQUEST_GAP_S)
                save_image(dataset, area, block, i, data, rows,
                           {"provider": "google", "pano_id": p["pano_id"], "date": p["date"],
                            "lat": round(p["lat"], 6), "lon": round(p["lon"], 6),
                            "heading": round(p["heading"]), "fov": FOV, "pitch": PITCH,
                            "credit": "© Google"})
            images[area][block] = rows
            print(f"google {area}/{block}: {len(rows)} images", flush=True)
    return images


def fetch_mapillary(dataset, points, images_per_block, token):
    headers = {"Authorization": f"OAuth {token}"}

    def request(url):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read())

    images = {}
    for area, blocks in points.items():
        all_lon = [p["lon"] for c in blocks.values() for p in c]
        all_lat = [p["lat"] for c in blocks.values() for p in c]
        pad = SNAP_M / 111_000 * 1.4
        bbox = f"{min(all_lon) - pad},{min(all_lat) - pad},{max(all_lon) + pad},{max(all_lat) + pad}"
        rows = request("https://graph.mapillary.com/images?" + urllib.parse.urlencode(
            {"bbox": bbox, "is_pano": "true",
             "fields": "id,geometry,compass_angle,captured_at,is_pano"})).get("data", [])
        photos = [{"id": r["id"], "lon": r["geometry"]["coordinates"][0],
                   "lat": r["geometry"]["coordinates"][1],
                   "heading": r.get("compass_angle") or 0.0,
                   "date": time.strftime("%Y-%m", time.gmtime(r.get("captured_at", 0) / 1000))}
                  for r in rows]
        images[area] = {}
        for block, candidates in blocks.items():
            found = []
            for p in candidates:
                px, py = TO_METRES.transform(p["lon"], p["lat"])
                near = [(math.hypot(*(a - b for a, b in zip(TO_METRES.transform(q["lon"], q["lat"]), (px, py)))), q)
                        for q in photos]
                near = [(d, q) for d, q in near if d <= SNAP_M]
                if near:
                    found.append({**p, "snap": min(near)[1]})
            chosen = spread(found, images_per_block)
            out = []
            for i, p in enumerate(chosen):
                q = p["snap"]
                detail = request(f"https://graph.mapillary.com/{q['id']}?fields=thumb_2048_url")
                with urllib.request.urlopen(detail["thumb_2048_url"], timeout=30) as response:
                    data = response.read()
                time.sleep(REQUEST_GAP_S)
                save_image(dataset, area, block, i, data, out,
                           {"provider": "mapillary", "pano_id": str(q["id"]), "date": q["date"],
                            "lat": round(q["lat"], 6), "lon": round(q["lon"], 6),
                            "heading": round(p["heading"]), "fov": FOV, "pitch": PITCH,
                            "credit": "© Mapillary"})
            images[area][block] = out
            print(f"mapillary {area}/{block}: {len(out)} images", flush=True)
    return images


def fetch_demo(dataset, points, images_per_block):
    today = date.today().isoformat()
    images = {}
    for area, blocks in points.items():
        images[area] = {}
        for block, candidates in blocks.items():
            chosen = spread(candidates, images_per_block)
            rows = []
            for i, p in enumerate(chosen):
                hue = int(sha256(f"{area}/{block}".encode()).hexdigest()[:4], 16) % 360
                svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='640' height='640'>"
                       f"<rect width='640' height='640' fill='hsl({hue},45%,70%)'/>"
                       f"<rect y='440' width='640' height='200' fill='hsl({hue},20%,45%)'/>"
                       f"<text x='320' y='330' font-family='sans-serif' font-size='44' "
                       f"text-anchor='middle' fill='#222'>{block}</text>"
                       f"<text x='320' y='390' font-family='sans-serif' font-size='26' "
                       f"text-anchor='middle' fill='#444'>DEMO view {i + 1} · {area}</text></svg>")
                save_image(dataset, area, block, i, svg, rows,
                           {"provider": "demo", "pano_id": None, "date": today,
                            "lat": round(p["lat"], 6), "lon": round(p["lon"], 6),
                            "heading": round(p["heading"]), "fov": FOV, "pitch": PITCH,
                            "credit": "placeholder"})
            images[area][block] = rows
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=("google", "mapillary", "demo"))
    parser.add_argument("--dataset", type=Path, default=dataset_root())
    parser.add_argument("--images-per-block", type=int, default=4)
    parser.add_argument("--only", help="fetch one area and merge into the existing manifest")
    parser.add_argument("--replace", action="store_true",
                        help="required when a manifest already exists")
    args = parser.parse_args()

    manifest_path = args.dataset / "manifest.json"
    for env, provider in (("PACE_GSV_API_KEY", "google"), ("PACE_MAPILLARY_TOKEN", "mapillary")):
        if args.provider == provider and not os.environ.get(env):
            raise SystemExit(f"set {env} for the {provider} provider")
    points = load_points(args.dataset)
    if args.only:
        points = {args.only: points[args.only]}
        if not manifest_path.exists():
            raise SystemExit("--only needs an existing manifest")
        manifest = json.loads(manifest_path.read_bytes())
        if manifest["provider"] != args.provider:
            raise SystemExit(f"manifest provider {manifest['provider']} != {args.provider}")
    else:
        if manifest_path.exists() and not args.replace:
            raise SystemExit("manifest.json already exists; pass --replace to refetch")
        manifest = None
    if args.provider == "google":
        images = fetch_google(args.dataset, points, args.images_per_block,
                              os.environ["PACE_GSV_API_KEY"])
    elif args.provider == "mapillary":
        images = fetch_mapillary(args.dataset, points, args.images_per_block,
                                 os.environ["PACE_MAPILLARY_TOKEN"])
    else:
        images = fetch_demo(args.dataset, points, args.images_per_block)
    if manifest is None:
        write_json(manifest_path, {"provider": args.provider,
                                   "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "size": IMAGE_SIZE, "fov": FOV, "pitch": PITCH,
                                   "radius_m": RADIUS_M, "images_per_block": args.images_per_block,
                                   "blocks": images})
    else:
        manifest["blocks"].update(images)
        manifest["additions"] = {**manifest.get("additions", {}),
                                 args.only: datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
