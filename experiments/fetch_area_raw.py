"""Download raw layers for one study area into the frozen NYC snapshot."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data/raw/nyc/2026-11-09"
RELEASE = "2026-10-21.0"  # match the frozen Overture release of the original snapshot


def curl(output, url):
    subprocess.run(["curl", "-fLsS", "--retry", "3", "--max-time", "300",
                    "-A", "PACE-Research-Pilot/1.0", url, "-o", str(output)], check=True)


def building_url(bounds):
    west, south, east, north = bounds
    ring = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    polygon = ",".join(f"{x} {y}" for x, y in ring)
    query = urlencode({"$where": f"intersects(the_geom, 'POLYGON (({polygon}))')",
                       "$order": "doitt_id", "$limit": 10000})
    return f"https://data.cityofnewyork.us/resource/5zhs-2jue.geojson?{query}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("area", help="area key in configs/areas.json")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    areas = json.loads((ROOT / "configs/areas.json").read_bytes())
    bounds = areas[args.area]["bounds"]
    bbox = ",".join(map(str, bounds))
    targets = {
        f"{args.area}_buildings.geojson": lambda p: curl(p, building_url(bounds)),
        f"{args.area}_osm.json": lambda p: curl(p, f"https://api.openstreetmap.org/api/0.6/map.json?bbox={bbox}"),
        f"{args.area}_places.geojson": lambda p: subprocess.run(
            ["overturemaps", "download", "-r", RELEASE, "--bbox", bbox,
             "-t", "place", "-f", "geojson", "-o", str(p)], check=True),
    }
    for name in targets:
        if (args.snapshot / name).exists():
            raise SystemExit(f"refusing to overwrite existing file: {name}")
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for name, fetch in targets.items():
        with TemporaryDirectory(dir=args.snapshot) as folder:
            temporary = Path(folder) / name
            fetch(temporary)
            temporary.replace(args.snapshot / name)
        print(f"downloaded {name}")
    manifest_path = args.snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.setdefault("additions", {})[args.area] = {
        "retrieved_at_utc": retrieved,
        "overture_release": RELEASE,
        "files": {name: sha256((args.snapshot / name).read_bytes()).hexdigest() for name in targets},
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"manifest updated: {args.area} @ {retrieved}")


if __name__ == "__main__":
    main()
