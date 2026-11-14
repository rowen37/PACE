"""Fetch the public inputs for the two PACE study windows."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
RELEASE = "2026-10-21.0"
NYCWALKS = (
    "https://www.dropbox.com/scl/fi/08is52ow5mowh8ieali8n/"
    "NYC_pednetwork_estimates_counts_2018-2019.geojson?dl=1&rlkey=vdgyuvyzzkgovg1ygnfib1afp"
)


def download(output, command):
    with TemporaryDirectory(dir=output.parent) as folder:
        temporary = Path(folder) / output.name
        subprocess.run([*command, "-o", str(temporary)], check=True)
        temporary.replace(output)


def curl(output, url):
    download(output, ["curl", "-fLsS", "--retry", "3", "--max-time", "120",
                      "-A", "PACE-Research-Pilot/1.0", url])


def building_url(bounds):
    west, south, east, north = bounds
    ring = [(west, south), (east, south), (east, north), (west, north), (west, south)]
    polygon = ",".join(f"{x} {y}" for x, y in ring)
    query = urlencode({
        "$where": f"intersects(the_geom, 'POLYGON (({polygon}))')",
        "$order": "doitt_id",
        "$limit": 10000,
    })
    return f"https://data.cityofnewyork.us/resource/5zhs-2jue.geojson?{query}"


def write_manifest(folder):
    manifest = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for path in sorted([*folder.glob("*.geojson"), *folder.glob("*_osm.json")]):
        with path.open("rb") as handle:
            manifest["files"][path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    areas = json.loads((HERE / "areas.json").read_text())
    curl(args.data / "nycwalks.geojson", NYCWALKS)
    for area, config in areas.items():
        bounds = config["bounds"]
        bbox = ",".join(map(str, bounds))
        curl(args.data / f"{area}_buildings.geojson", building_url(bounds))
        curl(args.data / f"{area}_osm.json",
             f"https://api.openstreetmap.org/api/0.6/map.json?bbox={bbox}")
        download(args.data / f"{area}_places.geojson", [
            "overturemaps", "download", "--release", RELEASE, "--bbox", bbox,
            "--type", "place", "-f", "geojson",
        ])
    write_manifest(args.data)


if __name__ == "__main__":
    main()
