"""Render the pilot candidate boundaries as a static map."""

import argparse
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as PlotPath
from shapely.geometry import box
from shapely.geometry.polygon import orient
from shapely.ops import transform

from cluster import HERE, SCALES, STEP1, TO_METRES, projected, read

AREAS = {"hudson_yards": "Hudson Yards", "soho": "SoHo"}
COLORS = ["#afd4c7", "#f6d898", "#bccced", "#f2bbab", "#d3b7e0", "#c9dca5",
          "#b9d9e7", "#e6c9a5", "#f3cce1", "#cbd4ce", "#d5ceec", "#b9d6b6"]
MALL = "#7345a0"
WALK = "#2e4664"


def draw_polygon(ax, geometry, **style):
    polygons = [geometry] if geometry.geom_type == "Polygon" else geometry.geoms
    for polygon in polygons:
        p = orient(polygon)
        rings = [PlotPath(ring.coords, closed=True) for ring in [p.exterior, *p.interiors]]
        ax.add_patch(PathPatch(PlotPath.make_compound_path(*rings), **style))


def plot(outputs, path):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 9))
    fig.patch.set_facecolor("#f9fafb")
    fig.subplots_adjust(left=0.05, right=0.975, top=0.86, bottom=0.10, wspace=0.12, hspace=0.29)
    fig.text(0.05, 0.963, "PACE  /  WALKABLE CANDIDATE AREAS", fontsize=21, fontweight="bold",
             color="#172d43")
    fig.text(0.05, 0.925, "Three nested scales in the same two Manhattan pilot windows",
             fontsize=11.5, color="#42566b")
    configs = read(STEP1 / "areas.json")
    for row, (area, label) in enumerate(AREAS.items()):
        folder = outputs / area
        blocks = [projected(f) for f in read(folder / "blocks.geojson")["features"]]
        candidates = [(projected(f), f["properties"])
                      for f in read(folder / "candidates.geojson")["features"]]
        entries = [(projected(f), f["properties"])
                   for f in read(folder / "gateways.geojson")["features"]]
        x0, y0, x1, y1 = transform(TO_METRES, box(*configs[area]["bounds"])).bounds
        for col, cap in enumerate(SCALES):
            ax = axes[row, col]
            ax.set_xlim(x0 - 12, x1 + 12)
            ax.set_ylim(y0 - 12, y1 + 12)
            ax.set_aspect("equal")
            ax.axis("off")
            for g in blocks:
                draw_polygon(ax, g, facecolor="#e9edf1", edgecolor="#fff", linewidth=0.45)
            level = [(g, p) for g, p in candidates if p.get("scale_max_blocks") == cap]
            ax.set_title(f"{label}  ·  Up to {cap} blocks", loc="left", fontsize=11.5,
                         fontweight="bold", color="#172d43", pad=17)
            avg = mean(p["block_count"] for _, p in level)
            ax.text(0, 1.01, f"{len(level)} street candidates   /   {avg:.1f} blocks on average",
                    transform=ax.transAxes, fontsize=8.5, color="#42566b")
            for i, (g, _) in enumerate(level):
                draw_polygon(ax, g, facecolor=COLORS[i % len(COLORS)], edgecolor="#40536b",
                             linewidth=0.62)
            for g, p in candidates:
                if p["kind"] == "mall":
                    draw_polygon(ax, g, facecolor="none", edgecolor=MALL, linewidth=1.7, zorder=4)
            points = {(round(g.x, 4), round(g.y, 4)) for g, p in entries
                      if p.get("scale_max_blocks") == cap}
            if points:
                x, y = zip(*sorted(points))
                ax.scatter(x, y, s=4, c=WALK, linewidths=0, zorder=3)
            for point, p in entries:
                if p["kind"] == "reviewed_entrance":
                    ax.scatter(point.x, point.y, marker="D", s=19, c=MALL, edgecolor="white",
                               linewidth=0.55, zorder=5)
            bar_x = x0 + 25
            bar_y = y0 + 25
            ax.plot([bar_x, bar_x + 100], [bar_y, bar_y], c="#40536b", linewidth=1.15)
            ax.text(bar_x + 50, bar_y + 12, "100 m", ha="center", fontsize=6.5, color="#40536b")
    fig.legend(handles=[
        Patch(facecolor=COLORS[0], edgecolor="#40536b", label="Street candidate"),
        Line2D([], [], linestyle="none", marker=".", color=WALK, label="Network gateway"),
        Patch(facecolor="none", edgecolor=MALL, label="Mall footprint"),
        Line2D([], [], linestyle="none", marker="D", markersize=4, color=MALL, label="Reviewed mall door"),
        Patch(facecolor="#e9edf1", label="Outside the clustering sample"),
    ], loc="lower left", bbox_to_anchor=(0.045, 0.049), frameon=False, ncols=5,
       fontsize=8.7, columnspacing=1.5)
    fig.text(0.05, 0.032, "Sources  /  MIT City Form Lab NYCWalks and © OpenStreetMap contributors",
             fontsize=8, color="#596d80")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_embeddings(outputs, path):
    threshold = 0.65
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8))
    fig.patch.set_facecolor("#f9fafb")
    fig.subplots_adjust(left=0.055, right=0.965, top=0.76, bottom=0.13, wspace=0.15)
    fig.text(0.055, 0.945, "PACE  /  AREAS LEARNED FROM BLOCK SIMILARITY", fontsize=17,
             fontweight="bold", color="#172d43")
    fig.text(0.055, 0.895, "Walk-connected groups with cosine similarity ≥ 0.65 for every block pair",
             fontsize=10, color="#42566b")
    configs = read(STEP1 / "areas.json")
    for ax, (area, label) in zip(axes, AREAS.items(), strict=True):
        folder = outputs / area
        blocks = read(folder / "block_embeddings.geojson")["features"]
        groups = [f for f in read(folder / "candidates.geojson")["features"]
                  if f["properties"]["embedding_threshold"] == threshold
                  and f["properties"]["block_count"] > 1]
        merged_count = sum(f["properties"]["block_count"] for f in groups)
        ax.set_title(f"{label}   ·   {len(groups)} groups, {merged_count} blocks merged", loc="left",
                     fontsize=10.5, color="#172d43", fontweight="bold", pad=8)
        ax.set_aspect("equal")
        ax.axis("off")
        x0, y0, x1, y1 = transform(TO_METRES, box(*configs[area]["bounds"])).bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        for f in blocks:
            fill = "#e2e8ef" if f["properties"]["enough_pois_for_merging"] else "#f1f3f5"
            draw_polygon(ax, projected(f), facecolor=fill, edgecolor="#aab6c1", linewidth=0.5)
        for i, f in enumerate(groups):
            draw_polygon(ax, projected(f), facecolor=COLORS[i], edgecolor=WALK, linewidth=1.1)
    fig.legend(handles=[
        Patch(facecolor=COLORS[0], edgecolor=WALK, label="Multi-block group"),
        Patch(facecolor="#e2e8ef", edgecolor="#aab6c1", label="Unmerged block"),
        Patch(facecolor="#f1f3f5", edgecolor="#aab6c1", label="Fewer than five commercial POIs"),
    ], loc="lower left", bbox_to_anchor=(0.05, 0.07), frameon=False, ncols=3, fontsize=8.5)
    fig.text(0.055, 0.05, "Sources  /  NYCWalks, © OpenStreetMap contributors, Overture Places, NYC Open Data",
             fontsize=7.5, color="#596d80")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--embedding", action="store_true")
    args = parser.parse_args()
    output = args.outputs or HERE / "outputs" / ("embedding" if args.embedding else "")
    draw = plot_embeddings if args.embedding else plot
    draw(output, args.map or output / "candidate_map.png")
