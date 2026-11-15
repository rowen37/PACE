# PACE — Place-Aware Citywalk Exploration

PACE learns pedestrian-perspective representations of city blocks and uses them to discover multi-scale, walk-connected commercial districts. Instead of recommending officially named areas, the system recommends **boundaries with descriptions** of how a place can be explored on foot. Every design choice is validated against independent human judgments of street-level imagery rather than administrative or BID fences.

## Background

A citywalk begins with choosing an area. What makes an area worth walking is not the list of places it contains but how they are arranged for a person on foot: small storefronts one after another along a sidewalk, or a single lobby entrance; destinations chained by continuous public walkways, or interrupted by blank walls and curb cuts. Two facts motivate PACE:

**1. Official district boundaries are administrative artifacts, not experience boundaries.** BIDs and named districts exist for taxation, sanitation, and marketing. We measured this directly: in the study areas below, the official BID boundary separates street-level character no better than a random grouping of blocks — inside the fence, street fronts are as heterogeneous as across it. A recommender built on official fences inherits their blindness.

**2. Many of the most walkable commercial areas have no official boundary at all.** Retail agglomerates organically: a street that accumulates boutiques over decades, or a corridor where malls and street shops accrete around a transit node. These areas are real to walkers and to local knowledge, but invisible to any fence layer. A system that can only recommend inside recorded polygons cannot see them — and cannot answer the question a user actually asks: *"where does the good stretch start and end, and how long do I need?"*

PACE starts from these facts. The boundary that matters is the one a pedestrian experiences; where it is not recorded anywhere, it must be discovered from what shapes the walking experience itself — the composition of destinations, the morphology of street fronts, and the connectivity of the walk network.

## Scientific question

**Can a block representation learned from map data alone capture how a block is explored on foot, and can multi-scale district boundaries derived from that representation align with where independent human observers perceive street character to change — without official fences and without hand-set thresholds?**

Three operational sub-questions, each carrying a registered scope limit:

1. **Validity.** Do learned representations predict human judgments of street-level structure (frontage type, door density, walk continuity) above majority baselines, per view and per fusion? — measured by leave-one-out prediction against two-annotator consensus labels.
2. **Discovery.** Do profile-aligned partitions place group boundaries at perceived street-character transitions, including in areas that have no official boundary? — measured by within-group vs cross-group edge agreement, audited on the annotator who did not select the cut.
3. **Scale.** Can the appropriate district scale be chosen by human-labeled structure rather than an objective-function constant, and still adapt to a user's time budget? — a complete merge tree keeps every scale; the default level is selected by profile alignment and pruned per branch by walking time.

## Pipeline

```
map inputs          POI (Overture) · building footprints · walk network (NYCWalks)
                          │
view encoders       independent self-supervised encoders (augmentation consistency)
                          │
fusion              POI + walk, reliability-weighted:
                    r_view(block) = observed × cross-seed neighbor stability
                          │
partition           edge weights on walking-adjacent blocks →
                    complete-link merge tree (all scales preserved)
                          │
selection           tree level chosen by human-profile alignment (no thresholds)
                          │
serving             budget-pruned connected candidates:
                    tour time = internal walk length / 4.5 km/h
```

The building view was removed by ablation (it carries no street-level information); the annotation instrument and evaluation are described below.

## Experimental design

**Human labels.** Annotators rate each block from street-view imagery on three visible constructs: frontage type (Gehl active frontage), door density, and street-wall continuity. Blocks are sampled from camera points placed every 35 m along a block's walk segments. Consensus labels are the strict majority of two calibrated annotators; agreement is reported as per-field Cohen's κ, and instrument portability is tested against uncalibrated annotators.

**Representation checks.** Frozen view embeddings are scored by leave-one-out 1-NN prediction of the consensus labels, per field and per view, against majority-class baselines. View specialization (walk → continuity, POI → density) is established on development areas; a held-out area (Flushing, window frozen before any model output) measures transfer under geographic and commercial-fabric shift.

**Partition evaluation.** Official BID boundaries serve only as post-hoc geometric references; profile-purity and within-group vs cross-group edge-agreement audits showed they carry no street-experience structure. Partition quality is instead measured by human profiles: a selected layer must place group boundaries where annotators perceive the street texture to change. Layer selection uses one annotator's labels and is audited with the other's (selection/audit separation).

**District candidates.** The complete-link merge tree keeps every scale. The default level is selected by profile alignment (smoothed Welch t on within- vs cross-group edge agreement); user time budgets then prune the tree per branch to the finest candidate whose internal walking tour fits. No similarity cutoffs, target group counts, or stopping tolerances exist anywhere in the pipeline.

## Data

All raw layers live in a single frozen NYC snapshot (`data/raw/nyc/2026-11-09/`, SHA-256 manifest included).

| Area | Role | Blocks | Commercial POI | Frozen |
| --- | --- | ---: | ---: | --- |
| SoHo | development | 67 | 2,479 | 2026-11-09 |
| Hudson Yards | development | 34 | 512 | 2026-11-09 |
| Flushing | held-out judge | 69 | 1,466 | 2026-11-14 (before any model output) |

Sources: Overture Places, NYC building footprints, MIT NYCWalks pedestrian network, OpenStreetMap, Google Street View (imagery per-block, pano IDs and capture dates recorded).

## Repository layout

```text
configs/        study-area windows (frozen bounds)
data/           raw snapshot, processed block inputs, annotations, splits
docs/           protocols, experiment reports, open-questions ledger
experiments/    runnable entry points for every pipeline stage
outputs/        frozen experiment artifacts with input hashes
src/pace/       model code
archive/        pre-reorganization experiments (kept as historical evidence)
```

## Setup

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python experiments/pedestrian_rgscl.py --epochs 1 --output outputs/env_smoke
```

Street-view fetching requires `PACE_GSV_API_KEY` (Google) or `PACE_MAPILLARY_TOKEN`.

## Documentation

- [Dataset specification](docs/DATASET_SPEC.md) — sources, coverage, field dictionaries
- [Annotation protocol](docs/STREETVIEW_ANNOTATION_PROTOCOL.md) — instrument design and quality gates
- [Partition benchmarks](docs/WHOLE_BLOCK_BENCHMARK.md) · [fusion comparison](docs/FUSION_PARTITION_BENCHMARK.md) · [Infomap and parcel ceiling](docs/MAPPLUTO_AND_INFOMAP_PILOT.md)
- [Profile-guided level selection](docs/PROFILE_LEVEL_SELECTION.md) · [view ablation](docs/VIEW_ABLATION.md) · [E4 connected candidates](docs/E4_CONNECTED_CANDIDATES.md)
- [Held-out area verdict](docs/FLUSHING_HELDOUT_VERDICT.md)
- [Open scientific questions](docs/OPEN_SCIENTIFIC_QUESTIONS.md) — registered gaps and their scope limits
- [Task and experiment SOP](docs/TASK_AND_EXPERIMENT_SOP.md) · [audit reports](docs/audit_structure.md)

## Principles

1. **No hand-set thresholds** — no similarity cutoffs, target group counts, or stopping tolerances; every constant declares its provenance.
2. **Freeze before evaluate** — inputs, predictions, and labels are hashed; results are never overwritten, only superseded in new output directories.
3. **Selection/audit separation** — the labels that choose a method are never the labels that grade it.
