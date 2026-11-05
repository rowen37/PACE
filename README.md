# PACE: Place-Aware Citywalk Exploration

## Project Information

**Student:** Qiwen Bian

## Background and Research Question

A Citywalk often begins with choosing an area to explore. A shopping district can contain hundreds of similar listings. How it feels on foot depends on how those places are organized. Shops can share an indoor concourse, line several connected streets, or form a mix of both. PACE uses neighborhood embeddings to describe these patterns so users can choose an area that suits how they want to explore. Once there, they choose their own route and stops.

**Research question:** How can a neighborhood embedding capture different ways of exploring a city on foot and help users choose an area for their Citywalk?

## Data Sources

New York City will serve as the development site. Selected areas in Shanghai will be used to test transfer with fields available in both cities.

| Data type | United States — New York City | China — Shanghai |
| --- | --- | --- |
| Walking network | [NYCWalks](https://cityform.mit.edu/projects/nycwalks) provides citywide sidewalks, crossings, and footpaths. OpenStreetMap (OSM) adds street classifications. | The [OSM China extract](https://download.geofabrik.de/asia/china.html) provides streets and mapped footpaths. Crossings will be checked in the study areas. |
| Commercial points of interest (POIs) | [Overture Places](https://docs.overturemaps.org/guides/places/) provides locations and category hierarchies. [NYC storefront records](https://data.cityofnewyork.us/d/92iy-9c3n) support spot checks. | Overture Places and OSM provide locations and categories. Sample checks will cover mall tenants and street-facing shops. |
| Buildings and entrances | [NYC building footprints](https://data.cityofnewyork.us/d/5zhs-2jue), mapped entrances, and public mall directories. | [Overture Buildings](https://docs.overturemaps.org/guides/buildings/), mapped entrances, and public mall directories. |
| Exploration profile labels | Manually label sampled areas for indoor shopping and street browsing. An area can have both labels. | Apply the same labeling criteria independently to the Shanghai sample. |

## Proposed AI Methods

| Engineering implementation | Method category |
| --- | --- |
| **Build the walking graph and link shops to buildings and streets.** Use spatial joins to associate POIs with building footprints. Connect verified entrances to the graph built from sidewalks and crossings. | Spatial joins and network construction · Vector data operations |
| **Generate candidate areas at multiple scales.** Use graph-constrained agglomerative clustering with average linkage to merge walkably adjacent blocks based on edge betweenness, node degree, dead-end ratio, and street orientation. Include mall complexes as candidates linked to public entrances. | Spatially constrained clustering · Unsupervised learning |
| **Encode each area.** Use embedding lookup for broad and fine POI categories. Apply two [GraphSAGE](https://proceedings.neurips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html) layers to average neighboring features by relationship type across shops, buildings, and streets. Mean-pool building and street representations separately, then append the share of shops in the largest complex and the length of streets with storefronts. | Graph neural networks · Deep neural networks |
| **Learn and test exploration profiles.** Train the encoder and a linear prediction head with binary cross-entropy for each profile. Hold out entire districts. Compare POI summaries, POI summaries plus spatial features, and those features plus the learned embedding. Use areas with similar POI mixes to check whether the embedding captures differences in how people can explore them. | Supervised learning · Ablation study |
| **Apply the walking budget at request time.** The user selects a start, a time budget, and an exploration profile. Use [single-source Dijkstra](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_dijkstra.html) to calculate walking times to candidate entrances. Keep areas that leave time to explore and rank them by the selected profile. | Graph search · Constraint-based filtering |

## Expected Outcomes

An interactive map will show several feasible areas, their boundaries, suggested entrances, walking times, and features that describe how each area can be explored. A short evaluation will examine which features distinguish exploration profiles, how well they transfer to Shanghai, and whether users find the descriptions helpful when choosing where to walk.
