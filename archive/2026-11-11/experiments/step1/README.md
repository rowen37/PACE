# PACE 第一步实验 · 纽约步行图与商铺关联

本次试验选取 Hudson Yards 和 SoHo 周边各一个矩形样区。样区坐标见 [areas.json](areas.json)。以 NYCWalks 为底图，用 Overture Places 的购物类 POI 关联 NYC 建筑轮廓，并在每个样区抽查一个公开入口。

## 2026 年 11 月 8 日试跑

| 指标 | Hudson Yards | SoHo |
| --- | ---: | ---: |
| NYCWalks 裁剪路段 | 501 | 667 |
| 步行路段总长度 | 27.22 km | 31.33 km |
| 关联到 OSM 命名街道的路段 | 354 / 501（70.7%） | 639 / 667（95.8%） |
| 图的最大连通分量 | 339 / 345 节点（98.3%） | 420 / 424 节点（99.1%） |
| 购物类 POI（置信度 ≥ 0.80） | 163 | 1,173 |
| 唯一匹配到建筑轮廓 | 130 / 163（79.8%） | 902 / 1,173（76.9%） |
| 未落入任何建筑轮廓 | 33 | 271 |
| 未落入轮廓且距建筑 ≤ 5 m | 11 | 183 |
| 单个建筑轮廓内最多的购物 POI | 49（30.1%） | 9（0.8%） |
| 抽查入口接入最大连通分量 | 1 / 1 | 1 / 1 |

- **Hudson Yards** 的购物中心主入口沿 OSM 人行道和过街线走到接入点，末端对齐 NYCWalks 的偏移为 1.14 m，整条接入线长 13.69 m。点落面将 Aritzia、Dior 和 Lululemon 都关联到同一建筑轮廓（DOITT ID 1284821）。该轮廓也包括其他用途，49 条记录中包含购物中心自身的聚合 POI。 [官方到访指引](https://www.hudsonyardsnewyork.com/stories/how-get-hudson-yards) · [官方商铺目录](https://www.hudsonyardsnewyork.com/shop) · [OSM 门点](https://www.openstreetmap.org/node/11137273983)
- **SoHo** 的 Camper（110 Prince Street）主入口接到 Prince Street 人行道，接入线长 1.49 m。Camper 另一家分店（221 Bowery）的 POI 离最近建筑轮廓 3.18 m，按本次点落面规则保留为未匹配。 [Camper 门店目录](https://www.camper.com/en_US/shops/usa/new-york) · [OSM 门点](https://www.openstreetmap.org/node/2547223508)

单体建筑内的购物 POI 集中度在这两个样区差异明显，可以作为后续验证的空间特征。两个窗口的边界、楼层信息缺失、重复或聚合 POI 都会影响该数值。建筑关联是平面几何关系。入口可达性只验证到建筑门点，建筑内通行和逐户临街入口仍需核验。

## 方法

1. 用 NYCWalks 的 EPSG:6538 米制坐标裁剪步行线，以共享线端建无向加权图。保留来源拓扑，几何上相交的线不会自动新增路口。每条边用线长作为权重。NYCWalks 的基础路网时间早于当前 POI 和 OSM 数据。 [MIT City Form Lab](https://cityform.mit.edu/projects/nycwalks)
2. 从 OSM 提取命名街道。步行线上的三个采样点都须在同一街道线 18 m 以内，中点附近的方向余弦绝对值须至少为 0.9。这是街道名称和道路分类的几何匹配，未达到条件的路段保留在步行图中。 [© OpenStreetMap 贡献者](https://www.openstreetmap.org/copyright)
3. 保留 Overture 2026-10-21.0 版中分类树以 `shopping` 为根、置信度至少 0.80 的 POI。只有点落在唯一建筑轮廓上或内部时才分配建筑 ID。5 m 指标用于诊断坐标偏差。 [Overture Places](https://docs.overturemaps.org/guides/places/) · [NYC 建筑轮廓](https://data.cityofnewyork.us/City-Government/BUILDING/5zhs-2jue)
4. 入口采用官方公开信息和 OSM 入口坐标交叉核对。记录入口对应的市政建筑轮廓。在指定街道的 NYCWalks 边上插入接入节点，要求对齐距离不超过 15 m 并筛查接入线穿过建筑的情况。Hudson Yards 使用已标绘的步行接近线，Camper 使用门到人行道的短接线。输出里的 `building_entrance_nodes` 表示 POI 所在建筑有已接入的入口，连接到门点之后的楼层和室内路线仍需单独建模。

## 复现

在仓库根目录运行。当前工作区的 `experiments/step1/data/` 已保存本次源数据和 SHA-256 清单。新环境可用 `fetch.py` 下载，NYC 建筑轮廓和 OSM 使用下载当时的快照，Overture 固定到指定发布版。

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/step1/requirements.txt
# 如果本地没有数据，运行 python experiments/step1/fetch.py
python experiments/step1/run.py
```

[metrics.json](outputs/metrics.json) 是机器可读结果。每个样区目录都有 `graph.geojson`、`buildings.geojson`、`shops.geojson` 和 `entrances.geojson`。图可以按 `u`、`v`、`edge_id` 和 `length_m` 重建。原始数据和生成的 GeoJSON 留在本地工作区。
