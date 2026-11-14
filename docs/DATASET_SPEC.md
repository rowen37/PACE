# PACE–NYC v0：区域表征数据集

**决定：第一版用现有纽约地图快照构建 RGSCL 基线，并用公共步行网络和可核验的入口、目的地接入关系补充行人视角。** 这是 PACE 自建的开发数据集。地图底座已经在本地；通路补标和独立体验标签尚未完成。室内走廊和沿街动线只作为核对连接方式的两个实例。

## 范围与版本

- **地理范围**：纽约曼哈顿 Hudson Yards 和 SoHo 的两个现有矩形窗口，不是两片街区的全行政范围。精确边界沿用 [areas.json](../archive/2026-11-11/experiments/step1/areas.json)。两个窗口均已用于开发，保留为开发与构图验证区。正式测试区需在训练和选参前另行冻结。
- **原始快照**：`snapshot_id=nyc-2026-11-09`，路径 [`data/raw/nyc/2026-11-09/`](../data/raw/nyc/2026-11-09)。[manifest.json](../data/raw/nyc/2026-11-09/manifest.json) 的采集完成时间是 **2026-11-09 01:30:59 UTC（纽约当地 11 月 8 日 20:30:59 EST）**，本次盘点校验了全部 7 个文件。采集日期不等于各数据层的观测日期。
- **样本单位**：一条表征样本对应一个街区，主键为 `(snapshot_id, area_id, block_id)`。POI、建筑、门点和步行段保留各自的来源 ID。

| 样区 | 边界 `[西, 南, 东, 北]`（经纬度） | 窗口面积 | 合格街区面积与数量 |
| --- | --- | ---: | ---: |
| Hudson Yards | `[-74.0050, 40.7480, -73.9952, 40.7568]` | 0.8088 km² | 0.4339 km² / 24 个 |
| SoHo | `[-74.0045, 40.7195, -73.9930, 40.7263]` | 0.7337 km² | 0.5151 km² / 53 个 |

沿用归档构建规则：命名公共道路划出原子街区，剔除接触窗口边缘、面积小于 500 m² 或缺少步行主连通部分的块。第一版保留全部 77 个合格块，不因某块 POI 少或缺少室内信息而删除。原始道路围出的块只是样本单位，最终片区边界另由表征实验决定。

| 数据源 | 版本或文件 | 用途 |
| --- | --- | --- |
| [Overture Places](https://docs.overturemaps.org/guides/places/) | `2026-10-21.0`；两区 `*_places.geojson` | 商业 POI 的类别、坐标、名称和地址。首轮固定沿用 `shopping`、`food_and_drink`、`lifestyle_services` 和置信度至少 0.8 的历史样本口径；名称和地址用于核对实体。 |
| [NYC Building Footprints](https://data.cityofnewyork.us/City-Government/BUILDING/5zhs-2jue/about_data) | 2026-11-09 UTC 采集快照中的两区 `*_buildings.geojson`；原记录保留 `last_edited_date` | 建筑轮廓、形态、POI 归属候选。点落唯一轮廓记为几何匹配；多重或未匹配均保留状态。 |
| [MIT NYCWalks](https://cityform.mit.edu/projects/nycwalks) | `nycwalks.geojson`；基础网络经 2019 年航片验证 | 室外步行拓扑和路段长度。先沿用该底座，试点连接需核对是否仍可通行；首轮不使用其中的估算人流。 |
| [OpenStreetMap](https://wiki.openstreetmap.org/wiki/Simple_Indoor_Tagging) | 同一快照中的两区 `*_osm.json` | 原子街区划分所需道路、公共入口及通路候选；保留 `entrance`、`access`、`level` 等来源信息。门点进入可走图前需核实用途与接入。 |

0.8 是旧基线的 POI 质量筛选值；上面的街区样本量和后面的覆盖统计都对应旧口径。

**下一轮不设人工阈值。** 步行网沿用来源已明确记录的拓扑；入口只凭同一来源的共享节点、已绘制通路或可追溯的独立证据接入。缺失接入证据时记 `unknown`，不按门点离楼或人行道的距离补线。聚合使用连续相似度，不设相似度截断值，也不使用 BID 调整划分。旧基线里的 3 米门点检查、15 米吸附、500 m² 街区筛选、5 米共享边界筛选和 0.8 POI 置信度筛选属于历史协议，不能把复用旧图的结果称为全流程无人工阈值。

新输入可用 `.venv/bin/python experiments/prepare_block_inputs.py --output data/processed/nyc-2026-11-09-block-inputs` 从原始快照重建。保留未被研究窗口截断的道路围合面，不按面积删块或按道路条带宽度重画；街区间只按源步行网的连接生成邻接，不按共同边界的米数筛选。商业 POI 沿用原有类别范围和分类层级，`confidence` 只保留为来源字段；建筑不采用 RGSCL 原文的 20 m² 和每区至少 10 个对象的筛选。新输出与历史预测的街区 ID 和口径不通用。

## 实际数据量和覆盖率

以下由冻结原文件及归档街区规则复核。数据层的要素数与独立街区样本数含义不同。

| 指标 | Hudson Yards | SoHo | 合计 |
| --- | ---: | ---: | ---: |
| 原始 Overture POI（全类别） | 2,416 | 8,272 | 10,688 |
| 达到当前类别及置信度口径的商业 POI | 444 | 2,057 | 2,501 |
| 落入合格街区的商业 POI | 260 | 1,547 | 1,807 |
| 其中 `shopping` / `food_and_drink` / `lifestyle_services` | 107 / 134 / 19 | 935 / 459 / 153 | 1,042 / 593 / 172 |
| 其中唯一建筑匹配 | 191（73.5%） | 1,161（75.0%） | 1,352（74.8%） |
| 其中未匹配 / 多重匹配 | 69 / 0 | 386 / 0 | 455 / 0 |
| 原始建筑轮廓 | 400 | 1,285 | 1,685 |
| 与合格街区有面积交集的唯一建筑 | 168 | 853 | 1,021 |
| NYCWalks 窗口裁剪后的线段 | 501（27.224 km） | 667（31.329 km） | 1,168（58.553 km） |
| OSM 原始 node / way / relation | 15,882 / 3,025 / 352 | 13,098 / 2,856 / 294 | 28,980 / 5,881 / 646 |
| 划出街区 / 剔除窗口边缘 / 合格街区 | 45 / 21 / 24 | 74 / 21 / 53 | 119 / 42 / 77 |
| OSM 带 `entrance` 标签 / 已核验且接入室外网 | 68 / 1 | 41 / 1 | 109 / 2 |
| 已核验室内通路 / 独立街区体验标签 | 0 / 0 | 0 / 0 | 0 / 0 |

NYCWalks 原文件有 315,577 条全市线要素；研究只用表里的两个窗口裁剪结果。裁剪线段数在门点插入和街区切分之前统计。1,807 条 POI 不是 1,807 个独立训练区域；有效地理规模仍为两个开发窗口。

## 精度与已知缺口

- **POI**：原始文件 10,688 条记录都带源 ID、点坐标和 `confidence`；`taxonomy` 非空为 9,287 条（86.9%）。Overture 的 `confidence` 衡量地点存在的信号，不是定位误差、营业状态或建筑归属准确率。当前没有地面核验样本可估计定位误差。唯一建筑匹配的 74.8% 是几何匹配覆盖率，未匹配的 455 条仍保留。
- **建筑**：[NYC 官方元数据](https://github.com/CityOfNewYork/nyc-geo-metadata/blob/main/Metadata/Metadata_BuildingFootprints.md#2-data-quality-and-specifications)给摄影测量更新的轮廓约 ±2 英尺（约 ±0.61 m）的估计位置精度。两区原始轮廓中 1,376 条为摄影测量，308 条为人工绘制，1 条来源空缺。该精度不能推广到所有轮廓或推导出的 POI 归属，也没有室内隔断信息。
- **步行网与入口**：几何计算使用米制 EPSG:6538。NYCWalks 经 2019 年航片验证，不能视为 2026 年逐路段实测；试点动线须核对现况。OSM 要素 `timestamp` 是编辑时间。109 个入口标签并非 109 个公共商店门；两区各只有一个门已经通过旧流程的核验和接网。
- **室内**：两个窗口均无 `indoor=corridor` 或 `highway=corridor` 要素。资料只支持楼层时只记楼层；室内路径和 `unknown` 分开记录，不能把空白当成不通行。

## 字段字典

**原文件字段。** GeoJSON 的 `geometry` 与顶层 `id`（如果存在）独立于 `properties`。

| 原始表 | 实际包含的字段 | 首轮取用 |
| --- | --- | --- |
| Overture Places | 顶层 `id`, `geometry`；`basic_category`, `taxonomy`（`primary`, `hierarchy`, `alternates`）, `confidence`, `names`, `addresses`, `operating_status`, `brand`, `sources`, `version`, `websites`, `emails`, `phones`, `socials` | ID、点、分类和来源；名称/地址核对实体，`confidence` 用于筛选。联系方式不进模型。 |
| NYC Building Footprints | `geometry`, `doitt_id`, `objectid`, `bin`, `base_bbl`, `mappluto_bbl`, `name`, `feature_code`, `geom_source`, `shape_area`, `shape_length`, `height_roof`, `ground_elevation`, `construction_year`, `last_edited_date`, `last_status_type` | `doitt_id` 作来源 ID，使用轮廓计算米制形态；保留几何来源与编辑日期作质量记录。 |
| NYCWalks | 顶层 `id`, `geometry`；`id`, `FID`, `X__GUID`, `__GUID`, `Shape_Leng`, `Shape_Le_1`；`CountLoc`, `CntLocWKND`；`Wkdy_AM_CT`, `Wkdy_MD_CT`, `Wkdy_PM_CT`, `Wknd_AM_CT`, `Wknd_MD_CT`, `Wknd_PM_CT`；`predwkdyAM`, `predwkdyMD`, `predwkdyPM`, `predwkndAM`, `predwkndMD`, `predwkndPM`；`AMN_AMN`, `AMN_MTA`, `HME_AMN`, `HME_JOB`, `HME_MTA`, `HME_PRK`, `HME_SCH`, `JOB_AMN`, `JOB_MTA`, `PRK_MTA` | 源线段 ID、几何、由几何计算的米制长度及端点拓扑。人流字段保留在原文件。 |
| OSM API | `type`, `id`, `version`, `timestamp`, `changeset`, `uid`, `user`, `tags`（可缺）；node 用 `lon`, `lat`，way 用 `nodes`，relation 用 `members` | 要素 ID、空间和关系结构、公共通行及室内相关标签。编辑者字段不进入处理后的表。 |

**拟生成的建模与标注表。** 下列是字段契约，不代表文件已生成。街区相关表统一带 `snapshot_id`, `area_id`, `block_id`；节点跨块共享时保留同一来源 ID。

| 表（每行单位） | 列 |
| --- | --- |
| `blocks`（一个街区） | `geometry`, `area_m2`, `eligible`, `exclusion_reason`, `poi_count`, `unique_building_match_count`, `walk_length_m` |
| `pois`（一个地点） | `poi_id`, `geometry`, `name`, `address`, `category_l1`, `category_l2`, `category_path`, `confidence`, `source_version` |
| `buildings`（一个建筑在一个街区的关联） | `building_id`, `geometry`, `area_m2`, `perimeter_m`, `radius_m`, `orientation_rad`, `compactness`, `fractality`, `concavity`, `elongation`, `geom_source`, `source_edited_at` |
| `poi_building_matches`（一个 POI 的归属状态） | `poi_id`, `building_id`（无法唯一匹配时为空）, `candidate_count`, `match_status`（`unique/none/ambiguous`）, `match_method`, `evidence_id`（有人工核验时） |
| `walk_nodes` / `walk_edges`（一个节点 / 一条步行或接入边） | 节点：`node_id`, `node_type`, `geometry`, `level`, `source_feature_id`；边：`edge_id`, `from_node_id`, `to_node_id`, `edge_type`, `source_segment_id`, `geometry`, `length_m`, `access_status`, `evidence_id`。没有测量时，长度和楼层允许空。 |
| `access_evidence`（一条门点、归属或通路证据） | `evidence_id`, `from_id`, `to_id`, `relation_type`, `level_from`, `level_to`, `access_status`（`public/restricted/unknown`）, `verification_status`, `source_ref`, `observed_at`, `verified_at` |
| `block_labels`（一名标注者对一个街区的独立记录） | `annotator_id`, `indoor_continuity`, `street_continuity`, `evidence_ref`, `observed_at`, `unknown_reason`；两维分别记 `present/absent/unknown`，可以同时为 `present`。 |
| `block_similarity_triplets`（一名标注者的一次独立比较） | `triplet_id`, `snapshot_id`, `anchor_block_key`, `candidate_b_key`, `candidate_c_key`, `scenario_id`, `annotator_id`, `choice`（`b/c/tie/unknown`）, `reason`, `evidence_ref`, `observed_at`；街区 key 含 `area_id` 和 `block_id`。 |

## 第一批补标数据

| 试点 | 固定范围 | 来源与入库标准 |
| --- | --- | --- |
| 室内 | The Shops at Hudson Yards，Level 2 [Uniqlo](https://www.hudsonyardsnewyork.com/shop/uniqlo) 公共店门所在走廊，两侧截至首个可核实的公共岔口 | [商场目录](https://www.hudsonyardsnewyork.com/shop?tid=61)核对商户和楼层。公共走廊与门的实际接续由有日期的楼层图、导览牌或现场观察核实。只确认到楼层时，记楼层，不生成通行边。 |
| 沿街 | SoHo Prince Street，**Wooster Street 至 Greene Street，Camper 门店所在侧**；以 [Camper，110 Prince St](https://www.camper.com/en_IS/shops/usa/new_york/store_shoes_110_prince_st-T812) 为起点 | 现有步行底图和已核验门点为起点。用近期街景、门店资料或现场核对额外店门及其人行道接入。两个试点优先核对服装、鞋履等可比较类别。 |

两段动线内的门和公共通路按证据逐条登记，完成后报告实标量与未知量。补标保存实际观测日期和核验日期；若无法证明资料适用于快照附近时段，保持未知，若发现实体或路径确实变化则另开版本，避免将新状态回填到旧快照。店铺归属与步行通路分别记录。

## 评价数据与发布条件

- **构图核验**：检查门点和公共步行边可追溯，逐条核对两处示范动线。缺失率单独报告，避免把地图更完整误读成更适合室内探索。
- **表征评价**：两名标注者依据独立观察，在相同的步行探索情境下比较街区，记录从哪里进入、公共路径如何连接、能否串联多个目的地，允许平局和未知。室内、沿街观察是辅助描述。建图用过的同一段通路不直接当成表征测试标签；正式比较用未见地区的街区三元组，流程见 [SOP](TASK_AND_EXPERIMENT_SOP.md#行人视角的第一组对照)。
- **训练输入边界**：首轮数据集只含地图、建筑、POI 与核验关系。Foursquare、Massive-STEPS、用户打卡、估算人流和用户偏好都不进入这一版。推荐评估日后单独采集与地图同期的用户选择。
- **产物位置**：原始快照保留在 `data/raw/`；节点、边、匹配与覆盖统计写 `data/processed/`；人工证据和独立体验标签写 `data/annotations/`；冻结的地理划分写 `data/splits/`。

地图底座现在即可用于原版 RGSCL 表征和构图诊断。只有核验通路入库后，才比较加入室内关系的变体；独立体验标签完成后，才能评价表征是否学到「怎么逛」。
