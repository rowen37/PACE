# 地块几何上限与 Infomap 对照

沿用 [上一轮](FUSION_PARTITION_BENCHMARK.md)的 SoHo 67 块和 Hudson Yards 34 块。两个实验分开：地块实验只核对覆盖率和几何上限；Infomap 实验在原有街区和边权上只换算法。BID 是管理边界，用来做事后几何比较。

## MapPLUTO 26v2：地块有多全？

从 [NYC Planning 的官方图层](https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/ArcGIS/rest/services/MAPPLUTO/FeatureServer/0)按两个研究区的外包矩形下载，图层数据最后更新元数据为 2026-09-09 UTC。返回条数与分别调用官方 count API 一致；BBL 无重复，多边形有效。只按原有研究范围裁切，不按 BID 裁切地块。

| 区域 | 范围内地块（外包矩形返回条数） | 街区研究范围有地块覆盖 | 可评估 BID 面积有地块覆盖 | Overture POI 落入地块 |
|---|---:|---:|---:|---:|
| SoHo | 859（1,226） | 66.85% | 约 100% | 2,088 / 2,479（84.2%） |
| Hudson Yards | 249（375） | 72.55% | 98.75% | 456 / 512（89.1%） |

有任何地块覆盖的街区分别是 53 / 67 和 33 / 34。地块图层表达的是登记地块；没有地块覆盖的地方不等于下载遗漏，面积统计也不能直接判定每个缺口的用途。

| 区域 | 整街区 IoU 上限 | 地块 IoU 上限 | 地块最佳组合 |
|---|---:|---:|---:|
| SoHo | 0.6165 | 0.999998 | 91 宗 |
| Hudson Yards | 0.6169 | 0.9578 | 87 宗 |

两种尺度都用同一个参考面：原研究街区范围内的 BID 面积（SoHo 7.20 公顷，Hudson Yards 15.03 公顷）。给定互不重叠的原子单元，按 BID 面积覆盖比例排序并枚举前缀可求最优 IoU。Hudson Yards 源地块重叠仅 0.000017 平方米；已对选中地块求真实并集后复算。该结果不要求步行连通、不训练模型；这接近只能说明这份 SoHo BID 面在几何上几乎可拼成地块。不能把上限当成模型预测准确率或实际步行边界。

## Infomap：在同一张图上换分区目标

用官方 Infomap 2.16.0 的标准两层无向模式，单步随机游走，不调分辨率、不设相似度阈值和片区数。用等权拼接实验冻结的两区三个种子（13、29、53）、步行邻接以及边权 `(1 + cosine) / 2`。每个街区保留一次，所有输出组都验证步行连通。先写出分组，再打开 BID，在每套分组中按同样的 IoU 口径匹配参考组；地区内对三个种子取平均。

| 区域 | 方法 | IoU | 多圈（公顷） | 其中完全在 BID 外的整块（公顷） | 漏圈（公顷） |
|---|---|---:|---:|---:|---:|
| SoHo | 模块度 | 0.482 | 6.41 | 2.54 | 0.71 |
| SoHo | Infomap | 0.500 | 2.91 | 0.25 | 2.14 |
| Hudson Yards | 模块度 | 0.328 | 9.72 | 4.47 | 7.08 |
| Hudson Yards | Infomap | 0.350 | 9.01 | 3.60 | 6.85 |

SoHo 的 Infomap 匹配组在三个种子完全相同；种子 13 的模块度结果（0.604）仍更好。SoHo 的 `b014` 和 `b009` 仍被三次划入匹配组。Hudson Yards 的 Infomap 在种子 53 落后于模块度（0.285 对 0.376）。平均指标不足以证明它普遍更好。

**随机游走实例。** SoHo 种子 13，从 `b023` 出发，各邻边权重除以该节点的边权总和：

| 下一块 | 边权 | 概率 |
|---|---:|---:|
| `b014` | 0.700 | 22.3% |
| `b020` | 0.885 | 28.2% |
| `b028` | 0.711 | 22.7% |
| `b037` | 0.843 | 26.9% |

一旦走到只有一条邻边的 `b014`，下一步只能回 `b023`。Infomap 用随机游走的长期流量计算地图编码长度，无须实际抽样模拟行人；把这条往返边留在同组，走这条边就不用记一次跨组。模型没有真实人流、目的地或停留时间。对 `b014` 的观察与这种目标一致，不等于证明所有度为一的节点都应该留在组里。[Infomap 原论文](https://arxiv.org/abs/0707.0609)

```sh
.venv/bin/python experiments/infomap_partition_benchmark.py prepare --source outputs/fusion_partition_topology_v2 --output outputs/infomap_partition_repeat
.venv/bin/python experiments/infomap_partition_benchmark.py score --source outputs/fusion_partition_topology_v2 --output outputs/infomap_partition_repeat
.venv/bin/python experiments/mappluto_geometry_ceiling.py --source outputs/fusion_partition_topology_v2 --parcels data/raw/nyc/mappluto_live_snapshot --output outputs/mappluto_geometry_repeat
```

[Infomap 分种子结果](../outputs/infomap_partition_pilot/metrics.json) · [MapPLUTO 几何结果](../outputs/mappluto_geometry_pilot/geometry.json) · [原始数据来源、查询和哈希](../data/raw/nyc/mappluto_live_snapshot/manifest.json)
