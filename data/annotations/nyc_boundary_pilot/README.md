# 纽约现成围栏：PACE 小样本分区对照

这一小包用来对照 PACE 能否把现有商业片区内的街区连在一起。围栏只进入评估，不用于训练表征、构图、选择超参数或决定何时停止合并。

- [商业区边界](commercial_bids.geojson)：从纽约市小企业服务局 [Business Improvement Districts（7jdm-inj8）](https://data.cityofnewyork.us/Business/Business-Improvement-Districts/7jdm-inj8) 中取两个正式 BID 多边形，保留原几何和原始属性。该图层标注的数据更新日期为 2024-10-08，与本地地图快照分别记录日期。
- [商场轮廓](hudson_yards_mall_osm.geojson)：从 PACE 冻结快照提取 [© OpenStreetMap 贡献者 way 398440363](https://www.openstreetmap.org/way/398440363) 的 `shop=mall` 轮廓。用于检查单体建筑与街区尺度的差别。[OpenStreetMap 数据许可](https://www.openstreetmap.org/copyright)。
- [来源与校验值](manifest.json)：原文件及摘取后文件的 SHA-256。

| 围栏 | 有面积交集的可用街区 | 围栏内街区质心 | 两端质心均在内的步行邻接 / 跨围栏邻接 | 任意整街区并集的 IoU 上限 |
| --- | ---: | ---: | ---: | ---: |
| SoHo Broadway BID | 10 / 53 | 10 | 13 / 9 | 0.610 |
| Hudson Yards / Hell’s Kitchen BID | 18 / 24 | 16 | 21 / 11 | 0.616 |
| The Shops at Hudson Yards | 1 / 24 | 0 | 不适用 | 0.156 |

计算使用归档 step2 的原子街区和步行邻接，EPSG:6538 投影。BID 的质心入围街区分别形成一个连通分量。IoU 参考面取正式边界与可用街区并集的交集；SoHo 可见参考面积约 72,000 m²（全围栏的 95%），Hudson Yards 约 150,300 m²（44%），因此后者只做窗口内比较。上限用任意整街区并集计算，不强制连通，用来说明街区粒度限制；它不是模型得分。

第一轮可直接评估已有 embedding：对同一个围栏，把两端都在内的步行邻接作为正例、一内一外作为负例，以余弦相似度排序，按围栏分别报 AP；两端都在围栏外的边不判为负例。真正生成片区后，再比较固定合并流程各候选与参考面的 IoU，并与上述粒度上限一起报。两项均不需要设定相似度阈值。

BID 是商区管理边界；不代表每条街都步行专用，也不代表居民或游客对片区的唯一认知。两份 BID 足够跑通和诊断这项对照，不能用来推断城市范围的准确率。[NYC DOT 步行广场](https://data.cityofnewyork.us/Transportation/NYC-DOT-Pedestrian-Plazas-Polygon/k5k6-6jex) 提供单独的步行空间多边形层，可以在扩展窗口时另立评估；本包未纳入。
