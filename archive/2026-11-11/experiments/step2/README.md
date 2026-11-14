# PACE 第二步实验 · 多尺度候选区域

沿用第一步的 Hudson Yards 和 SoHo 样区与本地数据快照。测试受步行邻接约束的平均链接聚类能否生成多尺度、边界连续且可以从路网进入的候选区。

## 2026 年 11 月 8 日试跑

| 指标 | Hudson Yards | SoHo |
| --- | ---: | ---: |
| 纳入聚类的完整街区 | 24 | 53 |
| 共享至少 5 m 边界的街区对 | 39 | 88 |
| 其中有直接步行连接 | 38 | 86 |
| 最多 2 街区档 · 候选数（平均街区数） | 15（1.6） | 30（1.8） |
| 最多 5 街区档 · 候选数（平均街区数） | 7（3.4） | 13（4.1） |
| 最多 10 街区档 · 候选数（平均街区数） | 4（6.0） | 7（7.6） |
| 最大档 · 实际街区数范围 | 3–8 | 4–9 |
| 独立商场候选区 | 1 | 0 |

三个尺度合计 76 条街区候选记录。导出后按每个候选区重建区内步行图，所有候选区都得到一个连续且有效的多边形，区内纳入的步行路段互通，并至少有一个路网边界接入点。同档候选区覆盖各自样区内全部纳入街区，每个小尺度候选区都完整包含在某个更大尺度候选区内。各档是同一批街区的嵌套结果，有些区域会原样出现在相邻尺度。

Hudson Yards 的 The Shops 单独保留为商场候选区。边界使用 OSM 标绘的商场轮廓，接入点沿用第一步核对过的 Hudson Boulevard 主入口。轮廓覆盖 41 条购物 POI 记录，这个口径与第一步按市政建筑轮廓计数不同。 [OSM 商场轮廓](https://www.openstreetmap.org/way/398440363) · [官方到访指引](https://www.hudsonyardsnewyork.com/stories/how-get-hudson-yards)

## 方法与边界检查

1. 将 OSM 命名街道中心线与窗口边界多边形化。双向车道分别画线时，中间会出现狭长面。本次将宽度不超过 25 m、长宽比至少为 2，且至少 65% 周界沿同名街道的面归并给共享边界最长的邻近街区。Hudson Yards 修正 5 处，SoHo 修正 12 处。裁到窗口边缘或面积小于 500 m² 的街区排除在聚类样本之外。 [© OpenStreetMap 贡献者](https://www.openstreetmap.org/copyright)
2. 在街区边界切分第一步最大连通分量里的 NYCWalks 路段，保留路网的原有节点和连边关系。两个街区须共享至少 5 m 边界，且各自主要的步行连通分量在该边界上共享一个路网节点，才可直接合并。修正狭长面后仍有 3 对几何邻居缺少这种连接，见 `adjacency.geojson`。这些位置仍需逐点核对过街条件。 [MIT City Form Lab NYCWalks](https://cityform.mit.edu/projects/nycwalks)
3. 将两个样区的街区特征合并标准化。特征包括按长度加权的步行边介数、原路网节点平均度、断头节点比例，以及按长度加权的轴向方向。方向编码用 `cos(2θ)` 和 `sin(2θ)`，把同一直线的两个行走方向视为等价。聚类按跨组街区特征的平均欧氏距离逐步合并，每个尺度在上个尺度的分组基础上继续。
4. 路网边界接入点要求接通候选区内外的步行路段。街区接入点由图推导，商场接入点有独立的公开资料和门点支持。街区边界仍受 OSM 中心线的平面化和道路带归并规则影响。本轮连通性检验也沿用第一步路网的覆盖范围。

## 复现与输出

使用第一步的 Python 环境和数据快照。从仓库根目录运行以下命令。绘图另需 Matplotlib。

```sh
python experiments/step2/cluster.py
python -m pip install matplotlib==3.11.2
python experiments/step2/plot.py
```

- [候选区域地图](outputs/candidate_map.png)
- [机器可读指标](outputs/metrics.json)
- 每个样区目录包含 `blocks.geojson`、`adjacency.geojson`、`road_strips.geojson`、`candidates.geojson`、`gateways.geojson`、`network.geojson` 和 `merges.json`。可用 `network.geojson` 的 `u`、`v`、`edge_id` 和 `length_m` 重建图，核查相邻街区的连接证据与候选接入点。

## 用街区 embedding 决定合并

这次沿用前面划出的 77 个基本街区和步行邻接关系。先用商铺、餐饮、生活服务的类别比例、商业点密度、商业点在同一栋楼里的集中程度和步行线路密度，拟合一个六维 PCA embedding。类别和空间两组特征在降维前具有相同的总方差。经归一化后用余弦相似度比较 embedding。输入里没有街区坐标或现成的商圈标签。商圈边界来自入选街区的拼接。

候选合并必须有跨组步行连接，且合并后组内任意一对街区的余弦相似度都达到阈值。这是完全链接规则，用于防止两个很不相似的街区经第三个街区串在一起。本轮没有指定每片的街区数。数据少于 5 条商业 POI 的 20 个街区用总体类别分布补齐后也生成了向量，但暂时作为单街区保留。至少 5 条的 57 个街区用于拟合并可参与合并。建筑匹配少于 3 条时，用训练样区的中位数补齐建筑集中度。

| 最低相似度 | Hudson Yards 合并区（涵盖街区数） | SoHo 合并区（涵盖街区数） |
| --- | ---: | ---: |
| 0.55 | 1（5） | 11（34） |
| 0.65 | 2（5） | 10（29） |
| 0.80 | 2（5） | 6（15） |

在 0.65 档，Hudson Yards 的两片分别有 2 和 3 个街区，SoHo 的十片各有 2–6 个街区。SoHo 的 `b021` 和 `b024` 都以服装类商业点为主，相似度约 0.96，合进同一片。另一对可步行连通的邻居 `b030` 和 `b032` 相似度约 −0.60，分开保留。这个试跑检验了 embedding 能否决定合并，探索风格仍需用人工标注验证。阈值是试验设定，样区内拟合的结果还需要在新样区检验。

```sh
python experiments/step2/embedding.py
python experiments/step2/plot.py --embedding
```

[相似度合并地图](outputs/embedding/candidate_map.png) · [指标](outputs/embedding/metrics.json) · [拟合参数](outputs/embedding/model.json)。每个样区另有 `block_embeddings.geojson`、`neighbor_similarity.geojson`、`candidates.geojson` 和 `gateways.geojson`，可核对各街区的向量、邻接关系、相似度和最终分组。
