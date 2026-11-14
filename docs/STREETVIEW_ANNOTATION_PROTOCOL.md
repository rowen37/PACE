# 街景街区标注协议（nyc_streetview_v1）

E3 表征评价的人工标注。范式沿革：v1 锚定三元组（"哪个更像 A"）→ 试标后发现复合相似度判断超出人的可靠处理范围（需同时评估锚、两个候选再比相似），且"串联能力"从照片不可判。**v2 改为逐块类别标签**：每屏一个街区、一个字段、若干桶；块间相似度由画像事后计算，不由标注者脑内聚合。方法先例：Gehl Public Space/Public Life 调查即逐段清点临街面类型，而非相似度比较。

## 评分字段（v2，现行）

每屏展示一个街区的全部街景（沿步行段布点、每块至多 4 张），问一个字段：

| 字段 | 构念锚点 | 桶 |
| --- | --- | --- |
| 街面类型 frontage | Gehl 活跃临街面（active frontage） | A 临街小店为主 / B 混合 / C 大入口为主 / D 几乎无商业街面 / X 看不出 |
| 店门密度 door_density | frontage 密度 | A 一个挨一个 / B 明显间隔 / C 零星或没有 / X |
| 沿街连续性 continuity | street wall 连续性（Lynch 路径意象） | A 连续 / B 有断口 / C 破碎 / X |

通用规则：以这组照片能看到的为准；判断不了选 X，不猜；不看品牌好坏与新旧。**「串联能力」不设照片标注**：它是路线层面的量，从步行网+POI 位置计算（规划中的构图实验），看照片的人判不了。

「都断开怎么办」类两难在打标签制下消失：两块都标"破碎"即有效信息，相似度=画像距离。

## 历史数据与采样设计

- v1 三元组（99 题，含 S1 POI 相近走法不同 / S2 反向 / S3 随机分层、跨区混合、10% 换序重复）保留在 `triplets.json`，标注者 001 的整体印象判断保留在 `responses/`；其比对结论作为协议演进的记录，不再扩充。S1/S2/S3 的分层逻辑将来用于构造块对难度分层。
- v2 评分题：`ratings.json`，100 块 × 3 字段 + 15 道重复题 = 315 题，顺序固定混洗（种子 20261116）。重复题测自一致性，不提示。

## 质量与评价

1. **标注者间**：逐字段 Cohen's κ（X 计入）；试标 κ≥0.4 定稿措辞，正式 κ<0.6 不出共识。共识=非 X 严格多数。
2. **自一致性**：重复题一致率；`elapsed_ms` 分布筛查极速作答。
3. **模型评价（块对层面）**：块画像 = (frontage, door_density, continuity)；预测任务为逐字段留一 1-NN（冻结向量余弦近邻），对照多数类基线——这是比三元组一致率更硬的监督指标。
4. **模型评价（相似度层面）**：同画像块对 vs 异画像块对的 embedding 余弦区分度（AUC），按 v1 的 S1/S2 分层报告。
5. 两个开发区仅作试标；正式评价在预先冻结的新地区进行（[SOP](TASK_AND_EXPERIMENT_SOP.md) E3）。

## 复现

```sh
.venv/bin/python experiments/streetview_points.py                      # 机位（无需 key）
.venv/bin/python experiments/streetview_fetch.py google                # 真实街景，需 PACE_GSV_API_KEY
.venv/bin/python experiments/streetview_ratings.py                     # v2 评分题（现行）
.venv/bin/python experiments/streetview_triplets.py                    # v1 三元组（历史）
.venv/bin/python experiments/streetview_annotation_server.py           # 打开 http://127.0.0.1:8787
.venv/bin/python experiments/streetview_evaluate.py                    # 质检 + κ + LOO 预测，写 evaluation.json
```

服务器优先服务 `ratings.json`（评分题），否则回退三元组。标注结果只追加，不回写。密钥：Google Cloud Console 开通 Street View Static API 后设 `PACE_GSV_API_KEY`；或 Mapillary 设 `PACE_MAPILLARY_TOKEN`。当前规模（100 块 × 4 图 + 元数据）在 Google 每月赠金额度内。
