# PACE 数据

| 目录 | 内容 |
| --- | --- |
| `raw/` | 按城市和快照日期保存的原始数据。保留下载文件和来源清单。 |
| `processed/` | 清洗、关联、构图等处理结果。 |
| `annotations/` | 人工核验和探索画像标注。 |
| `splits/` | 训练、验证、测试样本 ID 和地理划分。 |

## 已有快照

第一版区域表征实验的数据源、样区和待补标内容见 [PACE–NYC v0 数据规格](../docs/DATASET_SPEC.md)。

`raw/nyc/2026-11-09/` 来自原 `experiments/step1/data/`，采集时间按原清单的 UTC 日期命名。包含 Hudson Yards、SoHo 的建筑轮廓、OSM 数据和 Overture Places，以及 NYCWalks 步行网络。

**Flushing 扩展（2026-11-14 加入同一快照目录）**：Main St 商圈窗口（边界冻结于 [configs/areas.json](../configs/areas.json)，冻结先于该区任何模型输出）。`flushing_*.geojson/json` 三层由 [fetch_area_raw.py](../experiments/fetch_area_raw.py) 下载，Overture 沿用原快照 release `2026-10-21.0`；哈希与下载时间记录在 [manifest.json](raw/nyc/2026-11-09/manifest.json) 的 `additions.flushing`，与原 2026-11-09 采集时间分开注明。

[manifest.json](raw/nyc/2026-11-09/manifest.json) 保存原采集时间和 7 个源文件的 SHA-256。迁移已逐一核对。旧样区边界和入口核验配置保存在 [归档 areas.json](../archive/2026-11-11/experiments/step1/areas.json)。

旧实验生成的图和预测结果位于 `archive/2026-11-11/experiments/step*/outputs/`，可用于追溯当时的处理过程。

[纽约现成围栏小样本](annotations/nyc_boundary_pilot/README.md) 含两个官方 BID 多边形及一处商场轮廓，供片区评估和尺度核验。

[街景街区相似性标注](annotations/nyc_streetview_v1/README.md) 是锚定三元组标注数据集（E3），含机位采样、街景抓取、三元组构建与本地打分界面，协议见[标注协议](../docs/STREETVIEW_ANNOTATION_PROTOCOL.md)。

原始数据和处理结果默认留在本地，由根目录 `.gitignore` 排除。标注、划分清单和配置可以纳入版本控制，便于复现实验。
