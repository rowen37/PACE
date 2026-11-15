# Flushing 留出区判决（地理迁移）

Flushing 窗口于任何模型输出之前冻结（[configs/areas.json](../configs/areas.json)），数据、切块、街景、两人标注（001/002）均在模型接触该区之前完成。编码器按冻结协议在开发区训练（SoHo→Flushing 主判，Hudson Yards→Flushing 三角验证），三种子。无手设阈值。

## 判决一：画像预测不迁移（原始口径）/ 大幅衰减（均衡口径）

原始准确率（共识标签，LOO-1NN）：

| 字段 | POI | 步行 | POI+步行融合 | 多数类基线 |
| --- | --- | --- | --- | --- |
| 店门密度 | 49% | 58% | 60% | **67%** |
| 沿街连续性 | 45% | 57% | 66% | 57% |

Flushing 标签严重偏斜（约半数块为非商业/破碎形态，多数类 67%），原始准确率对多数类系统有利。**均衡准确率**（各类召回平均，随机线 33%）：

| 字段 | POI | 步行 | 融合 |
| --- | --- | --- | --- |
| 店门密度 | 40-42% | 47% | **48-52%** |
| 沿街连续性 | 45-47% | 45-48% | **56-63%** |

读法：**融合超随机线 15-30 个点——结构有残留；开发区的"单视角分工"（POI 单独 83% 管密度）完全没有迁移**（单视角全部退到 40-48%）。换训练区（HY）结论不变，排除训练区选择问题。

## 判决二：画像选层迁移不稳定

001 选层 / 002 审计（组内-跨组边分离）：三种子分别 +0.107（输给模块度逐块 +0.161）、**+0.314（大胜，模块度 −0.021）**、+0.027（平手）。均值领先（+0.149 vs +0.056）但不再 3/3 全胜（SoHo 是 3/3）；选中层跨种子摆动大（17/33/56 组）。

## 归因边界（诚实版）

- **嵌入端确有衰减**：同协议同标注口径下，开发区 81%/76% → 留出区 48-66%（均衡口径）
- **测量端同时在退化**：Flushing 001-002 κ=0.52（SoHo 0.36/HY 0.60）、共识子集 n=28-43、类别偏斜——两边都在变差，无法把差距全部记在嵌入头上
- **可用结论**："表征在未见地区保留部分与人类感知一致的结构（融合层）"；**不可用结论**："单视角分工可迁移"、"高准确率可迁移"

## 复现

```sh
.venv/bin/python experiments/flushing_embeddings.py                       # soho → flushing
.venv/bin/python experiments/flushing_embeddings.py --train hudson_yards # 三角验证
.venv/bin/python experiments/streetview_profile_levels.py --areas flushing \
    --source outputs/flushing_embeddings --output outputs/profile_level_selection_flushing
```

冻结产物：`outputs/flushing_embeddings/`、`outputs/flushing_embeddings_from_hudson_yards/`、`outputs/profile_level_selection_flushing/`。
