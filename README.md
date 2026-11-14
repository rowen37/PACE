# PACE — Place-Aware Citywalk Exploration

PACE 以 RGSCL 的 POI 与建筑双视角为基线，加入公共入口、步行通路和目的地接入关系，学习行人视角的街区表征。连通片区生成与 Citywalk 推荐是这项表征的下游应用。

2026-11-11 整理工作区，准备重新搭建实验。

当前任务定义、三方代码审计和分阶段实验见 [任务与下一步 SOP](docs/TASK_AND_EXPERIMENT_SOP.md)。审计发现并修复旧 RGSCL 的跨地区建筑图缓存错配；旧 RGSCL 外区成绩暂不能用于模型优劣判断，待固定历史参数重新评估。原始数据和历史输出保留。

## 实验环境

在项目根目录，用 Python 3.12 创建本地环境：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

[requirements.in](requirements.in) 列出直接依赖，[requirements.txt](requirements.txt) 固定完整版本。当前环境在 macOS arm64、Python 3.12.15 上验证，用于读取冻结快照、构图和运行模型。历史下载、绘图和作者脚本的额外依赖见各自归档说明。

已有本地快照时，可用一次短运行验证三视角实验入口。每次指定一个新的输出目录：

```sh
.venv/bin/python experiments/pedestrian_rgscl.py --epochs 1 --output outputs/env_smoke
```

## 目录

```text
PACE/
├── data/
│   ├── raw/nyc/2026-11-09/   # 已有纽约原始数据及来源校验清单
│   ├── processed/           # 新流程生成的清洗数据与图
│   ├── annotations/         # 人工标注
│   └── splits/              # 固定的训练、验证、测试划分
├── src/pace/                # 新流程的公共代码
├── configs/                 # 数据处理与实验配置
├── docs/                    # 任务定义、代码审计与实验 SOP
├── experiments/             # 新实验入口
├── outputs/                 # 新实验指标、模型和图表
├── references/              # 论文及作者参考代码
└── archive/2026-11-11/       # 原实验、结果与原项目说明
```

原始数据以独立快照保存，处理后的数据写入 `data/processed/`，标注和划分分别写入对应目录。实验配置引用数据路径，运行结果按实验名称写入 `outputs/`。

数据说明见 [data/README.md](data/README.md)。旧实验的使用方式和迁移记录见 [archive/2026-11-11/README.md](archive/2026-11-11/README.md)。
