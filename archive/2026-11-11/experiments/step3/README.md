# PACE 第三步 · GATv2 对照试跑

这一轮检验图注意力能否学习街区内的商业空间组织。具体任务是只看部分商业 POI，预测同一街区中被遮住的商业 POI 有多集中。建筑集中度是可重复计算的代理指标。步行体验仍需要用独立商场目录、临街入口和实际走访评估。

## 实验设置

- 沿用 Hudson Yards 和 SoHo 两个样区的 77 个完整街区。轮流以一个样区训练、整个另一样区测试。每个街区随机遮住约一半商业 POI。被遮住的点、类别和匹配关系从模型输入移除，建筑轮廓和现有步行图仍作为公开底图输入。
- 在被遮住且能唯一匹配建筑的 POI 中任取两条记录，计算它们落在同一建筑轮廓里的比例。只有能组成这样的 POI 对的遮盖样本才有可计算目标。POI 归属由空间叠加推得，结果取决于源数据的覆盖和位置精度。
- GAT 与 GATv2 共用两层异构图编码器、六维街区向量和线性预测头。节点包括店铺、建筑、步行段和街区，关系包括店铺落入建筑、步行段相连、已核验的建筑入口接入步行段，以及三类节点各自属于哪个街区。输入特征、遮盖划分、训练轮数和随机种子一致。两种注意力算子的参数数目略有差异。 [GAT 算子](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html) · [GATv2 算子](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html)
- PCA 对照用同一批可见点计算类别比例、建筑集中度、匹配率和密度等统计量，在训练样区重新拟合六维 PCA，再用线性头预测。直接估计对照沿用可见匹配点的同楼比例。可见点无法组成一对时，用训练街区的目标均值。直接估计检验这个任务是否确实需要图模型。
- 每个训练街区最多有 8 次遮盖，每个测试街区最多 5 次，图模型用 3 个初始化种子并平均预测。主指标是先在同一街区内对遮盖的绝对误差取均值，再在街区间等权平均。误差可读成比例的百分点，越低越好。

## 2026 年 11 月 9 日结果

平均绝对误差（百分点，按街区等权）。

| 方法 | 测试 Hudson Yards，训练 SoHo | 测试 SoHo，训练 Hudson Yards |
| --- | ---: | ---: |
| 可见 POI 同楼比例 | **19.5** | **8.0** |
| PCA | 24.6 | 23.2 |
| GAT | 28.8 | 54.5 |
| GATv2 | 33.2 | 63.2 |
| 可计算目标的测试街区数 | 11 | 49 |

77 个街区里有 60 个至少在一次测试遮盖中能计算同楼比例。训练 Hudson Yards、测试 SoHo 时，GATv2 对测试街区的平均预测是 71.8%，实际平均为 14.6%。三个 GATv2 初始化种子单独计算的 SoHo 误差为 59.7–65.7 个百分点。这个方向的训练数据只有 11 个可计分街区，训练和测试样区的建筑集中度差别也很大。结果表明这套试跑配置未能在该代理任务上跨样区胜过简单统计。入口证据只有第一步核对的两个门点，各自接入两个步行段。

下一次评估应增加能整片留出的独立商业片区，并用独立核验的商场与临街样本验证探索类型。后续推荐效果以用户在相同候选区和时间预算下的实际选择与体验为准。

## 复现

在含有第一步数据快照及输出的工作区，从仓库根目录运行。

```sh
python -m pip install -r experiments/step3/requirements.txt
python experiments/step3/run.py
```

[指标](outputs/metrics.json) · [Hudson Yards 逐街区预测](outputs/hudson_yards_predictions.json) · [SoHo 逐街区预测](outputs/soho_predictions.json)

## GATv2 调参复查

每轮先将训练样区沿跨度最大的坐标轴分成三个相邻的地带，轮流留出其中一带做验证。每组设置在第 5、15、30、45 轮上分别评分，按内部验证的街区等权绝对误差选配置。三组设置分别为原配置（宽度 16、学习率 0.004、权重衰减 0.01）、只降低学习率到 0.001，以及宽度 8、学习率 0.004、权重衰减 0.05。内部搜索用一个初始化种子。确定设置后，用整个训练样区和三个种子重新训练，复用原先的整片测试和遮盖样本。这是同一批样区的诊断性复查。

| 整片测试 | 内部验证选出的配置 | 原 GATv2 MAE | 调参后 MAE | 可见点比例 MAE |
| --- | --- | ---: | ---: | ---: |
| Hudson Yards | 学习率 0.001、45 轮 | 33.2 | 26.4 | **19.5** |
| SoHo | 原学习率 0.004、5 轮 | 63.2 | 42.8 | **8.0** |

单位为百分点。用 Hudson Yards 训练时，内部验证三带分别只有 2、5、4 个可计分街区。早停后对 SoHo 的平均预测由 71.8% 降到 52.3%，仍高于 SoHo 的 14.6%。这次搜索改善了两个方向，但在同一批数据上，简单统计仍更准。

```sh
python experiments/step3/tune.py
```

[调参复查指标](outputs/tuning/metrics.json) 包含各配置的内部验证成绩、街区划分和外部测试误差。

## 隔离测试后的 GATv2 调参诊断

保留原有数据、图结构、POI 遮盖目标和评分方式。七组配置各在第 1、5、15、30、45 轮记一次成绩。相对原配置，分别测试较低学习率、较窄模型、更高权重衰减、特征 dropout、将训练损失从 BCE 换为 MAE，以及窄模型配合中等权重衰减。除最后一组联合设置外，每组只改一个因素。

两次 `select` 各自只读取对应的训练样区数据。内部验证使用三个地理条带，清除训练侧与验证侧共享建筑轮廓或原始 NYCWalks 线段的街区。类别词表只由每折可见的训练 POI 构建，验证和测试张量不含目标标签。内部按两个初始化的平均预测选择街区等权 MAE 最小的配置，并将完整排名和选定配置写入各自的选择文件。两个文件都存在后，`test` 才按锁定配置训练三个初始化并评分。两个外层样区是前面已报告过的样区，因此本轮是诊断性复查。

```sh
python experiments/step3/audit_tuning.py select --training-area hudson_yards
python experiments/step3/audit_tuning.py select --training-area soho
python experiments/step3/audit_tuning.py test
```

脚本保留已写出的选择文件和外层评分。重新运行整套实验时，可以用同一个新的 `--output` 目录执行三条命令。

### 诊断结果

| 整片测试 | 原 GATv2 MAE | 上一轮调参 MAE | 本轮隔离调参 MAE | 可见点比例 MAE |
| --- | ---: | ---: | ---: | ---: |
| Hudson Yards | 33.2 | 26.4 | 30.2 | **19.5** |
| SoHo | 63.2 | 42.8 | 54.8 | **8.0** |

单位为百分点。两边的内部验证均选中直接以 MAE 训练，保留原宽度、学习率和权重衰减，在第 30 轮停止。选择按各自内层验证成绩确定。上一轮的配置和分数作为历史参照。

- **过拟合。** Hudson Yards 内部，原配置从第 15 轮继续到第 45 轮，平均训练 MAE 从 20.4 降到 5.4，验证 MAE 从 38.9 升到 48.9。清除共享实体后三折分别只有 7、3、4 个可计分训练街区。
- **跨区预测偏差。** Hudson Yards 训练目标均值为 53.8%，SoHo 外层测试目标均值为 14.6%。调参后，模型对 SoHo 的平均预测为 62.9%。反方向也有偏差，SoHo 训练后对 Hudson Yards 的平均预测为 28.1%，实际为 55.3%。
- **排序和校准。** 在 SoHo，调参后的 GATv2 街区排序 Spearman 相关为 0.25，直接统计为 0.81。在 Hudson Yards，二者分别为 0.76 和 0.78。对 SoHo，除了整体高估，街区相对集中程度也排序不准。直接统计和任务的目标形式相同，分别估计同一个街区的两批随机遮盖样本，因此这是一个贴合代理任务的基线。

[锁定配置后的测试指标](outputs/tuning_audit/metrics.json) · [Hudson Yards 内层搜索](outputs/tuning_audit/hudson_yards_selection.json) · [SoHo 内层搜索](outputs/tuning_audit/soho_selection.json)

## 建筑级辅助任务对照

复用原来的遮盖样本与整区留出，比较直接统计、正则化回归、GATv2，以及加建筑级辅助任务的同一 GATv2。回归与两组 GATv2 使用相同的街区统计特征，包括可见同楼比例、该比例能否计算、可见匹配数、类别构成和底图汇总特征。两组 GATv2 共用原有异构图。特征归一化和词表均由每折的训练数据拟合。

辅助任务给定可见且唯一匹配建筑的店铺，让模型在同一街区的建筑轮廓中识别它的归属。评分使用主模型共享的店铺类别和建筑形态投影。已知归属边仅用于提供辅助标签，辅助评分读不到归属边、楼内店铺数或隐藏 POI。只有训练批次运行辅助任务。这项对照检验的是这一个辅助目标带来的增量。

内部验证沿用三条地理带和共享建筑及路段清除规则。在每个训练区单独选岭回归正则强度，先选出普通 GATv2 的停止轮数，再在同一轮数比较建筑辅助权重。权重搜索包括零。整区测试报告锁定的最佳正权重配置与零权重配置，便于直接判断辅助项的效果。每组图模型使用相同的三个初始化种子。主指标是街区等权 MAE，分组按可见匹配数报告，配对误差先在街区内平均。

```sh
python experiments/step3/building_auxiliary.py select --training-area hudson_yards
python experiments/step3/building_auxiliary.py select --training-area soho
python experiments/step3/building_auxiliary.py test
```

复跑时对三条命令指定同一个新的 `--output` 目录。

| 方法 | 测试 Hudson Yards（11 个街区） | 测试 SoHo（49 个街区） |
| --- | ---: | ---: |
| 直接用可见同楼比例 | **19.48** | **8.00** |
| 正则化回归 | 33.74 | 35.73 |
| GATv2＋街区统计 | 31.31 | 18.80 |
| GATv2＋街区统计＋建筑辅助任务 | 30.74 | 19.03 |

数值为街区等权 MAE，单位是百分点。辅助项在 Hudson Yards 降低 0.58 个百分点（8/11 个街区改善），在 SoHo 增加 0.23 个百分点（15/49 个街区改善）。两边可见匹配数为 0–1 的组均未改善，分别仅有 3 个和 2 个街区。训练区内部的最佳正权重相比零权重只改善约 0.18 和 0.002 个百分点（分别在 Hudson Yards 和 SoHo 选择）。本轮暂不把辅助项纳入主模型。

两个外层样区此前已用于诊断，结论限于这两区和本轮同楼比例代理任务。测试行逐一与原试跑核对街区、遮盖种子、目标值和直接统计预测。

[完整指标与按可见匹配数分组](outputs/building_auxiliary/metrics.json) · [Hudson Yards 训练区选参](outputs/building_auxiliary/hudson_yards_selection.json) · [SoHo 训练区选参](outputs/building_auxiliary/soho_selection.json)

## RGSCL 双视角实验

使用同一批 PACE 样区和 POI 遮盖，复用 [RGSCL 作者发布的模型代码](../../references/RGSCL/code/model.py) 中的两路 GAT、单头 GAT 增强分支、注意力融合和判别器。POI 图用可见点的类别和位置。建筑图用轮廓的八个形态指标和质心。每个街区内分别做 Delaunay 连边，两路各设一个街区汇总节点。对重合、共线或数量不足的点保留节点，使用可用的几何邻接和汇总边，避免从原测试集删掉稀疏街区。

按[论文 §3.3](https://doi.org/10.1080/13658816.2025.2585320)实现区内跨视角对比和区间对比，区内损失权重为 0.7。每个预训练批次对每个训练街区只使用一份可见遮盖，避免同一街区的两份遮盖互为负样本。每折词表、建筑特征归一化、预训练和预测头拟合都只使用训练侧。冻结编码器后用宽度为 16 的 MLP 预测同楼比例。另一路在 MLP 输入中加入可见同楼比例、是否缺失及匹配数。每一路单独在训练区的三条地理验证带选择预训练和预测头停止轮数，外层用三个种子平均。比较方法使用已完成的建筑辅助对照输出，测试逐行核对街区、遮盖种子、标签和直接统计预测。

| 方法 | 测试 Hudson Yards（11 个街区） | 测试 SoHo（49 个街区） |
| --- | ---: | ---: |
| 直接用可见同楼比例 | **19.48** | **8.00** |
| GATv2＋街区统计 | 31.31 | 18.80 |
| RGSCL | 24.54 | 31.98 |
| RGSCL＋可见同楼统计 | 25.54 | 32.95 |

数值为街区等权 MAE，单位是百分点。RGSCL 在 Hudson Yards 好于 GATv2，在 SoHo 更差，两边都没有超过直接统计。在只匹配到 0–1 个可见 POI 的遮盖样本中，Hudson Yards 的 RGSCL MAE 为 32.31，直接统计回退到训练区均值后为 39.20（3 个街区）；SoHo 对应为 69.50 和 47.74（2 个街区）。这两区以前已用于诊断，且 GATv2 使用了额外的归属与步行图特征，因此这是同任务的方案对照，不能把差异单独归因于编码器。

```sh
python experiments/step3/rgscl.py select --training-area hudson_yards
python experiments/step3/rgscl.py select --training-area soho
python experiments/step3/rgscl.py test
```

脚本需要作者模型文件以及前一轮 `outputs/building_auxiliary` 中的逐街区结果。复跑时对三条命令指定同一个新的 `--output` 目录。

[完整指标](outputs/rgscl/metrics.json) · [Hudson Yards 逐遮盖预测](outputs/rgscl/hudson_yards_predictions.json) · [SoHo 逐遮盖预测](outputs/rgscl/soho_predictions.json)
