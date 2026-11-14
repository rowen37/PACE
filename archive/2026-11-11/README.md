# 2026-11-11 实验归档

| 位置 | 内容 |
| --- | --- |
| `experiments/step1/` | 数据获取、基础图和入口核验，以及原输出。 |
| `experiments/step2/` | 街区构建、embedding 和合并试验，以及原输出。 |
| `experiments/step3/` | GAT、GATv2、调参、建筑辅助任务、RGSCL，以及原输出。 |
| `project_proposal.md` | 整理前的根目录 README。 |
| `migration_manifest.json` | 98 个迁移文件的新旧路径、字节数和 SHA-256。 |

原始数据实际存放于根目录 `data/raw/nyc/2026-11-09/`。本目录 `experiments/step1/data` 的相对符号链接指向该快照，`references` 的相对符号链接指向根目录的参考资料。旧脚本的相对路径由这两个链接衔接。

复查旧实验时，以本归档目录为工作目录，使用各步骤 README 中的命令。例如：

```sh
cd archive/2026-11-11
python experiments/step3/rgscl.py --help
```

依赖版本记录在 `experiments/step1/requirements.txt` 和 `experiments/step3/requirements.txt`。执行模型脚本需要安装对应依赖。已有选参和测试输出保留在原有子目录中，重跑使用脚本的 `--output` 参数指定新的输出位置。
