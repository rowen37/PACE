# nyc_streetview_v1：街景街区相似性标注

锚定三元组（A；B vs C）街景标注数据集，用于 E3 表征评价。任务定义、判据卡、采样规则与打分计划见[标注协议](../../../docs/STREETVIEW_ANNOTATION_PROTOCOL.md)。

当前状态：占位图（demo）数据已就绪，机位 618 个（SoHo 422 / Hudson Yards 196），覆盖 100 个街区，三元组 99 题。运行 `experiments/streetview_fetch.py google --replace`（需 `PACE_GSV_API_KEY`）替换为真实街景后即可正式标注；标注界面：

```sh
.venv/bin/python experiments/streetview_annotation_server.py
# 打开 http://127.0.0.1:8787，输入标注者 ID 即可开始
```

`responses/` 下每位标注者一个 JSONL，只追加不覆盖。
