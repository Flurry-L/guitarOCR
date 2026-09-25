# 训练与评测

先完成 [数据生产](data.md) 和 [训练依赖安装](setup.md#glm-训练环境)。训练配置位于对应处理阶段，默认路径与 `datagen.run --phase all` 的输出一致。训练产物写入 `output/`，发布权重位于 `weights/`。

## 训练

```bash
uv run --no-sync python -m measure_ocr.train
uv run --no-sync python -m document_info.train
```

这两个入口使用 LLaMA-Factory。可以传 `--config` 和 `key=value` 参数覆盖数据目录、输出目录和训练参数。多卡使用 `FORCE_TORCHRUN=1`、`NPROC_PER_NODE`。不要覆盖已发布的模型目录。

```bash
tools/paddlex-venv/bin/python -m layout.train -c layout/configs/train.yaml -o Global.mode=train
```

`layout/configs/train_sparse.yaml` 用于进一步微调稀疏谱行。先构建相应稀疏数据，确认 `Global.dataset_dir` 与初始 checkpoint 路径，再运行。它们是两阶段训练配置，推理只加载一份最终版面权重。

## 分开报告不同评测条件

### 版面定位

```bash
tools/paddlex-venv/bin/python -m layout.evaluate   --model-dir weights/pp_doclayout_v3_score_sparse   --dataset-dir database/gp8_measure_sequence_v2/datasets/layout   --postprocess --output output/evaluation/layout.json
```

报告框的 AP、召回与小节数量正确的页面数。此结果不代表音符准确率。

### 谱面信息

```bash
uv run --no-sync python -m document_info.evaluate --output output/evaluation/info.jsonl
```

### 正确裁图、标注前文

```bash
uv run --no-sync python -m measure_ocr.evaluate   --manifest database/gp8_measure_sequence_v2/manifests/test.jsonl   --context-source gold --max-samples 600   --predictions output/evaluation/gold.jsonl --metrics output/evaluation/gold-metrics.json
```

这里的上一小节上下文来自标注，评估单个裁图的能力。

### 正确裁图、预测前文

```bash
uv run --no-sync python -m measure_ocr.evaluate   --manifest database/gp8_measure_sequence_v2/manifests/test.jsonl   --context-source predicted --max-samples 0 --max-sources 20   --predictions output/evaluation/sequences.jsonl --metrics output/evaluation/sequences-metrics.json
```

按 source / mode / measure_index 顺序执行，前文使用实际预测。首小节从 START 开始；无效输出反馈为带待检查标记的休止小节。续跑会重建预测上下文。不得随机抽散小节，因此此模式要求 `--max-samples 0`，可用 `--max-sources` 限制完整曲谱数量。结果仍以正确裁图为输入。

### 完整 PDF / 图片流程

```bash
uv run --no-sync python -m pipeline.evaluate   --cases examples/cases.json --output output/evaluation/examples   --layout-python tools/paddlex-venv/bin/python
```

每个案例指定 inputs、expected_m2、mode，可指定 layout_source。输入路径相对 cases 文件。报告模型清单、依赖版本、输入校验值、耗时、峰值已分配显存、小节数和待检查列表。小节数量相同时按阅读顺序对齐评测；数量不同时记录定位失败，不输出具有误导性的逐小节分数。

`examples/` 是四小节功能示例，不是独立评测集。正式效果报告应另外记录数据来源、分组划分、完整曲谱指标和人工修订量。`measure_ocr/configs/release_gate.json` 的阈值是目标，不是已测得成绩。

## 发布模型

更新权重时同步维护 `weights/manifest.json`、对应模型卡、训练配置、source_catalog 的版本或哈希，以及采用上述哪种条件的评测报告。使用 `guitarocr-check --hashes` 检查实际文件。基座 revision 固定，LoRA 与版面权重随项目提供。
