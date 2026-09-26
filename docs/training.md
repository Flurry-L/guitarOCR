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

## 三种谱面的统一定位模型（两类前置模型）

`layout/configs/train_multimode.yaml` 从已发布的 TAB 检查点继续训练 `measure` / `tempo_region` 两类框，同时学习 tab、notation、both。混合谱每个小节只有一个合并框，不增加谱面类型检测类别。

已完成的 12 轮训练选用第 8 轮权重，保存在 `weights/pp_doclayout_v3_score_multimode`，作为后续四类模型的初始化来源。分谱面测试结果、源谱分组和实际训练配置见[模型卡](../weights/pp_doclayout_v3_score_multimode/README.md)。

```bash
tools/paddlex-venv/bin/python -m layout.train -c layout/configs/train_multimode.yaml
# 验证集用于选择检查点；测试集只在选好检查点后评估
tools/paddlex-venv/bin/python -m layout.evaluate \
  --model-dir output/pp_doclayout_v3_score_multimode/best_model/inference \
  --dataset-dir database/gp8_layout_multimode_v1/datasets/layout \
  --split test --postprocess --output output/evaluation/multimode-test.json
```

训练入口显式使用 bbox AP 选择最佳权重，修正 PaddleX 默认 `target_metrics: mask` 与 `eval_mask: false` 的不一致；并保留 `gt_read_order` 数据字段。使用其他初始权重时覆盖 `Train.pretrain_weight_path`。每轮保存检查点和导出模型，训练完成后才选择发布到 `weights/`。

评估文件按 mode 分别报告 AP、召回、小节数完全正确的页面数及整谱数，同时保存模型／标注哈希、逐页数量和预测框。`--mode tab` 等参数可筛选模式，`--device gpu:7` 可指定评估设备。不带 `--postprocess` 时 AP 基于未阈值过滤的原始预测；带该选项时评估实际阈值和谱线后处理后的框。

## 分开报告不同评测条件

### 版面定位

```bash
tools/paddlex-venv/bin/python -m layout.evaluate   --model-dir weights/pp_doclayout_v3_score_typed   --dataset-dir database/gp8_layout_multimode_v1/datasets/layout_typed   --split test --postprocess --output output/evaluation/layout.json
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

## 四类版面模型：定位与谱面类型

`weights/pp_doclayout_v3_score_typed` 在前置两类模型基础上继续训练 8 轮，按验证集选择第 7 轮，同时预测小节位置和 `tab / notation / both` 类型。当前默认路径以 `shared/defaults.py` 和 `weights/manifest.json` 为准。

在上述三模式 inventory 上使用 `datagen.build_layout_data --typed-measures --include-test` 构建 `datasets/layout_typed`。保留历史两类数据和模型，便于独立比较定位效果。

```bash
tools/paddlex-venv/bin/python -m layout.initialize_typed \
  --source output/pp_doclayout_v3_score_multimode/best_model/best_model.pdema \
  --output output/layout-typed-initial.pdparams
tools/paddlex-venv/bin/python -m layout.train -c layout/configs/train_typed.yaml
tools/paddlex-venv/bin/python -m layout.evaluate \
  --model-dir output/pp_doclayout_v3_score_typed/best_model/inference \
  --dataset-dir database/gp8_layout_multimode_v1/datasets/layout_typed \
  --split test --postprocess --output output/layout-typed-test.json
```

初始化需要保留的训练检查点；这里使用与上一版发布推理模型对应的 EMA 权重。脚本仅扩展分类投影及类别 embedding，记录输入／输出哈希，其余张量保留。完成训练后等待进程正常退出再复制 best_model，不能仅根据 PaddleX 的 `done_flag` 判断导出文件已经写完。

评测同时报告合并小节类别后的定位 AP、`typed_detection` 四类 AP、按页投票的类型准确率和 IoU 0.50 匹配小节的类型准确率。未匹配框的影响由 AP／召回率体现；没有任何类型预测的页面计为类型识别失败。当前模型和完整记录见[四类模型卡](../weights/pp_doclayout_v3_score_typed/README.md)。

## 发布模型

更新权重时同步维护 `weights/manifest.json`、对应模型卡、训练配置、source_catalog 的版本或哈希，以及采用上述哪种条件的评测报告。使用 `guitarocr-check --hashes` 检查实际文件。基座 revision 固定，LoRA 与版面权重随项目提供。

## 扩充三种谱面的继续训练

`gp8_joint_v3` 的来源隔离、标签清理和增强规模见[数据说明](data.md)。对应配置为 `layout/configs/train_joint_v3.yaml`、`document_info/configs/train_v3.yaml` 和 `measure_ocr/configs/train_v3.yaml`。小节模型从 v2 LoRA 继续训练；在 H100 上使用 BF16、FlashAttention 2、每卡 16 张图及两步梯度累积，8 卡全局批量为 256。

先单独完成缓存，避免误把 LLaMA-Factory 的 `tokenized_path` 当成“仅预处理后退出”的开关。标签、提示或上下文规则改变时必须使用新的缓存目录：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m shared.tokenize_training \
  --config measure_ocr/configs/train_v3.yaml \
  --output database/gp8_joint_v3/datasets/measure_ocr/tokenized_v3_context_checked
PATH="$PWD/.venv/bin:$PATH" FORCE_TORCHRUN=1 NPROC_PER_NODE=8 OMP_NUM_THREADS=1 \
  .venv/bin/python -m measure_ocr.train --config measure_ocr/configs/train_v3.yaml
```

本轮环境使用 Torch 2.14.0+cu130、Transformers 5.8.0 和 FlashAttention 2.8.3.post1。FA2 是训练加速依赖，不是普通推理的必需项；其他环境可传入 `flash_attn=sdpa`。此机器的 FA2 从源码针对 SM90 编译：CUDA 13.0，C++20（Torch 2.14 头文件要求），`MAX_JOBS=32 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE`，不替换现有 Torch。安装后必须验证真实小节数据的前向、反向以及最长样本的显存占用。

版面继续训练的初始化使用 `best_model.pdparams`，这是 Paddle 保存的 EMA 选择权重；`.pdema` 存放恢复训练用的原始参数，不能直接传给本项目的预训练加载入口。训练与旧模型对比使用相同的生成条件和样本，先按验证结果选择权重，再做独立测试。扫描退化集作为单独的压力测试，不能称为真实扫描测试集。选中的本地发布权重附带训练记录、数据来源哈希和新旧评估；历史版本保留用于回退。

小节评估中的 `overall` / `by_mode` 统计交给后续流程的最终序列；`raw_overall` / `raw_by_mode` 统计重试结束后的原始 OCR 输出。连续预测时，无效小节会被替换为休止占位，因此报告格式／约束合法率时应使用 `raw_*` 并同时报告 `needs_review`，不能把占位后的合法率当作模型合法率。单次解码与重试评估也应分别说明。

`measure_ocr.evaluate_parallel --batch-size 8` 可在每张 GPU 上批量生成互相独立的标注前文样本；连续预测前文必须保持 `--batch-size 1`。批量输入使用左侧 padding，校验过未填充 token、图像像素和图像网格与逐条处理完全相同。BF16 批量矩阵运算仍可能改变边缘 token 的贪心选择，因此新旧模型必须使用相同 batch size 重评，不能混用逐条基线与批量候选结果；batch size 也纳入恢复签名与评估记录。

谱头副标题修复使用 `document_info/configs/train_v3_headers.yaml`：从 v2 LoRA 继续，学习率 `1e-5`，两轮、8 卡全局批量 96；23,808 条训练、1,584 条验证。先用相同 tokenization 命令处理该配置，输出至 `database/gp8_headers_v3/datasets/tokenized_info_mixed`，然后调用 `document_info.train --config document_info/configs/train_v3_headers.yaml`。与其他分布式任务并行时需指定不同 `MASTER_PORT`。检查点选择仍使用验证集，真实中文谱头回归也是发布门槛。

本轮小节训练已完成：最多两轮的计划在第 5,000 步早停（约 1.554 轮，8,468 秒），使用第 3,500 步最小验证损失 0.0327563 的权重。谱头修复版完成两轮 496 步、选择第 400 步；版面模型完成六轮、选择第 4 轮。三项训练均使用 8 张 H100，独立测试在权重选择后执行。最终对照和限制见[扩充训练报告](training-v3-report.md)。

`core_exact_rate` 同时要求小节元数据、节奏和音符字段一致；`exact_match_rate` 比较规范化 M2 的完整内容，还包括奏法。需要只检查音符或节奏时，应分别看 `note_fields_exact_rate` 和 `rhythm_exact_rate`。连续评估的失败占位保留前一小节 feel，但正常预测缺省 feel 仍表示直拍；修改运行时会更新恢复签名，旧预测不能直接当作新运行续跑。
