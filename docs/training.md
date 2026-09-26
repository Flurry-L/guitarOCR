# 训练与评测

先完成[数据生产](data.md)和[训练环境安装](setup.md#glm-训练环境)。下面使用当前默认模型对应的配置；训练数据、缓存与检查点不随仓库提供，需要先生成。实测结果见[扩充训练报告](training-v3-report.md)。

## 选择配置

| 任务 | 当前配置 | 初始化来源 |
| --- | --- | --- |
| 小节定位与谱面类型 | `layout/configs/train_joint_v3.yaml` | 上一版四类模型的训练检查点 |
| 曲名、作者、调弦和速度 | `document_info/configs/train_v3_headers.yaml` | 谱面信息 v2 LoRA |
| 小节音符与节奏 | `measure_ocr/configs/train_v3.yaml` | 小节 v2 LoRA |

配置中的数据路径对应已记录的数据版本。更换语料时，应同时修改数据、缓存和输出路径；新训练写入 `output/`，选定后再更新 `weights/`。默认模型与历史版本见[模型目录](../weights/README.md)。

## 训练 GLM-OCR

两个入口使用 LLaMA-Factory，接受 `--config` 和 `key=value` 覆盖参数。以下配置使用 8 张 H100；小节任务全局批量 256，谱头任务全局批量 96。

先生成缓存。修改标签、提示词或前文规则后必须换用新的缓存目录，`tokenized_path` 本身不是预处理后退出的开关。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 uv run --no-sync python -m shared.tokenize_training \
  --config measure_ocr/configs/train_v3.yaml \
  --output database/gp8_joint_v3/datasets/measure_ocr/tokenized_v3_context_checked
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 uv run --no-sync python -m shared.tokenize_training \
  --config document_info/configs/train_v3_headers.yaml \
  --output database/gp8_headers_v3/datasets/tokenized_info_mixed
```

依次训练两个任务：

```bash
PATH="$PWD/.venv/bin:$PATH" FORCE_TORCHRUN=1 NPROC_PER_NODE=8 OMP_NUM_THREADS=1 \
  uv run --no-sync python -m measure_ocr.train --config measure_ocr/configs/train_v3.yaml
PATH="$PWD/.venv/bin:$PATH" FORCE_TORCHRUN=1 NPROC_PER_NODE=8 OMP_NUM_THREADS=1 \
  uv run --no-sync python -m document_info.train --config document_info/configs/train_v3_headers.yaml
```

本次使用 Torch 2.14.0+cu130、Transformers 5.8.0、BF16 和 FlashAttention 2.8.3.post1。其他环境可传 `flash_attn=sdpa`；减少 GPU 数量时需重新计算全局批量。并行运行多个分布式任务时，分配不同 GPU 和 `MASTER_PORT`。

H100 的 FA2 从源码针对 SM90 编译，使用 CUDA 13.0、C++20，以及 `MAX_JOBS=32 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE`。安装后应检查真实样本的前向、反向和最长输入显存占用。

## 训练版面模型

```bash
tools/paddlex-venv/bin/python -m layout.train -c layout/configs/train_joint_v3.yaml
```

该配置需要 `output/pp_doclayout_v3_score_typed/best_model/best_model.pdparams`。这是保留的训练检查点，仓库中的推理权重不能直接替代。已有其他兼容检查点时，用 `-o Train.pretrain_weight_path=/path/to/checkpoint.pdparams` 指定。使用 Paddle 保存的 `best_model.pdparams`（EMA 选择权重）；`.pdema` 是恢复训练的原始参数。

训练入口按验证集 bbox AP 选择检查点，并保留 `gt_read_order` 字段。训练进程正常退出后再复制导出的最佳模型，确保文件写入完成。

历史训练路径依次为纯 TAB、三模式两类、三模式四类。初始化和配置保存在各[历史模型卡](../weights/README.md#历史版本)及对应 YAML 中；两类转四类的工具是 `layout.initialize_typed`。

## 评测版面与谱面信息

版面评测使用实际阈值和后处理，分别报告框 AP、召回率、小节数正确页面及类型准确率：

```bash
tools/paddlex-venv/bin/python -m layout.evaluate \
  --model-dir weights/pp_doclayout_v3_score_joint_v3 \
  --dataset-dir database/gp8_joint_v3/datasets/layout_typed \
  --split test --postprocess --threshold 0.25 \
  --output output/evaluation/layout.json
```

`--mode tab|notation|both` 可筛选谱面，`--device gpu:7` 可指定设备。不带 `--postprocess` 时评估原始检测框，应与实际流程的指标分别记录。

谱面信息按字段完全匹配评测：

```bash
uv run --no-sync python -m document_info.evaluate \
  --dataset database/gp8_headers_v3/datasets/info_mixed/document_info_test.json \
  --output output/evaluation/info.jsonl
```

## 评测小节识别

使用正确裁图和标注前文，评估独立小节：

```bash
uv run --no-sync python -m measure_ocr.evaluate_parallel \
  --manifest database/gp8_joint_v3/datasets/measure_ocr/manifests/test.jsonl \
  --context-source gold --max-samples 1800 --batch-size 8 \
  --output output/evaluation/measure-gold
```

使用正确裁图和模型预测前文，评估连续识别：

```bash
uv run --no-sync python -m measure_ocr.evaluate_parallel \
  --manifest database/gp8_joint_v3/datasets/measure_ocr/manifests/test.jsonl \
  --context-source predicted --max-samples 0 --max-sources 8 --batch-size 1 \
  --output output/evaluation/measure-sequences
```

并行入口默认使用 8 张 GPU，可通过 `--gpus 0,1` 等参数调整。连续评估必须保留完整曲谱，从首小节开始，因此要求 `--max-samples 0` 和 `--batch-size 1`。失败输出替换为带待检查标记的休止占位，续跑会重建预测前文。

新旧模型比较时保持样本、解码参数、重试次数和 batch size 一致。BF16 批量运算可能改变边缘 token 的选择，不能混用逐条与批量结果。扫描退化集应单独报告，它衡量模拟退化下的表现。

| 指标 | 含义 |
| --- | --- |
| `core_exact_rate` | 小节元数据、节奏、音符字段全部匹配 |
| `exact_match_rate` | 规范化小节文本完整匹配，含奏法 |
| `note_fields_exact_rate` / `rhythm_exact_rate` | 分别比较音符字段和节奏 |
| `raw_overall` / `raw_by_mode` | 重试后、占位处理前的 OCR 输出指标 |
| `overall` / `by_mode` | 交给后续流程的最终小节指标 |

报告格式与约束合法率时使用 `raw_*`，并列出 `needs_review`，避免把休止占位计为模型成功识别。上述两种小节评测都以正确裁图为输入，页面到 GP5 的表现需要另测。

## 评测完整流程

```bash
uv run --no-sync python -m pipeline.evaluate \
  --cases examples/cases.json --output output/evaluation/examples \
  --layout-python tools/paddlex-venv/bin/python
```

案例清单包含 `inputs`、`expected_m2`（参考小节文本路径）、`mode`，可指定 `layout_source`；文件路径相对案例清单。报告记录模型、依赖、输入哈希、耗时、峰值已分配显存和待检查小节。小节数量一致时按阅读顺序比较；数量不一致时记录定位失败。

`examples/` 是四小节功能示例。正式评测还应记录独立来源、整谱指标和人工修订量。`release_gate.json` 中的阈值是目标，实测结果在评测报告中。

## 更新默认模型

先按验证集选择权重，再运行独立测试。更新 `weights/manifest.json`、模型卡、训练配置、来源划分哈希和评测记录，最后执行 `uv run --no-sync guitarocr-check --hashes`。标签或运行逻辑改变后需重新检查续跑签名，旧预测不能直接作为新运行的结果。
