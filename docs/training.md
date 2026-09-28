# 训练与评测

先完成[数据生产](data.md)和[训练环境安装](#训练环境)。训练数据、缓存与检查点不随仓库提供，需要先生成。实测结果见[评测报告](model-evaluation.md)。

## 训练环境

先按[安装说明](setup.md#手动安装模型推理)安装推理依赖，再安装 LLaMA-Factory `0.9.6.dev0` 的固定源码提交：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git tools/LLaMA-Factory
git -C tools/LLaMA-Factory checkout 97b32d3133b501432141a82949d5c7bc4d94f23a
uv pip install --python .venv/bin/python -e tools/LLaMA-Factory
```

安装训练依赖后，训练使用 `uv run --no-sync`。需要更新本项目的可编辑安装时，执行 `uv pip install --python .venv/bin/python --no-deps -e .`，保留已装的训练框架。

### Paddle 训练组件

先安装[独立 Paddle 环境](setup.md#独立-paddle-环境)，再安装 PP-DocLayoutV3 使用的 PaddleDetection 训练组件：

```bash
PIP_CONSTRAINT="$PWD/layout/constraints.txt" \
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
tools/paddlex-venv/bin/paddlex --install PaddleDetection
```

`layout/constraints.txt` 保持 NumPy、OpenCV、pycocotools 与当前工作环境一致。初始化检查点和训练命令见下文。

## 训练配置

每项任务使用一份 `configs/train.yaml`。GLM-OCR 配置以仓库中对应的 `merged/` 模型为起点，训练新的 LoRA；导出时必须使用同一合并模型作为基座。版面配置需要兼容的六类训练检查点，放在 `tools/models/layout-checkpoint.pdparams`。仓库中的 `inference.pdiparams` 是推理权重，不能替代训练检查点。

下面使用[数据生产](data.md#混合后继续训练)中的混合数据目录。更换语料时同时修改数据、缓存和输出路径，并保留已有乐器、排版、谱号和移调任务的样本。训练结果写入 `output/`，评测后再更新 `weights/`。已发布权重的实际训练参数记录在各模型目录的 `training.json` 中。

## 训练 GLM-OCR

两个入口使用 LLaMA-Factory，接受 `--config` 和 `key=value` 覆盖参数。每卡批量与梯度累积见配置文件；全局批量随 GPU 数量变化。整页分轨任务使用更高的图像分辨率，训练 MTP 时也应传入相同的 `--image-max-pixels`。

先生成缓存。修改标签、提示词或前文规则后必须换用新的缓存目录，`tokenized_path` 本身不是预处理后退出的开关。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 uv run --no-sync python -m shared.tokenize_training \
  --config measure_ocr/configs/train.yaml \
  --output database/score_support/measure_canonical
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 uv run --no-sync python -m shared.tokenize_training \
  --config document_info/configs/train.yaml \
  --output database/score_support/info_crop_rehearsal/balanced
```

依次训练两个任务：

```bash
PATH="$PWD/.venv/bin:$PATH" FORCE_TORCHRUN=1 NPROC_PER_NODE=8 OMP_NUM_THREADS=1 \
  uv run --no-sync python -m measure_ocr.train --config measure_ocr/configs/train.yaml
PATH="$PWD/.venv/bin:$PATH" FORCE_TORCHRUN=1 NPROC_PER_NODE=8 OMP_NUM_THREADS=1 \
  uv run --no-sync python -m document_info.train --config document_info/configs/train.yaml
```

已发布模型使用 Torch 2.14.0+cu130、Transformers 5.8.0、BF16 和 FlashAttention 2.8.3.post1。其他环境可传 `flash_attn=sdpa`；减少 GPU 数量时需重新计算全局批量。并行运行多个分布式任务时，分配不同 GPU 和 `MASTER_PORT`。

H100 的 FA2 从源码针对 SM90 编译，使用 CUDA 13.0、C++20，以及 `MAX_JOBS=32 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE`。安装后应检查真实样本的前向、反向和最长输入显存占用。

## 训练版面模型

```bash
tools/paddlex-venv/bin/python -m layout.train -c layout/configs/train.yaml
```

用 `-o Train.pretrain_weight_path=/path/to/checkpoint.pdparams` 指定其他训练检查点。使用 Paddle 保存的 `best_model.pdparams`；`.pdema` 是恢复训练的原始参数。

若从官方预训练模型开始，可将该路径设为官方 `PP-DocLayoutV3_pretrained.pdparams`，类别数保持为 6，并按验证结果调整学习率和轮数。训练数据需包含谱号和移调框，构建方法见[数据生产](data.md#谱号与移调)。

训练入口按验证集 bbox AP 选择检查点，并保留 `gt_read_order` 字段。训练进程正常退出后再复制导出的最佳模型，确保文件写入完成。

## 评测版面与谱面信息

版面评测使用实际阈值和后处理，分别报告框 AP、召回率、小节数正确页面及类型准确率：

```bash
tools/paddlex-venv/bin/python -m layout.evaluate \
  --model-dir weights/layout \
  --dataset-dir database/scores/datasets/layout \
  --split test --postprocess --threshold 0.25 \
  --output output/evaluation/layout.json
```

`--mode tab|notation|both` 可筛选谱面，`--device gpu:7` 可指定设备。不带 `--postprocess` 时评估原始检测框，应与实际流程的指标分别记录。

谱面信息按字段完全匹配评测：

```bash
uv run --no-sync python -m document_info.evaluate \
  --dataset database/headers/datasets/info_mixed/document_info_test.json \
  --output output/evaluation/info.jsonl
```

## 评测小节识别

新模型先批量读取印刷拍号和调号，再使用当前及相邻小节图像独立解码，最后统一连接延音线。评测保留每首曲谱的完整序列，使用模型预测的拍号与调号，不提供标注前文。裁图、乐器、调弦与移调上下文来自标注；完全自动流程另用完整 PDF 评测。

```bash
uv run --no-sync python -m measure_ocr.evaluate_scores \
  --manifest database/parallel_score_corrected/manifest_test.jsonl \
  --adapter weights/measure_ocr --gpus 0,1,2,3,4,5,6,7 \
  --max-scores 0 --batch-size 32 \
  --output output/evaluation/measure-scores
```

vLLM 模型路径从适配器的 `inference.json` 读取；也可用 `--model` 指定合并模型。`--speculative-tokens 2` 启用已训练的 MTP 草稿层。每张 GPU 处理完整曲谱，单首曲谱内批量解码小节。`--legacy` 保留旧模型串行前文方案的对照入口。

比较模型时保持样本、解码参数、重试次数和 batch size 一致。BF16 批量运算可能改变边缘 token 的选择，不能混用逐条与批量结果。扫描退化集应单独报告，它衡量模拟退化下的表现。

加入其他乐器后，固定验证和测试清单应包含每个乐器、排版及弦数组合，并按来源分散抽样。小节报告的 `by_instrument` 和 `by_strings` 用于检查新乐器是否改善、已有吉他能力是否下降。鼓按可见符号的规范编号比较，钢琴单谱表的结果不能推广到双谱表或总谱。连续识别另用完整来源测试，不能把零散裁图拼成序列。

`by_pitched_family` 将钢琴及键盘音色（GM 0 至 7）与其他旋律乐器分别计分。源文件的 MIDI 音色只用于评测分组，不作为模型输入；移调乐器的结果不能代替钢琴指标。

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
  --cases /path/to/cases.json --output output/evaluation/pages \
  --layout-python tools/paddlex-venv/bin/python
```

案例清单包含 `inputs`、`expected_m2`（参考小节文本路径）、`mode`，可指定 `layout_source`；文件路径相对案例清单。报告记录模型、依赖、输入哈希、耗时、峰值已分配显存和待检查小节。小节数量一致时按阅读顺序比较；数量不一致时记录定位失败。

评测候选权重时用 `--adapter`、`--info-adapter`、`--layout-model-dir` 指定目录，无需先替换发布文件。案例可明确提供 `instrument`、`tuning` 或 `midi_program`；报告会列出这些人工信息，区分自动识别和人工指定的条件。单个案例失败会保留错误及已完成阶段的结果。

评测应使用预留测试曲源的完整谱面，记录独立来源、整谱指标和人工修订量。

## 更新默认模型

先按验证集选择权重，再运行独立测试。更新 `weights/manifest.json` 中的文件路径和大小、训练参数及实际评测记录，再执行 `uv run --no-sync guitarocr-check`。标签或运行逻辑改变后需重新检查续跑签名，旧预测不能直接作为新运行的结果。

谱面信息适配器完成首行乐器识别训练和评测后，在模型目录加入 `capabilities.json`，内容为 `{"staff_profile":true}`，运行时才会请求这项结果。旧适配器继续使用原有谱头与速度任务。服务端任务保留提交时的模型路径，更新权重后应检查新任务和已有任务的续跑。

移调训练还需在谱面信息和小节适配器中声明 `"pitch_context":true`。小节适配器同时学习 `ottava` 事件标记。声明 `"written_pitch":true` 的适配器对带音高前文的旋律五线谱输出记谱音高，运行时根据移调量和事件八度标记换算，再交给校对和导出。训练数据用 `datagen.written_pitch_data` 构建，不能只给旧适配器增加声明。继续训练时保留三种排版和已有乐器的样本，分别验证原有小节识别、移调量、谱号、八度范围以及 GP5 导出后的实际音高。

版面类别按顺序为 `measure_tab`、`measure_notation`、`measure_both`、`tempo_region`、`clef_region`、`transposition_region`。从四类权重继续训练时，先用 `python -m layout.extend_classes --source 原权重.pdparams --output 扩展权重.pdparams --classes 6` 保留已有分类参数，再把训练配置的 `num_classes` 设为 6。旧版面测试集没有新类别标注，新增类别须在带谱号和移调标注的测试集上单独报告。
