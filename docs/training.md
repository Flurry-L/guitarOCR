# 训练与评测

先完成[数据生产](data.md)和[训练环境安装](#训练环境)。训练数据、缓存与检查点不随仓库提供，需要先生成。实测结果见[评测报告](model-evaluation.md)。

## 入口与产物

- 共享视觉语言模型：`research.training.measures` 同时训练小节与谱面信息任务
- 版面检测：`research.training.layout` 使用独立 Paddle 配置和环境
- 拍号／调号辅助分类器：`research.training.state`；`research.training.mtp` 是主模型后的可选蒸馏步骤，不属于普通应用启动
- 数据预处理：`research.training.tokenize_training` 只建 token 缓存；数据生产与来源划分见[数据说明](data.md)，不在训练器内重做

入口负责参数与任务选择，`research/training/training.py` 负责 LLaMA-Factory 启动和配置边界，`research/training/worker.py` 负责模型专用训练策略。项目自定义选项由训练 worker 消费，不透传给 LLaMA-Factory；关闭对应特性仍可保留配置中的参数。 拍号数据读取统一在 `research/data/signature_data.py`，训练和评测不再互相导入；MTP 层结构与检查点映射在 `research/models/mtp.py`，蒸馏循环留在 `research/training/mtp.py`。OCR 训练、拍号训练／评测、MTP 和 token 缓存入口的 `--help` 不加载训练框架；版面训练仍使用 PaddleX 自己的命令行。

检查点先写 `output/`，评测后再使用 `research.export.export_glm`、`research/export/export_auxiliary.py` 或 `research/export/export_gguf.py` 转为部署格式，最后更新 `weights/` 的模型与分发清单。训练依赖不进入桌面应用资源，部署模型也不等于可继续训练的检查点。这里只列训练／评测契约，设备支持和首次下载见[安装说明](setup.md)。

## 训练环境

训练使用独立的 `.venv`。先安装训练依赖，再安装 LLaMA-Factory `0.9.6.dev0` 的固定源码提交：

```bash
uv sync --locked --package guitarocr-research --python 3.11 --extra glm-ocr --extra training --extra dev
git clone https://github.com/hiyouga/LLaMA-Factory.git tools/LLaMA-Factory
git -C tools/LLaMA-Factory checkout 97b32d3133b501432141a82949d5c7bc4d94f23a
uv pip install --python .venv/bin/python -e tools/LLaMA-Factory
```

安装训练依赖后，训练使用 `uv run --no-sync`。需要更新本项目的可编辑安装时，执行 `uv pip install --python .venv/bin/python --no-deps -e ./scorelib -e ./gpbridge -e ./research`，保留已装的训练框架。

### Paddle 训练组件

只有训练或重新导出版面模型时需要 Paddle；部署使用 ONNX。建立独立环境，并安装 PP-DocLayoutV3 使用的 PaddleDetection 训练组件：

```bash
uv venv --python 3.11 tools/paddlex-venv
uv pip install --python tools/paddlex-venv/bin/python paddlepaddle-gpu==3.2.0 \
  --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install --python tools/paddlex-venv/bin/python 'paddlex[ocr]==3.7.2' \
  -c research/configs/layout/constraints.txt
PIP_CONSTRAINT="$PWD/research/configs/layout/constraints.txt" \
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
tools/paddlex-venv/bin/paddlex --install PaddleDetection
```

`research/configs/layout/constraints.txt` 保持 NumPy、OpenCV、pycocotools 与当前工作环境一致。初始化检查点和训练命令见下文。

## 训练配置

OCR 共用 `research/configs/measures/train.yaml`，以 `weights/score_ocr/merged` 为起点，同时训练小节、声部分组、页眉、谱号及标注任务。导出 LoRA 时必须使用本次训练的同一基座。版面配置单独位于 `research/configs/layout/train.yaml`，需要兼容的十一类训练检查点，放在 `tools/models/layout-checkpoint.pdparams`。仓库中的 `inference.pdiparams` 是推理权重，不能替代训练检查点。

下面使用[数据生产](data.md#混合后继续训练)中的混合数据目录。更换语料时同时修改数据、缓存和输出路径，并保留已有乐器、排版、谱号和移调任务的样本。训练结果写入 `output/`，评测后再更新 `weights/`。已发布权重的实际训练参数记录在各模型目录的 `training.json` 中。

## 训练 GLM-OCR

训练入口使用 LLaMA-Factory，接受 `--config` 和 `key=value` 覆盖参数。每卡批量与梯度累积见配置文件；全局批量随 GPU 数量变化。整页分轨任务使用更高的图像分辨率，训练 MTP 时也应传入相同的 `--image-max-pixels 2016000`。

先生成缓存。修改标签、提示词或前文规则后必须换用新的缓存目录，`tokenized_path` 本身不是预处理后退出的开关。

```bash
uv run --no-sync python -m research.data.unified_data --assemble-training \
  --inputs database/unified_score --output database/unified_score_visible \
  --staff-data database/instrument_training/datasets/staff_visible
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 uv run --no-sync python -m research.training.tokenize_training \
  --config research/configs/measures/train.yaml \
  --output database/unified_score_visible/tokenized
```

混合输入保留完整小节语料，增加指法图、五线谱／TAB 配对与页眉的训练比例；验证及测试集不重复采样。`--assemble-training` 组合已生成的 `ocr_*`、`extra_*`、`refinement_*`、`paired_complete_*` 和 `headers_*` 数据。其中扩展标注由 `research.data.unified_data --refinement` 生成，页眉由 `research.data.header_rehearsal` 生成。和弦图使用实际印出的点、指法和横按标注，不能从和弦名推导指法。

迁移已有数据时，可传 `--inputs 原任务目录 --output 新目录 --text-sources 文字来源映射.json`，在组装时仅替换已确认有差异的文字字段。输出目录中已有的 `text_sources.json` 会自动使用。映射按图片路径保存 `[源标签路径, 排版模式, 原始小节索引]`。文字来自实际排版模型；和弦名称另与 PDF 中真正印出的文字对应，例如内部的 `C7M` 可能印成 `Cmaj7`。对应时要求小节内和弦数量和根音一致，节拍位置仍来自原始事件。训练的 `dataset_dir` 与 `tokenized_path` 必须同时指向新数据，不能沿用修改前的缓存。

训练共享 OCR：

```bash
PATH="$PWD/.venv/bin:$PATH" FORCE_TORCHRUN=1 NPROC_PER_NODE=8 OMP_NUM_THREADS=1 \
  uv run --no-sync python -m research.training.measures --config research/configs/measures/train.yaml
```

已发布模型使用 Torch 2.14.0+cu130、Transformers 5.8.0、BF16 和 FlashAttention 2.8.3.post1。其他环境可传 `flash_attn=sdpa`；减少 GPU 数量时需重新计算全局批量。并行运行多个分布式任务时，分配不同 GPU 和 `MASTER_PORT`。

小节训练按每卡 token 预算组批，相邻样本共享图像编码；训练和推理从模型的 `score_image_policy.json` 读取相同的等比缩放及白边填充规则。音乐词表的嵌入和输出层使用独立学习率，压缩字段按原 token 长度加权，避免一个音高或节奏字段压缩后在损失中权重过低。词表扩展由 `research.models.music_vocab --structured` 完成，只从训练标签补充高频字段；新增词元先进行音乐事件序列化预热，再参与图像训练。

主模型导出后，可用 `research.training.mtp --model 合并模型 --tokenized 训练缓存 --output 输出目录 --epochs 1 --token-budget 131072 --image-max-pixels 2016000 --eval-every 400 --early-stopping-patience 2` 蒸馏原生 MTP 层。训练使用与部署一致的图像位置编码，验证覆盖完整验证集。`validation_agreement` 是草稿与主模型的 token 一致率，实际接受率和速度需另用完整曲谱推理测量。

H100 的 FA2 从源码针对 SM90 编译，使用 CUDA 13.0、C++20，以及 `MAX_JOBS=32 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE`。安装后应检查真实样本的前向、反向和最长输入显存占用。

## 训练版面模型

完成基础语料及和弦数据生成后，合并原生和组合总谱的版面标注：

```bash
uv run --no-sync python -m research.data.unified_layout --output database/unified_layout
uv run --no-sync python -m research.data.unified_layout --base database/unified_layout \
  --chords database/chord_training/layout --output database/unified_layout_complete
uv run --no-sync python -m research.data.header_layout --source database/unified_layout_complete \
  --output database/unified_layout_headers
tools/paddlex-venv/bin/python -m research.training.layout -c research/configs/layout/train.yaml
```

用 `-o Train.pretrain_weight_path=/path/to/checkpoint.pdparams` 指定其他训练检查点。使用 Paddle 保存的 `best_model.pdparams`；`.pdema` 是恢复训练的原始参数。

`header_layout` 使用原有来源划分，把字体边界和原生 PDF 文字坐标标注的谱头接到小节页面上。曲名、副标题、署名、调弦文字及其他谱头文字新增五类；合成前移除旧页未标注的谱头，保留已有音乐区域。

从旧六类检查点续训时，先运行 `research.inference.layout.extend_classes --source /path/to/best_model.pdparams --output tools/models/layout-checkpoint.pdparams --classes 11`，保留原六类权重并扩展分类头。从官方预训练模型开始则直接指定官方检查点，类别数仍由十一类数据配置决定。

训练入口按验证集 bbox AP 选择检查点，并保留 `gt_read_order` 字段。训练进程正常退出后再复制导出的最佳模型，确保文件写入完成。

## 评测版面与谱面信息

版面评测使用实际阈值和后处理，分别报告框 AP、召回率、小节数正确页面及类型准确率：

```bash
tools/paddlex-venv/bin/python -m research.evaluation.layout.evaluate \
  --model-dir weights/layout \
  --dataset-dir database/unified_layout_headers \
  --split test --postprocess --threshold 0.25 \
  --output output/evaluation/layout.json
```

`--mode tab|notation|both` 可筛选谱面，`--device gpu:7` 可指定设备。不带 `--postprocess` 时评估原始检测框，应与实际流程的指标分别记录。

谱面信息按字段完全匹配评测：

```bash
uv run --no-sync python -m research.evaluation.information.evaluate_parallel \
  --dataset database/unified_score/headers_test.jsonl \
  --adapter weights/score_ocr --gpus 0,1,2,3 \
  --output output/evaluation/headers
```

## 评测小节识别

按输入选择入口，报告不能混用：`research.evaluation.measures.evaluate` / `evaluate_parallel` 对固定裁图样本评测（默认标注前文，适合旧模型对照）；`research.evaluation.measures.evaluate_scores` 对完整小节序列使用预测前文；`research.evaluation.pipeline.evaluate_scores` 从同一曲源清单找到完整 PDF，额外评测检测和谱面信息；`research.evaluation.pipeline.evaluate_ensembles` 消费组合总谱目录；`research.evaluation.pipeline.evaluate` 消费手工案例清单。并行入口只是分片与汇总，不是另一套模型。

新模型先批量读取印刷拍号和调号，再使用当前及相邻小节图像独立解码，最后统一连接延音线。评测保留每首曲谱的完整序列，使用模型预测的拍号与调号，不提供标注前文。裁图、乐器、调弦与移调上下文来自标注；完全自动流程另用完整 PDF 评测。

```bash
uv run --no-sync python -m research.evaluation.measures.evaluate_scores \
  --manifest database/score_support_rehearsal/manifest_test.jsonl \
  --adapter weights/score_ocr --gpus 0,1,2,3,4,5,6,7 \
  --max-scores 0 --batch-size 64 \
  --output output/evaluation/measure-scores
```

vLLM 模型路径从适配器的 `inference.json` 读取；也可用 `--model` 指定合并模型。`--speculative-tokens` 控制 MTP 草稿长度，设为 `0` 可禁用；应在相同完整曲谱上比较速度及输出质量后选择。完整曲谱按小节数分配给各张 GPU，单首曲谱内批量解码小节。`--legacy` 保留旧模型串行前文方案的对照入口。

比较模型时保持样本、解码参数、重试次数和 batch size 一致。BF16 批量运算可能改变边缘 token 的选择，不能混用逐条与批量结果。扫描退化集应单独报告，它衡量模拟退化下的表现。

拍号／调号分类器的图片预处理默认使用最多 4 个 CPU 线程，保持原变换和输入顺序；设置 `GUITAROCR_STATE_PREPROCESS_WORKERS=1` 可固定为单线程。计时对照需同时记录该设置，区分模型解码和图片预处理带来的收益。

加入其他乐器后，固定验证和测试清单应包含每个乐器、排版及弦数组合，并按来源分散抽样。小节报告的 `by_instrument` 和 `by_strings` 用于检查新乐器是否改善、已有吉他能力是否下降。鼓按可见符号的规范编号比较，钢琴单谱表的结果不能推广到双谱表或总谱。连续识别另用完整来源测试，不能把零散裁图拼成序列。

`by_pitched_family` 将钢琴及键盘音色（GM 0 至 7）与其他旋律乐器分别计分，缺少 MIDI 编号的旋律小节列入 `unknown`，避免漏计。对照模型时每组样本数必须一致。源文件的 MIDI 音色只用于评测分组，不作为模型输入；移调乐器的结果不能代替钢琴指标。

| 指标 | 含义 |
| --- | --- |
| `note_content_onset_duration.f1` | 音符内容、声部、起点和实际时值同时匹配；五线谱比较音高，纯 TAB 比较弦品 |
| `fingering_onset_duration.f1` | TAB 或混合谱的弦、品、声部、起点和时值同时匹配 |
| `technique.f1` | 奏法及参数是否附着在正确音符或事件上 |
| `core_exact_rate` | 小节元数据、节奏、音符字段全部匹配 |
| `exact_match_rate` | 规范化小节文本完整匹配，含奏法 |
| `note_fields_exact_rate` / `rhythm_exact_rate` | 分别比较音符字段和节奏 |
| `raw_overall` / `raw_by_mode` | 重试后、占位处理前的 OCR 输出指标 |
| `overall` / `by_mode` | 交给后续流程的最终小节指标 |

报告格式与约束合法率时使用 `raw_*`，并列出 `needs_review`，避免把休止占位计为模型成功识别。上述两种小节评测都以正确裁图为输入，页面到 GP5 的表现需要另测。

## 评测完整流程

```bash
uv run --no-sync python -m research.evaluation.pipeline.evaluate \
  --cases /path/to/cases.json --output output/evaluation/pages \
  --layout-python tools/paddlex-venv/bin/python
```

案例清单包含 `inputs`、`expected_m2`（参考小节文本路径）、`mode`，可指定 `layout_source`；文件路径相对案例清单。报告记录模型、依赖、输入哈希、耗时、峰值已分配显存和待检查小节。小节数量一致时按阅读顺序比较；数量不一致时记录定位失败。

评测候选权重时用 `--adapter`、`--info-adapter`、`--layout-model-dir` 指定目录，无需先替换发布文件。案例可明确提供 `instrument`、`tuning` 或 `midi_program`；报告会列出这些人工信息，区分自动识别和人工指定的条件。单个案例失败会保留错误及已完成阶段的结果。

评测应使用预留测试曲源的完整谱面，记录独立来源、整谱指标和人工修订量。

## 更新默认模型

先按验证集选择权重，再运行独立测试。更新 `weights/manifest.json` 中的文件路径和大小、训练参数及实际评测记录，再执行 `uv run --no-sync guitarocr-check`。标签或运行逻辑改变后需重新检查续跑签名，旧预测不能直接作为新运行的结果。

共享 OCR 的 `capabilities.json` 声明已训练的任务和状态读取器。导出时保留这些配置；新增任务须完成训练及评测后再声明支持。更新权重后应检查新任务和已有任务的续跑。

音高上下文使用 `"pitch_context":true`，并学习 `ottava` 事件标记。声明 `"written_pitch":true` 的模型输出记谱音高，运行时根据明确的移调量和事件八度标记换算，再交给校对和导出。训练数据用 `research.data.written_pitch_data` 构建。继续训练时保留三种排版和已有乐器的样本，分别验证小节识别、移调量、谱号、八度范围以及 GP5 导出后的实际音高。

版面类别顺序见 `research/common/layout_labels.py`：三种小节、速度、谱号、一般标注，以及曲名、副标题、署名、调弦和其他谱头文字。一般标注包含和弦、指法图、奏法文字及真正的移调说明，由共享 OCR 分类后决定用途；谱头文字按区域类型逐框转写。历史 `transposition_region` 输入仍可读取。
