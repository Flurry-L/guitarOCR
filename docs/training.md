# 数据、训练与评测

训练依赖留在开发环境；桌面端使用 GGUF 和 ONNX。

## 模型文件

以下目录均位于 `weights/`：

| 目录 | 内容 |
| --- | --- |
| `score_ocr/merged/` | 可训练的 OCR 权重、词表、图像策略、信息适配器和拍调号检查点 |
| `score_ocr/gguf/` | 部署用语言模型、视觉编码器和信息适配器 |
| `auxiliary/` | 部署用 ONNX 和可见奏法阈值配置 |
| `layout/` | 可继续导出的 Paddle 推理模型 |
| `checkpoints/` | 按职责保存的辅助网络源检查点 |

`checkpoints/chord_evidence/` 的 primary / secondary 同时参与一致性判定；名称定位使用单个 `staff_label/` 模型。职责见[识别机制](../README.md#识别机制)。

`distribution.json` 是部署文件清单，本地槽位为 `current`。辅助模型只保存在 `auxiliary/`，`score_ocr/merged/capabilities.json` 统一声明所用模型；导出到独立目录时再收集依赖。输入输出约定由代码和 ONNX 内嵌元数据提供。`score_image_policy.json` 规定等比缩放等输入行为。采用模型后替换对应职责文件，清理其余检查点、渲染副本和缓存，不按实验代次复制模型目录。

## 数据生成

训练数据统一写入 `database/score/`，包含 `train.jsonl`、`validation.jsonl`、`test.jsonl`、数据清单和 `images/`。图片路径相对数据目录。原谱与来源分组放在 `database/sources/`，检查点和评测输出写入 `output/`，这些生成内容不进入 Git。

### 数据与真值

- 同一原曲及其排版、字体、移调和扫描变体留在同一 split；独立 test 不参与调参和阈值选择。
- 音符标签对应实际打印内容。五线谱输出书面音高，TAB 输出可见弦品；不要从源文件复制不可见的隐藏参数。
- 裁图、训练和推理均保持长宽比。上下文只帮助解释目标小节，不混入相邻小节答案。
- PM、let ring、装饰音和鼓件等由实际渲染证据确认可见性。看不清的字段保持未知，不用“没有”替代。
- 结构真值保留乐器、谱表和小节对应关系；图示库与实际拍点分开标注。

`research/data/run.py` 组织选曲、原生谱面渲染及裁图；`build_measure_data.py` 和 `training_samples.py` 生成监督样本；`technique_visibility.py`、`pack_score.py` 检查谱面可见性。通用造谱使用 `synthetic_scores`（GP）和 `engraved_scores`（MusicXML / Verovio）；后者同时生成小节、拍调号、信息和结构训练数据，包含不同字体、谱行间距、乐器与节奏。

信息样式使用 `information_styles`，谱号和拍调号裁图使用 `symbols clef` / `symbols signature`；和弦造谱使用 `synthetic_scores --profile chords`，共享 GP 保存与来源划分流程。标题增强集中在 `headers` 的 `text` / `layout` 子命令。各生成器输出新样本，通过 `pack_score` 与原生样本混合，不在生成器里绑定某轮实验的回放比例或旧数据目录。

小节训练统一读取目标、上一小节、下一小节三张图片，不生成上一小节的答案文本。五线谱与混合谱的书面音高由实际 GP8 谱号、移调和变调夹状态产生；缺少原生字段或音高基准时拒绝生成，不按乐器名猜八度。旧导出缺少 `display_mode` 时也必须重做。

### 安装导出环境

数据生产需要用户提供 Guitar Pro 8 运行目录。Windows 安装 Python；Linux 另需可运行的 Wine、Xvfb、Windows Python 和初始化好的 Wine prefix。原生 DLL 的构建见 [GP8 导出器](gpbridge-build.md)。这些依赖仅用于数据生产。

alphaTab 合成渲染需要安装 research 的训练依赖：`uv sync --locked --package guitarocr-research --python 3.11 --extra training`，然后运行 `uv run --no-sync python -m playwright install chromium` 安装渲染浏览器。

从原始谱文件生成数据，先在临时目录导出和校验，再汇总到训练入口：

```bash
python -m research.data.run \
  --corpus database/sources/files --source-catalog database/sources/catalog.json \
  --output output/source_data \
  --runtime /path/to/GuitarPro8 \
  --wine-prefix-template /path/to/wine-prefix \
  --wine-python /path/to/windows-python/python.exe
```

`--source-catalog` 让重新选择的原谱继续使用保留的 family / split；缺失或冲突的来源记录会被拒绝。只有全新语料才省略它并建立新的划分。需要只调用导出器时使用 `python -m research.data.export_scores --help`；manifest 每条记录都要明确 `display_mode`，输出包含 PDF、`layout.json` 和 `official-score.json`。Windows 运行导出器无需 Wine 参数。

合成谱写入 `DIR` 后，使用 `export_scores --manifest DIR/manifest.jsonl --source-root DIR/sources --output-dir DIR/native-export`，并传入同样的运行环境参数；随后执行 `research.data.run --output DIR --phase crop` 和 `--phase datasets`。源谱与导出目录相互独立。需要核对 GP8 导入是否改变音符时，运行 `python -m research.data.native_alignment --source DIR --output DIR/native-alignment.json`。

### 汇总与检查

各任务生产并完成来源分组后，通过 `pack_score` 汇总到统一目录。每个 split 可以传入多个 JSON 数组或 JSONL 文件，直接混合小节、标题、谱表等任务；不另设补训混合脚本，也不隐式重复某类样本。小节 manifest 提供可见性及几何证据。未验证的样本被丢弃，仅留下拒绝原因计数；图片复制到新数据目录。

```bash
python -m research.data.pack_score \
  --train /path/to/measures_train.jsonl /path/to/document_info_train.json \
  --validation /path/to/measures_validation.jsonl /path/to/document_info_validation.json \
  --test /path/to/test.jsonl --measure-manifests /path/to/measures.jsonl \
  --output database/score
python -m research.data.pack_score --validate database/score
```

输出目录必须尚不存在。生成器负责来源划分及真值质量；汇总器检查已有 `provenance.family/split` 是否冲突、同一图片是否跨 split，并检查小节可见性和图片引用。来源字段未写入对话样本的生成器仍须在上游清单中保证原曲不交叉。

首次训练前分词；数据、图片、词表或预处理配置变化后，删除旧分词缓存并重新生成。

谱行延续数据由 `research.data.part_continuation` 统一生成。`--engraved` 接收当前渲染器的 `{split}/*/score.json` 目录；`--native-info` 配合 `--exports` 接收原生谱表清单与 GP 导出目录。两种来源可同时提供，同曲不同轨在生成时组成候选组。生成器保持来源划分，每份数据只写一次；`--limit` 显式限制每种来源、每个 split 的文档数，默认使用全部。排版谱面的名称遮罩来自 SVG 标签节点，原生 GP 页面只使用真实图像及导出轨迹。

原生导出共用一条处理流程：`inventory` 整理页面与可见标注，`build_measure_data` 生成小节，`build_info_data` 生成标题、速度、谱号及可见移调标记，`build_layout_data` 生成版面数据。合成 GP 谱也通过同一流程处理。

## 主 OCR

使用 `research/configs/measures/train.yaml`，默认从 `weights/score_ocr/merged/` 继续训练，数据为 `database/score/`。数据更新后需要重新训练和独立评测，不能把数据修正当成模型准确率提升。

```bash
python -m research.data.pack_score --validate database/score
python -m research.training.measures \
  --config research/configs/measures/train.yaml --tokenize-only
python -m research.training.measures
```

缓存直接使用 LLaMA-Factory 的 `tokenized_path`，已有缓存会继续使用。数据、图片、模型输入策略或词表变化时，由使用者删除旧 `database/score/tokenized/` 后重新运行 `--tokenize-only`；程序不再计算输入指纹或自动拒绝旧缓存。`save_total_limit: 2` 限制训练过程检查点数量。训练产物写入 `output/score_ocr/`，允许复用输出目录；确认采用后再导出并替换 `weights/` 的对应职责文件。

训练与推理统一使用等比缩放、书面音高、可见监督和 `M` 小节文本。词表与模型参数一同训练和导出，token ID 必须与词嵌入对应。旧文本协议直接拒绝，不设置兼容解码分支。

## 版面与质量排序

`research/configs/layout/train.yaml` 从 `weights/checkpoints/layout/model.pdparams` 继续训练 11 类检测器，数据目录为重新生成的 `database/layout/`。输入等比补白，完整小节框必须覆盖所需内容。

版面数据统一通过 `build_layout_data` 构建。`--source` 接收一个或多个原生 inventory，`--coco` 接收已有的 11 类数据集，可混合真实谱面、排版变体和短页。它保留原曲 split，重排图片及标注 ID；旧的两类标签须重新生成。扫描增强继续使用 `scan_augment`，完整短页使用 `layout_complete_pages`。

```bash
python -m research.data.build_layout_data \
  --source output/source_data/inventory \
  --coco output/engraved_layout output/short_pages \
  --include-test --output database/layout
```

`--source` 和 `--coco` 至少提供一种。版面数据优先链接来源图片，训练期间保留这些来源目录。

质量模型只学习现有候选的质量。`research.training.layout_quality` 接收 JSONL 页面，每行包含 `image`、`family`、`split`、实际执行的 `views`、`candidates` 和带类别的 `truth` 框。训练和验证禁止共用来源 family，目标按同类别 IoU 连续计算。候选坐标保持不变。

```bash
python -m research.training.layout_quality train \
  --train /path/to/train-pages.jsonl --validation /path/to/validation-pages.jsonl \
  --device cuda --output output/layout_quality
python -m research.training.layout_quality export \
  --checkpoint output/layout_quality/model.pt --output output/layout_quality/layout_quality.onnx
```

## 专项网络

| 职责 | 训练 / 导出入口 |
| --- | --- |
| 拍号、调号 | `research.training.state`、`calibrate_state` / `research.export.export_auxiliary` |
| 结构关系 | `research.training.structure_relation`：`train`、`evaluate`、`export` |
| 谱表类型与弦数 | `research.training.staff_evidence`：`train`（包含校准和导出）、`evaluate` |
| 谱行延续、谱号 | `research.training.part_continuation`、`clef_spatial`：各自提供 `train`、`evaluate`、`export` |
| 名称定位、乐器名分类 | `research.training.staff_label`、`instrument_text`：训练使用 `--data DIR --output DIR`，导出使用 `--export MODEL_DIR` |
| 和弦图证据 | `research.training.chord_evidence` → `research.training.chord_calibration` 的 `reader`、`consensus` 子命令 → `research.training.chord_calibration export` |
| 事件定位 | `research.training.technique_event_grounding`：`train`、`export` |
| PM / let 可见关系 | `research.training.technique_span_relations`：`train`、`export` |

各入口及子命令的 `--help` 列出数据、初始化模型和输出参数。结构关系数据使用 `research.training.structure_relation prepare` 的 `--train`、`--validation`、`--test` 读取 `engraved_scores` / `information_styles` 输出的 `structure_{split}.jsonl`。每行直接保存干净页面 `image`、谱行中心 `centers`、`row_modes`、结构标签 `expected.rows`、原曲 `family` 和 `split`。`train --task system` / `train --task part` 分别训练现用的两个分支，`--initial` 加载对应检查点；`evaluate --checkpoint ... --manifest ...` 接受独立评测清单，`export` 显式接收两份权重和验证清单。

谱行延续的 `export --model DIR --data DIR` 在验证集上同时校准有名称和无名称的候选组，完成 ONNX 数值核对并保存阈值；`evaluate` 直接使用导出模型和阈值。拍调号训练通过多次 `--data DIR` 混合来源，只学习实际印出的拍调号。

奏法数据使用 `research.data.technique_scores --output DIR` 渲染合成谱，或 `research.data.technique_native_grounding --source-root GP_EXPORT --output DIR` 读取真实 GP 导出；一次生成 `grounding/` 和 `relations/` 两套任务清单，分别包含事件坐标和可见跨度。两种训练入口的 `--data` 都接受多个对应任务目录，直接混合来源；`export --candidate MODEL_DIR --data DATA_DIR ...` 使用验证集核对 ONNX 与训练网络的输出。独立评测与阈值选择集中在 `research.evaluation.techniques` 的 `evaluate` / `calibrate` 子命令，评测读取导出的 ONNX。

采用奏法模型时，将校准配置与两份网络一起安装；产品统一应用独立 PM / let 判断。版面检测同时需要检测网络和质量网络。

和弦训练通过 `--reader primary` 或 `--reader secondary` 选择两个现用读图器，`--initial` 严格加载相应检查点；奏法训练使用同一个联合网络。两者都更新全部参数，不提供历史架构迁移或实验冻结层开关。

和弦训练、校准和评测共用 `research.data.chord_evidence.ChordEvidenceData`：训练读取 train，选择模型和校准读取 validation，评测读取 test；来源与指型分组仍在生成器内保持隔离。`chord_calibration reader --model DIR --data database/score` 为单个读图器校准并导出，`consensus --model PAIR_DIR --data database/score --output agreement.json` 校准已导出的双读图器。`--font` 可指定当前使用的字体。

独立和弦评测使用 `python -m research.evaluation.chords --model DIR --data database/score --output output/chords.json`。也可用 `--cases crops.json` 单独检查真实裁图：JSON 数组每项为 `image`、`kind`（diagram／text／other）和图示的 `expected`，图片路径相对清单。评测直接比较参考答案，报告原始正确率、接受结果的精度及错误样例，不按文件名修改标签。

数据生产与评测仍需实际乐谱及独立来源；被清除的实验结果不作为可用数据。专项输出先写临时目录，通过真实输入评测及 Python / 原生推理一致性检查后，再更新当前模型和阈值配置。

谱表类型与弦数训练使用 `staff_evidence train` 子命令和生成器的专项清单，明确提供 `--train`、`--validation` 和 `--initialize`。每条记录须保留 `provenance.family` 与 `provenance.mode`；主 OCR 打包后的纯对话样本不提供这些来源字段，不能直接代替专项清单。

## 采用条件

同时检查目标任务和原有能力：整页漏检 / 合框、事件身份、音高与节奏、未知字段、声部和拍点关联、保存重开及 GP5 / MusicXML 导出。参数量或局部裁图分数不能代替整谱结果。

评测记录输入范围、所用模型、设备、页数及小节数，分别报告整页与参考裁图结果。阈值只用验证集选择，再用来源独立的测试集评估；失败、漏检、无法匹配、未知真值和人工补框均保留在报告中。当前小节测试集为 `database/score/test.jsonl`。编译检查和少量实际操作用于确认程序可运行，模型准确率需要上述独立评测。

整谱识别与评测使用正在运行的原生后端，模型资源配置见[应用构建](native-packaging.md)。例如：

```bash
cargo run --locked -p guitarocr-backend -- \
  --assets ui/workbench --resources desktop/src-tauri/resources/native \
  --models tools/model-cache --port 7860
# 在另一个终端运行：
python -m research.inference.run score.pdf --output output/recognition
python -m research.evaluation.pipeline \
  --scores /path/to/rendered_scores --output output/evaluation
```

默认端点为 `http://127.0.0.1:7860`，可用 `--endpoint` 指定本地或受信网络中的无账号 API。识别结果保存项目 ID、M 文本、ScoreIR 和可用的 MusicXML；在工作台打开同一项目即可核对和导出 GP5。中断后通过 `--session ID --output 新目录` 续跑，后端依据保存的 `ocr_task` 状态和小节日志继续。外部替换 PDF、模型或其他识别输入后，应重新开始识别。进度、修改和导出校验均由原生项目管理。

常用评测入口：

- `research.evaluation.layout`：实际版面检测链，报告漏框、多框和谱面类型。
- `research.evaluation.measures`：用小节 manifest 中与训练一致的参考裁图及状态，评测原始模型输出的音高、节奏和奏法。
- `research.evaluation.information`：标题、速度、谱表、谱号与注释的参考裁图评测，按任务选择信息适配器。报告原始输出的字段、文字和和弦指标，失败保留在分母中。
- `research.evaluation.pipeline`：调用实际软件做整页和多声部识别。`--scores` 接收渲染器的 `score.json`、包含这些文件的目录，或原生数据生成器的新小节 JSONL（须有 `source_pdf` 和完整小节序列）；漏检保留在分母中。

参考裁图评测使用 vLLM 直接读取尚未导出的训练权重；默认环境是 `tools/vllm-venv/bin/python`，可用 `GUITAROCR_VLLM_PYTHON` 指定。它评估单个模型的训练效果；整谱指标来自原生软件的 GGUF / ONNX 流程。评测输出由 `--output` 指定。

主模型格式转换入口在 `research/export/`，小型专项网络的 ONNX 导出与训练共用任务入口。本地模型清单按职责列出文件；发布下载附件时才生成文件大小和完整性校验，不把历史实验编号、训练输出路径或固定哈希写进产品逻辑。

统一 OCR 的 GGUF 导出使用 `python -m research.export.export_gguf --llama-cpp /path/to/llama.cpp --model weights/score_ocr/merged --output output/gguf/score_ocr`；信息适配器另用 `research.export.export_information_adapter` 转换。两者使用所提供的 llama.cpp 源码，不要求固定提交。
