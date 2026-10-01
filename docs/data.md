# 数据生产

`database/` 保存生成数据，由 Git 忽略。页面使用 Guitar Pro 原生渲染，导出环境的安装步骤见[下文](#安装导出环境)。

## 选入口，不重复搭流水线

- 已有 GP 曲库：`research.data.run` 负责选源 → 原生渲染 → 小节裁图 → 任务数据集，按下面的 `--phase` 续做
- 已有人工确定的来源划分：先用 `research.data.prepare_catalog`，随后使用同一个 render / crop / datasets 流程
- 原创多乐器、双谱表与复调数据：`research.data.engraved_scores` 生成谱面及标签，再由任务构建器消费；不依赖 GP 原生导出
- 已有多组任务样本：`research.data.unified_data` 汇集共享 OCR 训练输入，`research.data.unified_layout` 汇集版面数据；训练从[训练说明](training.md)进入

数据层的交付是清单、图像、标签和 `dataset_info.json`，不启动优化器、不安装推理 runtime，也不改变用户项目。`research/data/catalog.py` 负责来源／family 划分；`research/data/training_samples.py` 负责训练消息序列化；各 builder 负责自己的可见标签。提示词与图像规范直接复用对应推理模块，避免训练和推理偷偷使用不同输入。

曲源标签、训练样本和项目 `score.json` 是不同用途的文件；乐谱语义共用[IR 约定](score-text.md#表示与职责边界)，不能把模型训练消息当成可编辑项目备份。Guitar Pro / Wine 只用于下述原生数据生产，用户识别已有 PDF 不需要这些工具。

## 生成三种排版

```bash
uv run --no-sync python -m research.data.run \
  --corpus /path/to/gp-files --output database/scores \
  --source-count 2000 --phase all --typed-measures \
  --runtime /path/to/GuitarPro8 \
  --wine-prefix-template /path/to/wine-prefix \
  --wine-python /path/to/wine-python/python.exe --workers 4
```

`all` 依次选源、渲染、生成小节数据，再生成版面和谱面信息数据。 `all` / `render` 在写输出前检查必需的渲染参数；独立阶段只导入自己需要的构建器。支持 GP3 / GP4 / GP5 / GTP，默认导出 TAB、五线谱和混合谱三种排版。可重复指定 `--mode tab|notation|both` 筛选。

默认选取 8 至 128 小节、至少 16 个音符的曲谱。可用 `--minimum-measures`、`--maximum-measures` 和 `--source-count` 调整选源范围。

```text
database/scores/
  labels/                  源曲谱标签
  prepared/                单音轨 GP5
  native-export/           原生 PDF、layout.json、official-score.json
  source_catalog.json      source_id、family、split
  source_splits.json       source_id 到 split 的映射
  crops/                   小节裁图
  manifests/               小节清单
  llamafactory/            小节模型训练输入
  inventory/               页面图与布局清单
  datasets/layout/         COCO 版面标注
  datasets/document_info/  谱面信息裁图和训练输入
```

已有渲染与裁图时，运行 `--phase datasets` 补建版面和信息数据。其他阶段及参数见 `python -m research.data.run --help`。

## 来源分组

版面与共享 OCR 的各项任务共用 `source_catalog.json`，由 `research/data/catalog.py` 统一创建和核对。同一曲谱及其转调、重排版变体应使用同一个 `family`，整体进入 train、validation 或 test。默认按文件哈希分组，内容相同但文件不同的曲谱需要提前指定 family。

完成 select 后，可在标签的 `family` 字段填写分组，再首次运行 crop / datasets。已有 catalog 必须包含全部选中来源，跨 split 的 family 会报错。修改分组后需重新生成对应数据。

已有明确划分的语料可提供 `sources` 列表，每项包含 `source_path`、`family`、`split`，再执行：

```bash
uv run --no-sync python -m research.data.prepare_catalog \
  --catalog /path/to/input_catalog.json --output database/scores --workers 8
```

随后按上文提供 runtime 和 Wine 参数执行 `--phase render`。单音轨 GP5 会原样复制，显示模式由原生导出器切换。

## 加入贝斯、鼓和其他乐器

[`010qwe/guitar`](https://huggingface.co/datasets/010qwe/guitar) 的 `TabDatasetGP5` 按乐器存放拆分后的音轨。先检查文件，再按曲源选取，不能将同曲的不同音轨随机分到训练和测试中。该数据的文件名末尾八位标识用于关联曲源；选源器还检查作者、曲名和音符指纹，并保留已有模型的来源划分。

```bash
uv run --no-sync python -m research.data.audit_instruments \
  --corpus /path/to/TabDataset --output database/instrument_audit.jsonl
uv run --no-sync python -m research.data.instrument_catalog \
  --audit database/instrument_audit.jsonl --splits /path/to/source_splits.json \
  --previous database/scores --output database/instruments
uv run --no-sync python -m research.data.prepare_catalog \
  --catalog database/instruments/input_catalog.json --output database/instruments
```

划分文件格式为 `{"source_splits":{"曲源标识":"train"}}`，取值还包括 `validation`、`test`，兼容 `dev`。输入 catalog 可用 `track_index` 指定原文件中的音轨。吉他和贝斯渲染三种排版，鼓与其他旋律乐器只渲染五线谱。为 `database/instruments` 执行上文的原生 render 后，先校验标签，再裁图：

```bash
uv run --no-sync python -m research.data.curate_instruments \
  --source database/instruments --output database/instrument_training
uv run --no-sync python -m research.data.run \
  --output database/instrument_training --phase crop
uv run --no-sync python -m research.data.curate_measure_data \
  --source database/instrument_training \
  --output database/instrument_training/datasets/measure_ocr --workers 16
```

鼓使用打击乐五线谱。标签中的音高字段保存 General MIDI 鼓件编号，不表示旋律音高。Guitar Pro 默认鼓组中，部分不同编号打印成相同符号，无法从谱图恢复原音色，统一为可见符号对应的编号：33→37，34/39/40→38，41→45。35 和 36 保持区分。没有验证过可见符号的编号，以及同时重叠成一个符号的重复鼓音，会剔除整个来源并记录原因。具体规则见 [`scorelib/python/scorelib/percussion.py`](../scorelib/python/scorelib/percussion.py)。

整理器还把源标签与 Guitar Pro 原生导入结果逐小节比较，剔除音高或起点发生变化的来源。GP5 导入可能补齐跨拍延音，也可能改变有问题的连音，不能只依据源文件解析成功就接受标签。这个检查不覆盖全部打印奏法。已有小节数据可用 `python -m research.data.native_alignment --source database/scores --output output/alignment.json` 核对；混合时传入 `--exclude-report output/alignment.json`，同步排除该来源的原图、重复难例和增强图，再重新生成训练缓存及固定评测子集。

少量钢琴或人声 GP 文件用变调夹字段记录整体升高的音程。选源时将它归入音符和装饰音的实际音高，原值保存在 `source_capo`；吉他、贝斯仍单独保存变调夹。修改这项标签规则后应重新选源和裁图，保留原来的曲源划分。

谱面信息另加一项任务，识别首行的乐器名称、打击乐谱号和 TAB 线数。标签依据实际打印内容，不用文件中隐藏的 MIDI 音色推测页面上的乐器；无名称的旋律五线谱标为通用音高谱，弦数留空。

首行输入保留开头的小节，并将左侧竖排乐器简称转正、放大；训练裁图和运行时使用同一个处理函数。输出同时保留实际印出的乐器名称，少见简称在训练中增加采样，验证和测试不重复。首行谱号上方漏检的短文字会单独重读，变调夹位置必须由完整的文字指令确认。

```bash
uv run --no-sync python -m research.data.build_staff_data \
  --source database/scores --source database/instrument_training \
  --output database/instrument_training/datasets/staff
```

`research.data.mix_measure_data`、`research.data.mix_info_data`、`research.data.mix_layout_data` 将新增样本与已有吉他数据混合，检查 family 是否跨划分。小节验证和测试按乐器、排版和弦数固定抽样；训练混合保留旧数据，评测仍分别报告各组结果。命令参数见各模块的 `--help`。

`research.data.ensemble_pages` 将保持原划分的原生谱面片段组合为总谱页面。`research.data.engraved_scores` 从独立乐谱中间表示生成多乐器、钢琴双谱表和复调作品，使用三种字体排版，同时输出小节、谱号、元信息和整页分轨标签。`research.data.ensemble_layout_data` 合并版面数据；`research.data.structure_data --engraved database/engraved_scores --compact --output database/score_support/info_structure_compact` 构建分轨和谱面信息数据；`research.data.score_support_data --engraved database/engraved_scores --rehearsal-focus --output database/score_support_rehearsal` 保留三种排版并增加复调、打击乐和奏法样本。再运行 `research.data.info_crops`，加入仅来自训练曲谱的偏移、缩放和压缩裁图。完整 PDF 由 `research.evaluation.pipeline.evaluate_ensembles` 评测。

组合页面保留小节原始宽高比，按同一时间位置最宽的小节分配列宽，并从源曲开头选取连续小节，使谱号、拍号和调号可见。`python -m research.data.ensemble_pages --output database/ensemble_quality` 默认生成 2,400／240／240 份训练、验证和测试总谱。`python -m research.data.quality_data --output database/score_quality_profiles` 将它与旧组合谱、复习样本、TAB 技法定位样本和音乐词表预热数据合并。技法定位按原生起点和实际主音品数字形关联，排除装饰音和颤音旁注。

数据构建器通过 `research.data.source_profile` 从原始 GP 文件恢复旧标签缺失的乐器、调弦和弦数，纠正旧贝斯样本的吉他默认谱号。重读已经组合的页面时保留页面打印的声部身份，只修复源图谱号与弦数信息；音乐目标不随元数据修复而改变。

## 谱号与移调

谱号裁图依据原生 PDF 字形标注。混合谱中的 TAB 符号单独标为 `tab`，不覆盖五线谱的谱号或改变音高。

从已完成原生音符核验的数据生成移调变体，保留原曲的 family 和划分：

```bash
uv run --no-sync python -m research.data.prepare_pitch_data \
  --source database/instrument_training --excluded output/alignment.json \
  --output database/pitch --per-group 200
uv run --no-sync python -m research.data.export_scores \
  --manifest database/pitch/manifest.jsonl --source-root database/pitch \
  --output-dir database/pitch_native --runtime /path/to/GuitarPro8 \
  --wine-prefix-template /path/to/wine-prefix \
  --wine-python /path/to/wine-python/python.exe --workers 8
uv run --no-sync python -m research.data.build_pitch_data \
  --source database/pitch --native-export database/pitch_native \
  --output database/pitch/datasets/native_pitch
```

生成器写入可见的整轨移调说明及局部 8va、8vb、15ma、15mb 标记，再由 Guitar Pro 排版。标注使用原生谱号框、八度线范围和 PDF 文字位置。标签核验会检查原生导入后的移调量、音高和起点；没有打印移调信息的移调乐器页面不进入小节训练。输出包含六类版面标注、信息裁图和带移调前文的小节样本。

用 `--named-instruments` 可将数字说明替换为“Trumpet in Bb”等乐器名称。以 `database/pitch` 为 `--source`，指定新的输出目录，再执行同样的原生导出和数据构建。两种文字都应进入训练；用 `research.data.mix_layout_data --new-repeat` 可增加少量新增训练页的采样次数，验证和测试仍各保留一份。

音高标签沿用[小节文本格式](score-text.md)的定义，吉他、贝斯的变调夹单独记录。局部八度标记另存为事件效果，导出时恢复标记，不再平移音符。训练数据中的谱号、整轨移调、局部八度范围应分别计数；只见过 G 谱号和 F 谱号的模型不能据此宣称已经验证 C 谱号。

## 版面标注

页面清单记录小节框、谱面类型和速度区域。混合谱的一个小节框同时覆盖五线谱与 TAB。

```bash
uv run --no-sync python -m research.data.inventory \
  --gp8-export database/scores --output database/scores/inventory
uv run --no-sync python -m research.data.build_layout_data \
  --source database/scores/inventory --output database/scores/datasets/layout \
  --typed-measures --include-test
uv run --no-sync python -m research.data.build_info_data \
  --source database/scores/inventory \
  --output database/scores/datasets/document_info --include-test
```

类别顺序为 `measure_tab`、`measure_notation`、`measure_both`、`tempo_region`、`clef_region`、`annotation_region`。上述基础流程生成前四类标注，谱号和标注框由 `research.data.build_pitch_data` 生成，再用 `research.data.mix_layout_data` 合并。最后一类包括和弦、指法图、奏法和移调说明，不能把所有框都标为移调。`--include-test` 单独输出测试标注，训练器只读 train 和 val。

`track-index.jsonl` 记录 source_id、family、split、mode、renderer、source_track、folder 和 errors。每个文件夹内的 `layout-pages.jsonl` 记录页面图与 annotations；框为原图像素 `[x0,y0,x1,y1]`。WebUI 使用 `[x,y,width,height]`。

## 整理小节与模拟扫描

先整理基础小节数据并生成扫描增强：

```bash
uv run --no-sync python -m research.data.curate_measure_data \
  --source database/scores --output database/scores/datasets/measure_ocr --workers 16
uv run --no-sync python -m research.data.scan_augment \
  --source database/scores/datasets/layout \
  --output database/scores/datasets/layout_train_scan --workers 16
```

小节整理器检查标签及前文，保存剔除记录、来源计数、清单哈希和技巧覆盖，生成训练增强与固定验证子集。验证或测试中存在无效标签时会排除整个来源，以保留完整序列。改动标签或提示词后需重建训练缓存。

退化包括缩放、模糊、阴影、噪声和 JPEG 压缩，页面框坐标保持不变。默认只增强 train；可用 `--split val --split test --fraction 1 --replace` 生成单独的退化评测目录。脚本拒绝覆盖已有增强目录。

小节退化评测集可单独生成：

```bash
uv run --no-sync python -m research.data.scan_augment \
  --measure-manifest database/scores/datasets/measure_ocr/manifests/validation.jsonl \
  --output database/scores/datasets/measure_scan_eval --workers 8
```

## 谱头数据

标题、副标题和作者变体沿用原曲源的划分。生成后仍需由 Guitar Pro 渲染三种排版：

```bash
uv run --no-sync python -m research.data.augment_headers \
  --source database/scores --output database/headers
# 为 database/headers 执行 research.data.run --phase render，提供上文的 runtime 和 Wine 参数。
uv run --no-sync python -m research.data.build_info_data \
  --gp8-export database/headers \
  --output database/headers/datasets/document_info --include-test
uv run --no-sync python -m research.data.augment_info \
  --source database/scores/datasets/document_info/llamafactory \
  --extra database/headers/datasets/document_info/llamafactory \
  --output database/headers/datasets/info_mixed
```

作者缺失时目标为 `null`，副标题保留为独立元数据。中文字段通过原生 UTF-8 属性附属文件设置，需同时检查元数据和实际字形，见[原生工具说明](native-build.md)。

旧 GP 文件还需通过 `text_encoding` 声明原始编码，原生排版前统一解码曲名、轨名、段落、自由文字及和弦名。小节构建器以实际排版所用的原生模型校正文字字段，保持音符与节奏不变。历史图片中的乱码不能配正常中文答案；应修正这些图片对应的文字标签，或重新排版后再使用正常文字。

和弦数据由 `research.data.chord_scores` 生成带节拍位置的 GP 源谱，原生排版后由 `research.data.chord_supplement` 提取可见名称及指法图。指法图标签来自实际绘出的点、空弦／闷弦、手指和横按；和弦名不能代替指法证据。`research.data.chord_annotations` 补充四至八弦图形，`research.data.paired_staves` 补充分离的五线谱与 TAB 配对。曲源及其排版变体共享同一数据划分。

## 混合后继续训练

完成上面的基础数据、首行乐器、谱号与移调数据构建后，合并为训练配置使用的目录：

```bash
uv run --no-sync python -m research.data.mix_measure_data \
  --new database/pitch/datasets/native_pitch/measure \
  --replay database/scores/datasets/measure_ocr \
  --output database/pitch/datasets/measure_mixed
uv run --no-sync python -m research.data.mix_info_data \
  --source database/headers/datasets/info_mixed \
  --source database/instrument_training/datasets/staff \
  --source database/pitch/datasets/native_pitch/info \
  --output database/pitch/datasets/info_mixed
uv run --no-sync python -m research.data.mix_layout_data \
  --new database/pitch/datasets/native_pitch/layout \
  --replay database/scores/datasets/layout_train_scan \
  --output database/pitch/datasets/layout_mixed
uv run --no-sync python -m research.data.written_pitch_data \
  --source database/pitch/datasets/measure_mixed \
  --output database/pitch/datasets/measure_written
```

可把已核验的乐器混合集合作为 `--replay`；命令中的基础吉他集只是可复现的起点。小节混合需传入相关原生音符核验的 `--exclude-report`，重复运行前换用新的输出和缓存目录。合并版面数据前，各数据集必须采用相同的六类顺序。

最后一步把带音高前文的五线谱训练目标换成记谱音高。运行时由代码将结果换回实际音高，保留原来的小节文本和导出格式。TAB、混合谱和鼓谱的输出语义不变。

当前模型的数据规模和来源分组见[模型目录](../weights/README.md)，测试范围见[评测报告](model-evaluation.md)。生成的扫描退化属于图像增强，不能替代真实扫描件评测。

## 安装导出环境

安装 Wine、虚拟显示和辅助工具：

```bash
sudo apt-get install -y wine wine64 xvfb xauth x11-xkb-utils util-linux
mkdir -p gpbridge/runtime
export WINEPREFIX="$PWD/gpbridge/runtime/wine-prefix"
export WINEARCH=win64
xvfb-run -a wineboot --init
```

下载并安装 Wine 内的 Python 3.11 和 PDF 依赖：

```bash
curl -fL https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe \
  -o gpbridge/runtime/python-installer.exe
xvfb-run -a wine gpbridge/runtime/python-installer.exe \
  /quiet InstallAllUsers=0 'TargetDir=C:\Python311' Include_pip=1 Include_test=0
xvfb-run -a wine "$WINEPREFIX/drive_c/Python311/python.exe" \
  -m pip install "pypdfium2>=5.13,<6" "pdfplumber>=0.11.10,<1"
```

Guitar Pro 8 使用官方安装程序安装到同一 Wine prefix；在有桌面的会话中运行其安装程序，完成安装和激活。原生 DLL 需与使用的 GP8 版本匹配：

```bash
wine /path/to/GuitarPro8-setup.exe
```

原生 DLL 的构建方法见[原生工具说明](native-build.md)。
