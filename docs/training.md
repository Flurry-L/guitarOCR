# 数据、训练与评测

训练依赖留在开发环境；桌面端使用 GGUF 和 ONNX。

## 模型文件

以下目录均位于 `weights/`：

| 目录 | 内容 |
| --- | --- |
| `score_ocr/merged/` | 可训练的单图观察 OCR 权重、词表、图像策略及产品能力声明 |
| `score_ocr/gguf/` | 部署用小节观察模型和视觉编码器 |
| `information/` | 谱面信息模型，含 HF 权重、GGUF、视觉编码器与谱表／谱号适配器 |
| `auxiliary/` | 部署用版面、信息与结构 ONNX |
| `layout/` | 可继续导出的 Paddle 推理模型 |
| `checkpoints/` | 按职责保存的辅助网络源检查点 |

`checkpoints/chord_evidence/` 的 primary / secondary 同时参与一致性判定；名称定位使用单个 `staff_label/` 模型。职责见[识别流程](../README.md#识别流程)。

指法几何通过双读图器的一致性与校准门槛后，采用其结构结果；信息模型负责名称读取。专用读图器已确认的几何可以覆盖语言模型的不同读数，同时保留差异记录；两个专用读图器之间仍有分歧或名称未确认时，则保留待核对。

`distribution.json` 是部署文件清单，本地槽位为 `current`。辅助模型只保存在 `auxiliary/`，`score_ocr/merged/capabilities.json` 统一声明所用模型；导出到独立目录时再收集依赖。输入输出约定由代码和 ONNX 内嵌元数据提供。`score_image_policy.json` 规定等比缩放等输入行为。采用模型后替换对应职责文件，清理其余检查点、渲染副本和缓存，不按实验代次复制模型目录。

## 数据生成

单图观察训练数据写入 `database/observations/`，包含三个 split 的对话样本、`manifest_{split}.jsonl` 和数据说明。合并清单引用各生成目录中的图片，训练时保留这些目录；`pack_score` 可将图片一并复制到独立数据目录。原谱与来源分组放在 `database/sources/`，检查点和评测输出写入 `output/`，这些生成内容不进入 Git。

### 数据与真值

- 同一原曲及其排版、字体、移调和扫描变体留在同一 split；独立 test 不参与调参和阈值选择。
- 音符标签对应实际打印内容。五线谱输出谱位和显式升降记号，TAB 输出可见弦品；没有印出的品位为 null。源文件的完整音乐答案只用于组谱评估。
- 裁图、训练和推理均保持长宽比。小节输入仅一张图，不附相邻图片、有效调弦、前节答案或源文件中的音乐状态。
- PM、let ring、装饰音和鼓件等由实际渲染证据确认可见性。看不清的字段保持未知，不用“没有”替代。
- 结构真值保留乐器、谱表和小节对应关系；图示库与实际拍点分开标注。

小节采用包含和弦标注的音乐区域：左右按小节线，上下保留本节和弦名称、指法图、加线、符干、连线和奏法。指法图向左右跨界时，不为收全网格而把邻小节的音符纳入主框；完整图另由 `annotation_region` 保存，两个框允许重叠。训练和产品均不擦除网格或整个标注区域，避免误删同一区域的 PM、let ring、八度线、力度或其他音乐标记。裁图范围应根据本次实际绘制内容建立，不能仅按裸谱线高度确定。

产品补全已有框的图示上下文时，先用原框直接交集与谱表／谱行归属确定候选。仅被裁图留白碰到、或横跨多个谱行的图，不触发自动扩框；已有谱线带也用于避免新框吞入另一行正文。归属不明时保留原图及独立指法图，允许用户修框。

小节观察只把谱表中的演奏事件写入 `voices`。和弦名称必须在图中可读且有本节拍点依据，写为该事件的 `chord:NAME`（保留升降号、扩展音和斜杠低音，IR 保留字符使用百分号编码）；指法图的手指号、起始品位、圆点和空弦／静音符号不生成音符或节奏。小节观察不输出 `diagram:`，完整指法几何由专用读图器提供。谱首和弦字典、邻节图示和看不到名称的网格不生成本节和弦名称。

组谱后，指法图只能匹配已经识别出的和弦事件。人工关联优先，其次是同页、本小节可见且归属唯一的同名图；跨入两个小节时，由实际识别出的同名事件进一步确定归属。全曲其他位置的同名图不自动替代当前按法。同节同名多种按法、多个候选归属或读图冲突保留选择项；独立指法库不会增加演奏轨道。训练需同时覆盖名称与图示、仅名称、无标注，以及边界截断、相邻图示和奏法共存；相同乐句的外观变体保持同一来源划分。《春日影》只用于真实输入检查，不进入训练集。评测单列 `visible_chords`，按实际名称与起点匹配，不把时值错误或等价百分号编码重复算作和弦错误；音符、节奏及已有奏法另行检查。

`research/data/run.py` 组织选曲、原生谱面渲染及裁图；`build_measure_data.py` 保留源小节清单。只有具备 `observation` 的记录才能经 `training_samples.py` 生成小节训练答案；完整 M 不能直接转换为可见标签。通用造谱使用 `synthetic_scores`（GP）和 `engraved_scores`（MusicXML / Verovio）；后者还生成拍调号、信息和结构训练数据。

信息样式使用 `information_styles`，谱号和拍调号裁图使用 `symbols clef` / `symbols signature`；和弦造谱使用 `synthetic_scores --profile chords`，共享 GP 保存与来源划分流程。标题增强集中在 `headers` 的 `text` / `layout` 子命令。各生成器输出新样本，通过 `pack_score` 与原生样本混合，不在生成器里绑定某轮实验的回放比例或旧数据目录。

小节模型输出 JSON：`state` 保存本图印出的状态标记，`voices` 保存局部声部、事件及可见音符，`relations` 保存连线与延伸线片段。五线谱最低谱线为位置 0，每个线间向上加 1；临时记号 null 表示没印，`unknown` 表示有但看不清。关系端点可以引用本节音符，也可以是 `left` / `right`，类型不明时保持 null。

观察生成按 `assembly.observation_schema()` 进行 JSON 约束解码，约束字段、类型及合法奏法名称；参数值再由 IR 校验，未知信息仍可输出 null。音乐内容的正确性另行评估；生成被长度上限截断时仍记为失败。

`scorelib/src/observation.rs` 根据谱表顺序、观察和显式辅助结果进行状态继承与关系连接，研究端的 `research/inference/assembly.py` 仅调用原生绑定。缺节、歧义和未知保留在结果中，不生成休止占位。越界音符引用、声部冲突等坏关系单独报错，不参与相邻小节的类型传播。音符和节奏已知时可输出附有未解决问题的 M 草稿；缺少品位、音高或时值时仍不补猜，因此 M 可用率不代表完整正确率。共用组谱的音高基准为书面音高，八度范围另行保留。产品后端在组谱后按乐器移调、谱号八度与变调夹转换音高，再保存或导出；不直接把书面音高当实音。产品已使用单图观察接口，未知字段可保存、补全，局部重识别和人工校正都会重新组谱。

PM / let ring / 八度延伸线先按音乐结构连接：同一谱表的小节相邻、模型明确观察到离开和进入本节的端点、声部已对应、已知类型与数值不冲突，且双方候选唯一时就连接。模型给出的 `lane` 只用于判断上下顺序，不把其小数精度当作连接距离；虚线长短、粗细、段数和绝对高度都不作为组谱门槛。推理不调用手写像素虚线筛选器。

存在多个竞争范围时，只有两侧数量相同、上下顺序均明确、每对结构兼容且处于同一已知谱行，才按顺序配对。换行可能改变线高和上下排布，跨行多候选不靠排序猜。模型仍负责识别奏法类型、本图内的起止事件和延续端点；组谱不会把本节结束强改为延续，也不会因一次像素漏检而丢掉已读出的本地奏法。重复或歧义观察保留原文，不把“只测到一条”当作另一个观察必假的证据。观察缺失或顺序误读仍可能造成错误关联，需要通过训练及原生谱面评估改善。

不同渲染器的样本共享可见事实与关系约定，同一原曲的不同渲染样式保持在同一 split，各次渲染使用独立的 `source_id`。清单中的 `renderer` 明确记录 `guitarpro`、`alphatab` 或 `verovio`；评测按渲染器分别报告，缺少来源的记录列为 `unknown`。合成谱面的高分不能替代 GP8 原生验证。

原生 GP8 观察适配读取同次 PDF 导出的品位、休止符、连线绘制轨迹和实际文字。延音的隐藏品位通过真实附着视图确认不可见，本地答案保留 null；括号内重印的品位则照常读取。三连音等节奏比例要求实际数字和括线证据；GP 多声部没有印出的细分节奏保留未知，不能用源谱补答案。源谱中的完整音符与奏法仅作组谱评测参考。

PM / let ring 支持只见文字、并行范围和多声部归属不明；文字、端点或声部看不清时分别保留未知。无标签续线不抄上一节的名称。H/P 与滑音跨裁图时也使用关系片段，未知弧线、斜线和范围延伸线分别记录形状。GP 某些跨行滑音不在下一行重画入线：组谱只有找到下一节唯一同弦、时刻连续且声部确定的可见起音时才补连接，并记录端点来自组谱推断；本地观察不补画关系。没有唯一目标时保留开放关系及已知起点奏法。以上几何证据只用于编译训练标签，推理不运行这套 PDF 标注程序。

当前可见观察数据由 `observations` 生成。以下数量指渲染视图数，每份再裁成多个小节；同一原曲的字体、排版和标注变体继承同一 split，不能把视图数当作独立原曲数。

```bash
python -m research.data.observations --mode notation \
  --train 300 --validation 30 --test 30 --output database/observations-notation
python -m research.data.observations --mode tab \
  --train 1000 --validation 100 --test 100 --output database/observations-tab
python -m research.data.observations --mode tab-chords \
  --train 1024 --validation 128 --test 128 --output database/observations-chords
python -m research.data.observations --mode octave \
  --train 256 --validation 32 --test 32 --output database/observations-octave
python -m research.data.observations \
  --merge database/observations-notation database/observations-tab database/observations-chords database/observations-octave \
  --output database/observations
```

`tab-chords` 每四份视图分别呈现无标注、仅名称、完整指法图和名称被裁断的图；再与奏法外观组合，同一音乐 family 不跨 split。Guitar Pro 的谱内网格需用 `synthetic_scores --format gp` 输出带明确样式的 `.gp` 后实际渲染：GP8 导入 GP5 时可能忽略谱内网格设置，不能只凭源文件开关认定 PDF 中已画出指法图。

五线谱标签读取实际 SVG 谱位、升降记号与连线；TAB 标签读取实际绘制的数字、弧线与延伸线。数据包含跨节隐藏品位、部分和弦延音、PM / let ring 多小节及跨行延续、8va / 8vb / 15ma / 15mb 以及首标签被遮掉的未知范围。TAB 混合 3/4、4/4、6/8、整小节休止、无延音的连梁和 H/P、滑音，独立变化排版宽度、缩放、延伸线粗细及其与谱表的距离。H/P、两音间滑音与滑入／滑出使用各自的 IR 名称，归属实际接触的音符，并与延音和延伸线组合出现；只有文字而无可见延长线的 PM / let ring 记录可见起点，结束位置保留 null。源谱只用于关联图形与事件、提供评测答案，不能把源谱中未印出的品位或奏法名称写入局部答案。关系按进入本节、本节内、离开本节排序，端点引用当前答案中从 0 开始的事件和音符数组下标；休止和隐藏品位的续音同样占用事件下标。

补充新原谱时用 `--index-offset` 指定尚未使用的起始编号，它也改变随机种子；已有原谱及其变体仍须保持原 split。根据旧评测结果调整过数据或训练后，使用新留出的原谱做最终独立评测，旧 test 保留作历史结果。

TAB 数据独立变化编号、调弦说明和乐器名的显示，并保留实际打印的力度。相同音乐的显示变体与有／无延伸线配对属于同一 family；H/P、滑音的位置与音程独立采样，包含相邻品位。裁图覆盖实际绘制的标记，不把编号或调弦文字当作音符标签。

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

各任务生产并完成来源分组后，通过 `pack_score` 汇总到统一目录。每个 split 可以传入多个 JSON 数组或 JSONL 文件，直接混合小节、标题、谱表等任务；不另设补训混合脚本，也不隐式重复某类样本。完整 M 和多图样本不进入新训练集；图片复制到新数据目录。

```bash
python -m research.data.pack_score \
  --train /path/to/measures_train.jsonl /path/to/document_info_train.json \
  --validation /path/to/measures_validation.jsonl /path/to/document_info_validation.json \
  --test /path/to/test.jsonl \
  --output database/observations_packed
python -m research.data.pack_score --validate database/observations_packed
```

输出目录必须尚不存在。生成器负责来源划分及真值质量；汇总器检查已有 `provenance.family/split` 是否冲突、同一图片是否跨 split，并检查单图输入、观察答案格式和图片引用。来源字段未写入对话样本的生成器仍须在上游清单中保证原曲不交叉。

首次训练前分词；数据、图片、词表或预处理配置变化后，删除旧分词缓存并重新生成。

谱行延续数据由 `research.data.part_continuation` 统一生成。`--engraved` 接收当前渲染器的 `{split}/*/score.json` 目录；`--native-info` 配合 `--exports` 接收原生谱表清单与 GP 导出目录。两种来源可同时提供，同曲不同轨在生成时组成候选组。生成器保持来源划分，每份数据只写一次；`--limit` 显式限制每种来源、每个 split 的文档数，默认使用全部。排版谱面的名称遮罩来自 SVG 标签节点，原生 GP 页面只使用真实图像及导出轨迹。

原生导出共用一条处理流程：`inventory` 整理页面与可见标注，`build_measure_data` 生成小节，`build_info_data` 生成标题、速度、谱号及可见移调标记，`build_layout_data` 生成版面数据。合成 GP 谱也通过同一流程处理。

已有原生导出可直接生成受支持的小节观察，再通过 `observations --merge` 与其他渲染器合并：

```bash
python -m research.data.build_measure_data \
  --native-manifest /path/to/native/document-manifest.jsonl \
  --output database/observations-native
```

`--scopes-only` 只保留具备实际 PM / let 范围证据的小节，便于补充范围数据；来源 family 和 split 仍来自同一原生清单。

`synthetic_scores --profile native-coverage` 生成包含本地、跨节和跨行延音、H/P、滑音、三连音、多声部及 PM / let 组合的 GP 源谱，沿用上面的原生导出流程。`native_summary.json` 按 split 统计实际接受的图片标签，包括隐藏品位、未知节奏、并行范围和各种滑音；源文件中请求了某种效果不能计作已覆盖。训练、验证与测试使用独立原曲，开发时先验证训练／验证来源，再运行保留的测试来源。

CPU 回归检查集中在一个文件，包含组谱合同与 PDF 取证。真实 GP 集成检查使用上述 profile 的导出目录：

```bash
GUITAROCR_NATIVE_COVERAGE=/path/to/native-coverage/track \
  python -m unittest discover -s tests -p test_observation_assembly.py -v
```

## 主 OCR

使用 `research/configs/measures/train.yaml`，默认从 `weights/score_ocr/merged/` 继续训练，数据为 `database/observations/`。当前续训配置为 2 轮、学习率 `1e-5`，词嵌入与输出层的独立学习率为 `5e-5`。数据更新后需要重新训练和独立评测，不能把数据修正当成模型准确率提升。

```bash
python -m research.training.measures \
  --config research/configs/measures/train.yaml --tokenize-only
python -m research.training.measures --check
python -m research.training.measures
```

`--check` 只用 CPU：解析实际训练参数、加载训练依赖和词表、读取已有缓存，并用首条及最短／最长样本实际处理图片和标签；不会加载模型权重或启动训练。它不能替代 GPU 上的前向、反向和显存检查。`CUDA_VISIBLE_DEVICES` 指定参与训练的卡，默认每张可见卡一个进程；启动前确认所选 GPU 已空闲。

缓存直接使用 LLaMA-Factory 的 `tokenized_path`，已有缓存会继续使用。数据、图片、模型输入策略或词表变化时，由使用者删除旧 `database/observations/tokenized/` 后重新运行 `--tokenize-only`；程序不再计算输入指纹或自动拒绝旧缓存。`save_total_limit: 2` 限制训练过程检查点数量。训练产物写入 `output/observation_ocr/`；新一轮微调用空输出目录。入口拒绝隐式恢复旧优化器，继续同一轮训练须显式指定 `resume_from_checkpoint`，模型评估通过后再决定产品接入。

每轮训练使用独立输出目录，完成合并后分别评估单图观察、组谱和实际 GGUF 推理，再采用到 `weights/score_ocr`。小节微调可能遗忘信息任务；谱面信息使用独立的 `weights/information`，不能因为小节指标提高就同时覆盖信息模型。

训练与推理共用单图观察提示词，模型输出局部 JSON，组谱后才产生 M。词表与模型参数一同保存，token ID 必须与词嵌入对应。相邻样本分组与重复图片共享计算已经移除。

一轮训练结束并合并权重后，可删除中间检查点；后续微调从合并模型开始，使用新的输出目录，例如 `model_name_or_path=weights/score_ocr/merged output_dir=output/observation_training`。

合并 LoRA 后可直接运行研究管线。输入清单每行包含 `id`、`image`、`mode`、`source_id`、`part_id`、`staff_id`、`bar_index`，并保留版面给出的 `source_page`（或 `page`）与 `system_index`。延伸线依照前述音乐结构与同谱行上下顺序连接，跨行的多候选歧义保留。顺序和谱表身份来自版面整理，不向模型输入相邻图像或完整答案。

奏法名称遵循 IR：H/P 使用 `hammer`；两音间的连奏滑音和换把滑音分别为 `sl`、`ss`；从下／上滑入为 `sib`、`sia`，向下／上滑出为 `sod`、`sou`。PM／let 的起点是文字标记的事件，不是文字后第一段虚线下的事件。生成数据需要覆盖这些奏法与延音、技术延伸线同时出现的情况，并包含相似乐句中没有这些符号的样本；只有实际绘制的符号才能成为观察标签。

```bash
python -m research.export.export_glm --base weights/score_ocr/merged \
  --adapter output/observation_ocr --output output/observation_ocr/merged
python -m research.inference.run --manifest /path/to/crops.jsonl \
  --model output/observation_ocr/merged --output output/observations
python -m research.evaluation.measures --manifest database/observations/manifest_test.jsonl \
  --model output/observation_ocr/merged --output output/observation-test
```

推理保存原始观察与组谱结果；`--observations` 可读取已有预测，仅在 CPU 上重新组谱。`--auxiliary` 接受按小节 id 对齐的独立状态预测或声部映射；评测的 `--oracle-auxiliary` 仅用于单独测量真值辅助下的上限，默认不用。指标分别统计可见音符、节奏、技法及其归属、关系、未知品位乱填和组谱前后音符结果；能生成 M 不代表所有延续关系均已确定。

`rhythm` 只统计真值起点和时值均已知的事件；部分未知的事件另按 `known_rhythm_fields` 统计已知字段，并用 `unknown_rhythm` 报告未知字段是否保留、乱填或漏失。未知起点的音符只在同局部声部、事件数一致时按事件顺序对应，不靠交换重复音来提高得分；不能对应的保留为漏失。

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

质量模型只学习现有候选的质量。`research.training.layout_quality` 接收 JSONL 页面，每行包含 `image`、`family`、`split`、实际执行的 `views`、`candidates` 和带类别的 `truth` 框。训练和验证禁止共用来源 family，目标按同类别 IoU 连续计算。候选坐标保持不变。小节框的内容范围变化时，检测器与质量网络需配套评估和更新；质量网络也要学习包含和弦标注的新框，否则会在融合前过滤真实小节。

训练时将真实低 IoU 负例适度重复到约一成，避免检测器变准后质量网络学成一律接受；验证与测试仍使用原始分布。

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

各入口及子命令的 `--help` 列出数据、初始化模型和输出参数。结构关系数据使用 `research.training.structure_relation prepare` 的 `--train`、`--validation`、`--test` 读取 `engraved_scores` / `information_styles` 输出的 `structure_{split}.jsonl`。每行直接保存干净页面 `image`、谱行中心 `centers`、`row_modes`、结构标签 `expected.rows`、原曲 `family` 和 `split`。`train --task system` / `train --task part` 分别训练现用的两个分支，`--initial` 加载对应检查点；`evaluate --checkpoint ... --manifest ...` 接受独立评测清单，`export` 显式接收两份权重和验证清单。

谱行延续的 `export --model DIR --data DIR` 在验证集上同时校准有名称和无名称的候选组，完成 ONNX 数值核对并保存阈值；`evaluate` 直接使用导出模型和阈值。拍调号训练通过多次 `--data DIR` 混合来源，只学习实际印出的拍调号；`export_auxiliary` 默认从 `weights/checkpoints/signature/` 读取权重、配置与校准，可用 `--signature-model DIR` 指定其他训练目录。

奏法训练随小节观察任务进行；`research.data.technique_visibility` 负责从实际渲染中提取可见性证据。

版面检测同时需要检测网络和质量网络。

和弦训练通过 `--reader primary` 或 `--reader secondary` 选择两个现用读图器，`--initial` 严格加载相应检查点，更新全部参数。

和弦训练、校准和评测共用 `research.data.chord_evidence.ChordEvidenceData`：训练读取 train，选择模型和校准读取 validation，评测读取 test；来源与指型分组仍在生成器内保持隔离。`chord_calibration reader --model DIR --data database/score` 为单个读图器校准并导出，`consensus --model PAIR_DIR --data database/score --output agreement.json` 校准已导出的双读图器。`--font` 可指定当前使用的字体。

独立和弦评测使用 `python -m research.evaluation.chords --model DIR --data database/score --output output/chords.json`。也可用 `--cases crops.json` 单独检查真实裁图：JSON 数组每项为 `image`、`kind`（diagram／text／other）和图示的 `expected`，图片路径相对清单。评测直接比较参考答案，报告原始正确率、接受结果的精度及错误样例，不按文件名修改标签。

数据生产与评测仍需实际乐谱及独立来源；被清除的实验结果不作为可用数据。专项输出先写临时目录，通过真实输入评测及 Python / 原生推理一致性检查后，再更新当前模型和阈值配置。

谱表类型与弦数训练使用 `staff_evidence train` 子命令和生成器的专项清单，明确提供 `--train`、`--validation` 和 `--initialize`。每条记录须保留 `provenance.family` 与 `provenance.mode`；主 OCR 打包后的纯对话样本不提供这些来源字段，不能直接代替专项清单。

## 采用条件

同时检查目标任务和原有能力：整页漏检 / 合框、事件身份、音高与节奏、未知字段、声部和拍点关联、保存重开及 GP5 / MusicXML 导出。参数量或局部裁图分数不能代替整谱结果。

评测记录输入范围、所用模型、设备、页数及小节数，分别报告整页与参考裁图结果。阈值只用验证集选择，再用来源独立的测试集评估；失败、漏检、无法匹配、未知真值和人工补框均保留在报告中。当前研究小节测试集由 `database/observations/manifest_test.jsonl` 指定。编译检查和少量实际操作用于确认程序可运行，模型准确率需要上述独立评测。

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
- `research.evaluation.measures`：单图局部观察及跨小节组谱评估，分别报告可见事实、未知信息乱填和组谱结果。完整答案只用于评分。
- `research.evaluation.information`：标题、速度、谱表、谱号与注释的参考裁图评测，按任务选择信息适配器。报告原始输出的字段、文字和和弦指标，失败保留在分母中。
- `research.evaluation.pipeline`：调用实际软件做整页和多声部识别。`--scores` 接收渲染器的 `score.json`、包含这些文件的目录，或原生数据生成器的新小节 JSONL（须有 `source_pdf` 和完整小节序列）；漏检保留在分母中。

参考裁图评测使用 vLLM 直接读取尚未导出的训练权重；默认环境是 `tools/vllm-venv/bin/python`，可用 `GUITAROCR_VLLM_PYTHON` 指定。它评估单个模型的训练效果；整谱指标来自原生软件的 GGUF / ONNX 流程。评测输出由 `--output` 指定。

主模型格式转换入口在 `research/export/`，小型专项网络的 ONNX 导出与训练共用任务入口。本地模型清单按职责列出文件；发布下载附件时才生成文件大小和完整性校验，不把历史实验编号、训练输出路径或固定哈希写进产品逻辑。

统一 OCR 的 GGUF 导出使用 `python -m research.export.export_gguf --llama-cpp /path/to/llama.cpp --model weights/score_ocr/merged --output output/gguf/score_ocr`；信息模型独立从 `weights/information/merged` 导出到 `weights/information/gguf`，其适配器另用 `research.export.export_information_adapter` 转换。两者使用所提供的 llama.cpp 源码，不要求固定提交。
