# 数据生产

`database/` 是生成数据与中间结果目录，由 Git 忽略。随仓库保留的是生成代码、格式约定和自编示例。PDF / 图片识别不需要训练数据。

## 从 GP 文件生成训练数据

先按 [setup.md](setup.md#数据导出环境) 安装 Wine、Windows Python 与 Guitar Pro 8。Linux 生产流程使用原生导出器取得 PDF、官方布局和音符标签。

```bash
uv run --no-sync python -m datagen.run \
  --corpus /path/to/gp-files --output database/gp8_custom \
  --source-count 2000 --phase all --typed-measures \
  --runtime /path/to/GuitarPro8 \
  --wine-prefix-template /path/to/wine-prefix \
  --wine-python /path/to/wine-python/python.exe --workers 4
```

`--typed-measures` 生成包含谱面类型的四类版面标注。生成后将训练 YAML 中的数据路径改为本次输出目录。

`all` 依次选源、渲染、生成小节数据，然后生成版面及谱面信息数据。GP 源文件支持 GP3 / GP4 / GP5 / GTP；默认原生导出 `tab`、`notation`、`both` 三种排版；可重复指定 `--mode` 筛选。默认筛选 8–128 小节、至少 16 个音符的曲谱；小型自编集可调整 `--minimum-measures`、`--maximum-measures` 和 `--source-count`。

```text
database/gp8_custom/
  labels/                  源曲谱解析标签
  prepared/                单音轨 GP5
  native-export/           原生 PDF、layout.json、official-score.json
  source_catalog.json      所有阶段共用的 source_id / family / split
  source_splits.json        source_id 到 split 的简化映射
  crops/                   小节裁图
  manifests/               train / validation / test 小节清单
  llamafactory/            小节模型训练输入
  inventory/               track-index.jsonl、页面图与 layout-pages.jsonl
  datasets/layout/         COCO 版面标注
  datasets/document_info/  谱面信息裁图与 LLaMA-Factory 输入
```

已有渲染与裁图时，可执行 `--phase datasets` 补建版面和信息数据。其他阶段为 `select`、`render`、`crop`、`relabel`、`relabel-labels`，具体参数见 `python -m datagen.run --help`。

## 来源分组与数据划分

三个模型使用同一份 `source_catalog.json`，`test` 来源不会补入版面或信息训练。`family` 表示同一首源谱或其变体，应整体进入一个 split。生成器默认按曲谱文件哈希分组；转调、重排版等不同文件若属于同一曲谱，必须在开始划分前给它们相同的 family。文件哈希无法自动识别音乐内容相同的变体。

已有 `source_splits.json` 会被保留。已有 catalog 必须恰好包含所有选中的来源；任何跨 split 的 family 都会报错。普通文件夹名、中文路径均可使用，不要求 `train-` 前缀。

```json
{"schema_version":"1.0","sources":[
  {"source_id":"曲谱标签中的编号","family":"song-a","split":"train","source_path":"/scores/a.gp5"}
]}
```

完成 `--phase select` 后，可在标签的 `family` 字段注明曲谱分组，再首次运行 crop / datasets 生成 catalog。修改已存在的 catalog 时需要重新生成对应数据集，不能把旧训练产物当作同一数据版本。

## 单独生成页面清单

```bash
uv run --no-sync python -m datagen.inventory \
  --gp8-export database/gp8_custom --output database/gp8_custom/inventory
uv run --no-sync python -m datagen.build_layout_data \
  --source database/gp8_custom/inventory \
  --output database/gp8_custom/datasets/layout --typed-measures --include-test
uv run --no-sync python -m datagen.build_info_data \
  --source database/gp8_custom/inventory \
  --output database/gp8_custom/datasets/document_info --include-test
```

也可以省略 `--source`，直接对两个构建器传 `--gp8-export`；它们会自动生成 inventory。旧 inventory 可以继续使用。

`track-index.jsonl` 每行包含 source_id、family、split、mode、renderer、source_track、folder、errors；`folder/layout-pages.jsonl` 每行包含 page_index（从 0 开始）、image（相对 inventory）、image_size、page_bbox_mm 和 annotations。标注 `box` 是原图像素 `[x0,y0,x1,y1]`，类别为 measure / tempo_region。WebUI 的框使用 `[x,y,width,height]`，两者不要混用。

稀疏谱行可使用 `build_layout_data --sparse-native-root /path/to/documents` 补充。目录中的 `sources.json` 可显式指定每个文档的 family、split；缺省时按文档名称分组并稳定划分。涉及同谱变体时，应提供显式分组。

## 验证

`tests/test_data_pipeline.py` 使用三首自编源谱和符合原生格式的导出夹具，验证三个构建器、任意 corpus 名称及 family 隔离；它不替代真实 GP8 渲染验收。真实原生构建与导出检查见 [native-build.md](native-build.md)。

## 已分组曲谱与三种版面

已有来源划分时，提供 `sources` 列表，每项包含 `source_path`、`family`、`split`（train / validation / test）；每个 family 选一份源文件。以下入口保留来源划分，单音轨 GP5 原样复制，由原生导出器切换显示模式，避免重写罕见奏法时丢失信息：

```bash
uv run --no-sync python -m datagen.prepare_catalog --catalog /path/to/input_catalog.json \
  --output database/gp8_layout_multimode_v1 --workers 8
# 按上文提供 runtime / Wine 参数运行 --phase render
uv run --no-sync python -m datagen.inventory --gp8-export database/gp8_layout_multimode_v1 \
  --output database/gp8_layout_multimode_v1/inventory --workers 8
uv run --no-sync python -m datagen.build_layout_data --source database/gp8_layout_multimode_v1/inventory \
  --output database/gp8_layout_multimode_v1/datasets/layout --include-test
```

页面清单构建默认识别已有的三种模式；显式传 `--mode` 时要求每个来源均有对应导出。COCO images 保留 mode / source_id / family / renderer。`--include-test` 单独写入 `instance_test.json`，训练器只读 train 和 val。混合谱的 measure 框覆盖同一小节的五线谱与 TAB，两者不会被当作两个小节。

## 同时监督谱面类型

四类版面模型直接使用页面的原生 `mode` 标注小节类别，不需要人工分类。以下命令复用 inventory 的图像和框坐标，保留原有来源划分：

```bash
uv run --no-sync python -m datagen.build_layout_data \
  --source database/gp8_layout_multimode_v1/inventory \
  --output database/gp8_layout_multimode_v1/datasets/layout_typed \
  --include-test --typed-measures
```

类别顺序为 `measure_tab`、`measure_notation`、`measure_both`、`tempo_region`。缺少有效页面类型会报错；不能根据文件夹名字猜测类型。`datagen.run --typed-measures` 也可在 datasets / all 阶段生成四类标注。不传该选项时保留旧两类格式，供历史训练配置使用。

## 扩充数据与退化样本

`database/gp8_joint_v3` 使用 2,799 份源谱，按曲目分成 2,400 / 199 / 200 个训练、验证、测试来源。三个版面均由 Guitar Pro 8 原生导出，共 30,693 页。新增来源包括 1,767 份真实 GP、17 份其他真实 GP 和 216 份技巧合成谱；另外复用历史来源，剔除一份与训练集标题身份冲突的验证来源。这里的“真实”指真实 GP 曲目重新排版，不能等同于真实扫描件测试。

| 数据 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: |
| 原生页面 | 26,192 | 2,337 | 2,164 |
| 合法小节裁图 | 587,019 | 54,138 | 49,551 |
| 文档信息裁图 | 15,029 | 1,224 | 1,176 |

曲目身份检查结合原有来源划分、音乐事件指纹及标题别名。所有版面、扫描变体与难例重复都沿用源谱 split。训练时另加入 6,548 页扫描退化图、147,212 张退化小节图及 89,318 个有上限的技巧难例重复，共 823,549 条小节训练样本。剔除 3 条损坏时值目标及 3 条含有损坏历史上下文的训练样本；验证、测试序列未因标签或上下文无效而删减。

```bash
uv run --no-sync python -m datagen.run --phase crop --output database/gp8_joint_v3 --workers 16
uv run --no-sync python -m datagen.curate_measure_data --source database/gp8_joint_v3 \
  --output database/gp8_joint_v3/datasets/measure_ocr --workers 16
uv run --no-sync python -m datagen.build_info_data --source database/gp8_joint_v3/inventory \
  --output database/gp8_joint_v3/datasets/document_info --include-test
uv run --no-sync python -m datagen.scan_augment --source database/gp8_joint_v3/datasets/layout_typed \
  --output database/gp8_joint_v3/datasets/layout_train_scan --workers 16
```

退化包括轻度缩放、模糊、阴影、噪声、JPEG 压缩；页面大小和框坐标不变。训练只增强 train，验证、测试保留原图。另用 `--split val --split test --fraction 1 --replace` 创建单独的退化评测目录；不得将它混回训练。脚本拒绝覆盖已有增强目录。

小节整理器保存无效标签清单、各 split 的来源与模式计数、清单哈希和难例覆盖。若验证／测试存在无效标签，会排除整个来源以保留完整的自回归序列；本次没有此类排除。验证快集固定为 900 个分层样本，用于训练期监测。改动标签后需重建 ShareGPT 数据和训练缓存；图像文件未变不代表标签版本未变。可见装饰音的语义与限制见[小节文本格式](score-text.md)。

小节 OCR 也可构建单独的扫描退化验证集，保持标签与干净裁图一致；默认抽取同一批按来源、谱面类型分层的 900 条验证样本。连续预测上下文的评估需使用 `--max-samples 0` 保留完整序列。

```bash
uv run --no-sync python -m datagen.scan_augment \
  --measure-manifest database/gp8_joint_v3/datasets/measure_ocr/manifests/validation.jsonl \
  --output database/gp8_joint_v3/datasets/measure_scan_eval --workers 8
```
## 谱头补充数据（2026-09）

`datagen.augment_headers` 从已分组的主语料选取 120 个训练、30 个验证、30 个测试音乐来源，只改标题、副标题、作者等元数据，并保留前四小节。180 个中英文变体各由 Guitar Pro 渲染 TAB、五线谱和混合谱，共 540 份文档；它们不增加独立音乐来源数量，也没有将验收文件 `jixian.pdf` 的内容用于训练。中文字段通过原生 UTF-8 元数据附属文件设置，见[原生构建说明](native-build.md)。

生成谱头变体后，用上文的 runtime／Wine 参数运行 render，再构建裁图和混合数据：

```bash
uv run --no-sync python -m datagen.augment_headers \
  --source database/gp8_joint_v3 --output database/gp8_headers_v3
# 在此对 database/gp8_headers_v3 执行 datagen.run --phase render
uv run --no-sync python -m datagen.build_info_data \
  --gp8-export database/gp8_headers_v3 \
  --output database/gp8_headers_v3/datasets/document_info --include-test
uv run --no-sync python -m datagen.augment_info \
  --source database/gp8_joint_v3/datasets/document_info/llamafactory \
  --extra database/gp8_headers_v3/datasets/document_info/llamafactory \
  --output database/gp8_headers_v3/datasets/info_mixed
```

原生元数据逐字段核对通过，谱头文本可见性过滤无拒绝；得到训练 711、验证 180、测试 180 条谱头／速度裁图（9 个训练速度区域因源谱标注条件未纳入）。`datagen.augment_info` 保留原有 15,029 条训练样本，增加其中 25% 的扫描退化版本，并将新增谱头及其退化版本各重复 6 次。最终训练 23,808 条、验证 1,584 条、测试 1,536 条；验证和测试没有重复加权。副标题不是作者，作者缺失时目标仍为 `null`。

该补充源于初版 Info v3 在真实中文谱头上把副标题当作者的退步。初版候选与评估记录保留，最终默认模型见[模型目录](../weights/README.md)。所有生成退化图都是鲁棒性增强，不等同于真实扫描训练数据。来源、增强和清单哈希在 `database/gp8_headers_v3/datasets/info_mixed/summary.json`。
