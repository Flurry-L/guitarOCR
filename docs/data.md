# 数据生产

`database/` 保存生成数据，由 Git 忽略。仓库提供生成代码和自编示例，识别 PDF 或图片无需训练数据。渲染只使用 Guitar Pro，安装步骤见[数据导出环境](setup.md#数据导出环境)。

## 生成三种排版

```bash
uv run --no-sync python -m datagen.run \
  --corpus /path/to/gp-files --output database/scores \
  --source-count 2000 --phase all --typed-measures \
  --runtime /path/to/GuitarPro8 \
  --wine-prefix-template /path/to/wine-prefix \
  --wine-python /path/to/wine-python/python.exe --workers 4
```

`all` 依次选源、渲染、生成小节数据，再生成版面和谱面信息数据。支持 GP3 / GP4 / GP5 / GTP，默认导出 TAB、五线谱和混合谱三种排版。可重复指定 `--mode tab|notation|both` 筛选。

默认选取 8 至 128 小节、至少 16 个音符的曲谱。小型自编集可调整 `--minimum-measures`、`--maximum-measures` 和 `--source-count`。

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

已有渲染与裁图时，运行 `--phase datasets` 补建版面和信息数据。其他阶段及参数见 `python -m datagen.run --help`。

## 来源分组

三个模型共用 `source_catalog.json`。同一曲谱及其转调、重排版变体应使用同一个 `family`，整体进入 train、validation 或 test。默认按文件哈希分组，内容相同但文件不同的曲谱需要提前指定 family。

完成 select 后，可在标签的 `family` 字段填写分组，再首次运行 crop / datasets。已有 catalog 必须包含全部选中来源，跨 split 的 family 会报错。修改分组后需重新生成对应数据。

已有明确划分的语料可提供 `sources` 列表，每项包含 `source_path`、`family`、`split`，再执行：

```bash
uv run --no-sync python -m datagen.prepare_catalog \
  --catalog /path/to/input_catalog.json --output database/scores --workers 8
```

随后按上文提供 runtime 和 Wine 参数执行 `--phase render`。单音轨 GP5 会原样复制，显示模式由原生导出器切换。

## 版面标注

页面清单记录小节框、谱面类型和速度区域。混合谱的一个小节框同时覆盖五线谱与 TAB。

```bash
uv run --no-sync python -m datagen.inventory \
  --gp8-export database/scores --output database/scores/inventory
uv run --no-sync python -m datagen.build_layout_data \
  --source database/scores/inventory --output database/scores/datasets/layout \
  --typed-measures --include-test
uv run --no-sync python -m datagen.build_info_data \
  --source database/scores/inventory \
  --output database/scores/datasets/document_info --include-test
```

四类顺序为 `measure_tab`、`measure_notation`、`measure_both`、`tempo_region`，类型来自原生页面标注。`--include-test` 单独输出测试标注，训练器只读 train 和 val。

`track-index.jsonl` 记录 source_id、family、split、mode、renderer、source_track、folder 和 errors。每个文件夹内的 `layout-pages.jsonl` 记录页面图与 annotations；框为原图像素 `[x0,y0,x1,y1]`。WebUI 使用 `[x,y,width,height]`。

## 整理小节与模拟扫描

以下目录与训练配置一致：

```bash
uv run --no-sync python -m datagen.curate_measure_data \
  --source database/scores --output database/scores/datasets/measure_ocr --workers 16
uv run --no-sync python -m datagen.scan_augment \
  --source database/scores/datasets/layout \
  --output database/scores/datasets/layout_train_scan --workers 16
```

小节整理器检查标签及前文，保存剔除记录、来源计数、清单哈希和技巧覆盖，生成训练增强与固定验证子集。验证或测试中存在无效标签时会排除整个来源，以保留完整序列。改动标签或提示词后需重建训练缓存。

退化包括缩放、模糊、阴影、噪声和 JPEG 压缩，页面框坐标保持不变。默认只增强 train；可用 `--split val --split test --fraction 1 --replace` 生成单独的退化评测目录。脚本拒绝覆盖已有增强目录。

小节退化评测集可单独生成：

```bash
uv run --no-sync python -m datagen.scan_augment \
  --measure-manifest database/scores/datasets/measure_ocr/manifests/validation.jsonl \
  --output database/scores/datasets/measure_scan_eval --workers 8
```

## 谱头数据

标题、副标题和作者变体沿用原曲源的划分。生成后仍需由 Guitar Pro 渲染三种排版：

```bash
uv run --no-sync python -m datagen.augment_headers \
  --source database/scores --output database/headers
# 为 database/headers 执行 datagen.run --phase render，提供上文的 runtime 和 Wine 参数。
uv run --no-sync python -m datagen.build_info_data \
  --gp8-export database/headers \
  --output database/headers/datasets/document_info --include-test
uv run --no-sync python -m datagen.augment_info \
  --source database/scores/datasets/document_info/llamafactory \
  --extra database/headers/datasets/document_info/llamafactory \
  --output database/headers/datasets/info_mixed
```

作者缺失时目标为 `null`，副标题保留为独立元数据。中文字段通过原生 UTF-8 属性附属文件设置，需同时检查元数据和实际字形，见[原生工具说明](native-build.md)。

当前模型的数据规模和来源分组见[模型目录](../weights/README.md)，测试范围见[评测报告](model-evaluation.md)。生成的扫描退化属于图像增强，不能替代真实扫描件评测。
