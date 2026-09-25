# 数据生产

`database/` 是生成数据与中间结果目录，由 Git 忽略。随仓库保留的是生成代码、格式约定和自编示例。PDF / 图片识别不需要训练数据。

## 从 GP 文件生成三套数据

先按 [setup.md](setup.md#数据导出环境) 安装 Wine、Windows Python 与 Guitar Pro 8。Linux 生产流程使用原生导出器取得 PDF、官方布局和音符标签。

```bash
uv run --no-sync python -m datagen.run   --corpus /path/to/gp-files --output database/gp8_measure_sequence_v2   --source-count 2000 --phase all   --runtime /path/to/GuitarPro8   --wine-prefix-template /path/to/wine-prefix   --wine-python /path/to/wine-python/python.exe --workers 4
```

`all` 依次选源、渲染、生成小节数据，然后生成版面及谱面信息数据。GP 源文件支持 GP3 / GP4 / GP5 / GTP；当前原生训练数据目标是纯 TAB。默认筛选 8–128 小节、至少 16 个音符的曲谱；小型自编集可调整 `--minimum-measures`、`--maximum-measures` 和 `--source-count`。

```text
database/gp8_measure_sequence_v2/
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
uv run --no-sync python -m datagen.inventory   --gp8-export database/gp8_measure_sequence_v2   --output database/gp8_measure_sequence_v2/inventory
uv run --no-sync python -m datagen.build_layout_data   --source database/gp8_measure_sequence_v2/inventory   --output database/gp8_measure_sequence_v2/datasets/layout
uv run --no-sync python -m datagen.build_info_data   --source database/gp8_measure_sequence_v2/inventory   --output database/gp8_measure_sequence_v2/datasets/document_info
```

也可以省略 `--source`，直接对两个构建器传 `--gp8-export`；它们会自动生成 inventory。旧 inventory 可以继续使用。

`track-index.jsonl` 每行包含 source_id、family、split、source_track、folder、errors；`folder/layout-pages.jsonl` 每行包含 page_index（从 0 开始）、image（相对 inventory）、image_size、page_bbox_mm 和 annotations。标注 `box` 是原图像素 `[x0,y0,x1,y1]`，类别为 measure / tempo_region。WebUI 的框使用 `[x,y,width,height]`，两者不要混用。

稀疏谱行可使用 `build_layout_data --sparse-native-root /path/to/documents` 补充。目录中的 `sources.json` 可显式指定每个文档的 family、split；缺省时按文档名称分组并稳定划分。涉及同谱变体时，应提供显式分组。

## 验证

`tests/test_data_pipeline.py` 使用三首自编源谱和符合原生格式的导出夹具，验证三个构建器、任意 corpus 名称及 family 隔离；它不替代真实 GP8 渲染验收。真实原生构建与导出检查见 [native-build.md](native-build.md)。
