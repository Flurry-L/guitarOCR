# 参与开发

## 本地检查

```bash
uv sync --locked --python 3.11 --extra webui --extra dev
uv run --no-sync ruff check datagen layout document_info measure_ocr gp5_export pipeline shared webapp scripts examples tests
uv run --no-sync python -m unittest discover -s tests -v
```

检查覆盖 GP5 导出、项目保存与恢复、数据划分、版面类型和下载校验，不需要模型或 GPU。CI 只保留手动触发的同一组检查。界面修改后，本地检查导入、画框、保存、导出和刷新恢复。

训练环境使用 `uv run --no-sync`，避免同步命令移除额外训练依赖。模型评测命令见[训练说明](docs/training.md)。

## 流程与代码

`pipeline.run` 依次调用 `layout`、`document_info`、`measure_ocr`、`gp5_export`。各步骤也可通过 `python -m <步骤>.run` 单独执行，返回自己的 `manifest.json` 路径。下游校验清单的 `schema_version` 和 `stage`，字段类型见 `shared/schema.py`。

| 目录 | 职责 |
| --- | --- |
| `datagen/` | GP 选源、Guitar Pro 渲染、标签和数据划分 |
| `layout/` | 页面渲染、小节与谱面类型检测、裁图 |
| `document_info/` | 曲名、作者、调弦和速度 |
| `measure_ocr/` | 小节识别、前文传递和续跑 |
| `gp5_export/` | 指法、奏法和 GP5 写出 |
| `shared/` | 小节文本、音乐约束、默认配置和模型调用 |
| `webapp/` | 本地网页、编辑流程和项目存储 |
| `scripts/` | 安装、启动与发布打包 |

小节记录保存各自的谱面类型，OCR 按该类型选择提示词。模型池共用一个 GLM 基座，两项 OCR 任务切换适配器。Paddle 在独立环境中运行。

`webapp/static/` 使用原生 ES modules，无需前端构建。项目写操作携带读取时的 revision；服务端通过 `If-Match` 检查，避免覆盖其他页面的新修改。识别失败的小节需要人工确认后才能导出。

项目迁移使用网页中的项目备份 ZIP；CLI 清单含本机路径，不适合直接搬到另一台电脑。续跑会核对输入、模型和识别参数，输入变化时使用新输出目录。

## 打包与依赖

取得当前权重后执行 `uv run --no-project --python 3.11 scripts/package_release.py`，生成启动 ZIP 和校验文件。发布前检查包内启动文件和示例导出，更新 CHANGELOG。启动 ZIP 不包含测试和 CI 配置。

模型变更需更新 `weights/manifest.json`、模型卡、训练配置和评测结果。数据、模型缓存、用户项目及私人曲谱放在 Git 忽略的目录中。

当前 Transformers 5.8.0 有[已记录的依赖公告](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)。升级时需同时验证 LLaMA-Factory 的版本兼容性及模型训练、推理、保存行为。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
