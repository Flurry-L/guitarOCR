# 处理流程与代码结构

每一步通过 `run(...) -> Path` 返回自己的 `manifest.json` 路径，也可以通过 `python -m <步骤>.run` 单独执行。总入口 `pipeline/run.py` 负责按顺序调用各步骤；算法、模型和后处理属于各步骤自己的实现。

## 结果清单

步骤清单共有 `schema_version: "1.0"` 和 `stage` 字段。下游读取时检查这两个字段，避免把其他步骤的文件误当作输入。清单及其中引用的文件要一起保留；当前清单使用绝对路径，移动整个输出目录后需重新生成对应路径。

| 步骤 | 输入参数 | 结果的主要字段 |
| --- | --- | --- |
| `layout.run` | PDF／图片路径列表 | `mode`、`inputs`、`info_source`、`regions`、`records` |
| `document_info.run` | `layout` 清单 | `layout`、`document_metadata`、`title`、`artist`、`tuning_used`、`capo`、`predictions` |
| `measure_ocr.run` | `layout`、`info` 清单 | `mode`、`measures`、`score_text`、`recognition_log`、`records`、标题／作者／调弦等信息 |
| `gp5_export.run` | `recognition` 清单 | `gp5`、`encoding_report` |

`layout.records` 按阅读顺序列出小节编号、页码、谱行编号、位置框、裁图路径和页面来源，并保存各小节的 `mode`、`mode_source`，模型检测时还保留 `detected_mode` 和 `score`。`pages` 保存页面类型；顶层 `mode` 是文档的汇总类型，不覆盖各小节类型。`regions` 列出页眉和速度裁图。PDF 的几何定位、图片模型定位及回退逻辑在 `layout/` 内完成。

谱面信息步骤根据 `info_source` 选择图片 OCR 或 PDF 信息提取，并处理用户传入的标题、作者、调弦和变调夹。小节 OCR 按各记录的类型选择提示词和校验方式，使用解析后的调弦，逐小节传递上一小节上下文；类型变化时将视觉上下文重置为 `START`。没有小节类型字段的历史清单沿用顶层类型。无效输出经重试仍失败时，以整小节休止占位并标记待检查。

总结果清单使用 `schema_version: "3.0"`，在 `stages` 中记录各步骤的状态和清单位置。失败时保留失败步骤及错误，后续步骤不执行。成功时汇总最终小节文本、GP5、识别日志路径。

## 重跑和恢复

- 修改 GP5 导出逻辑后，可以只运行 `gp5_export.run`，直接读取已有的小节 OCR 清单。
- 修改小节识别后，可以只运行 `measure_ocr.run`，复用版面与信息结果。
- `measure_ocr.run --resume` 校验裁图、上游清单、模型路径及主要模型文件状态、识别参数；一致时复用已经接受的小节。
- 总入口的 `--resume` 仍执行前两步，再复用小节结果。全部小节已接受时，小节识别步骤不加载 GLM 模型。
- 若输入或参数改变，请使用新输出目录，或省略 `--resume` 重做小节识别。

## 代码结构

| 目录 | 职责 |
| --- | --- |
| `datagen/` | GP 选源、原生渲染、标签和数据划分 |
| `layout/` | 页面渲染、区域检测、类型判断和裁图 |
| `document_info/` | 曲名、作者、调弦和速度读取 |
| `measure_ocr/` | 小节识别、前文传递和失败重试 |
| `gp5_export/` | 指法、奏法映射和 GP5 写出 |
| `pipeline/` | 命令行整谱流程 |
| `webapp/` | 本地网页、校对、任务和项目管理 |
| `shared/` | 小节解析、音乐约束、模型调用和公共配置 |
| `scripts/` | 安装、启动、校验与打包 |

训练和评测入口位于对应任务目录。所有模块共用根目录的 `pyproject.toml` 和 `uv.lock`。`database/`、`output/`、`tools/` 分别存放数据、运行结果和本地工具，由 Git 忽略。

## 版面检测与类型

自动类型使用 PP-DocLayoutV3，矢量 PDF 也先转为页面图像。模型输出三类小节和速度区域；谱头取首个小节上方区域。页面和文档类型按小节检测分数加权汇总，各小节自己的类型用于 OCR。

CLI 保留 `--layout-source auto|image|geometry`。手动指定整份谱面类型且 source 为 auto 时，矢量 PDF 优先使用几何定位；显式 geometry 或旧两类权重使用谱线规则补充类型。网页只提供谱面类型和区域编辑，定位来源使用 auto。

## 外部依赖

两套 LoRA 和图片版面模型随仓库发布在 `weights/`，通过 Git LFS 获取。GLM 模型仅在实际推理时加载；GP5 导出和纯几何版面处理可以独立运行。PaddleX 使用独立 Python 环境，通过 `layout.detector` 子进程调用。完整安装命令见 [setup.md](setup.md)。

GP8 导出器既支持 `python -m datagen.export_scores`，也支持 `python datagen/export_scores.py`。后一种入口用于 Wine 内的 Python，原生 DLL 仍从 `datagen/native-bin/` 加载。C++ 源码、符号定义和构建脚本位于 `datagen/native-source/`，构建命令见 [native-build.md](native-build.md)。

## 交互式编辑

前端使用原生 HTML、CSS 和 ES modules，无需构建。`webapp/workflow.py` 管理持久化会话，调用上述阶段；`layout/edit.py` 验证原图坐标并重新裁图。Web 请求和单队列在 `webapp/app.py`，前端文件在 `webapp/static/`。每次修改产生新阶段目录，更新会话指针并失效下游引用。`review_measures` 保存需人工确认的占位小节；小节记录包含 `needs_review`、`fallback_reason` 和 `manually_edited`。

整谱与交互流程通过 `shared.glm_backend.BackendPool` 复用基座；独立步骤仍能自行加载模型。总清单在存在占位小节时使用 `needs_review`，默认 GP5 路径为 null；显式允许导出也保留该状态和小节列表。

清单字段类型集中在 `shared/schema.py`。Web 项目 revision 用于乐观并发检查：更新区域、信息、小节、重识别、导出和删除都需携带 `If-Match: <revision>`。任务状态独立保存在 job.json；成功的手动修复会清除旧错误。识别中断保存 ocr_task 与逐小节日志，用于同机续跑。

`webapp/projects.py` 负责项目 ZIP。包内文件有 SHA-256 清单，内部引用转换为 project:// 相对路径；导入时校验路径、大小和哈希并建立新项目编号。跨机器迁移采用此入口；独立 CLI 阶段清单仍使用本机路径。

面向用户的文本路径为 `score_text`，内容以 `MEASURE` 开头；阶段清单的旧内部序列字段继续保留，以兼容项目和既有调用。显示名称转换由 `shared/score_text.py` 处理，不改变模型训练协议。
