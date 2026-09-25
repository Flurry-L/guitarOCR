# GuitarOCR

将吉他谱 **PDF（矢量或扫描件）和多张图片转换为可编辑的 GP5 文件**。在本地网页中逐步检查小节框、谱面信息和音符，保存人工修改后继续处理。

当前面向规则排版乐谱。图片版面模型针对纯六线谱 TAB；五线谱和混合谱提供几何定位及识别入口，效果需要另行验证。手写谱和明显透视畸变不在现有训练范围内。

![GuitarOCR 工作台](docs/images/webui.png)

## 快速开始

目前尚未发布可下载的 Release 安装包，请先通过 Git 获取源码和模型权重，再运行启动脚本。

### 获取源码

Windows 先安装 [Git for Windows](https://gitforwindows.org/) 和 [Git LFS](https://git-lfs.com/)，然后打开 PowerShell。Ubuntu / Debian 可在终端安装所需工具：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ca-certificates libgl1 libglib2.0-0
```

在可写目录中执行以下命令（Windows 和 Linux 相同）：

```bash
git lfs install
git clone --depth 1 --single-branch --branch agent/guitar-pro-end-to-end https://github.com/Flurry-L/guitarOCR.git
cd guitarOCR
git lfs pull
```

`git lfs pull` 获取两套 LoRA 和版面权重；GLM-OCR 基座在首次启动时按固定版本下载。GitHub 的 **Code → Download ZIP** 或自动生成的 **Source code.zip** 可能只包含 LFS 指针，而且没有安装器补下载权重所需的 Git 信息，请使用上述完整检出方式。

### Windows

1. 完成上述检出后，在 `guitarOCR` 文件夹中双击 **`start.bat`**。首次自动准备 Python 3.11、依赖和 GLM-OCR 基座，并校验模型权重；按提示等待安装完成。
2. 浏览器会打开 **http://127.0.0.1:7860**。以后仍双击同一个文件启动。

不需要提前安装 Python、Node.js 或 CUDA Toolkit。GPU 加速需要已安装兼容的 NVIDIA 驱动；自动模式在驱动低于 580 或没有 NVIDIA 显卡时选择 CPU。首次需要联网下载数 GB；中断后可重跑脚本。使用期间保留启动窗口。

### Linux x64

完成上述检出后，在 `guitarOCR` 目录中运行：

```bash
bash start.sh
```

`install.bat` / `bash install.sh` 可以只安装、不启动。常用参数、手动安装和硬件验证范围见 [安装说明](docs/setup.md)；下载失败、显存不足等见 [故障排查](docs/troubleshooting.md)。

## 转换一份乐谱

**上传 → 调整小节框 → 校对标题、调弦和速度 → 识别并校对小节 → 下载 GP5**。

- PDF 按页展开；多张图片可调整顺序。用缩略图、页码选择或上一页 / 下一页查看全部页面。
- 自动定位有误时，可移动、缩放、删除和补画小节框。
- 识别可以停止后继续，也可以重试指定小节。识别失败的小节会标成待检查，确认后才能导出。
- 修改曲名、作者、速度或变调夹会更新导出信息；修改调弦或框时，需要重新处理相关识别结果。
- 保存后可刷新继续。下载完整项目 ZIP，可在另一台安装了 GuitarOCR 的电脑上恢复原图和编辑结果。

先用自编的 [示例 PDF](examples/demo.pdf) 或 [示例图片](examples/demo.png) 试一次；[预期 GP5](examples/expected.gp5) 和 [M2 文本](examples/expected.m2) 可用于对照。M2 是本项目记录音符、节奏和奏法的文本格式。详细操作见 [WebUI 使用说明](docs/webui.md)。

## 项目结构

```text
数据生产：datagen → 版面数据 / 谱面信息数据 / 小节数据
乐谱转换：PDF / 图片 → layout → document_info → measure_ocr → gp5_export
                      └──────── pipeline 串联各步骤 ────────┘
```

| 目录 | 职责 |
| --- | --- |
| `datagen/` | 选源、GP8 导出、清单、裁图、标签和统一数据划分 |
| `layout/` | PDF 渲染、小节与速度区域定位、裁图 |
| `document_info/` | 标题、作者、调弦和速度读取 |
| `measure_ocr/` | 顺序识别小节、音乐约束校验、M2 输出 |
| `gp5_export/` | 指法与奏法映射、GP5 写出 |
| `pipeline/` | 命令行整谱流程及状态汇总 |
| `webapp/` | 本地工作台、人工编辑、任务与项目管理 |
| `shared/` | M2、共用模型调用、默认参数、环境检查 |
| `scripts/` | 安装、启动和发布打包 |
| `weights/` | 随项目发布的模型、模型卡与校验清单 |
| `examples/`、`tests/`、`docs/` | 示例、回归验证和文档 |

一套 PP-DocLayoutV3 负责定位；两套 GLM-OCR LoRA 分别读取谱面信息和小节内容，运行时共用一个基座。训练和评测入口跟随对应处理阶段。`database/`、`output/`、`tools/` 分别用于训练数据、运行结果和本地环境，均由 Git 忽略。

## 命令行与开发

手动配置开发环境后，整谱识别和 M2 导出示例：

```bash
uv run --no-sync guitarocr-gp examples/demo.pdf --output output/demo
uv run --no-sync python -m gp5_export.writer examples/expected.m2 output/demo.gp5 --mode tab
uv run --no-sync guitarocr-check --hashes
```

- [安装与外部依赖](docs/setup.md) · [步骤接口](docs/workflow.md) · [M2 格式](docs/m2.md)
- [数据生产](docs/data.md) · [训练与评测](docs/training.md) · [模型说明](weights/README.md)
- [参与开发](CONTRIBUTING.md) · [原生构建](docs/native-build.md) · [验证记录](docs/validation.md)

发布目标、裁图评测和完整乐谱效果分别记录。现有自动测试和小样例运行验证流程行为，不代表逐音准确率。GLM-OCR、Paddle、Qt 等第三方组件以及 Guitar Pro 软件保留各自许可；PDF / 图片转 GP5 无需安装 Guitar Pro，重新生成训练数据时才使用它。

项目许可证尚未选定，模型与原生工具的授权材料仍需补齐。公开发布前的待办与本轮修复见 [开源审核记录](docs/open-source-audit.md)，上游许可及授权范围见 [第三方说明](THIRD_PARTY_NOTICES.md)。
