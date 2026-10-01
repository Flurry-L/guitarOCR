# GuitarOCR

把乐谱 PDF 或图片转换为可编辑的音乐。在多轨工作台对照原谱校对，导出 GP5、MusicXML 或完整项目备份。

![多轨乐谱工作台](docs/assets/score-editor-0.1.webp)

支持 TAB、五线谱、五线谱＋TAB，以及吉他、贝斯、鼓、钢琴和其他旋律乐器的多轨谱面。谱面信息、分轨和小节识别共用一个 OCR 模型；辅助模型负责版面检测和拍号／调号。识别结果包含音符、节奏、奏法、和弦名称和指法图，实际效果及评测条件见[模型评测](docs/model-evaluation.md)。

## 使用

桌面应用默认在本机运行，也可以连接自己的服务器。客户端使用 Tauri 和 Rust，随包提供 ONNX Runtime、llama.cpp 与 PDFium，不安装或下载 Python。首次识别确认下载约 1.775 GB 模型；打开示例、校对和导出无需模型。模型缓存与项目独立于应用版本，更新应用不重复下载权重。

[下载发布版本](https://github.com/Flurry-L/guitarOCR/releases) · [安装与设备选择](docs/setup.md) · [自托管部署](docs/server.md)

1. 导入 PDF 或图片，选择「开始识别」；也可先检查区域和谱面信息。
2. 在任务中心查看进度、取消或续跑，已完成结果会保留。
3. 按音轨浏览谱面，对照原图编辑小节、和弦与指法，保存并确认。
4. 导出 GP5、MusicXML 或项目 ZIP，之后可以从项目库继续编辑。

首次使用可打开内置示例。详细操作见[工作台说明](docs/webui.md)。识别和导出不需要安装 Guitar Pro。

## 仓库布局

| 目录 | 职责 |
| --- | --- |
| `scorelib/` | 乐谱 IR、音乐规则、GP5／MusicXML 导出；Rust 核心与 Python 接口 |
| `gpbridge/` | Guitar Pro 原生操控与数据读取，隔离的 Wine／并行会话；C++ 与 Python 接口 |
| `research/` | Python 数据生产、模型、训练、实验推理、评测和 ONNX／GGUF 转换 |
| `weights/` | 当前生产权重、训练与评测记录、部署模型清单 |
| `engine/` | 原生模型加载、设备选择、推理排队、并发和取消 |
| `backend/` | 共用项目业务、识谱流程、编辑保存、HTTP 与服务端账号 |
| `ui/` | 共用工作台、账号页面、桌面启动界面 |
| `desktop/` | Tauri 窗口、系统集成和客户端打包 |
| `deploy/` | 自托管服务配置；服务与桌面使用同一个后端 |
| `scripts/` | 跨模块构建与发布工具 |

桌面和自托管服务复用 `backend → engine + scorelib`；训练迭代通过 Python 使用 `scorelib` 和 `gpbridge`。UI 不加载模型，桌面壳不实现另一套项目业务。生成数据、临时产物和工具环境分别写入 Git 忽略的 `database/`、`output/`、`tools/`。

## 开发入口

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Flurry-L/guitarOCR.git
cd guitarOCR
# Rust 1.90+；先运行工作台，导入示例即可编辑和导出
cargo run --locked -p guitarocr-backend -- --assets ui/workbench --port 7860
```

打开 `http://127.0.0.1:7860`。启用本机识别和构建安装包见[原生打包](docs/native-packaging.md)。独立 Python 研究环境：

```bash
uv sync --locked --package guitarocr-research --python 3.11 --extra dev
uv run --no-sync python -m research.data.run --help
uv run --no-sync python -m research.training.measures --help
uv run --no-sync python -m research.inference.pipeline.run --help
```

[参与开发](CONTRIBUTING.md) · [数据生产](docs/data.md) · [训练与评测](docs/training.md) · [乐谱表示](docs/score-text.md) · [模型说明](weights/README.md)

项目许可证尚未选定，第三方授权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
