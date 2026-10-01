# GuitarOCR

将乐谱 PDF 或图片识别为可编辑的乐谱，对照原图校对后导出 GP5、MusicXML 或项目备份。

![乐谱编辑器](docs/assets/score-editor-0.1.webp)

支持 TAB、五线谱、五线谱＋TAB，以及吉他、贝斯、鼓、钢琴等乐器的多轨谱面。可校对音符、节奏、奏法、和弦名和指法图。[识别效果与评测条件](docs/model-evaluation.md)。

## 使用

[下载应用](https://github.com/Flurry-L/guitarOCR/releases/tag/v0.1.0) · [安装与设备要求](docs/setup.md) · [自托管部署](docs/server.md)

提供 Apple 芯片 Mac 和 Windows 客户端。Windows 自动选择设备，兼容的 NVIDIA 显卡按需获取加速组件。Linux、Intel Mac 和自托管服务从源码构建。

1. 打开应用，导入 PDF 或图片，点击「开始识别」。首次识别需下载约 1.8 GB 模型。
2. 对照原谱修改音符，按「确认并继续」逐小节核对。
3. 导出 GP5 或 MusicXML；需要保留原图和校对进度时，下载项目 ZIP。

「项目」中可搜索乐谱、查看识别进度、停止或继续处理。内置示例可直接练习编辑，无需下载模型。详细操作见[使用说明](docs/webui.md)。

## 代码入口

| 目录 | 职责 |
| --- | --- |
| `scorelib/` | 乐谱 IR、音乐规则、GP5／MusicXML 转换；Rust 核心与 Python 接口 |
| `gpbridge/` | Guitar Pro 操控与原始数据导出，Wine 会话和并行调度 |
| `research/` | 数据生产、模型、训练、实验推理、评测和模型转换 |
| `weights/` | 当前权重、评测结果和部署清单 |
| `engine/` | 模型运行时、设备管理、推理调度 |
| `backend/` | 项目、识谱流程、编辑保存、HTTP 与账号 |
| `ui/` | 共用工作台、账号管理、桌面启动页 |
| `desktop/` | Tauri 窗口、系统集成和打包 |
| `deploy/` | 自托管服务配置 |
| `scripts/` | 构建与发布工具 |

先运行编辑器：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Flurry-L/guitarOCR.git
cd guitarOCR
cargo run --locked -p guitarocr-backend -- --assets ui/workbench --port 7860
```

需要 Rust 1.90+。打开 `http://127.0.0.1:7860`，导入示例即可编辑和导出；启用识别见[应用构建](docs/native-packaging.md)。

[参与开发](CONTRIBUTING.md) · [数据生产](docs/data.md) · [训练与评测](docs/training.md) · [乐谱表示](docs/score-text.md) · [模型说明](weights/README.md)

项目许可证尚未选定，第三方授权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
