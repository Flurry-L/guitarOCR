# GuitarOCR

把乐谱 PDF 或图片转换为可编辑的音乐。自动识别后，在多轨工作台对照原谱校对，导出 GP5、MusicXML 或完整项目备份。

工作台提供一键识别、项目库、任务中心和明暗主题；桌面端、本地网页与多人服务使用同一套乐谱编辑器。

![多轨乐谱工作台：音轨导航、谱面编辑与原谱对照](docs/assets/score-editor-0.1.webp)

吉他和贝斯支持 TAB、五线谱和五线谱＋TAB，自动判断谱面类型；鼓、钢琴及其他旋律乐器使用五线谱。总谱按乐器、谱表和共同小节时间轴组织，支持钢琴双谱表和混合记谱。识别结果保存为独立的结构化乐谱，可导出 GP5 或 MusicXML；GP5 对超出单轨弦数和声部限制的内容拆轨保存。具体效果见[模型评测](docs/model-evaluation.md)。模型采用通用 safetensors 格式，并非 TensorRT 专用；另提供 [GGUF 可选模型](docs/setup.md#gguf-可选后端)。各平台的本地与远程支持见[平台支持](docs/setup.md#平台支持)。

当前源码默认模型使用等比例乐谱图像、音乐词表和 MTP 4，并行识别小节。评测分别列出音符节奏、指法、奏法和完整 PDF 的效果。0.1 发布版采用当前工作台和模型。

## 下载与启动

推荐使用 NVIDIA GPU 自行部署，或连接已部署的 GPU 服务。手机和低端电脑可以通过网页使用服务器；本机 CPU 识别较慢。显存、内存和驱动要求见[硬件要求](docs/setup.md#硬件要求)。

桌面端使用 Tauri，可连接服务器，也可在本机识别或校对。连接服务器无需安装 Python 或下载模型。

| 系统 | 安装包 |
| --- | --- |
| Windows x64 | [安装程序](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_x64-setup.exe) |
| macOS Apple Silicon | [DMG](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_aarch64.dmg) |
| macOS Intel | [DMG](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_x64.dmg) |
| Linux x64 | [DEB](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_amd64.deb) · [AppImage](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_amd64.AppImage) |

安装包尚未签名或公证。本机 GPU 识别支持 Windows / Linux x64；macOS 可连接服务器，或在本机校对和导出。使用方式见[安装说明](docs/setup.md)。

也可下载 [GuitarOCR-0.1.0.zip](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR-0.1.0.zip)，完整解压到可写文件夹，无需 Git：

- Windows 双击 `start.bat`。
- Linux x64 在项目目录运行 `bash start.sh`。

桌面安装包和启动 ZIP 为轻量启动包。首次本机识别自动下载约 5.7 GB 的当前模型，以及 Python 和运行依赖；下载可续传，建议预留 50 GB 磁盘。连接服务或仅校对无需下载模型。安装后打开 http://127.0.0.1:7860，使用期间保留启动窗口。中断后重跑同一脚本即可。

Python、Python 包和 PyTorch 默认使用国内源，失败时尝试官方源。ZIP 下载仍需访问 GitHub，uv 和 Windows C++ 运行库从官方站点下载。安装失败见[故障排查](docs/setup.md#故障排查)。

## 转换乐谱

1. 导入 PDF 或图片，调整顺序，选择「开始识别」。也可先导入，逐步检查区域和谱面信息。
2. 在任务中心查看进度、停止或继续识别。关闭网页后，运行中的服务会继续处理。
3. 进入校对，按音轨查看谱面；用小节导航定位待检查内容，对照原图编辑并确认。
4. 导出 GP5、MusicXML，或保存项目 ZIP。在项目库搜索、筛选并继续已有项目。

详细操作见[工作台说明](docs/webui.md)。转换乐谱无需安装 Guitar Pro。

## 部署多人服务

首页无需登录，提交识别时登录或注册。GPU 按队列处理任务，关闭网页后继续运行，结果保存在账号中。管理员可管理用户、查看用量和队列，并在网页安装仓库更新。

部署需要 Linux 和 NVIDIA GPU，步骤见[服务端部署](docs/server.md)。

## 源码与开发

安装 Git 和 [Git LFS](https://git-lfs.com/) 后执行：

```bash
git lfs install
git clone --depth 1 --single-branch --branch main https://github.com/Flurry-L/guitarOCR.git
cd guitarOCR
git lfs pull
```

GitHub 自动生成的 Source code ZIP 可能只含权重指针，直接使用请下载上面的启动 ZIP；启动器自动获取该版本的完整权重。

仓库按产品职责组织，共用推理和乐谱表示，不在各界面重复实现音乐规则：

| 层次 | 入口 |
| --- | --- |
| 工作台、主题与乐谱编辑器 | `webapp/` |
| 账号、队列与多人服务 | `server/` |
| 桌面窗口与本机环境管理 | `desktop/` |
| 转换流程、项目保存与迁移 | `pipeline/` |
| 版面、信息、音符识别与导出 | `layout/`、`document_info/`、`measure_ocr/`、`gp5_export/` |
| 中间表示、模型运行与数据生产 | `shared/`、`datagen/`、`weights/` |

只开发界面或校对已有项目，可不加载模型启动：

```bash
uv run --no-sync guitarocr-web --edit-only
```

完成[推理环境安装](docs/setup.md#手动安装模型推理)后，可从命令行运行：

```bash
uv run --no-sync guitarocr-gp /path/to/score.pdf --output output/score
uv run --no-sync guitarocr-check
```

| 需要做什么 | 文档 |
| --- | --- |
| 理解代码、检查和打包 | [参与开发](CONTRIBUTING.md) |
| 编辑或导入小节文本 | [格式参考](docs/score-text.md) |
| 生成训练数据 | [数据生产](docs/data.md) |
| 训练和评测 | [训练说明](docs/training.md) |
| 查看模型文件与参数 | [模型说明](weights/README.md) |

项目许可证尚未选定，第三方授权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
