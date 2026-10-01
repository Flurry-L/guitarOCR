# GuitarOCR

把乐谱 PDF 或图片转换为可编辑的音乐。自动识别后，在多轨工作台对照原谱校对，导出 GP5、MusicXML 或完整项目备份。

工作台提供一键识别、项目库、任务中心和明暗主题；桌面端、本地网页与多人服务使用同一套乐谱编辑器。

![多轨乐谱工作台：音轨导航、谱面编辑与原谱对照](docs/assets/score-editor-0.1.webp)

吉他和贝斯支持 TAB、五线谱和五线谱＋TAB，自动判断谱面类型；鼓、钢琴及其他旋律乐器使用五线谱。总谱按乐器、谱表和共同小节时间轴组织，支持钢琴双谱表和混合记谱。识别结果保存为独立的结构化乐谱，可导出 GP5 或 MusicXML；GP5 对超出单轨弦数和声部限制的内容拆轨保存。具体效果见[模型评测](docs/model-evaluation.md)。模型采用通用 safetensors 格式，并非 TensorRT 专用；本机轻量路径使用 [GGUF 模型](docs/setup.md#gguf-本机后端)。各平台的本地与远程支持见[平台支持](docs/setup.md#平台支持)。

谱面信息、分轨和小节识别共用一个 OCR 模型，通过不同提示词执行任务。模型使用等比例图像和音乐词表，并行识别小节；Linux 原生 GPU 后端支持 MTP 加速。和弦名称、指法图与移调指令分别识别，编辑器可修改和弦及横按，保存到项目并导出。

## 下载与启动

本机识别是默认方向：自动选择 CPU、Apple Metal 或 NVIDIA CUDA，远程服务作为可选增强。手机和低端电脑也可通过网页使用服务器；CPU 的速度需要按实际设备验证。显存、内存和驱动要求见[硬件要求](docs/setup.md#硬件要求)。

桌面端使用 Tauri，可连接服务器，也可在本机识别或校对。连接服务器无需安装 Python 或下载模型。

| 系统 | 安装包 |
| --- | --- |
| Windows x64 | [安装程序](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_x64-setup.exe) |
| macOS Apple Silicon | [DMG](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_aarch64.dmg) |
| macOS Intel | [DMG](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_x64.dmg) |
| Linux x64 | [DEB](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_amd64.deb) · [AppImage](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR_0.1.0_amd64.AppImage) |

以上链接为此前发布的 0.1.0 安装包，不包含本次源码中的原生客户端改动。当前桌面客户端已改为 Tauri＋Rust 应用服务，使用 ONNX Runtime 与 llama.cpp GGUF；分发和运行不携带、不下载 Python。模型按需校验和下载，远程连接仍为可选入口。Python 数据、训练、评测、推理及服务器工具独立保留。当前源码的构建与验证范围见[安装说明](docs/setup.md)及[原生打包](docs/native-packaging.md)，不将未验证的平台称为可发布版本。

也可下载 [GuitarOCR-0.1.0.zip](https://github.com/Flurry-L/guitarOCR/releases/download/v0.1.0/GuitarOCR-0.1.0.zip)，完整解压到可写文件夹，无需 Git：

- Windows 双击 `start.bat`。
- Linux x64 在项目目录运行 `bash start.sh`。

原生桌面包包含应用和经校验的本机运行组件；GGUF 与 ONNX 模型约 1.775 GB，首次识别先确认下载，已有缓存不重复下载。仅校对或连接服务不下载模型。桌面缓存与项目保留在应用数据目录；安装器和 Python 研究脚本的路径、依赖分开，详见[安装说明](docs/setup.md#模型环境与更新)。

上面的 ZIP 与 `start.*` 是既有 Python 命令行/研究入口，不是新的原生桌面分发方式。研究环境继续使用原有镜像及官方源回退。

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
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --single-branch --branch main https://github.com/Flurry-L/guitarOCR.git
cd guitarOCR
```

GitHub 自动生成的 Source code ZIP 可能只含权重指针，直接使用请下载上面的启动 ZIP；启动器自动获取该版本的完整权重。

原生桌面开发从[客户端快速检查](CONTRIBUTING.md#原生客户端快速检查)和[原生打包](docs/native-packaging.md)开始，使用 Node.js 与 Rust，不需要 Python 或模型。

下面是独立 Python 研究入口；先运行 `bash install.sh` 准备研究环境。只校对已有项目时可不加载模型启动：

```bash
uv run --no-sync guitarocr-web --edit-only
```

完成[推理环境安装](docs/setup.md#python-研究与服务脚本)后，可从命令行运行：

```bash
bash scripts/bootstrap.sh run -- -m pipeline.run /path/to/score.pdf --output output/score
bash scripts/bootstrap.sh check
```

| 需要做什么 | 文档 |
| --- | --- |
| 理解代码、检查和打包 | [参与开发](CONTRIBUTING.md) |
| 编辑或导入小节文本 | [格式参考](docs/score-text.md) |
| 生成训练数据 | [数据生产](docs/data.md) |
| 训练和评测 | [训练说明](docs/training.md) |
| 查看模型文件与参数 | [模型说明](weights/README.md) |

项目许可证尚未选定，第三方授权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
