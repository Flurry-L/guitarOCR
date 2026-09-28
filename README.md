# GuitarOCR

把乐谱 PDF 或图片转换为可编辑的 GP5 文件。在工作台检查小节框、校对音符，再下载到 Guitar Pro 中继续编辑。

吉他和贝斯支持 TAB、五线谱和五线谱＋TAB，自动判断谱面类型；鼓、钢琴及其他旋律乐器使用五线谱。识别单位为单轨、单谱表，GP5 最多导出七弦。模型主要使用 Guitar Pro 原生页面训练，识别结果需要校对，具体效果见[模型评测](docs/model-evaluation.md)。

## 下载与启动

推荐使用 NVIDIA GPU 自行部署，或连接已部署的 GPU 服务。手机和低端电脑可以通过网页使用服务器；本机 CPU 识别较慢。显存、内存和驱动要求见[硬件要求](docs/setup.md#硬件要求)。

桌面端使用 Tauri，可连接服务器，也可在本机识别或校对。连接服务器无需安装 Python 或下载模型。

| 系统 | 安装包 |
| --- | --- |
| Windows x64 | [安装程序](https://github.com/Flurry-L/guitarOCR/releases/download/v0.2.2/GuitarOCR_0.2.2_x64-setup.exe) |
| macOS Apple Silicon | [DMG](https://github.com/Flurry-L/guitarOCR/releases/download/v0.2.2/GuitarOCR_0.2.2_aarch64.dmg) |
| macOS Intel | [DMG](https://github.com/Flurry-L/guitarOCR/releases/download/v0.2.2/GuitarOCR_0.2.2_x64.dmg) |
| Linux x64 | [DEB](https://github.com/Flurry-L/guitarOCR/releases/download/v0.2.2/GuitarOCR_0.2.2_amd64.deb) · [AppImage](https://github.com/Flurry-L/guitarOCR/releases/download/v0.2.2/GuitarOCR_0.2.2_amd64.AppImage) |

安装包尚未签名或公证。本机 GPU 识别支持 Windows / Linux x64；macOS 可连接服务器，或在本机校对和导出。使用方式见[安装说明](docs/setup.md)。

也可下载包含完整模型的 [GuitarOCR-0.2.2.zip](https://github.com/Flurry-L/guitarOCR/releases/download/v0.2.2/GuitarOCR-0.2.2.zip)，完整解压到可写文件夹，无需 Git：

- Windows 双击 `start.bat`。
- Linux x64 在项目目录运行 `bash start.sh`。

桌面安装包和启动 ZIP 均包含 GLM-OCR 基座、两套 OCR 适配器和版面模型，约 2 GB。首次本机识别会联网安装 Python 和运行依赖，仍需下载数 GB，建议预留 25 GB 磁盘。安装后打开 http://127.0.0.1:7860，使用期间保留启动窗口。中断后重跑同一脚本即可。

Python、Python 包和 PyTorch 默认使用国内源，失败时尝试官方源。ZIP 下载仍需访问 GitHub，uv 和 Windows C++ 运行库从官方站点下载。安装失败见[故障排查](docs/setup.md#故障排查)。

## 转换乐谱

1. 导入 PDF 或图片，多文件可调整顺序。
2. 自动检测小节框，检查位置、阅读顺序和谱面类型。
3. 读取并核对曲名、乐器、调弦、移调和速度。
4. 在整谱中点选音符，用工具栏或键盘编辑。红色表示识别失败，黄色表示需要核对。
5. 下载 GP5，或下载项目备份，稍后继续编辑。

识别可停止和继续，也可重试单个小节。详细操作见[工作台说明](docs/webui.md)。转换 PDF 或图片无需安装 Guitar Pro。

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

GitHub 自动生成的 Source code ZIP 可能只含权重指针，直接使用请下载上面的启动 ZIP。

完成[推理环境安装](docs/setup.md#手动安装模型推理)后，可从命令行运行：

```bash
uv run --no-sync guitarocr-gp /path/to/score.pdf --output output/score
uv run --no-sync guitarocr-check --hashes
```

| 需要做什么 | 文档 |
| --- | --- |
| 理解代码、检查和打包 | [参与开发](CONTRIBUTING.md) |
| 编辑或导入小节文本 | [格式参考](docs/score-text.md) |
| 生成训练数据 | [数据生产](docs/data.md) |
| 训练和评测 | [训练说明](docs/training.md) |
| 查看模型文件与参数 | [模型说明](weights/README.md) |

项目许可证尚未选定，第三方授权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
