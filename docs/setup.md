# 安装与启动

从 [Release](https://github.com/Flurry-L/guitarOCR/releases/latest) 下载对应系统的桌面安装包或启动 ZIP。开发者也可[克隆源码](../README.md#源码与开发)。

## 桌面应用

Windows 运行安装程序，macOS 打开 DMG 后将应用拖入 Applications，Linux 安装 DEB 或运行 AppImage。支持 Windows x64、macOS 12.3 及更新版本、Ubuntu 22.04 及更新版本；macOS 同时提供 Intel 和 Apple Silicon 包。

启动后选择：

- **连接 GPU 服务**：填写已部署服务的 HTTPS 地址，登录后提交识别。关闭窗口后服务器仍继续处理。
- **使用本机 GPU**：支持 Windows / Linux x64，模型已随安装包提供，首次联网安装 Python 和运行依赖。
- **仅校对与导出**：下载 Python 和基础依赖，恢复项目备份后编辑和导出 GP5。macOS 也可使用。

安装包尚未签名或公证，系统可能提示未知发布者。Windows 缺少 WebView2 时，安装程序会从 Microsoft 下载；本机 GPU 环境还需 [Microsoft Visual C++ x64 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)。

GP5 和备份保存到系统「下载」目录，重名时自动另取名称；没有下载目录时使用应用数据目录。点击「打开数据目录」可查看项目、环境和日志。升级保留此目录，删除它会同时删除本机项目。退出应用会停止本机任务，退出前请保存编辑。

## 脚本安装

将 ZIP 完整解压到可写文件夹，以下命令在该文件夹执行。首次安装后自动打开 http://127.0.0.1:7860，以后复用环境。使用期间保留启动窗口，中断后重跑同一脚本。

| 操作 | Windows | Linux x64 |
| --- | --- | --- |
| 安装并启动 | `start.bat` | `bash start.sh` |
| 只安装或修复 | `install.bat` | `bash install.sh` |
| 使用 CPU | `start.bat --device cpu` | `bash start.sh --device cpu` |
| 使用 NVIDIA GPU | `start.bat --device cuda` | `bash start.sh --device cuda` |
| 换端口 | `start.bat --port 7861` | `bash start.sh --port 7861` |
| 不打开浏览器 | `start.bat --no-browser` | `bash start.sh --no-browser` |

脚本安装 uv、Python 3.11 和固定版本依赖。NVIDIA 驱动 580 或更新版本可自动使用 CUDA 13，否则使用 CPU。Linux CUDA 安装使用 GPU 版面检测及独立 vLLM 推理环境；其他安装使用 Transformers 和独立 Paddle 环境。无需提前安装 Python 或 CUDA Toolkit。本次完整流程和性能评估使用 Linux、H100。

Ubuntu / Debian 系统依赖：

```bash
sudo apt-get install -y curl ca-certificates libgl1 libglib2.0-0
```

环境保存在 `tools/webui-venv/`、`tools/webui-paddle-venv/` 和 Linux CUDA 使用的 `tools/vllm-venv/`，项目在 `output/webui/`，日志在 `output/logs/`。

## 硬件要求

推荐自行部署 NVIDIA GPU，或连接已部署的 GPU 服务。远程用户的设备只需能打开网页和上传文件。

| 本机方式 | 建议配置 | 主要影响因素 |
| --- | --- | --- |
| GPU 识别 | 建议 24 GB 显存、32 GB 内存 | 两套常驻 OCR 引擎、视觉缓存及小节批量 |
| CPU 识别 | 建议 32 GB 内存，适合试用 | CPU 性能、内存带宽，速度较慢 |

当前源码版本建议为模型、独立推理环境和编译缓存预留 50 GB 磁盘。GLM-OCR 基座约 13.3 亿参数，两个 OCR 任务分别提供合并模型和 LoRA 适配器。大图、长小节和并发任务会增加显存与内存需求。

工作台在启动时预热模型，之后同一首谱内的小节批量识别。H100 的首次加载、预热后识别和完整 PDF 结果见[模型评测](model-evaluation.md)。远程服务还需计算上传与排队时间。

## 下载源与模型路径

下载失败时自动尝试备用源，模型下载支持续传和 SHA-256 校验。

| 内容 | 默认来源 | 备用来源 |
| --- | --- | --- |
| Python 3.11 | 南京大学镜像 | Astral GitHub Release |
| Python 包和 Paddle | 清华 PyPI 镜像 | PyPI |
| PyTorch | 上海交大镜像 | PyTorch 官方索引 |
| GLM-OCR 基座 | 魔搭 | Hugging Face |

已有 uv 配置优先用于普通依赖；`UV_PYTHON_INSTALL_MIRROR` 可指定 Python 来源，`HF_ENDPOINT` 可指定模型来源。ZIP、uv 和 Windows C++ 运行库仍需访问各自的官方站点。

桌面安装包和启动 ZIP 已包含完整模型，约 2 GB。首次本机识别会展开包内模型并记录版本，之后直接复用本地文件。桌面窗口显示当前安装步骤、正在下载的组件和包内模型的展开进度。Python 与运行依赖仍需联网安装。源码安装才需要下载模型，版本和文件哈希见 `weights/manifest.json`。

默认模型路径相对安装目录解析，可用 `--model`、`--adapter`、`--info-adapter`、`--layout-model-dir` 覆盖。多人服务将模型位置写入配置文件。

## 手动安装轻量编辑与导出

安装 uv 后执行：

```bash
uv sync --locked --python 3.11 --extra webui
uv run --no-sync guitarocr-web --device cpu
```

基础安装不拉取 Torch。这个环境可以导入、画框、校对和导出；自动识别还需要下面的模型环境。

## 手动安装模型推理

Ubuntu / Debian 可先安装系统工具：

```bash
sudo apt-get install -y git git-lfs curl build-essential libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.12.17/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git lfs install
git lfs pull
uv sync --locked --python 3.11 --extra glm-ocr --extra webui
uv run --no-sync hf download zai-org/GLM-OCR --revision ca5d8b3e287e52589e37c28385d9655ee4372f9d --local-dir tools/models/GLM-OCR
```

同步一次后，文档中的运行命令统一使用 `uv run --no-sync`，避免不同 extras 组合反复移除 UI 或训练依赖。`uv sync` 的默认 Torch wheel 在 Linux 使用 CUDA 13；CPU / Windows GPU 用户推荐用一键安装器选择对应官方 wheel。

### 独立 Paddle 环境

Linux NVIDIA GPU 版：

```bash
uv venv --python 3.11 --seed tools/paddlex-venv
uv pip install --python tools/paddlex-venv/bin/python "paddlepaddle-gpu==3.2.0" --index https://www.paddlepaddle.org.cn/packages/stable/cu126/
git clone https://github.com/PaddlePaddle/PaddleX.git tools/PaddleX
git -C tools/PaddleX checkout ffb64904d23708863ff5b8da312a5cbd52a7f462
uv pip install --python tools/paddlex-venv/bin/python -e "tools/PaddleX[ocr,cv]" "numpy==1.26.4" "Pillow>=12.3,<13" "pypdfium2>=5.13,<6" "pdfplumber>=0.11.10,<1"
uv pip install --python tools/paddlex-venv/bin/python --no-deps -e .
tools/paddlex-venv/bin/python -c "import paddle; paddle.utils.run_check()"
uv run --no-sync guitarocr-check --hashes --device cuda --layout-python tools/paddlex-venv/bin/python
uv run --no-sync guitarocr-web --layout-python tools/paddlex-venv/bin/python
```

CPU 版把 Paddle 安装命令换成 `uv pip install --python tools/paddlex-venv/bin/python "paddlepaddle==3.2.0"`。Windows 对应解释器路径是 `tools/paddlex-venv/Scripts/python.exe`。推理不需要 Wine、Guitar Pro 或原生 DLL。

## 服务地址

单用户工作台没有登录认证，默认仅监听本机。远程设备可用 `ssh -L 7860:127.0.0.1:7860 user@server` 转发访问；公网部署使用[多人服务](server.md)。

`--host` 设置监听地址，`--allow-host scores.example` 允许自定义主机名，可重复传入。`0.0.0.0` 不会自动允许所有主机名。Host 检查防止 DNS 重绑定，不能代替登录认证。

## 故障排查

先查看启动窗口里的第一条报错，完整日志在 `output/logs/`。修复后重跑安装或启动脚本，会复用已下载内容。

| 问题 | 处理 |
| --- | --- |
| 权重是小文本文件或提示 LFS 指针 | 使用 Release 的启动 ZIP；Git 用户运行 `git lfs pull` |
| Python、依赖或模型下载失败 | 检查日志中的地址和网络，重跑脚本。代理可使用 `HTTPS_PROXY` |
| uv 下载失败 | 检查 astral.sh / GitHub 的访问和系统时间 |
| GPU / CUDA 不可用 | 更新 NVIDIA 驱动至 580 或更新版本，或用 `--device cpu` 启动 |
| 显存不足 | 关闭其他占用显存的程序、检查过大的小节框，或连接显存更大的 GPU 服务 |
| Windows 缺 C++ Runtime | 重跑安装脚本，或安装上面的 Microsoft C++ 运行库 |
| Linux 缺 libGL.so.1 或 libglib | `sudo apt-get install libgl1 libglib2.0-0` |
| 端口 7860 被占用 | 打开已有工作台，或加 `--port 7861` |
| 自动框、音符不正确或无法导出 | 按[工作台说明](webui.md)校对并确认待检查小节 |
| 换电脑后项目路径失效 | 在原工作台下载项目备份 ZIP，在新工作台恢复 |

脚本环境检查：Windows 运行 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 check`，Linux 运行 `bash scripts/bootstrap.sh check`。

手动环境检查：`uv run --no-sync guitarocr-check --hashes --device cuda --layout-python tools/paddlex-venv/bin/python`。轻量编辑环境可加 `--core`。
