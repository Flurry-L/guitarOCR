# 安装与启动

从 [Release](https://github.com/Flurry-L/guitarOCR/releases/latest) 下载对应系统的桌面安装包或启动 ZIP。开发者也可[克隆源码](../README.md#源码与开发)。

## 桌面应用

Windows 运行安装程序，macOS 打开 DMG 后将应用拖入 Applications，Linux 安装 DEB 或运行 AppImage。支持 Windows x64、macOS 12.3 及更新版本、Ubuntu 22.04 及更新版本；macOS 同时提供 Intel 和 Apple Silicon 包。

启动后选择：

- **连接 GPU 服务**：填写已部署服务的 HTTPS 地址，登录后提交识别。关闭窗口后服务器仍继续处理。
- **使用本机 GPU**：支持 Windows / Linux x64，首次下载约 5.7 GB 模型并安装 Python 和运行依赖，之后复用本地文件。
- **仅校对与导出**：下载 Python 和基础依赖，恢复项目备份后编辑和导出 GP5。macOS 也可使用。

![0.1 桌面启动器：连接服务、本机识别或仅校对](assets/desktop-0.1.webp)

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

## 平台支持

权重格式是 Hugging Face safetensors。NVIDIA 本机推理使用 vLLM（Linux）或 Transformers；版面检测使用 Paddle。没有依赖某一块 H100 的 TensorRT engine。安装器目前使用 CUDA 13，需 NVIDIA 580 或更新驱动。

| 设备 | 0.1 安装包可用功能 | 本机识别 |
| --- | --- | --- |
| Windows / Linux x64 + NVIDIA | 完整工作台 | 支持；性能评测在 Linux H100 上完成 |
| Windows / Linux x64 CPU | 完整工作台 | 启动 ZIP 支持 CPU，较慢 |
| macOS Apple Silicon / Intel | 连接 GPU 服务、校对、导出 | DMG 不包含 Mac 本机识别后端 |
| AMD / Intel GPU | 连接 GPU 服务、校对、导出 | 当前安装器不安装 ROCm / Vulkan 推理环境 |
| Android / iPhone / iPad | 浏览器连接 GPU 服务 | 没有本地推理 APK / IPA |

跨平台转换需要同时适配 OCR 的视觉编码器、音乐词表、图像预处理、版面检测和拍号／调号分类器。只转换语言模型不能完成整份谱面的识别。Apple Silicon 可考虑 MLX / Metal；其他 GPU 和移动设备可考虑 llama.cpp 的 Vulkan / Metal 后端。0.1 另提供经过 Linux CUDA 验证的 GGUF 可选后端；Metal / Vulkan 和手机真机尚未验证，详情见下节。

## GGUF 可选后端

Release 中的 `GuitarOCR-0.1-measure_ocr-GGUF.zip` 与 `GuitarOCR-0.1-document_info-GGUF.zip` 是两项 OCR 的可选模型包，每包约 1.9 GB。语言部分使用 Q8_0，视觉编码器保留 F16，音乐词表和图像缩放策略随包提供。两个任务的视觉编码器不同，必须各自加载配套文件。它们不是 APK / IPA，也不包含版面检测、拍号／调号分类器或推理程序。

使用已验证的 [llama.cpp 提交](https://github.com/ggml-org/llama.cpp/tree/8019dc563b1ecbae6b161a70c3a1359f1b206c1e) 构建 `llama-server`。其后端可选择 CUDA、Metal 或 Vulkan，构建方式见[上游说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。本项目已在 Linux CUDA 上完成双 OCR 后端、整份 PDF 和 GP5 导出的运行验证；没有把这个结果当作 Mac 或手机真机验证。Mac DMG 仍提供远程识别及本机编辑。

解压两个 GGUF ZIP 后，分别在两个终端启动（Windows 程序名为 `llama-server.exe`）：

```bash
llama-server -m measure_ocr/model-Q8_0.gguf --mmproj measure_ocr/vision-F16.gguf --host 127.0.0.1 --port 8081 -ngl 99 -c 16384 --parallel 1 --jinja
llama-server -m document_info/model-Q8_0.gguf --mmproj document_info/vision-F16.gguf --host 127.0.0.1 --port 8082 -ngl 99 -c 8192 --parallel 1 --jinja
```

在已经安装完整 CPU 推理环境的项目目录中，指定这两个服务并启动工作台。以下命令适用于 Linux shell；Windows 在 PowerShell 中用 `$env:变量名="值"` 设置同名变量。

```bash
export GUITAROCR_BACKEND=llamacpp
export GUITAROCR_LLAMA_MEASURE_URL=http://127.0.0.1:8081
export GUITAROCR_LLAMA_INFO_URL=http://127.0.0.1:8082
tools/webui-venv/bin/python -m webapp.app --device cpu --layout-python tools/webui-paddle-venv/bin/python
```

这里 `--device cpu` 控制版面检测和拍号／调号分类器；两个 OCR 的设备由各自的 llama-server 决定。接入层保留等比例图像预处理、相邻小节图片和音乐词表，不加载两套原始 OCR 权重。已有安装中的原始权重仍保留。

同一批随机抽取的 256 个测试集小节对照：原 vLLM 后端的音符＋起点＋时值 F1 为 98.48%，GGUF 为 97.34%；两者文本语法合法率均为 100%。这是转换回归抽查，不能替代[完整模型评测](model-evaluation.md)。原生 MTP 参数保留在 GGUF 中，但当前 llama.cpp GLM-OCR 运行时不使用它们；默认后端继续使用 vLLM 的 MTP 加速，GGUF 为手动选择项。

重新转换仓库中的当前模型：

```bash
uv run --no-sync python scripts/export_gguf.py --llama-cpp /path/to/llama.cpp --output output/gguf
```

该脚本需要安装 llama.cpp 转换器要求的 Python 包，且检查上面固定的运行时提交。手机本地版还需要将版面检测、分类器、页面处理及模型内存调度一起移植，并在 Android / iOS 设备上验证；当前可直接使用浏览器连接 GPU 服务。

## 硬件要求

推荐自行部署 NVIDIA GPU，或连接已部署的 GPU 服务。远程用户的设备只需能打开网页和上传文件。

| 本机方式 | 建议配置 | 主要影响因素 |
| --- | --- | --- |
| GPU 识别 | 建议 24 GB 显存、32 GB 内存 | 两套常驻 OCR 引擎、视觉缓存及小节批量 |
| CPU 识别 | 建议 32 GB 内存，适合试用 | CPU 性能、内存带宽，速度较慢 |

当前源码版本建议为模型、独立推理环境和编译缓存预留 50 GB 磁盘。GLM-OCR 基座约 13.3 亿参数，两个 OCR 任务分别提供完整合并模型。大图、长小节和并发任务会增加显存与内存需求。

工作台在启动时预热模型，之后同一首谱内的小节批量识别。H100 的首次加载、预热后识别和完整 PDF 结果见[模型评测](model-evaluation.md)。远程服务还需计算上传与排队时间。

## 下载源与模型路径

Python 和依赖下载失败时尝试备用源；发布模型从 GitHub Release 下载，支持续传并校验清单中的文件大小。

| 内容 | 默认来源 | 备用来源 |
| --- | --- | --- |
| Python 3.11 | 南京大学镜像 | Astral GitHub Release |
| Python 包和 Paddle | 清华 PyPI 镜像 | PyPI |
| PyTorch | 上海交大镜像 | PyTorch 官方索引 |
| 当前推理模型 | GitHub Release（自动下载）或 Git LFS | 无需原始基座 |
| 训练用 GLM-OCR 基座 | 魔搭 | Hugging Face |

已有 uv 配置优先用于普通依赖；`UV_PYTHON_INSTALL_MIRROR` 可指定 Python 来源，`HF_ENDPOINT` 可指定模型来源。ZIP、uv 和 Windows C++ 运行库仍需访问各自的官方站点。

桌面安装包和启动 ZIP 的模型版本固定在发布清单中。首次本机识别从同一 Release 下载原始 safetensors 分片和版面模型，支持断点续传；后续直接复用。桌面窗口显示下载和安装进度。连接服务器或仅校对不会下载模型。Python 与运行依赖仍需联网安装。当前源码通过 Git LFS 获取完整任务模型，默认推理无需原始基座；版本和文件大小见 `weights/manifest.json`。

默认模型路径相对安装目录解析，可用 `--model`、`--adapter`、`--info-adapter`、`--layout-model-dir` 覆盖。多人服务将模型位置写入配置文件。

## 手动安装轻量编辑与导出

安装 uv 后执行：

```bash
uv sync --locked --python 3.11 --extra webui
uv run --no-sync guitarocr-web --edit-only --device cpu
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
uv run --no-sync guitarocr-check --device cuda --layout-python tools/paddlex-venv/bin/python
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

手动环境检查：`uv run --no-sync guitarocr-check --device cuda --layout-python tools/paddlex-venv/bin/python`。轻量编辑环境可加 `--core`。
