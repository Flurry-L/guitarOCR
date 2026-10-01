# 安装与启动

[现有 0.1.0 发布包](https://github.com/Flurry-L/guitarOCR/releases/tag/v0.1.0)早于本次原生客户端改动。当前源码桌面分发使用 Tauri＋Rust 服务，不包含或下载 Python；构建见[原生打包](native-packaging.md)。Python ZIP/脚本仍供研究、命令行与服务部署使用。

## 桌面应用

Windows 运行 EXE，macOS 将 DMG 内的应用拖入 Applications，Linux 安装 DEB 或运行 AppImage。当前源码构建目标为 Windows x64、macOS 13.4 及更新版本、Ubuntu 22.04 及更新版本；本地构建的候选包可能要求更新的系统，不能与公开构建的兼容范围混用。

- **连接 GPU 服务**：输入服务地址。本机不安装 Python、推理引擎或模型。
- **在这台电脑上识别**：使用已校验的本机 llama.cpp＋ONNX Runtime 组件，根据实际编译后端、可见设备和内存选择 CPU、Metal 或 CUDA。首次识别确认约 1.775 GB 模型校验/补齐；不安装 Python 或编译工具链。
- **Python 研究/服务环境**：独立保留 Transformers / vLLM 等命令行入口；不会被桌面端隐式启动。
- **仅校对与导出**：直接使用包内 Rust 服务恢复项目备份、编辑及导出，无需下载模型。

![0.1 桌面启动器](assets/desktop-0.1.webp)

安装包尚未签名或公证。公开 GGUF 引擎附件尚待维护者发布；维护者可在候选安装包中内置同平台、校验通过的引擎。现有 0.1.0 包与当前源码的能力不同，见下方平台支持。Windows 缺少 WebView2 时由安装程序下载；本机运行需 [Microsoft Visual C++ x64 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)。

「打开数据目录」可查看项目、模型缓存、运行环境和日志。更新应用保留这些内容。退出应用会停止本机任务，退出前请保存编辑。

## Python 研究与服务脚本

完整解压 ZIP 到可写目录。Windows 使用 `start.bat` / `install.bat`，Linux 使用 `bash start.sh` / `bash install.sh`，参数相同：

```bash
bash start.sh                         # 自动选择本机引擎并启动
bash start.sh --device cpu            # CPU 识别
bash start.sh --device cuda           # NVIDIA GPU 识别
bash install.sh --engine vllm --device cuda          # Linux GPU
bash install.sh --engine transformers --device cuda # Windows / Linux GPU
bash start.sh --port 7861 --no-browser
```

脚本自动安装 Python 3.11 和依赖，无需安装 CUDA Toolkit。CUDA 13 需要 NVIDIA 580 或更新驱动；自动模式在驱动不满足时选择 CPU。启动后访问 http://127.0.0.1:7860，保留启动窗口。下载中断后重跑相同命令即可续传。

## 模型、环境与更新


原生桌面将项目放在应用数据目录的 `projects/`，模型放在 `models/<模型版本>/`。GGUF/ONNX 共享同一代模型清单，缺失或损坏文件经确认后续传并校验 SHA-256。运行组件来自安装包，应用更新保留项目与模型。可用 `GUITAROCR_MODEL_CACHE` 指定共享缓存根目录，程序追加模型版本；也接受直接指定该版本目录。迁移时若新缓存尚未建立，会校验并复用应用数据目录内既有模型文件，不启动旧 Python 环境。

下面的 Python 运行环境表及 `tools/` 命令仅适用于研究/命令行安装，不是桌面运行依赖：

| 使用方式 | 所需模型下载 | 运行环境 |
| --- | --- | --- |
| 远程连接 / 仅校对 | 0 | 远程连接无需 Python；仅校对安装基础依赖 |
| Python vLLM / Transformers | 约 2.9 GB | 一套 Python 环境、一份 Torch；无 Paddle |
| llama.cpp | 约 1.8 GB | 一套轻量 Python 环境及 llama-server；无 Torch、Paddle、Transformers |

数字是模型文件字节数，不包含运行依赖。llama.cpp 的 Python 环境在 Linux 实测约 478 MiB，llama-server 另计；vLLM 整套 Python 依赖约 7.9 GiB。首次安装还需要下载缓存及临时空间，不能将模型大小当作安装总量。原生 GPU 建议预留 25 GB 磁盘、24 GB 显存及 32 GB 内存。

`tools/runtimes/<引擎>-<设备>/` 保存所选环境，`tools/install-state.json` 记录当前选择。`tools/model-cache/<模型版本>/` 保存模型，两个 OCR 引擎共用辅助模型。`GUITAROCR_MODEL_CACHE` 可指定其他磁盘上的共享缓存。模型版本由 `weights/distribution.json` 管理，运行依赖版本由 `scripts/runtime-manifest.json` 管理，与应用版本分开。

更新界面代码不会重装运行环境，模型版本未变化也不会重复下载。更换引擎时，旧环境保留以便切回；确认新环境可用并关闭本机服务后，可主动清理：

```bash
bash scripts/bootstrap.sh status  # 查看当前环境及旧环境占用
bash scripts/bootstrap.sh clean   # 只移除已列出的未使用运行环境
```

Windows 对应 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 status` / `clean`。清理命令不删除项目、训练环境或模型缓存。旧版 `webui-venv`、`webui-paddle-venv`、`vllm-venv` 和独立编辑环境会列为迁移后的可清理项。这些清理命令只适用于旧 Python 研究/脚本环境；新的原生桌面不使用 `backend/` 中的 Python。

## GGUF 本机后端

GGUF 是共享 OCR 模型的 llama.cpp 格式。语言权重量化为 Q8_0，视觉编码器保留 F16；所有 OCR 任务共用这一组文件。未执行的 MTP 层从 GGUF 中移除，原生 vLLM 模型保留 MTP。此路径只下载 GGUF、ONNX 辅助模型和小型配置，不下载原始 OCR safetensors。

使用固定的 [llama.cpp 提交](https://github.com/ggml-org/llama.cpp/tree/8019dc563b1ecbae6b161a70c3a1359f1b206c1e) 构建 `llama-server`，构建依赖见[上游说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。Linux CUDA 示例：

```bash
bash install.sh --engine llamacpp --device cuda --llama-server /absolute/path/to/llama-server
bash start.sh
```

Windows 将路径换成 `llama-server.exe`。CPU 构建使用 `--device cpu`；Apple Silicon 的 Metal 构建使用 `--device metal`。启动器自动管理一个共享 OCR 服务的端口、模型加载和退出。 受管本机服务只监听 loopback，并使用每次启动新建的短命认证密钥；密钥仅保存在 Python 后端内存中，关闭即丢弃，不写入项目或安装状态。模型端口禁用网页界面并限制跨源调用，工作台通过应用后端访问。自管服务可通过 `GUITAROCR_LLAMA_SCORE_URL` 指定地址。当前源码的桌面和脚本均接入 GGUF。注意：预编译 runtime 清单尚未录入已发布附件；维护者完成平台构建、许可证与 SHA-256 校验、真机测试和发布前，没有内置引擎的安装包会明确报缺包，不会假装成功或在用户电脑编译。带内置引擎的构建会先校验并展开到应用私有缓存，无需访问未发布的下载地址；模型仍按需获取。开发者可用固定提交构建的 `--llama-server` 路径验证，已安装且兼容的环境仍可离线复用。

实际准确率、速度及评测条件见[模型评测](model-evaluation.md)。原生与 GGUF 均使用同一套分轨、音高上下文、和弦编辑及导出程序。

重新导出：

```bash
uv run --no-sync python scripts/export_gguf.py --llama-cpp /path/to/llama.cpp
uv run --no-sync python scripts/export_auxiliary.py --paddle-python /path/to/paddle-env/bin/python
```

转换用 Torch、Paddle 和导出工具只装在构建机器，不随 GGUF 运行环境安装。

## 平台支持

| 平台 | 已提供的入口 | 本机识别状态 |
| --- | --- | --- |
| Windows / Linux x64 | 桌面、脚本 | GGUF CPU / CUDA 选择和缓存已接入；旧模型评测仅代表 Linux H100 条件 |
| macOS Apple Silicon | DMG、脚本 | Metal runtime 路径已接入，真机识别和安装包待验证 |
| macOS Intel | DMG、脚本 | CPU runtime 路径已接入，真机识别和安装包待验证 |
| AMD / Intel GPU | 桌面或浏览器连接服务 | llama.cpp Vulkan 需自行构建，未做真机验证 |
| Android / iOS | 浏览器连接服务 | 不提供本地推理 APK / IPA |

模型不依赖某一块 H100 的 TensorRT engine。llama.cpp 上游具有 Metal / Vulkan 后端，但本项目的 Linux 验证不代表已完成 Mac、其他 GPU 或手机真机验证。

当前云端原生服务已验证项目导入、校对保存、版本冲突、metadata 更新、GP5/项目导出和本机 HTTP 隔离；原生 ONNX/PDFium 做过小型边界验证。尚未运行整个原生自动识别的真实模型验收，也未完成系统安装、原生 GUI、Windows/Mac、Metal/CUDA 和跨发行版验收。现有云端 llama 组件要求 glibc ≥ 2.38，不可冒充 Ubuntu 22.04 通用发行包；正式发行须在对应基线构建。旧云端 Python DEB 不代表新的原生分发产物。

## 开发与手动运行

Python 研究用轻量编辑环境（原生桌面不执行以下命令）：

```bash
uv sync --locked --python 3.11 --extra webui
uv run --no-sync guitarocr-web --edit-only --device cpu
```

自动安装完成后，用当前选定环境执行命令行任务或部署服务：

```bash
bash scripts/bootstrap.sh run -- -m pipeline.run /path/to/score.pdf --output output/score
bash scripts/bootstrap.sh check
```

训练用 Torch / Paddle 环境见[训练说明](training.md)。训练与部署环境分开，日常推理无需安装训练依赖。

## 下载与故障排查

Python 使用南京大学镜像，普通 Python 包使用清华镜像，PyTorch 使用交大镜像；失败后尝试官方源，已有 uv 源配置优先。模型从 GitHub Release 下载并续传。源码默认可跳过 Git LFS 大文件，由安装器下载所选模型。

| 问题 | 处理 |
| --- | --- |
| 下载失败 | 查看 `output/logs/`，检查网络后重跑相同命令；支持 `HTTPS_PROXY` |
| CUDA 不可用 | 更新驱动至 580 或更新版本，或重新安装 `--device cpu` |
| 找不到 llama-server | 用 `--llama-server` 指定完整路径 |
| 显存不足 | 关闭其他占用显存的程序，或连接显存更大的服务 |
| 端口被占用 | 使用已有工作台，或加 `--port 7861` |
| 升级后仍有多套环境 | 先 `status` 查看，停止服务后 `clean` 清理 |

单用户工作台默认监听本机，没有登录认证。远程可用 SSH 转发，公网部署请使用[多人服务](server.md)。`--host` 和 `--allow-host` 用于监听和 Host 校验，不代替账号认证。
