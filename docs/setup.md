# 安装与启动

从 [GuitarOCR 0.1](https://github.com/Flurry-L/guitarOCR/releases/tag/v0.1.0) 下载对应系统的桌面安装包，或 `GuitarOCR-0.1.0.zip`。安装器按所选引擎下载模型；Release 下的模型附件由程序管理，不需要逐个下载。

## 桌面应用

Windows 运行 EXE，macOS 将 DMG 内的应用拖入 Applications，Linux 安装 DEB 或运行 AppImage。支持 Windows x64、macOS 12.3 及更新版本、Ubuntu 22.04 及更新版本。

- **连接 GPU 服务**：输入服务地址。本机不安装 Python、推理引擎或模型。
- **使用本机 GPU**：Windows 使用 Transformers，Linux 使用带 MTP 的 vLLM。只安装一种 OCR 引擎，版面与拍号／调号共用 ONNX Runtime。
- **使用 CPU**：Windows / Linux 使用 Transformers CPU 版，识别较慢。
- **仅校对与导出**：安装基础依赖，可恢复项目备份、编辑及导出；已有识别环境时直接复用。

![0.1 桌面启动器](assets/desktop-0.1.webp)

安装包尚未签名或公证。Windows 缺少 WebView2 时由安装程序下载；本机运行需 [Microsoft Visual C++ x64 运行库](https://aka.ms/vs/17/release/vc_redist.x64.exe)。

「打开数据目录」可查看项目、模型缓存、运行环境和日志。更新应用保留这些内容。退出应用会停止本机任务，退出前请保存编辑。

## 脚本安装

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

| 使用方式 | 所需模型下载 | 运行环境 |
| --- | --- | --- |
| 远程连接 / 仅校对 | 0 | 远程连接无需 Python；仅校对安装基础依赖 |
| 原生 vLLM / Transformers | 约 2.9 GB | 一套 Python 环境、一份 Torch；无 Paddle |
| llama.cpp | 约 1.8 GB | 一套轻量 Python 环境及 llama-server；无 Torch、Paddle、Transformers |

数字是模型文件字节数，不包含运行依赖。llama.cpp 的 Python 环境在 Linux 实测约 478 MiB，llama-server 另计；vLLM 整套 Python 依赖约 7.9 GiB。首次安装还需要下载缓存及临时空间，不能将模型大小当作安装总量。原生 GPU 建议预留 25 GB 磁盘、24 GB 显存及 32 GB 内存。

`tools/runtimes/<引擎>-<设备>/` 保存所选环境，`tools/install-state.json` 记录当前选择。`tools/model-cache/<模型版本>/` 保存模型，两个 OCR 引擎共用辅助模型。`GUITAROCR_MODEL_CACHE` 可指定其他磁盘上的共享缓存。模型版本由 `weights/distribution.json` 管理，运行依赖版本由 `scripts/runtime-manifest.json` 管理，与应用版本分开。

更新界面代码不会重装运行环境，模型版本未变化也不会重复下载。更换引擎时，旧环境保留以便切回；确认新环境可用并关闭本机服务后，可主动清理：

```bash
bash scripts/bootstrap.sh status  # 查看当前环境及旧环境占用
bash scripts/bootstrap.sh clean   # 只移除已列出的未使用运行环境
```

Windows 对应 `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 status` / `clean`。清理命令不删除项目、训练环境或模型缓存。旧版 `webui-venv`、`webui-paddle-venv`、`vllm-venv` 和独立编辑环境会列为迁移后的可清理项。桌面用户可在数据目录的 `backend/` 中用其 `tools/runtimes/` 内的 Python 执行 `scripts/launcher.py status` / `clean`。

## GGUF 可选后端

GGUF 是共享 OCR 模型的 llama.cpp 格式。语言权重量化为 Q8_0，视觉编码器保留 F16；所有 OCR 任务共用这一组文件。未执行的 MTP 层从 GGUF 中移除，原生 vLLM 模型保留 MTP。此路径只下载 GGUF、ONNX 辅助模型和小型配置，不下载原始 OCR safetensors。

使用固定的 [llama.cpp 提交](https://github.com/ggml-org/llama.cpp/tree/8019dc563b1ecbae6b161a70c3a1359f1b206c1e) 构建 `llama-server`，构建依赖见[上游说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。Linux CUDA 示例：

```bash
bash install.sh --engine llamacpp --device cuda --llama-server /absolute/path/to/llama-server
bash start.sh
```

Windows 将路径换成 `llama-server.exe`。CPU 构建使用 `--device cpu`。启动器自动管理一个共享 OCR 服务的端口、模型加载和退出。自管服务可通过 `GUITAROCR_LLAMA_SCORE_URL` 指定地址。GGUF 是脚本安装选项；桌面按钮使用原生后端。

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
| Windows / Linux x64 | 桌面、脚本 | 原生 GPU / CPU；运行验证在 Linux H100 完成 |
| macOS Apple Silicon / Intel | DMG | 远程识别、本机校对及导出 |
| AMD / Intel GPU | 桌面或浏览器连接服务 | llama.cpp Vulkan 需自行构建，未做真机验证 |
| Android / iOS | 浏览器连接服务 | 不提供本地推理 APK / IPA |

模型不依赖某一块 H100 的 TensorRT engine。llama.cpp 上游具有 Metal / Vulkan 后端，但本项目的 Linux 验证不代表已完成 Mac、其他 GPU 或手机真机验证。

## 开发与手动运行

轻量编辑环境：

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
