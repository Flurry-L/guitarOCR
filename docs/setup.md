# 安装与启动

先按 [README 的获取源码步骤](../README.md#获取源码) 使用 Git 和 Git LFS 克隆仓库、取得完整权重。当前没有可下载的 Release 安装包。以下命令都在仓库根目录执行；开发和重新训练可以使用后面的手动安装命令。

## 一键安装

Windows 在检出的 `guitarOCR` 文件夹中双击 `start.bat`。Linux x64 执行 `bash start.sh`。首次准备环境后自动打开浏览器，以后直接复用已安装环境；安装失败时保留日志并退出，重跑同一个脚本即可继续。

| 操作 | Windows | Linux |
| --- | --- | --- |
| 安装并启动 | `start.bat` | `bash start.sh` |
| 只安装或修复 | `install.bat` | `bash install.sh` |
| 强制使用 CPU 安装 | `install.bat --device cpu` | `bash install.sh --device cpu` |
| 安装 NVIDIA GPU 版本 | `install.bat --device cuda` | `bash install.sh --device cuda` |
| 换端口 | `start.bat --port 7861` | `bash start.sh --port 7861` |
| 服务器上不打开浏览器 | `start.bat --no-browser` | `bash start.sh --no-browser` |

脚本安装固定版本 uv 和 Python 3.11，不修改系统 Python。Windows 缺少 Microsoft C++ 运行库时，会先下载微软官方安装程序，系统可能弹出管理员确认；需要重启时脚本会明确提示。PyTorch 按 CPU / CUDA 13.0 选择官方 wheel，其余主环境依赖来自 `uv.lock`。自动选择 GPU 需要 NVIDIA 驱动 580 或更新版本；没有匹配驱动时使用 CPU。Paddle 版面检测默认使用独立的 CPU 环境，GLM 识别使用所选设备。需要加速版面检测的用户可按下方命令单独配置 Paddle GPU。

下载源：uv 来自 Astral，Python 由 uv 管理，Python 包来自 PyPI / PyTorch 官方索引，基座来自 Hugging Face，项目权重来自本次 Git 检出对应的 GitHub LFS 文件。基座 revision 和全部模型 SHA-256 见 `weights/manifest.json`。安装器不会更新到未知的最新基座。

安装器兼容默认配置和已有 uv 镜像配置。已有镜像优先用于依赖下载；若当前 uv 配置导致步骤失败，安装器会以默认配置和官方源重试该步骤。回退只作用于重试进程，保留代理、证书和缓存相关环境变量，不修改用户的全局配置文件。

环境位于 `tools/webui-venv/` 与 `tools/webui-paddle-venv/`；安装状态位于 `tools/install-state.json`，项目位于 `output/webui/`，日志位于 `output/logs/`。开发用 `.venv/` 和训练用 Paddle 环境独立管理。下载体积因设备而异，建议预留至少 20 GB 空间给环境、缓存和模型。

Ubuntu / Debian 的系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates libgl1 libglib2.0-0
```

## 已验证范围

| 环境 | 状态 |
| --- | --- |
| Linux x64 / Python 3.11 / CPU | 一键安装、重复安装、启动和模型依赖检查已实测 |
| Linux x64 / NVIDIA H100 | 一键 CUDA 安装、CPU 切换 CUDA，以及新环境 PDF / 图片到 GP5 实测通过；详见 [验证记录](validation.md) |
| Windows x64 / Python 3.11 | 提供双击脚本和 CI；完整模型安装仍需 Windows 实机验收 |
| macOS / ARM / AMD GPU | 不作为当前一键完整安装目标；可尝试轻量编辑与导出环境 |

尚未测得可承诺的最低内存、显存和消费级显卡耗时。CPU 可以运行 GLM，但耗时通常更长。

## 手动安装：轻量编辑与导出

安装 uv 后执行：

```bash
uv sync --locked --python 3.11 --extra webui --extra dev
uv run --no-sync guitarocr-web --device cpu
```

基础安装不拉取 Torch。这个环境可以导入、画框和执行格式与导出测试；自动读取音符还需要下面的模型环境。

工作台默认只接受 `localhost`、`127.0.0.1` 和 `::1` 主机名。自定义主机名需在 CLI 添加 `--allow-host scores.example`；该参数可重复传入，不支持通配符。`--host` 控制监听地址，指定具体地址会允许该地址，`0.0.0.0` / `::` 则不会自动允许所有主机名。工作台没有登录认证；Host 检查用于防止 DNS 重绑定，不能替代公网部署的身份认证。

## 手动安装：模型推理

Ubuntu / Debian 可先安装系统工具：

```bash
sudo apt-get install -y git git-lfs curl build-essential libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.12.17/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git lfs install
git lfs pull
uv sync --locked --python 3.11 --extra glm-ocr --extra webui --extra dev
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
uv pip install --python tools/paddlex-venv/bin/python -e "tools/PaddleX[ocr,cv]" "numpy==1.26.4" "Pillow>=12.3,<13" "PyMuPDF>=1.24,<2"
uv pip install --python tools/paddlex-venv/bin/python --no-deps -e .
tools/paddlex-venv/bin/python -c "import paddle; paddle.utils.run_check()"
uv run --no-sync guitarocr-check --hashes --device cuda --layout-python tools/paddlex-venv/bin/python
uv run --no-sync guitarocr-web --layout-python tools/paddlex-venv/bin/python
```

CPU 版把 Paddle 安装命令换成 `uv pip install --python tools/paddlex-venv/bin/python "paddlepaddle==3.2.0"`。Windows 对应解释器路径是 `tools/paddlex-venv/Scripts/python.exe`。推理不需要 Wine、Guitar Pro 或原生 DLL。

## GLM 训练环境

LLaMA-Factory 使用当前工作环境的 `0.9.6.dev0` 源码提交，支持本项目固定的 Transformers 5.8.0 和 PEFT 0.18.1：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git tools/LLaMA-Factory
git -C tools/LLaMA-Factory checkout 97b32d3133b501432141a82949d5c7bc4d94f23a
uv pip install --python .venv/bin/python -e tools/LLaMA-Factory

uv run --no-sync python -m measure_ocr.train
uv run --no-sync python -m document_info.train
```

安装训练依赖后，训练使用 `uv run --no-sync`。需要更新本项目的可编辑安装时，执行 `uv pip install --python .venv/bin/python --no-deps -e .`，保留已装的训练框架。

## 版面训练组件

先完成图片版面环境安装，再安装 PP-DocLayoutV3 使用的 PaddleDetection 训练组件：

```bash
PIP_CONSTRAINT="$PWD/layout/constraints.txt" \
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
tools/paddlex-venv/bin/paddlex --install PaddleDetection

tools/paddlex-venv/bin/python -m layout.train \
  -c layout/configs/train.yaml -o Global.mode=train
```

`layout/constraints.txt` 保持 NumPy、OpenCV、pycocotools 与当前工作环境一致。完整训练生成初始检查点后，可用 `layout/configs/train_sparse.yaml` 继续稀疏谱行微调；该配置中的初始检查点路径属于训练流程。

## 数据导出环境

安装 Wine、虚拟显示和辅助工具：

```bash
sudo apt-get install -y wine wine64 xvfb xauth x11-xkb-utils util-linux
mkdir -p datagen/runtime
export WINEPREFIX="$PWD/datagen/runtime/wine-prefix"
export WINEARCH=win64
xvfb-run -a wineboot --init
```

下载并安装 Wine 内的 Python 3.11 和 PDF 依赖：

```bash
curl -fL https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe \
  -o datagen/runtime/python-installer.exe
xvfb-run -a wine datagen/runtime/python-installer.exe \
  /quiet InstallAllUsers=0 'TargetDir=C:\Python311' Include_pip=1 Include_test=0
xvfb-run -a wine "$WINEPREFIX/drive_c/Python311/python.exe" \
  -m pip install "PyMuPDF>=1.24,<2"
```

Guitar Pro 8 使用官方安装程序安装到同一 Wine prefix；在有桌面的会话中运行其安装程序，完成安装和激活。原生 DLL 需与使用的 GP8 版本匹配：

```bash
wine /path/to/GuitarPro8-setup.exe
```

随后运行数据生产管线，`--runtime` 指向包含 `GuitarPro.exe` 的实际安装目录：

```bash
uv run --no-sync python -m datagen.run \
  --corpus /path/to/gp-files --output database/gp8_measure_sequence_v2 \
  --source-count 2000 --phase all \
  --runtime "$WINEPREFIX/drive_c/Program Files/Arobas Music/Guitar Pro 8" \
  --wine-prefix-template "$WINEPREFIX" \
  --wine-python "$WINEPREFIX/drive_c/Python311/python.exe" --workers 4
```

推理只需要主环境和图片版面环境；数据导出工具用于重新生成训练数据。

原生导出 DLL 的完整源码与 Windows 构建命令见 [native-build.md](native-build.md)。Web 工作台启动与操作见 [webui.md](webui.md)。
