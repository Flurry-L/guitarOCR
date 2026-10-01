# 安装与启动

## 桌面应用

从[发布页](https://github.com/Flurry-L/guitarOCR/releases)选择对应系统的安装包。Windows 使用 EXE，macOS 使用 DMG，Linux 使用 DEB 或 AppImage；源码构建见[原生打包](native-packaging.md)。以发布附件标注的系统版本和加速后端为准。

打开应用后选择「在这台电脑上使用」，即可进入工作台。可先打开内置示例，或恢复项目 ZIP 进行校对和导出。导入 PDF／图片并开始识别时，应用会提示下载所需模型。

客户端由 Tauri、Rust 后端和原生推理组件组成，不携带、不下载 Python。应用内附带 llama.cpp、ONNX Runtime 和 PDFium。仅使用远程服务时，本机无需下载模型；Android／iOS 可通过浏览器连接服务。

![桌面启动器](assets/desktop-0.1.webp)

## 硬件与平台

本机引擎根据包内后端、可见设备和可用内存自动选择设备。模型或加速设备初始化失败时，会在内存允许的情况下使用 CPU。

| 系统／设备 | 运行方式 |
| --- | --- |
| Linux、Windows x64 | CPU；CUDA 构建可使用 NVIDIA GPU |
| macOS Apple Silicon | Metal |
| macOS Intel | CPU |
| 手机或其他 GPU 平台 | 浏览器连接自托管服务 |

模型下载约 1.775 GB。建议至少 8 GB 系统内存、6 GB 可用磁盘；大谱面与多人并发需要更多内存。CUDA／Metal 内存不足时应减少并发。CPU 的识别速度取决于设备；Linux H100 的实测不能代表其他设备的速度。

Windows 需要 WebView2；安装器按配置处理 WebView2，原生组件使用相应系统运行库。macOS 原生组件基线为 13.4。Linux 包的系统要求由实际构建基线决定；在较新 Debian 构建的本地包不能当作 Ubuntu 22.04 兼容包。

## 模型、项目与更新

「打开数据目录」可以查看 `projects/`、`models/<模型版本>/` 和日志。模型由 `weights/distribution.json` 指定，缺失或损坏文件会校验、续传。更新应用不重新下载未变更的模型，也不改动已保存的项目。

使用 `GUITAROCR_MODEL_CACHE` 可指定共享模型缓存。服务端也可以通过 `--models DIR` 指定；缓存可在多个应用版本之间复用。退出桌面应用会停止其本机服务，已经保存的阶段和小节可继续处理。

GGUF 是 llama.cpp 使用的模型格式：语言模型使用 Q8_0，视觉编码器使用 F16。版面和拍号／调号使用 ONNX；所有 OCR 任务共用一组 GGUF。safetensors 训练权重保留在仓库中供研究使用，不会随客户端额外下载。

## 从源码运行

无模型编辑与导出：

```bash
cargo run --locked -p guitarocr-backend -- --assets ui/workbench --port 7860
```

准备完整原生资源后：

```bash
cargo run --locked --release -p guitarocr-backend -- \
  --assets ui/workbench --resources desktop/src-tauri/resources/native \
  --models /path/to/model-cache --port 7860
```

打开 `http://127.0.0.1:7860`。资源构建见[原生打包](native-packaging.md)，网络开放、账号和反向代理见[服务端部署](server.md)。

Python 只用于独立的[数据生产](data.md)、[训练和实验推理](training.md)，不会由桌面或自托管服务启动。
