# 原生应用构建与打包

客户端由 Tauri 壳、共用 Rust 后端、`ui` 和推理组件组成。模型按需下载，交付包不包含 Python 环境；下面的 Python 命令仅用于构建机上的资源收集。

## 构建流程

准备 Node.js 22、Rust 1.90+、CMake 和目标系统的 C/C++ 工具链，以及 [Tauri 系统依赖](https://v2.tauri.app/start/prerequisites/)。在仓库根目录执行：

```bash
npm ci --prefix desktop
cargo build --locked --release -p guitarocr-backend
```

推理组件使用 ONNX Runtime 1.23.2、PDFium 5.13.0，以及 `scripts/llamacpp-runtime.json` 固定的 llama.cpp 提交。取自官方发行件或从固定源码构建；资源收集器只提取原生文件及许可证，不安装 wheel 内的 Python 代码。

```bash
git clone https://github.com/ggml-org/llama.cpp.git --filter=blob:none /path/to/llama.cpp
git -C /path/to/llama.cpp checkout 8019dc563b1ecbae6b161a70c3a1359f1b206c1e
python scripts/build_llamacpp_runtime.py --source /path/to/llama.cpp \
  --build output/runtime-build --output output/runtime-artifacts --target linux-x64-cpu
python -m pip download --index-url https://pypi.org/simple --only-binary=:all: \
  --no-deps onnxruntime==1.23.2 pypdfium2==5.13.0 --dest output/native-wheels
python scripts/collect_native_assets.py --target linux-x64-cpu \
  --ort-archive /path/to/onnxruntime.whl --pdfium-wheel /path/to/pypdfium2.whl \
  --llama-archive /path/to/llamacpp.zip --llama-manifest output/runtime-artifacts/linux-x64-cpu.json \
  --output output/native-components
node scripts/prepare_native_desktop.mjs --manifest output/native-components/manifest.json --require-inference
npm run build --prefix desktop
python scripts/package_release.py
```

将示例归档路径换成上一步的实际产物。目标包括 `linux-x64-cpu`、`linux-x64-cuda`、`windows-x64-cpu`、`windows-x64-cuda`、`macos-arm64-metal` 和 `macos-x64-cpu`。CUDA 需要构建期 CUDA Toolkit；Linux 使用静态 CUDA 数学库，Windows 包含所需可再分发 DLL 和授权。可以用 `GUITAROCR_CUDA_ARCHS` 指定编译架构。Apple Silicon 构建嵌入 Metal shader。

`.github/workflows/desktop.yml` 按平台编排这套流程，输出安装包与服务端归档。产物位于根目录 `target/release/bundle/` 和 `output/releases/`。无推理组件的资源准备只用于编辑器开发；完整识别包必须使用 `--require-inference`。

## 资源与进程边界

`desktop/src-tauri/resources/` 仅为生成目录：

```text
native/  guitarocr-backend、运行组件、模型元信息与组件清单
ui/      workbench、accounts
licenses/  原生依赖、Cargo 依赖与源代码适配的授权
```

桌面启动界面来自 `ui/launcher`，由 Tauri 打包。所有原生 crate 共用根目录 Cargo workspace、lockfile 和 target。更改后端或 UI 后应重新准备资源再打包，避免带入旧二进制。

Tauri 只启动 `guitarocr-backend`，读取其 `GUITAROCR_READY` 回环地址；关闭应用时结束拥有的子进程组。后端组合 `engine` 和 `scorelib`。engine 拥有一个 llama-server，使用短期密钥访问回环接口，独立取消单个生成请求。ONNX Runtime 负责辅助模型，PDFium 负责 PDF 栅格化。用户的模型、项目和日志保存到应用数据目录。

组件清单记录文件、平台及完整性信息；收集器拒绝路径穿越、错误架构、损坏文件与缺失授权。补充的上游许可证位于 `desktop/native-licenses`。Pillow／OpenCV 图像变换适配的授权位于 `backend/licenses`；修改或打包时须一并保留。

## 平台验证

Linux 已实际运行原生 ONNX＋GGUF 识谱、两个账号并发、取消续跑、编辑保存及 GP5／项目导出。模型质量仍应以独立[评测报告](model-evaluation.md)为准，流程跑完不表示每个音符都正确。

每个平台的安装、设备加速和桌面交互应在对应系统验证；CI 配置本身不能代替真机结果。Linux 公共构建使用 Ubuntu 22.04；本地 Debian 构建可用 `scripts/build_linux_desktop.py --sysroot DIR`，它把实际系统库要求写入 DEB。macOS 原生组件最低 13.4；不要将本机较高的 glibc／SDK 基线包装成兼容更旧系统的发行件。

Guitar Pro 数据导出 DLL 属于独立的 `gpbridge`，构建见 [GP8 原生导出](gpbridge-build.md)，不进入用户的识谱应用。
