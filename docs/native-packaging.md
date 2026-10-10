# 原生应用构建与打包

客户端由 Tauri 壳、共用 Rust 后端、`ui` 和推理组件组成。模型按需下载，交付包不包含 Python 环境；下面的 Python 命令仅用于构建机上的资源收集。

## 构建流程

准备 Node.js 22、Rust 1.90+、CMake 和目标系统的 C/C++ 工具链，以及 [Tauri 系统依赖](https://v2.tauri.app/start/prerequisites/)。在仓库根目录执行：

默认构建读取 `weights/distribution.json`，该清单与仓库的图像策略、能力声明及已采用模型保持一致，普通 Cargo 构建和桌面资源准备使用同一份清单。执行：

```bash
npm ci --prefix desktop
npm run native:build --prefix desktop
```

构建时可用 `GUITAROCR_BUILD_MODEL_CATALOG` 或 `--model-catalog` 选择清单。运行时 `--models`／`GUITAROCR_MODEL_CACHE` 只选择缓存位置。后端和打包器均检查清单与模型的输入、输出和已声明能力是否匹配。

当前开发清单为 `local_unpublished`，模型按职责放在 `current` 槽位，不下载历史 release。准备本地模型：

```bash
python scripts/install_local_models.py
desktop/src-tauri/resources/native/guitarocr-backend --models tools/model-cache --resources desktop/src-tauri/resources/native
```

安装脚本优先使用硬链接，避免复制模型；替换 `weights/` 中的文件后重新运行。发布下载资源时再生成文件大小和校验值。公开下载的完整性检查与本地模型的更新机制分别由清单声明。

推理组件使用 ONNX Runtime 1.23.2、PDFium 5.13.0，以及 llama.cpp。前两者取自官方发行件，llama.cpp 使用提供的源码构建，不锁定 Git 提交。`scripts/llamacpp-runtime.json` 记录已发布运行组件的下载地址、大小和归档 SHA-256。资源收集器只提取原生文件及许可证，不安装 wheel 内的 Python 代码。

```bash
git clone https://github.com/ggml-org/llama.cpp.git --filter=blob:none /path/to/llama.cpp
python scripts/build_llamacpp_runtime.py --source /path/to/llama.cpp \
  --build output/runtime-build --output output/runtime-artifacts --target linux-x64-cpu
python -m pip download --index-url https://pypi.org/simple --only-binary=:all: \
  --no-deps onnxruntime==1.23.2 pypdfium2==5.13.0 --dest output/native-wheels
python scripts/collect_native_assets.py --target linux-x64-cpu \
  --ort-archive /path/to/onnxruntime.whl --pdfium-wheel /path/to/pypdfium2.whl \
  --llama-archive /path/to/llamacpp.zip --llama-manifest output/runtime-artifacts/linux-x64-cpu.json \
  --output output/native-components
node scripts/prepare_native_desktop.mjs --build \
  --manifest output/native-components/manifest.json --require-inference
npm run build --prefix desktop
```

将示例归档路径换成上一步的实际产物。目标包括 `linux-x64-cpu`、`linux-x64-cuda`、`windows-x64-cpu`、`windows-x64-cuda`、`macos-arm64-metal` 和 `macos-x64-cpu`。CUDA 需要构建期 CUDA Toolkit；Linux 使用静态 CUDA 数学库，Windows 包含所需可再分发 DLL 和授权。可以用 `GUITAROCR_CUDA_ARCHS` 指定编译架构。Apple Silicon 构建嵌入 Metal shader。

CPU 构建关闭 `GGML_NATIVE`，包含基础指令集和可按设备选择的优化模块。构建脚本检查 CMake 配置和所需库文件，并运行打包后的 `llama-server --version`；`build.json` 记录目标平台、构建参数和随包动态库。不同目标使用独立构建目录，避免将 CUDA、Metal 或特定 CPU 的产物混入其他安装包。

安装包保留运行组件的目录布局、授权与完整性清单。构建机上的加载检查不能替代目标平台的安装和推理验证；尤其需要核对 CPU 指令集、系统库版本和 GPU 驱动兼容性。

正式客户端只发布 Apple 芯片 Mac 与 Windows，安装包位于 `target/release/bundle/`。`.github/workflows/desktop.yml` 复用 `scripts/llamacpp-runtime.json` 中已发布的引擎，不再随客户端重复编译 CUDA。引擎升级才运行 `llamacpp-runtime.yml`，将 ZIP 发布到应用资源页并更新清单。Linux、Intel Mac 按上面的步骤本地构建；服务端完成资源准备后即可运行 `native/guitarocr-backend`，无需执行最后的桌面打包命令。无推理组件的资源准备只用于编辑器开发；完整识别包必须使用 `--require-inference`。

## 资源与进程边界

`desktop/src-tauri/resources/` 仅为生成目录：

```text
native/  guitarocr-backend、运行组件、模型元信息与组件清单
ui/      workbench、accounts
licenses/  原生依赖、Cargo 依赖与源代码适配的授权
```

桌面启动界面来自 `ui/launcher`，由 Tauri 打包。所有原生 crate 共用根目录 Cargo workspace、lockfile 和 target。更改后端或 UI 后应重新准备资源再打包，避免带入旧二进制。

Tauri 只启动 `guitarocr-backend`，读取其 `GUITAROCR_READY` 回环地址；关闭应用时结束拥有的子进程组。后端组合 `engine` 和 `scorelib`。engine 拥有一个 llama-server，使用短期密钥访问回环接口，独立取消单个生成请求。ONNX Runtime 负责辅助模型，PDFium 负责 PDF 栅格化。用户的模型、项目和日志保存到应用数据目录。Windows 通过系统 NVIDIA 驱动判断是否需要加速组件，按需下载到同一数据目录的 `runtimes/`；模型与设备无关，继续共用已有缓存。

若模型清单声明 `information_adapter`，同一 llama-server 额外载入约 15 MB 的语言 LoRA；视觉编码器和主模型继续共享。调用方明确指定乐器、谱号或带参考页的结构任务，音符和其他任务使用 base。每个请求独立设置 adapter scale，禁止全局切换；空 LoRA 列表会恢复服务端默认状态，因此已加载 adapter 时，base 请求显式发送 scale=0。GGUF adapter 与 base 一起进入模型清单和校验；声明缺文件、任务合同不支持或二进制元信息不一致均在识别前报错。Python 研究端的 Transformers/vLLM 使用同一语言 PEFT 资产，桌面只需要 GGUF。

下载归档整体校验 SHA-256；解包及准备资源时检查路径、文件大小、目标平台和所需组件，不再维护逐文件 SHA。模型、Pillow／OpenCV 适配及 Cargo 补充许可集中在根目录 `THIRD_PARTY_NOTICES.md`，随包复制到 `licenses/`；缺少自带许可文件的 Cargo 包列在 `desktop/license-supplements.json`。其余依赖的许可由打包器直接收集，界面的 alphaTab／Bravura 许可随 `ui/workbench/vendor/README.md` 提供。

## 平台验证

每个平台的安装、设备加速和桌面交互应在对应系统验证；CI 配置本身不能代替真机结果。Linux 构建先按 Tauri 文档安装系统依赖，再执行上面的资源准备与 `npm run build --prefix desktop`。macOS 原生组件最低 13.4；不要将本机较高的 glibc／SDK 基线包装成兼容更旧系统的发行件。

桌面 workflow 准备资源后直接构建安装包。安装后实际检查 PDF 显示、编辑保存、项目重开及 GP5 / MusicXML 导出，再用当前模型识别真实曲谱。一次性验证脚本、截图和记录放入临时目录或 `output/`，不作为常驻测试资源提交。

Guitar Pro 数据导出 DLL 属于独立的 `gpbridge`，构建见 [GP8 原生导出](gpbridge-build.md)，不进入用户的识谱应用。
