# 参与开发

## 原生客户端快速检查

界面、项目与原生客户端开发不需要 Python 或模型。先安装 Node.js 22、Rust 1.88+，在仓库根目录运行：

```bash
npm ci --prefix desktop
npm test --prefix desktop
node --test tests/score-editor.test.mjs tests/project-import.test.mjs
cargo test --locked --manifest-path desktop/native-service/Cargo.toml
cargo test --locked --manifest-path desktop/native-core/Cargo.toml
```

启动或打包还需要目标平台的 Tauri 系统依赖，见下方“桌面安装包”。`npm run resources` 必须在服务构建之后执行；修改 `webapp/static` 或 Rust 服务后也须重新准备资源，避免打入旧文件。仅编辑预览不需要原生推理组件；完整本机识别包的构建期获取与校验步骤见[原生打包](docs/native-packaging.md#acquiring-native-build-inputs)。

## Python 研究路径检查

```bash
uv sync --locked --python 3.11 --extra webui --extra dev
uv run --no-sync ruff check datagen layout document_info measure_ocr gp5_export pipeline shared webapp server scripts tests
uv run --no-sync python -m unittest discover -s tests -v
```

检查覆盖 GP5 导出、项目保存与恢复、数据划分、版面类型和下载校验，不需要模型或 GPU。模型替身须在应用启动前安装，避免后台 warmup 意外加载真实模型。界面修改后，本地检查导入、画框、保存、导出和刷新恢复。

训练环境使用 `uv run --no-sync`，避免同步命令移除额外训练依赖。模型评测命令见[训练说明](docs/training.md)。

## 先找负责这项工作的入口

| 职责 | 主入口与边界 |
| --- | --- |
| 数据生产 `datagen/` | `python -m datagen.run --help`；选源、渲染、裁图、标签。原创多谱表用 `datagen.engraved_scores`，汇集现成任务用 `datagen.unified_data`，详见[数据生产](docs/data.md) |
| 训练 | `measure_ocr.train` 训练共享 OCR，`layout.train` 训练检测器；`train_state` / `train_mtp` 是辅助训练，`document_info.train` 只是兼容别名，详见[训练说明](docs/training.md) |
| 评测 | 各模块 `evaluate*` 检查本阶段，`pipeline.evaluate*` 检查完整流程；固定裁图、连续小节和整页报告不能混用，[入口选择](docs/training.md#评测小节识别) |
| 推理四阶段 | `pipeline.run` → `layout.run` → `document_info.run` → `measure_ocr.run` → `gp5_export.run`，每阶段写自己的清单 |
| 乐谱 IR `shared/` | `m2.py` / `constraints.py` 管小节语义，`score_document.py` 管整谱投影；`musicxml.py` 和 `gp5_export/` 各管格式限制，见[表示边界](docs/score-text.md#表示与职责边界) |
| Python 应用服务 `pipeline/` | `Workspace` 管项目、阶段和编辑；`archive.py` 管 ZIP；`webapp/app.py` 是本机 HTTP 入口，`server/` 增加账号、队列、租约和部署 |
| 原生应用服务 `desktop/` | `native-core` 管算法、IR、模型与导出；`native-service` 管项目、任务和 HTTP；`src-tauri` / `ui` 只管桌面窗口、子进程与远程连接 |
| 共用工作台 `webapp/static/` | 三个服务复用同一套 ES modules；`server/static/` 只增加账号和管理页，`desktop/ui/` 只提供启动壳 |
| 工具与发布 `scripts/` | `launcher.py` 管 Python 环境；`export_*` 转模型；`prepare_native_desktop.mjs` 准备桌面资源。`install.*` / `start.*` 只转交 Python 启动器 |
| 模型与验证 | `weights/` 保存模型契约和评测记录，测试在 `tests/` 与 `desktop/*/tests`；`.github/workflows` 只编排转换、构建和候选附件，见[打包边界](docs/native-packaging.md) |

保留单仓库，因为数据、模型、格式和工作台必须一起演进；不再增设只转发调用的层。依赖以实际用途为准：数据构建和评测复用推理提示词、图像规范和 IR，推理不导入数据构建器或训练入口。`shared/` 的模型适配器、训练工具、环境检查是基础设施，不是乐谱语义；阅读 IR 不需要 Torch、GGUF 或 Web 服务。`datagen/native*` 的 GP8/Wine 导出器仅生产训练数据，与桌面原生运行时无关。

### Python 推理与项目服务

各推理阶段可独立运行。`shared/artifacts.py` 校验清单，`shared/schema.py` 描述交换字段。`layout/` 管区域和阅读顺序，`document_info/` 管乐器、调弦和移调，`measure_ocr/` 管生成、重试和续跑，`gp5_export/` 管指法分配和 GP5 限制。

谱面信息、分轨和小节任务共用 `weights/score_ocr`。`shared/glm_backend.py` 统一模型池及 native / vLLM / llama.cpp 选择；辅助模型部署用 ONNX，Paddle 留给训练和转换。训练和推理共享 `shared/crops.py` 裁图留白及 `shared/score_image.py` 图像规范，不能各自改变输入尺寸。

`Workspace` 是项目业务入口：`process` 共用本机和多人服务的一键识别／续跑决策；管理 revision、待续跑任务和人工编辑；先写新阶段文件，再原子替换 `session.json`，失败保留旧结果。`measure_ocr/result.py` 负责保存校对记录及重建导出投影，`pipeline/archive.py` 负责 ZIP 路径转换与恢复。原始记录与派生文件的关系见[表示边界](docs/score-text.md#表示与职责边界)。

本地 `webapp/app.py` 使用后台线程，多用户 `server/app.py` 加账号、配额与归属检查，`server/worker.py` 领取 SQLite 持久任务；两者复用 `Workspace`、`webapp/contracts.py` 请求契约与 `webapp/views.py` 展示数据。同项目写操作由本地线程锁或 `server/projects.py` 文件锁串行化，数据库租约标识任务归属；`Workspace` 内部锁不代替整个操作的互斥。浏览器只接收项目数据与下载 URL，不依赖磁盘路径。

### 工作台与桌面

`webapp/static/` 使用原生 ES modules，无前端构建。`app.js` 连接阶段和编辑器，`workspace.js` 管理项目导航与任务；`state.js` 区分已保存快照和草稿。`metadata-editor.js`、`boxes.js`、`measure-editor.js` 分别负责信息、区域、小节编辑。保存显式携带 revision，冲突保留草稿。

`score-model.js` 修改音符数据，`score-engraving.js` 生成 alphaTab 排版投影，`score-view.js` 处理点选和拖动；排版对象不写回识别记录。`theme.css` / `theme.js`、`shell.css`、`icons.js` 共用，`style.css` 管流程面板，`editor.css` 管编辑器。

原生客户端不启动 Python；音乐规则放在 core，项目事务放在 service，不写在桌面壳或 UI 中。

Python 与 Rust 必须各自实现运行所需算法，但共用 M2、stage schema 1.0、score/2、项目 ZIP 及工作台 API 形状，不另造插件协议。`desktop/native-core/tests/fixtures` 保存像素/语义金样，Python `test_m2_semantics` 与 Rust `tests/score.rs` 同读 IR 金样；修改规则时两端一起验证，不各改一份期望。GP5/MusicXML 的跨语言回读检查位于 `desktop/native-core/tests/verify_exports_oracle.py`，只供开发，不随客户端运行。模型推理的数值等价另测，不能由这些金样推断。

原始记录、OCR 可见标签和导出投影的隐藏字段有意不同；保持旧默认，先说明格式损失或产品选择，再改语义。

## 打包与依赖

Python 研究/服务模型路径由 `shared/defaults.py` 读取选定缓存；`scripts/launcher.py` 管其安装与环境复用，`shared/model_files.py` 管不加载模型的校验。桌面端独立使用原生模型缓存，模型清单仍来自同一份 `weights/distribution.json`。

`package_release.py` 生成 Python 启动 ZIP 及模型附件。桌面打包仅包含共用静态资源、Rust 服务和校验过的原生组件/许可，不复制 Python、训练工具或模型大文件。

部署模型由 `weights/distribution.json` 管理，更新模型时修改 `generation` 和文件清单。运行依赖由 `scripts/runtime-manifest.json` 与 `scripts/runtime-vllm.txt` 管理；修改依赖时提升对应运行环境 generation。界面及普通代码更新不修改这两个版本。服务端的在线更新仅复用兼容的依赖与模型，运行环境发生变化时通过安装器升级。

模型变更需更新 `weights/manifest.json`、模型说明、训练配置和评测结果。数据、模型缓存、用户项目及私人曲谱放在 Git 忽略的目录中。

当前 Transformers 5.8.0 有[已记录的依赖公告](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)。升级时需同时验证 LLaMA-Factory 的版本兼容性及模型训练、推理、保存行为。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### 桌面安装包

需要 Node.js 22、Rust 1.88+ 和目标系统的 [Tauri 构建依赖](https://v2.tauri.app/start/prerequisites/)。本地执行，不要求推送：

```bash
npm ci --prefix desktop
cargo fetch --locked --manifest-path desktop/src-tauri/Cargo.toml
cargo fetch --locked --manifest-path desktop/native-service/Cargo.toml
npm run native:build --prefix desktop
npm run resources --prefix desktop
npm run build --prefix desktop -- -- --locked
```

完整本机识别包需给资源脚本传入原生组件清单，并使用 `--require-inference` 检查组件/许可闭包。产物位于 `desktop/src-tauri/target/release/bundle/`。手动桌面工作流只产出未发布验证附件，不自动公开发布；无 Python 资源不等于所有平台已验收。

### 本机引擎与模型

`scripts/llamacpp-runtime.json` 固定引擎提交和已核验附件。`build_llamacpp_runtime.py` 构建，`package_llamacpp_runtime.py` 校验打包，`collect_native_assets.py` 汇集组件/许可；用户机器不编译。获取与验收见[原生打包](docs/native-packaging.md)，不能填猜测 URL 或哈希。

缺少系统开发包的 Debian 构建机可用 `python scripts/build_linux_desktop.py --help` 查看不提权的本地构建方式；Windows/macOS 仍需各自 SDK。`.github/workflows/desktop.yml` 和 `llamacpp-runtime.yml` 只生成候选附件；`auxiliary.yml` / `gguf.yml` 将核验模型上传到指定草稿发布，不能与公开发布混为一谈。

原创校对样本可用 `python -m scripts.create_demo` 重建；只检查项目校对与导出，不作为 OCR 推理评测。
