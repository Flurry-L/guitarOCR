# 参与开发

## 本地检查

```bash
uv sync --locked --python 3.11 --extra webui --extra dev
uv run --no-sync ruff check datagen layout document_info measure_ocr gp5_export pipeline shared webapp server scripts tests
uv run --no-sync python -m unittest discover -s tests -v
```

检查覆盖 GP5 导出、项目保存与恢复、数据划分、版面类型和下载校验，不需要模型或 GPU。界面修改后，本地检查导入、画框、保存、导出和刷新恢复。

训练环境使用 `uv run --no-sync`，避免同步命令移除额外训练依赖。模型评测命令见[训练说明](docs/training.md)。

## 从使用路径读代码

一次命令行转换从 `pipeline/run.py` 开始，依次调用 `layout.run`、`document_info.run`、`measure_ocr.run`、`gp5_export.run`。每个阶段返回自己的 `manifest.json` 路径，也可以单独运行。下游用 `shared/artifacts.py` 读取和检查清单，字段见 `shared/schema.py`。

| 要改的规则 | 所在位置 |
| --- | --- |
| 页面展开、区域检测、谱面类型、阅读顺序 | `layout/` |
| 自动读取及手动修改乐器、调弦、速度、移调 | `document_info/run.py`、`document_info/edit.py` |
| 小节生成、前文、重试和续跑 | `measure_ocr/recognizer.py`、`measure_ocr/run.py` |
| 小节校对、信息变化对音符的影响、结果文件生成 | `measure_ocr/result.py` |
| 音乐文本语法、音高含义、节奏和奏法约束 | `shared/m2.py`、`shared/pitch_context.py`、`shared/constraints.py`、`shared/techniques.py` |
| 指法分配及 Guitar Pro 文件表示 | `gp5_export/` |

小节记录保存各自的谱面类型，OCR 按该类型选择提示词。模型池分别加载两项 OCR 的完整合并模型，CUDA 优先使用 vLLM。发布安装使用 ONNX 辅助模型；Paddle 用于训练和原始模型对照。训练与推理的小节裁图留白共用 `shared/crops.py`。

网页编辑从 `pipeline/workspace.py` 的 `Workspace` 进入同一组阶段。它管理项目目录、当前阶段、待续跑任务和 revision。修改成功后先写出新的阶段结果，再原子替换 `session.json`；失败时仍指向上次保存的结果。哪些编辑会使后续结果失效，见[工作台说明](docs/webui.md#保存与重新处理)。

识别清单的 `records` 是保存和导出的依据。`measure_ocr/result.py` 同时生成模型文本、下载用的 `score.txt` 和结构化的 `score.json`。下载文件是这些记录的投影，不用于反向覆盖项目。`pipeline/archive.py` 负责项目 ZIP 的路径转换与恢复；备份格式保持兼容。

本地 `webapp/app.py` 使用单个后台执行线程；`server/app.py` 检查账号、配额与项目归属，`server/worker.py` 领取 SQLite 中的持久任务。两者调用 `Workspace`，并共用 `webapp/contracts.py` 的请求格式、revision 检查和 `webapp/views.py` 的展示数据。浏览器收到阶段是否完成及下载 URL，不依赖服务器磁盘路径。

同一项目的写操作由调用方串行化：本地入口使用线程锁，多用户服务使用 `server/projects.py` 的文件锁，并以数据库租约识别任务归属。`Workspace` 的内部锁只保护单次读取与替换，不替代整个操作的互斥。关闭网页不影响服务器任务；任务恢复按提交时的模型路径继续。

`webapp/static/` 使用原生 ES modules，无需前端构建。`app.js` 连接导入、阶段操作与编辑器，`workspace.js` 管理导航、项目库、任务和运行设置。`theme.css` / `theme.js`、`shell.css` 和 `icons.js` 是三种入口共用的主题、框架和图标源，桌面打包时由 `sync-assets.mjs` 同步；`style.css` 管理工作流程面板，`editor.css` 管理画框和乐谱编辑。`state.js` 区分已保存快照和编辑草稿；打开项目和接受保存结果是显式操作。`metadata-editor.js`、`boxes.js`、`measure-editor.js` 各自负责一个编辑面板。`score-model.js` 修改音符数据，`score-engraving.js` 将识别数据映射为 alphaTab 排版模型，`score-view.js` 根据排版后的坐标处理点选和拖动。排版模型不写回识别数据。保存时显式传入读取时的 revision，冲突后保留草稿。

桌面入口是 `desktop/ui/` 和 `desktop/src-tauri/src/main.rs`。Tauri 管理窗口、下载和本机子进程；本机通过 `scripts/desktop_runtime.py` 启动同一个网页工作台，远程窗口连接服务器。音乐处理规则不在桌面壳中重复实现。Tauri 的开发和构建钩子调用 `desktop/sync-assets.mjs`，将共用主题复制到桌面前端目录；生成文件不入库。

数据生产从 `datagen/run.py` 进入。`datagen/catalog.py` 在渲染前确定曲源与 family 的划分，三种排版和各项训练任务共用它。`datagen/export_scores.py` 管理 Guitar Pro 导出进程；`datagen/native/` 与 `native-source/` 处理原生通信、导出协议和打印信息。`native/annotations.py` 从原生结果读取可见标签与小节位置，数据构建器负责生成各任务样本。

`datagen/sampling.py` 管理训练难例和固定评测抽样，`datagen/training_samples.py` 管理 LLaMA-Factory 数据格式；音乐提示词仍在对应 OCR 模块。训练入口保留各任务的配置选择，GLM 训练共用 `shared/training.py`，Paddle 训练适配在 `layout/train.py`。训练参数与评测命令见[训练说明](docs/training.md)。

## 打包与依赖

模型路径由 `shared/defaults.py` 读取选定的模型缓存；显式传入路径优先。`scripts/launcher.py` 负责安装和环境复用判断，桌面启动器也使用它。`shared/model_files.py` 提供不加载模型的文件校验，安装、诊断和打包共用。

取得当前训练权重后执行 `uv run --no-project --python 3.11 scripts/package_release.py`，生成轻量启动 ZIP 及原生 OCR、ONNX 附件。`scripts/export_gguf.py` 生成去除未用 MTP 的 GGUF 文件。发布前验证解压包的干净安装、PDF 识别和导出。桌面资源范围由 `scripts/prepare_desktop.py` 定义，模型大文件不会嵌入安装包。

部署模型由 `weights/distribution.json` 管理，更新模型时修改 `generation` 和文件清单。运行依赖由 `scripts/runtime-manifest.json` 与 `scripts/runtime-vllm.txt` 管理；修改依赖时提升对应运行环境 generation。界面及普通代码更新不修改这两个版本。服务端的在线更新仅复用兼容的依赖与模型，运行环境发生变化时通过安装器升级。

模型变更需更新 `weights/manifest.json`、模型说明、训练配置和评测结果。数据、模型缓存、用户项目及私人曲谱放在 Git 忽略的目录中。

当前 Transformers 5.8.0 有[已记录的依赖公告](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)。升级时需同时验证 LLaMA-Factory 的版本兼容性及模型训练、推理、保存行为。第三方许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

谱面编辑逻辑使用 `node --test tests/score-editor.test.mjs` 检查。

### 桌面安装包

需要 Node.js 22、Rust stable、Python 3.11 和目标系统的 [Tauri 构建依赖](https://v2.tauri.app/start/prerequisites/)。提交并推送代码后执行：

```bash
npm ci --prefix desktop
cargo fetch --locked --manifest-path desktop/src-tauri/Cargo.toml
python scripts/prepare_desktop.py
npm run build --prefix desktop -- -- --locked
```

资源准备脚本记录当前提交，本机识别按该版本下载 Release 的模型附件；缺失的小型配置按构建提交修复。产物位于 `desktop/src-tauri/target/release/bundle/`。

也可手动运行 **Build desktop installers** 工作流，生成 Windows、两种 macOS 和 Linux 安装包；产物位于该次运行的 Artifacts。发布时将安装包、启动 ZIP 与模型附件放入同一个 Release；全部构建成功后公开发布。
