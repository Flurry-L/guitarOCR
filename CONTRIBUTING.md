# 参与开发

目录职责见 [README](README.md#代码入口)。修改先放进实际拥有该能力的模块，避免桌面、服务和研究脚本各维护一份音乐规则或项目业务。

以维护者能沿入口读懂输入、处理和输出为简化标准：删除没有调用者的代码、重复实现和纯转发层；相关的小函数跟随所属流程。避免单函数单文件，也避免把无关职责塞进大文件或压成难读的一行。不为未来设想增加扩展接口、兼容层或配置项。提交时说明删掉了什么复杂性和实际验证结果。

仓库精简先从软件、训练、导出和打包入口核对全仓依赖，再统一删除；删除后复查孤立的辅助函数、界面分支和测试资源。不能只凭文件大小、名字或没有直接导入就判定无用，也不能把一轮局部删减当成全仓整理完成。

## 依赖与边界

- `scorelib` 定义音乐语义与格式转换，不依赖训练、HTTP、模型或桌面。Python 的 `score.py` 集中调用 maturin／PyO3 扩展，`guitarpro.py` 负责 GP 文件对象适配；训练提示词属于 `research/data/prompts.py`。
- `gpbridge` 管 Guitar Pro 会话、原始数据和 Wine 工作进程；选曲、数据划分、可见标签与训练清单属于 `research/data`。
- `research` 中 `data` 造数据，`models` 放网络，`training` 按模型组织训练及配套校准、导出和专项评估，`evaluation` 放整谱与参考裁图评测，`export` 放独立的模型格式转换。专用 Dataset 和只被一个功能使用的辅助函数放在所属模块；共同使用的输入处理放在 `common`。`inference` 保留共享的模型读取器和原生 API 客户端；整谱识别、状态继承、人工修改与导出由 `backend` 统一负责。
- `engine` 管运行时、模型生命周期与调度，不认识项目、乐谱或账号。`backend` 组合图像处理、识谱、`scorelib` 和项目事务；本地与服务端模式只增加不同的访问边界。
- `ui/workbench` 是共用编辑器；`ui/accounts` 管登录和管理页；`ui/launcher` 是桌面入口。`desktop` 只管理窗口、进程、文件下载与系统交互。
- `weights` 保存经过选择的模型；候选检查点和评测输出写入 `output`。`scripts` 只放跨模块打包工具，不接管研究任务。

原生工程共用根目录 `Cargo.toml`／`Cargo.lock`。Python 工程共用根目录 `uv.lock`，三个包可以分别安装。

## 开发与检查

Node.js 22、Rust 1.90+：

```bash
cargo run --locked -p guitarocr-backend -- --assets ui/workbench --port 7860
cargo check --locked -p scorelib -p guitarocr-engine -p guitarocr-backend
cargo fmt --all -- --check
```

工作台从浏览器访问 `http://127.0.0.1:7860`，无模型也能打开示例和编辑导出。真实识别需要原生资源和模型；资源获取、Tauri 系统依赖及打包见[原生打包](docs/native-packaging.md)。

Python 数据与训练开发需要 Rust 构建 `scorelib` 的扩展：

```bash
uv sync --locked --package guitarocr-research --python 3.11 --extra dev
uv run --no-sync ruff check research gpbridge scorelib/python scripts
```

完整训练依赖见[训练说明](docs/training.md)。更新可编辑库时使用 `uv pip install --no-deps -e ./scorelib -e ./gpbridge -e ./research`，避免重装训练环境。

一次性选曲配额、回放比例和补训配方留在实验输出目录。可复用的数据能力加入对应生成器或汇总入口，同一任务的训练与评测复用数据加载和指标；不为每次实验新增一套命令、兼容入口和说明文件。

开发验证使用编译、静态检查和实际主要操作，不维护常驻测试套件或固定 fixtures。只检查本次修改涉及的行为：界面使用真实后端操作，识别质量使用真实曲谱和独立数据集评估；训练与推理预处理需核对一致性。一次性验证脚本、截图和实验记录留在临时目录或 `output/`，不进入 Git；使用与维护说明更新现有文档。

项目修改使用 revision 检查及原子保存。取消任务保留已经写入的阶段与小节；续跑依据 `ocr_task` 状态和小节日志。外部替换 PDF、模型或其他识别输入后，需重新开始识别。账号模式将项目隔离在各自目录，共享模型实例。重启后中断任务由用户继续，不自动重新提交所有任务。
