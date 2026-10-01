# 参与开发

目录职责见 [README](README.md#代码入口)。修改先放进实际拥有该能力的模块，避免桌面、服务和研究脚本各维护一份音乐规则或项目业务。

## 依赖与边界

- `scorelib` 定义音乐语义与格式转换，不依赖训练、HTTP、模型或桌面。Python 接口由 maturin／PyO3 构建，GP 文件对象适配位于 `scorelib/python/scorelib`。
- `gpbridge` 管 Guitar Pro 会话、原始数据和 Wine 工作进程；选曲、数据划分、可见标签与训练清单属于 `research/data`。
- `research` 中 `data` 造数据，`models` 放网络，`training` 训练，`inference` 实验推理，`evaluation` 评测，`export` 转换产物。提示词与图像规则由数据和推理复用。
- `engine` 管运行时、模型生命周期与调度，不认识项目、乐谱或账号。`backend` 组合图像处理、识谱、`scorelib` 和项目事务；本地与服务端模式只增加不同的访问边界。
- `ui/workbench` 是共用编辑器；`ui/accounts` 管登录和管理页；`ui/launcher` 是桌面入口。`desktop` 只管理窗口、进程、文件下载与系统交互。
- `weights` 保存经过选择的模型；候选检查点和测试输出写入 `output`。`scripts` 只放跨模块打包工具，不接管研究任务。

原生工程共用根目录 `Cargo.toml`／`Cargo.lock`。Python 工程共用根目录 `uv.lock`，三个包可以分别安装。

## 开发与检查

Node.js 22、Rust 1.90+：

```bash
cargo run --locked -p guitarocr-backend -- --assets ui/workbench --port 7860
node --test ui/tests/*.test.mjs desktop/tests/*.test.mjs
cargo test --locked -p scorelib -p guitarocr-engine -p guitarocr-backend
```

工作台从浏览器访问 `http://127.0.0.1:7860`，无模型也能打开示例和编辑导出。真实识别需要原生资源和模型；资源获取、Tauri 系统依赖及打包见[原生打包](docs/native-packaging.md)。

Python 数据与训练开发需要 Rust 构建 `scorelib` 的扩展：

```bash
uv sync --locked --package guitarocr-research --python 3.11 --extra dev
uv run --no-sync ruff check research gpbridge scorelib/python scripts
uv run --no-sync python -m unittest discover -s research/tests
```

完整训练依赖见[训练说明](docs/training.md#训练环境)。更新可编辑库时使用 `uv pip install --no-deps -e ./scorelib -e ./gpbridge -e ./research`，避免重装训练环境。

只运行与修改有关的检查。界面修改应实际检查导入示例、修改／取消、保存失败后保留草稿、导出和刷新恢复；推理修改应使用真实曲谱与真实模型，并保留输入、硬件和解码条件。既有小型检查用于保护边界，不能代替整谱评测。

项目修改使用 revision 检查及原子保存。取消任务保留已经写入的阶段与小节；续跑使用保存的记录。账号模式将项目隔离在各自目录，共享模型实例。重启后中断任务由用户继续，不自动重新提交所有任务。
