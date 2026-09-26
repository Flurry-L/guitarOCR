# 参与开发

## 开发环境和检查

```bash
uv sync --locked --python 3.11 --extra webui --extra dev
uv run --no-sync ruff check datagen layout document_info measure_ocr gp5_export pipeline shared webapp scripts examples tests
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync playwright install chromium
uv run --no-sync python -m unittest discover -s tests -p 'browser_*.py' -v
uv build --wheel --out-dir dist
uv run --no-sync python scripts/check_package.py dist
```

这组检查不需要 Torch、权重或 GPU。CI 在 Linux / Windows 上运行核心检查，在 Linux 上运行 Chromium 回归。GPU 验证使用 `python -m pipeline.evaluate`，命令见 docs/training.md。训练环境安装后使用 `uv run --no-sync`，避免同步命令移除额外训练依赖。

## 代码与接口

继续按处理阶段组织目录，训练和评测跟随对应阶段。`pipeline/` 与 `webapp/` 负责组织步骤。小节文本约束与默认路径放在 `shared/`；阶段清单类型在 `shared/schema.py`。

前端使用原生 ES modules，无需 Node.js 构建：app 组织流程，api 处理请求及版本头，pages 管理翻页，boxes 管理画框，measure-editor 管理音符编辑，state 保存页面状态。修改状态依赖时，应验证用户保存的结果是否保留。回归测试应覆盖具体的错误行为。

项目写操作要求读取时的 revision 作为 `If-Match` 请求头；版本冲突返回 409，缺少版本返回 428。不要通过自动刷新版本后重试覆盖来绕开冲突。

## 文档与模型

README 提供首次使用的完整路径；安装、数据、训练、格式参考各放一个平铺文档。修改命令时实际执行一次；缺少硬件或外部运行时时说明验证范围。模型变更要记录基座 revision、权重哈希、训练配置、数据分组与评测条件，区别目标门槛和实测数值。

中文文案使用常用词，按钮直接说明动作。技术细节放在开发文档中；用户操作页只保留完成当前任务所需的信息。修改界面后同步更新使用说明和截图。

## 准备发布包

取得完整 LFS 权重后执行：

```bash
uv run --no-project --python 3.11 scripts/package_release.py
```

生成 `output/releases/GuitarOCR-版本号.zip` 与 SHA-256 文件。包内包含源码、启动脚本、模型卡和当前默认的实际 LoRA / 版面权重，不包含本地环境、训练数据、用户项目或 GLM 基座。用户可直接解压运行 start。构建前校验权重，历史模型二进制不进入 ZIP。`release.json` 记录来源提交，供无 Git 环境修复缺失文件；`使用说明.txt` 提供首次启动步骤。

GitHub Actions 的 Build release ZIP 流程只产生可下载构件，不自动发布。Native 构建流程记录源码和 DLL 哈希，但仍需在目标 GP8 运行时验收。发布时更新 CHANGELOG、验证记录及模型清单，并对最终提交的新检出完成安装验收。

`database/`、`output/`、`tools/` 和本地环境由 Git 忽略。请勿提交私人曲谱、上传文件或训练缓存。第三方组件遵循各自许可。

公开发布前先处理 [开源审核记录](docs/open-source-audit.md) 中的许可、模型来源和依赖待办。项目所有者补充的根目录 `LICENSE*`、`NOTICE*` 与 `THIRD_PARTY_NOTICES.md` 会进入发布 ZIP；依赖包自身的许可文本仍须按实际分发内容保留。
