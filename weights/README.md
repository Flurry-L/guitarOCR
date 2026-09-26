# 模型

运行时使用三个任务模型，两项 OCR 共用一个 GLM-OCR 基座。

| 目录 | 用途 | 大小 |
| --- | --- | ---: |
| [layout](layout/README.md) | 小节定位与谱面类型 | 约 130.5 MB |
| [document_info](document_info/README.md) | 曲名、作者、调弦和速度 | 约 31.5 MB |
| [measure_ocr](measure_ocr/README.md) | 小节音符、节奏和奏法 | 约 31.5 MB |

版面模型直接预测 TAB、五线谱、混合谱的小节框及速度框。每个小节的类型传给 OCR，界面中可以纠正。准确率和已知错误见[评测报告](../docs/model-evaluation.md)。

启动 ZIP 包含上述权重。Git 用户执行 `git lfs pull`；GLM-OCR 基座由安装器下载至 `tools/models/GLM-OCR`。文件大小、SHA-256 和基座 revision 记录在 [manifest.json](manifest.json)，可运行 `uv run --no-sync guitarocr-check --hashes` 检查。

每个模型目录中的 `training.json` 记录已发布权重的训练参数，`evaluation.json` 记录评测结果。继续训练使用各任务的 `configs/train.yaml`，步骤见[训练说明](../docs/training.md)。模型许可见[第三方说明](../THIRD_PARTY_NOTICES.md)。
