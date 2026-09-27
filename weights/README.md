# 模型

运行时使用三个任务模型，两项 OCR 共用 GLM-OCR 基座。

| 目录 | 用途 | 大小 |
| --- | --- | ---: |
| `layout` | PP-DocLayout 检测小节类型及位置、速度、谱号和移调区域 | 约 130.5 MB |
| `document_info` | GLM-OCR LoRA 读取曲名、作者、乐器、调弦、速度和移调 | 约 31.5 MB |
| `measure_ocr` | GLM-OCR LoRA 读取音符、节奏、奏法和局部八度标记 | 约 31.5 MB |

版面模型预测 TAB、五线谱和混合谱，每个小节的类型传给 OCR。页面与整谱类型按检测分数汇总，投票占比不代表校准概率。界面可手动纠正。

桌面安装包和启动 ZIP 包含全部模型，首次本机识别时在本地展开，基座保存在 `tools/models/GLM-OCR`。Git 用户执行 `git lfs pull` 获取任务权重，基座由安装器下载。文件哈希和基座 revision 见 [manifest.json](manifest.json)，可运行 `uv run --no-sync guitarocr-check --hashes` 检查。

| 任务 | 训练参数 | 评测记录 | 标签与来源 |
| --- | --- | --- | --- |
| 版面 | [training.json](layout/training.json) | [evaluation.json](layout/evaluation.json) | [dataset_sources.json](layout/dataset_sources.json) |
| 谱面信息 | [training.json](document_info/training.json) | [evaluation.json](document_info/evaluation.json) | [capabilities.json](document_info/capabilities.json) |
| 小节 | [training.json](measure_ocr/training.json) | [evaluation.json](measure_ocr/evaluation.json) | [label_schema.json](measure_ocr/label_schema.json) |

准确率、评测条件和已知错误见[模型评测](../docs/model-evaluation.md)。继续训练见[训练说明](../docs/training.md)，使用各任务的 `configs/train.yaml`；输出含义见[小节文本格式](../docs/score-text.md)。模型授权见[第三方说明](../THIRD_PARTY_NOTICES.md)。
