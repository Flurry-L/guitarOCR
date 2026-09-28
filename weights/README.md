# 模型

默认流水线包含版面检测、谱面信息、小节识别和拍号／调号分类。两项 OCR 使用分别合并的 GLM-OCR 权重及经过任务训练的原生 MTP 层；CUDA 加速环境可通过 `uv run python -m scripts.setup_acceleration` 安装。

| 目录 | 用途 | 主要文件 |
| --- | --- | --- |
| `layout` | 检测小节类型及位置、速度、谱号和移调区域 | 约 130.5 MB 检测权重 |
| `document_info` | 曲名、作者、乐器、调弦、速度、谱号、移调和变调夹 | 约 31.5 MB LoRA；`merged/` 为合并模型 |
| `measure_ocr` | 同谱小节并行识别音符、节奏、奏法和局部八度标记 | rank-16、约 63 MB LoRA；`merged/` 为合并模型 |
| `measure_ocr/merged/state_reader` | 批量读取印出的拍号和调号 | 约 53.3 MB 分类器 |

两个合并模型各约 2.7 GB，包含处理器、主模型和 MTP 参数。`inference.json` 记录默认引擎及批量配置。没有 CUDA 加速环境时，程序使用基座与 LoRA 的 Transformers 后端。

小节模型读取当前及相邻小节图像。拍号、调号先批量识别，再按谱序传播；小节文本可以并行生成，跨小节延音关系在生成后统一处理。检测出的 TAB、五线谱或混合谱类型传给 OCR，界面支持手动纠正。

Git 用户通过 `git lfs pull` 获取任务权重；基座保存在 `tools/models/GLM-OCR`，由安装器下载。[manifest.json](manifest.json) 登记发布版本、基座 revision、模型文件及字节数，包含嵌套的合并模型和分类器。

| 任务 | 训练记录 | 评测记录 |
| --- | --- | --- |
| 版面 | [training.json](layout/training.json) | [evaluation.json](layout/evaluation.json) |
| 谱面信息 | [training.json](document_info/training.json) | [evaluation.json](document_info/evaluation.json) |
| 小节 | [training.json](measure_ocr/training.json) | [evaluation.json](measure_ocr/evaluation.json) |

实际准确率和计时条件见[模型评测](../docs/model-evaluation.md)，训练入口见[训练说明](../docs/training.md)，输出格式见[小节文本格式](../docs/score-text.md)，模型授权见[第三方说明](../THIRD_PARTY_NOTICES.md)。
