# 模型

默认流水线包含版面检测、谱面信息、小节识别和拍号／调号分类。两项 OCR 使用分别合并的 GLM-OCR 权重及经过任务训练的原生 MTP 层；CUDA 加速环境可通过 `uv run python -m scripts.setup_acceleration` 安装。

| 目录 | 用途 | 主要文件 |
| --- | --- | --- |
| `layout` | 检测小节类型及位置、速度、谱号和移调区域 | 约 130.5 MB 检测权重 |
| `document_info` | 页面分轨、曲名、作者、乐器、调弦、速度、谱号、移调和变调夹 | `merged/` 为完整模型 |
| `measure_ocr` | 各声部小节并行识别音符、节奏、奏法和局部八度标记 | `merged/` 为完整模型 |
| `measure_ocr/merged/state_reader` | 批量读取印出的拍号和调号 | 约 53.3 MB 分类器 |

两个合并模型各约 2.7 GB，包含处理器、主模型和 MTP 参数。`inference.json` 记录默认引擎及批量配置。没有 CUDA 加速环境时，Transformers 后端加载同一份完整模型。

小节模型使用增加 1,121 个词元的音乐词表，默认 MTP 4；`merged/music_vocabulary.json` 保存词元与训练权重，`merged/score_image_policy.json` 保存训练和推理共用的等比例缩放、谱线尺度与白边规则。拍号／调号图片预处理使用最多 4 个 CPU 线程。首次生成不加 M2 语法约束，失败重试时启用约束；这些设置不改变 Score IR 或导出格式。

页面结构模型把谱表归入乐器声部，钢琴双谱表共享声部，五线谱＋TAB 作为一份音乐内容识别。小节模型读取同一谱表的当前及相邻小节图像。拍号、调号先批量识别，再按谱序传播；小节文本并行生成，跨小节延音关系在生成后统一处理。检测出的 TAB、五线谱或混合谱类型传给 OCR，界面支持手动纠正。整谱以 `guitarocr.score/2` 保存，GP5 和 MusicXML 是该结构的导出格式。

Git 用户通过 `git lfs pull` 获取完整任务权重，默认推理无需另行下载原始 GLM-OCR 基座。[manifest.json](manifest.json) 登记发布版本、基座来源、模型文件及字节数，包含嵌套的合并模型和分类器。

| 任务 | 训练记录 | 评测记录 |
| --- | --- | --- |
| 版面 | [training.json](layout/training.json) | [evaluation.json](layout/evaluation.json) |
| 谱面信息 | [training.json](document_info/training.json) | [evaluation.json](document_info/evaluation.json) |
| 小节 | [training.json](measure_ocr/training.json) | [evaluation.json](measure_ocr/evaluation.json) |

实际准确率和计时条件见[模型评测](../docs/model-evaluation.md)，训练入口见[训练说明](../docs/training.md)，输出格式见[小节文本格式](../docs/score-text.md)，模型授权见[第三方说明](../THIRD_PARTY_NOTICES.md)。

## 可选 GGUF

0.1 Release 另提供两套 GGUF 模型包：语言模型 Q8_0 ＋配套 F16 视觉编码器，单任务约 1.9 GB。它们由本目录的完整训练权重转换，保留扩展音乐词表。`scripts/export_gguf.py` 可重新导出。

设置 `GUITAROCR_BACKEND=llamacpp` 后，两项 OCR 使用各自的本机 llama-server；原始后端保持默认。完整流程已在 Linux CUDA 运行，Metal / Vulkan 与手机真机未验证。当前 llama.cpp 保留但不执行本模型的 MTP 层。转换对照与启动命令见[GGUF 可选后端](../docs/setup.md#gguf-可选后端)。
