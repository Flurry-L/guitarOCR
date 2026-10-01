# 模型

谱面信息、页面分轨和小节识别共用一个 GLM-OCR 模型，以不同提示词选择任务。版面检测与拍号／调号分类使用较小的辅助模型。

| 目录 | 用途 |
| --- | --- |
| `score_ocr/merged` | 曲名、作者、乐器、调弦、谱号、分轨、音符、节奏、奏法、和弦名称及指法图 |
| `layout` | 检测 TAB、五线谱、混合谱小节，以及速度、谱号和一般标注区域 |
| `score_ocr/merged/state_reader` | 训练用拍号／调号分类器；部署使用对应 ONNX 文件 |
| `auxiliary` | 部署用版面检测和拍号／调号 ONNX 文件 |

训练与实验推理使用通用 safetensors，vLLM 与 Transformers 共用一份权重。客户端只获取部署用 GGUF 和 ONNX：Q8_0 语言模型、F16 视觉编码器及辅助模型合计约 1.775 GB。训练权重不随应用下载，原生运行组件随安装包提供。模型转换代码位于 `research/export/`，缓存规则见[安装说明](../docs/setup.md#保存与更新)。

音乐词表增加 1,121 个词元；`music_vocabulary.json` 记录映射，`score_image_policy.json` 记录训练及推理共用的等比缩放、谱线尺度和小标注放大规则。页面结构使用固定行数及可见连线约束；小节首次生成不加 M2 语法约束，失败重试时启用。研究环境的 vLLM 支持经过蒸馏的 MTP，GGUF 去掉不执行的 MTP 层。

页面按乐器和谱表组织共同的小节时间轴。钢琴双谱表共享一个乐器声部，五线谱＋TAB 配对为一份音乐内容。拍号、调号批量读取后传播；当前及相邻小节图像供各小节独立识别，跨小节延音在生成后协调。和弦指法图保留实际可见弦品、手指及横按，页眉和弦图作为指法库保存，只有小节中读到的和弦名称才附着到节拍。

整谱以 `guitarocr.score/2` 保存，GP5 和 MusicXML 是导出格式。和弦名称、指法图和 `let ring` 等奏法标注不会作为移调指令；音高变化需要明确的谱面证据，八度虚线只能延续已经确认的八度标记。

`manifest.json` 登记文件路径、字节数和训练来源；`distribution.json` 选择部署模型版本及附件。Git 克隆可用 `GIT_LFS_SKIP_SMUDGE=1` 跳过训练权重，安装器按需获取部署文件。默认推理无需另行下载原始 GLM-OCR 基座，也无需保留旧的两套 OCR 目录。

| 任务 | 训练记录 | 评测记录 |
| --- | --- | --- |
| 共享 OCR | [training.json](score_ocr/training.json) | [evaluation.json](score_ocr/evaluation.json) |
| 版面 | [training.json](layout/training.json) | [evaluation.json](layout/evaluation.json) |

准确率与计时条件见[模型评测](../docs/model-evaluation.md)，训练入口见[训练说明](../docs/training.md)，输出格式见[小节文本格式](../docs/score-text.md)，模型授权见[第三方说明](../THIRD_PARTY_NOTICES.md)。
