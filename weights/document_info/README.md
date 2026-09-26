---
base_model: zai-org/GLM-OCR
library_name: peft
tags: [image-text-to-text, music, lora]
---

# 谱面信息识别

该 LoRA 读取可见的标题、作者、调弦标签和速度。缺失字段输出 `null`。作者标签来自 Guitar Pro 的 artist 字段，复杂署名仍需人工核对。

训练包含 23,808 张裁图，验证 1,584 张，测试 1,536 张。曲源来自统一划分的主语料，另生成中英文标题、副标题和作者变体。扫描退化图只作为模拟增强。

8 张 H100、每卡批量 12、LoRA rank 8 / alpha 16、学习率 1e-5，共训练 496 步。按验证损失选择第 400 步。独立测试的标题、作者、调弦和速度分别正确 765/768、761/768、768/768 和 768/768。

详细参数见 [training.json](training.json)，字段指标及数据哈希见 [evaluation.json](evaluation.json)。[评测报告](../../docs/model-evaluation.md)说明测试范围和已知错误；继续训练见[训练说明](../../docs/training.md)。
