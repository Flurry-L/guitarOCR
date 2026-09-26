---
base_model: zai-org/GLM-OCR
library_name: peft
tags: [image-text-to-text, music, lora]
---

# 小节识别

该 LoRA 读取 TAB、五线谱或混合谱小节及其前文，输出音符、节奏和奏法。谱面类型由版面模型提供。

2,799 个曲源按 2,400 / 199 / 200 划分为训练、验证和测试。训练使用 587,019 张有效原生裁图、89,318 条技巧难例重复和 147,212 张模拟扫描裁图，共 823,549 条。完整验证和测试数据分别有 54,138、49,551 条，报告中的模型评测使用固定子集。

8 张 H100、LoRA rank 8 / alpha 16、学习率 2e-5、每卡批量 16、累积 2 步，全局批量 256。训练在 5,000 步早停，按验证损失选择第 3,500 步。固定 1,800 个独立测试小节的核心匹配率为 88.89%，完整匹配率为 80.44%。

详细参数见 [training.json](training.json)，指标、输入哈希和整谱验收见 [evaluation.json](evaluation.json)，标签的可见性依据见 [label_schema.json](label_schema.json)。[评测报告](../../docs/model-evaluation.md)解释指标与已知错误；继续训练见[训练说明](../../docs/training.md)。

输出格式见[小节文本参考](../../docs/score-text.md)。力度尚未按实际打印位置监督，训练语料中没有 trill 标签；格式合法仍可能存在音高错误。
