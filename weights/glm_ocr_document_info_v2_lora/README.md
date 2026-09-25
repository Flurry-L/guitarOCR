---
base_model: zai-org/GLM-OCR
tags: [image-to-text, music, lora]
---

# 谱面信息 LoRA

读取页眉中的曲名、作者、调弦名称，以及速度区域中的四分音符 BPM。区域位置由 layout 步骤提供。它与小节识别 LoRA 共用 GLM 基座。

- 基座：GLM-OCR 0.9B，revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`。
- LoRA：rank 8、alpha 16、dropout 0.05；包含视觉层，冻结多模态投影层。配置见 `document_info/configs/train.yaml`。
- 已核对的训练产物：`output/glm_ocr_document_info_v2_lora`，最终 global_step 735，3 个 epoch；其 adapter_model.safetensors 的 SHA-256 与当前发布文件一致。
- 本地对应数据：3912 个训练裁图，496 个验证裁图；包括页眉和速度区域。训练损失 0.1010、验证损失 0.00935，这些是损失值，不能解读为识别准确率。
- 数据生成：`datagen/inventory.py` 与 `datagen/build_info_data.py`。只使用可见标题与可见的四分音符速度标签，过滤重叠或无法可靠对应的区域。

历史数据清单尚未形成随权重发布的完整版本快照。新的数据生成统一复用 source_catalog，训练前应保存其哈希。模型卡中的数量来自当前本地数据与匹配权重的训练记录；新构建的数据规模取决于输入 corpus。

目前不能可靠识别任意文字、任意拍单位的速度或任意自定义调弦。未找到信息时可在工作台手填。文件校验值见上级 manifest.json；效果条件与示例实测见 `docs/training.md`、`docs/validation.md`。
