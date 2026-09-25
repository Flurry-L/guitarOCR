---
base_model: zai-org/GLM-OCR
tags: [image-to-text, music, lora]
---

# 小节识别 LoRA

输入单个小节裁图和上一小节上下文，输出 M2。随本仓库发布，不需要使用者自行训练或寻找适配器。

- 基座：GLM-OCR 0.9B，revision `ca5d8b3e287e52589e37c28385d9655ee4372f9d`。
- LoRA：rank 8、alpha 16、dropout 0.05；训练包含视觉层 LoRA，冻结多模态投影层。配置见 `measure_ocr/configs/train.yaml`。
- 文件大小与 SHA-256：见上级 `manifest.json`，适配器权重约 31.5 MB。
- 数据：GP 源谱解析标签与官方 GP8 渲染的小节裁图；通过源谱分组隔离训练、验证、测试，补充技巧难例。当前公开代码生成纯 TAB 数据。
- 当前发布适配器的完整历史训练清单、样本量和 checkpoint 步数尚未随可发布记录保留；不要将新的数据构建结果描述为该检查点的精确训练快照。后续训练应保存 source_catalog 哈希与训练记录。

支持规则排版的 TAB，五线谱和混合谱接口仍需独立效果验证。照片透视、手写谱、特殊排版和复杂技巧可能失败。结构校验通过也需要对照原谱，失败占位在项目中保留待检查状态。

评测区分正确裁图及标注前文、正确裁图及预测前文、完整 PDF / 图片流程。命令见 `docs/training.md`；可复现的自编示例实测见 `docs/validation.md`。发布门槛配置不是已达到的指标。尚未发布代表性完整测试集准确率或消费级显卡最低显存要求。
