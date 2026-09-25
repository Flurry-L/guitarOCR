---
tags: [object-detection, music, paddlex]
---

# TAB 版面定位模型

由 PP-DocLayoutV3 微调，检测两类区域：`measure`（小节）和 `tempo_region`（速度）。页眉根据首小节上方区域生成，可由用户调整。这个目录是一份最终推理模型。

训练配置：`layout/configs/train.yaml` 为基础训练，`train_sparse.yaml` 为稀疏谱行微调。当前发布文件来自历史 `output/pp_doclayout_v3_score_sparse/0/inference`；源码中的训练配置已整理为可复用路径，新训练应明确选择发布检查点。

当前本地基础数据集为 6248 页训练 / 839 页验证；稀疏增强集为 7748 页训练 / 989 页验证，含 188431 / 25047 个标注框。这些是本地数据快照统计，不构成独立测试集成绩。历史评测文件尚未携带完整的模型与数据版本签名，因此本模型卡不将其作为当前发布模型的可复现准确率声明。

页面按 180 DPI 渲染。推理经过阈值、阅读顺序和谱线证据后处理。PaddlePaddle 3.2.0 与 PaddleX 3.7.2 的独立环境可运行，安装器默认 CPU 版；GPU 环境见 docs/setup.md。

模型面向规则纯 TAB，手写、照片透视、五线谱或混合谱不属于此模型已验证的定位范围。遇到框错误，可在工作台调整。权重与推理配置的 SHA-256 见上级 manifest.json；当前示例验证见 docs/validation.md，完整 COCO 评测命令见 docs/training.md。
