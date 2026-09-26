---
tags: [object-detection, music, paddlex]
---

# 小节定位与谱面类型

PP-DocLayout 同时预测 `measure_tab`、`measure_notation`、`measure_both` 和 `tempo_region`。混合谱中同一小节的五线谱与 TAB 属于一个框。页面和整份谱的类型按检测分数汇总，投票占比不是校准概率。

训练使用 2,400 个曲源的 26,192 张原生页面和 6,548 张模拟扫描页面；验证 199 个曲源、2,337 页，测试 200 个曲源、2,164 页。三种排版均由 Guitar Pro 原生渲染，同一曲源的变体保持在同一划分。

8 张 H100、每卡批量 4、学习率 2e-5，训练 6 轮，按验证框 AP 选择第 4 轮。独立测试四类 AP@[0.50:0.95] 为 95.30%，小节数量完全正确的页面为 2148/2164。

详细参数见 [training.json](training.json)，指标与输入哈希见 [evaluation.json](evaluation.json)，曲源分组见 [dataset_sources.json](dataset_sources.json)。[评测报告](../../docs/model-evaluation.md)说明测试范围和已知错误；继续训练见[训练说明](../../docs/training.md)。
