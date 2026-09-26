# 模型

默认使用三项任务模型。两套 GLM-OCR LoRA 共用一个基座，版面模型独立运行。

| 任务 | 权重与模型卡 | 大小 |
| --- | --- | --- |
| 小节定位与谱面类型 | [PP-DocLayoutV3](pp_doclayout_v3_score_joint_v3/README.md) | 约 130.5 MB |
| 曲名、作者、调弦和速度 | [谱面信息 LoRA](glm_ocr_document_info_v3_headers_lora/README.md) | 约 31.5 MB |
| 小节音符与节奏 | [小节 LoRA](glm_ocr_measure_sequence_v3_lora/README.md) | 约 31.5 MB |

版面模型直接输出 TAB、五线谱、混合谱的小节框和速度框，将小节类型传给 OCR。训练规模、新旧对照和已知错误见[训练报告](../docs/training-v3-report.md)。

## 获取与校验

权重通过 Git LFS 获取：

```bash
git lfs install
git lfs pull
```

自动安装器会下载固定版本的 GLM-OCR 基座。手动环境执行：

```bash
uv run --no-sync hf download zai-org/GLM-OCR --revision ca5d8b3e287e52589e37c28385d9655ee4372f9d --local-dir tools/models/GLM-OCR
uv run --no-sync guitarocr-check --hashes
```

[manifest.json](manifest.json) 记录默认模型和基座的文件大小、SHA-256 与来源。两套 LoRA 使用基座的 processor 和 tokenizer；训练检查点中的优化器、日志和重复 tokenizer 不属于推理文件。

## 历史版本

以下权重保留供对照与回退，常规推理只使用上表中的默认版本。

| 历史模型 | 用途或替换原因 |
| --- | --- |
| [小节 v2](glm_ocr_measure_sequence_v2_lora/README.md) | 扩充训练的初始化与对照 |
| [谱面信息 v2](glm_ocr_document_info_v2_lora/README.md) | 谱头修复训练的初始化与对照 |
| [谱面信息初版 v3](glm_ocr_document_info_v3_lora/README.md) | 曾将副标题误作作者，保留失败记录 |
| [纯 TAB 版面模型](pp_doclayout_v3_score_sparse/README.md) | 两类框，面向纯 TAB |
| [三模式两类版面模型](pp_doclayout_v3_score_multimode/README.md) | 支持三种排版，类型需由谱线规则补充 |
| [三模式四类版面模型](pp_doclayout_v3_score_typed/README.md) | 同时检测位置和类型，作为扩充训练的初始化 |

模型授权与训练素材记录见[第三方说明](../THIRD_PARTY_NOTICES.md)。训练数据和优化器状态不随仓库提供；历史信息缺失处在模型卡中标明。
