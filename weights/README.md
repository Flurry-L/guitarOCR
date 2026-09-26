# 随项目发布的模型

本目录拟随项目发布小节识别 LoRA、谱面信息 LoRA 和支持三种谱面的版面模型。模型配置和推理所需权重都存放在本目录，大文件通过 Git LFS 管理。权重许可证与训练素材授权记录尚需项目所有者确认，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。

```bash
git lfs install
git lfs pull
```

| 步骤 | 目录 | 内容 |
| --- | --- | --- |
| 小节识别 | `glm_ocr_measure_sequence_v3_lora/` | GLM-OCR 小节 LoRA，约 31.5 MB |
| 谱面信息 | `glm_ocr_document_info_v3_headers_lora/` | GLM-OCR 页眉／速度 LoRA，约 31.5 MB |
| 版面定位与类型 | `pp_doclayout_v3_score_joint_v3/` | PP-DocLayoutV3 推理模型，约 130.5 MB；`measure_tab`、`measure_notation`、`measure_both` 和 `tempo_region` 四类 |

两套 LoRA 共用 GLM-OCR 基座，使用以下命令下载：

```bash
uv run --no-sync hf download zai-org/GLM-OCR --revision ca5d8b3e287e52589e37c28385d9655ee4372f9d --local-dir tools/models/GLM-OCR
```

推理和评测入口使用本目录内的 LoRA；版面定位默认使用 `weights/pp_doclayout_v3_score_joint_v3`，也可通过 `--layout-model-dir` 指定其他模型。默认自动识别每个小节的谱面类型，并将类型传给 GLM-OCR。上一版四类权重 `pp_doclayout_v3_score_typed/`、两类统一权重 `pp_doclayout_v3_score_multimode/` 和旧纯 TAB 权重 `pp_doclayout_v3_score_sparse/` 保留，便于对照和回退。基座的 processor/tokenizer 用于两套 LoRA，训练检查点中的优化器状态、日志和重复 tokenizer 文件不属于推理模型文件。

`manifest.json` 记录每个发布文件的大小、SHA-256 和已知训练产物来源。更新正式模型时，同步更新这份清单；训练中的检查点仍写入 `output/`。

原始基座模型及其第三方组件保留原有许可。训练样本不随模型目录发布。

## 校验与模型卡

上述权重通过 Git LFS 获取，步骤见 [README](../README.md#获取源码)。安装器自动校验权重并下载固定 revision 的基座。手动环境可执行 `uv run --no-sync guitarocr-check --hashes`。`manifest.json` 同时记录基座文件哈希。

- [小节识别模型卡](glm_ocr_measure_sequence_v3_lora/README.md)
- [谱面信息模型卡](glm_ocr_document_info_v3_headers_lora/README.md)
- [小节定位与谱面类型联合模型卡](pp_doclayout_v3_score_joint_v3/README.md)
- [上一版两类统一定位模型卡](pp_doclayout_v3_score_multimode/README.md)

训练数据不会被打包进权重目录。未留存的历史训练信息在模型卡中明确标注，后续发布应补齐版本化训练记录。

谱头 v2 和初版 `glm_ocr_document_info_v3_lora/` 保留供回退。初版 v3 因真实副标题／作者退步未采用，当前使用经过补充训练的 `glm_ocr_document_info_v3_headers_lora/`；测试中的改善和退步均见其模型卡。

小节 v2 保留供回退。三项默认权重均完成扩充训练与独立测试，数据规模、连续状态修复与剩余错误见[扩充训练报告](../docs/training-v3-report.md)。小节文本的显示名称见[格式说明](../docs/score-text.md)；内部协议保持兼容，不是额外模型。
