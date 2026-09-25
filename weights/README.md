# 随项目发布的模型

本目录拟随项目发布小节识别 LoRA、谱面信息 LoRA 和图片 TAB 版面模型。模型配置和推理所需权重都存放在本目录，大文件通过 Git LFS 管理。权重许可证与训练素材授权记录尚需项目所有者确认，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。

```bash
git lfs install
git lfs pull
```

| 步骤 | 目录 | 内容 |
| --- | --- | --- |
| 小节识别 | `glm_ocr_measure_sequence_v2_lora/` | GLM-OCR 小节 LoRA，约 31.5 MB |
| 谱面信息 | `glm_ocr_document_info_v2_lora/` | GLM-OCR 页眉／速度 LoRA，约 31.5 MB |
| 版面定位 | `pp_doclayout_v3_score_sparse/` | PP-DocLayoutV3 推理模型，约 130.5 MB；检测 `measure` 和 `tempo_region` |

两套 LoRA 共用 GLM-OCR 基座，使用以下命令下载：

```bash
uv run --no-sync hf download zai-org/GLM-OCR --revision ca5d8b3e287e52589e37c28385d9655ee4372f9d --local-dir tools/models/GLM-OCR
```

推理和评测入口使用本目录内的 LoRA；图片定位时指定 `--layout-model-dir weights/pp_doclayout_v3_score_sparse`。基座的 processor/tokenizer 用于两套 LoRA，训练检查点中的优化器状态、日志和重复 tokenizer 文件不属于推理模型文件。

`manifest.json` 记录每个发布文件的大小、SHA-256 和已知训练产物来源。更新正式模型时，同步更新这份清单；训练中的检查点仍写入 `output/`。

原始基座模型及其第三方组件保留原有许可。训练样本不随模型目录发布。

## 校验与模型卡

一键发布包包含上述权重，安装器自动校验并下载固定 revision 的基座。手动环境可执行 `uv run --no-sync guitarocr-check --hashes`。`manifest.json` 同时记录基座文件哈希。

- [小节识别模型卡](glm_ocr_measure_sequence_v2_lora/README.md)
- [谱面信息模型卡](glm_ocr_document_info_v2_lora/README.md)
- [版面模型卡](pp_doclayout_v3_score_sparse/README.md)

训练数据不会被打包进权重目录。未留存的历史训练信息在模型卡中明确标注，后续发布应补齐版本化训练记录。
