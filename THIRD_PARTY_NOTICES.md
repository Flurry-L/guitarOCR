# 第三方组件与授权状态

本文件记录发布审核中确认的上游许可和待补材料，不授予本项目源码、微调权重或曲谱素材新的使用许可。项目所有者尚未选定项目许可证；公开发布前需要添加完整 `LICENSE`，并明确模型、示例和原生工具的授权范围。

## 核心依赖

| 组件 | 已核对的许可 | 发布时需要考虑的事项 |
| --- | --- | --- |
| [PyMuPDF / MuPDF](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright) | GNU AGPL-3.0 或 Artifex 商业许可 | 本项目直接导入并使用该组件；选择项目及组合软件的许可时，需处理 AGPL 的适用义务或取得适用商业授权。不能仅给自有代码加 MIT 文本就认为全部依赖的授权已解决。 |
| [PyGuitarPro](https://github.com/Perlence/PyGuitarPro) | LGPL-3.0-only | 分发该依赖或修改版本时保留其许可、版权说明，并履行对应义务。 |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU | 保留上游许可及版权说明；其捆绑图像库可能另有许可。 |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause 及捆绑组件的其他许可 | 二进制发行中的 BLAS 等组件需按实际 wheel 核对。 |
| [FastAPI](https://github.com/fastapi/fastapi)、[Uvicorn](https://github.com/encode/uvicorn) | MIT、BSD-3-Clause | 保留相应上游说明。 |
| [Transformers](https://github.com/huggingface/transformers)、[PEFT](https://github.com/huggingface/peft) | Apache-2.0 | 框架许可与下载模型的权重许可分别核对。 |
| [PyTorch](https://github.com/pytorch/pytorch) | 多许可组合，见实际发行包 | CPU / CUDA wheel 捆绑内容不同；CUDA 相关包还有 NVIDIA 的条款。 |
| [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)、[PaddleX](https://github.com/PaddlePaddle/PaddleX) | Apache-2.0 | 框架许可不能替代 PP-DocLayoutV3 权重及微调数据的授权记录。 |

以上是主要直接依赖清单，不是传递依赖及二进制捆绑库的完整许可证清单。发布 ZIP 不附带 Python 环境，安装器会下载依赖；这并不免除核对项目使用方式与许可兼容性的要求。以实际安装版本所附的 `LICENSE`、`NOTICE` 和上游条款为准。

## 模型与曲谱

- GLM-OCR 基座来自 [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR/tree/ca5d8b3e287e52589e37c28385d9655ee4372f9d)。本地对应模型卡声明 MIT；正式分发时仍需保留该固定版本的授权材料。
- `weights/` 中两套 LoRA 与 PP-DocLayoutV3 微调模型尚未声明项目所有者授予的权重许可证。基座的许可、微调权重的许可和训练曲谱的授权是不同事项。
- 模型卡记录了部分训练来源与统计，尚缺完整、可追溯的数据授权清单。训练样本不随仓库发布，不能据此推断模型再分发条件已经满足。
- `examples/` 标注为项目自编示例；项目所有者仍需明确示例图、PDF、M2 和 GP5 使用哪一许可证。

## GP8 原生数据工具

`datagen/native-source/`、`datagen/native-bin/` 包含与 Guitar Pro 内部 ABI 配合的自定义代码、声明和 DLL。来源及构建状态见 [原生构建文档](docs/native-build.md)。

Guitar Pro 是 Arobas Music 的专有软件；本仓库不提供其安装程序或运行库。发布前需确认迁入代码的著作权归属、声明文件来源，以及这些工具的使用和再分发是否符合适用条款。Qt 使用版本、许可选项及链接方式也需结合真实构建核对。目前不能仅凭 DLL 哈希与来源副本相同，宣称完成了源码到二进制的可复现验证。

具体发布待办及本轮修复见 [开源审核记录](docs/open-source-audit.md)。
