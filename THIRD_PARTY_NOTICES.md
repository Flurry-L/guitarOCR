# 第三方组件与许可

项目自身的许可证尚未选定。以下组件遵循各自许可，项目许可证不能改变它们的授权条件。

| 组件 | 许可与使用要求 |
| --- | --- |
| [pypdfium2 / PDFium](https://github.com/pypdfium2-team/pypdfium2) | 封装采用 Apache-2.0 或 BSD-3-Clause，PDFium 采用 BSD 风格许可。二进制还附带第三方许可，分发时一并保留。 |
| [pdfplumber](https://github.com/jsvine/pdfplumber) / pdfminer.six | MIT；用于提取 PDF 文字与位置。 |
| [alphaTab 1.8.4](https://github.com/CoderLine/alphaTab/tree/v1.8.4) / Bravura | alphaTab 为 MPL-2.0；Bravura 为 SIL OFL-1.1。用于校对谱面的排版与点选，许可和对应源码地址保存在 `webapp/static/vendor/`。 |
| [Tauri](https://tauri.app) / [uv](https://github.com/astral-sh/uv) | MIT 或 Apache-2.0；桌面安装包附带组件许可。系统 WebView 由操作系统提供。 |
| ReportLab | BSD；仅用于生成测试 PDF。 |
| [PyGuitarPro](https://github.com/Perlence/PyGuitarPro) | LGPL-3.0-only。作为可替换的独立 Python 包使用；分发时保留许可及版权说明，并履行适用的修改与源码提供义务。 |
| Pillow、NumPy、FastAPI、Uvicorn | 分别为 MIT-CMU、BSD-3-Clause、MIT、BSD-3-Clause；保留各自的版权和许可文本。 |
| Transformers、PEFT、PaddlePaddle、PaddleX | Apache-2.0；模型权重的许可需单独确认。 |
| PyTorch、CUDA 及二进制捆绑库 | 以实际发行包所附的许可为准，NVIDIA 组件另有条款。 |

依赖安装在用户本机，许可证随相应软件包提供。未来分发完整离线环境时，也需包含其许可和版权材料。

## 模型与素材

GLM-OCR 基座的模型卡声明 MIT，来源固定为 [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR/tree/ca5d8b3e287e52589e37c28385d9655ee4372f9d)。魔搭下载源使用相同文件校验值。桌面安装包和启动 ZIP 附带原始模型卡及 MIT 许可文本，见 `weights/licenses/`。PP-DocLayout 使用 Apache-2.0，同目录附带其许可。

仓库中的微调权重尚未单独声明许可证。模型目录保留训练来源和评测记录，完整训练素材的授权记录仍需补齐。

## Guitar Pro 数据工具

Guitar Pro 是 Arobas Music 的专有软件，安装程序和运行库不随仓库提供。`datagen/native-source/` 及 DLL 使用其内部接口和 Qt，需要分别确认迁入代码的授权、接口使用条件及 Qt 的分发要求。源码和构建步骤见[原生工具说明](docs/native-build.md)。

部署用 ONNX Runtime 使用 MIT 许可，OpenCV 使用 Apache-2.0 许可，随 Python 依赖保留各自的许可证文件。分发的 ONNX 辅助模型是原版面与拍号／调号模型的格式转换，继续适用对应模型授权。llama.cpp 使用 MIT 许可；预编译 runtime 须附带固定提交的原始 LICENSE、所含依赖许可与真实校验清单。当前本地构建保留上游聚合许可及相关头文件许可文本；正式发布还需目标平台依赖和兼容性审计。GGUF 转换不改变原始模型许可。

`scripts/create_demo.py` 生成的 Harbor Light 曲例与原图为原创合成素材，可按 CC0-1.0 使用、修改和再分发；预置校对结果没有执行 OCR，不代表模型准确率。
