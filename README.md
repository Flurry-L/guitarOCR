# GuitarOCR

将乐谱 PDF 或图片识别为可编辑的乐谱，对照原图校对后导出 GP5、MusicXML 或项目备份。

![乐谱编辑器](docs/assets/score-editor-0.1.webp)

支持 TAB、五线谱、五线谱＋TAB，以及吉他、贝斯、鼓、钢琴等乐器的多轨谱面。可校对音符、节奏、奏法、和弦名和指法图。

当前技术路线与人工校对边界见下方[识别机制](#识别机制)。本地整理后的模型尚未发布到下方历史下载版本。

## 使用

[下载应用](https://github.com/Flurry-L/guitarOCR/releases/tag/v0.1.0) · [安装与设备要求](docs/usage.md) · [自托管部署](docs/server.md)

提供 Apple 芯片 Mac 和 Windows 客户端。Windows 自动选择设备，兼容的 NVIDIA 显卡按需获取加速组件。Linux、Intel Mac 和自托管服务从源码构建。

1. 打开应用，导入 PDF 或图片，点击「开始识别」。首次识别需下载约 1.8 GB 模型。
2. 对照原谱修改音符，按「确认并继续」逐小节核对。
3. 导出 GP5 或 MusicXML；需要保留原图和校对进度时，下载项目 ZIP。

「项目」中可搜索乐谱、查看识别进度、停止或继续处理。内置示例可直接练习编辑，无需下载模型。详细操作见[使用说明](docs/usage.md)。

## 代码入口

日常开发从下面五条路径进入，具体模型、训练参数和平台构建命令随各自文档展开。

| 要修改的功能 | 主入口 |
| --- | --- |
| 识别流程 | [`pipeline.rs`](backend/src/pipeline.rs) 组织页面、区域与小节；[`recognition.rs`](backend/src/recognition.rs) 调用识别器 |
| 编辑与保存 | [`app.js`](ui/workbench/app.js) 连接界面；[`edit.rs`](backend/src/edit.rs) 保存修改 |
| 音乐规则与导出 | [`score.rs`](scorelib/src/score.rs) 定义 IR；[`music_exports.rs`](scorelib/src/music_exports.rs) 导出 GP5／MusicXML；Python 通过 `scorelib` 调用同一核心 |
| 数据与训练 | [`data/run.py`](research/data/run.py) 组织选谱、渲染和裁图；[`training/measures.py`](research/training/measures.py) 训练主 OCR；辅助模型与评测见[训练说明](docs/training.md) |
| 运行与发布 | [`runtime.rs`](engine/src/runtime.rs) 管理模型和推理；[`desktop/main.rs`](desktop/src-tauri/src/main.rs) 管理窗口与后端；见[构建](docs/native-packaging.md)和[服务部署](docs/server.md) |

先运行编辑器：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Flurry-L/guitarOCR.git
cd guitarOCR
cargo run --locked -p guitarocr-backend -- --assets ui/workbench --port 7860
```

需要 Rust 1.90+。打开 `http://127.0.0.1:7860`，导入示例即可编辑和导出；启用识别见[应用构建](docs/native-packaging.md)。

## 识别机制

产品输入是 PDF 渲染出的页面或用户图片。PDF 字体、矢量线条和源工程只用于生产与核验训练真值，不作为产品推理的补救通道。

### 识别流程

1. **版面检测**：PP-DocLayout 检测三种小节区域和谱头、谱号、速度、标注。输入等比缩放并补白，保留原图坐标。有限的多尺度视图补充候选；质量网络结合图像和视图间的一致性，对已有框排序、去重，不生成新框或猜小节数。
2. **结构与身份**：学习谱行分组、声部关系、可见乐器名、谱表类型、谱号和跨行延续。名称只用一个模型定位，完整谱行和名称裁图一起交给 OCR 读取；不能确认时保留未知，要求人工指定。不合法的声部分配直接报错，不按重复行数、乐器类别或上一行模板补出身份。位置接近不等于同一乐器。
3. **音符与谱面信息**：共用一个 GLM-OCR 引擎，谱行与谱号任务按配置启用信息适配器。桌面、服务、命令行与整谱评测调用同一原生后端。小节首轮即使用 M 语法约束，读取目标图及相邻小节图，目标在前。格式或时序失败时，只重读目标图并保留可信谱面状态，避免错误草稿和邻图干扰；仅标注失败时保留已验证的音乐内容。拍号、调号按声部传播，书面音高在明确谱号、八度及乐器移调证据下转换，跨小节延音单独协调。
4. **和弦**：将指法图、谱头和弦库、正文和弦名称和实际拍点分别处理。指法图证据由两个小模型及一致性判断提供；不能因为页首有图就认定它在第一拍演奏。不确定的图、指法或拍点保留待确认。
5. **可见奏法**：将实际 OCR 事件定位到谱行图片，用完整当前谱行和可信的上一谱行读取 PM / let ring。每个效果独立判断“存在、缺失、未知”。自动结果写入可见效果层；未知保留已有信息，不重写原始音符的效果归属。
6. **IR 与人工校对**：整谱保存在 ScoreIR，单小节序列化为 `M` 文本。训练、推理、编辑和保存只接受当前协议，不迁移旧文本或旧识别日志。保存、重开、撤销与导出使用同一份音乐语义。用户修正框、声部、音符和标注；GP5 / MusicXML 是 IR 的导出结果。

### 能力边界

- 漏框、严重合框和完全未识别的谱行由用户补框或改框。质量网络只能选择它收到的候选。
- 图片中没有印出的播放参数、逐音力度、源文件隐藏开关不能可靠恢复，不作为正确标签。
- 没有文字标记、归属不明的延续线由用户核对范围。只有明确的八度标记进入音高前文。
- 陌生版式、模糊图像、复杂连线、跨系统身份和指法图细节仍可能识别错。语法正确不代表音符正确，导出前仍需对照原谱。
- GP5 的弦数、声部数、编码及指法图库表示能力有限；完整工作应保存项目 ZIP。

每项任务只保留当前采用的推理路径。识别失败后的小节重读、人工待核对状态和文件保存校验各有明确用途；未采用的实验选项、历史模板补救及其专属测试不进入主线。增加第二模型或额外重试前，应先用相同输入做对照，确认收益。

[参与开发](CONTRIBUTING.md) · [数据、训练与评测](docs/training.md) · [乐谱表示](docs/score-text.md)

项目许可证尚未选定，第三方授权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
