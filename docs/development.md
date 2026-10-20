# 开发、构建与训练

从源码启动工作台，再按需要准备识别组件、构建安装包或训练模型。

[项目主页](../README.md) · [算法与评测](algorithm-and-models.md) · [乐谱数据与 M 文本](ir-and-m-spec.md)

## 1. 源码快速启动

需要 **Rust 1.90+**。工作台内置 `Harbor Light` 示例，可以先体验编辑、试听和导出：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Flurry-L/guitarOCR.git
cd guitarOCR
cargo run --locked -p guitarocr-backend -- --assets ui/workbench --port 7860
```

浏览器访问 `http://127.0.0.1:7860` 即可使用。

启用 PDF/图片识别时，按第 2 节准备 ONNX Runtime、PDFium 和 `llama.cpp` 原生组件，然后指定 `--resources` 与 `--models`：

```bash
cargo run --locked -p guitarocr-backend -- \
  --assets ui/workbench \
  --resources desktop/src-tauri/resources/native \
  --models tools/model-cache \
  --port 7860
```

### 代码导览与本地检查

| 要修改的功能 | 建议阅读顺序 |
| --- | --- |
| 工作台编辑与试听 | `ui/workbench/measure-editor.js` → `score-model.js` → `score-engraving.js` / `score-effects.js`；`playback-plan.js` 生成并截取 MIDI，`audio-preview.js` 管理 AlphaSynth 播放 |
| 图片预处理 | `backend/src/recognition/image_policy.rs` 管理 OCR 尺寸策略，`backend/src/image_transforms.rs` 提供共享缩放；Python 裁图函数在 `research/common/` |
| 信息区域识别 | `scorelib/src/information/` 保存提示词与 Schema，`information.rs` 解析响应；Python 使用 `scorelib.information` |
| 任务取消与阶段保存 | `backend/src/tasks.rs` 管理任务状态，`pipeline.rs` 串联阶段；`Progress::commit` 协调结果写入与取消 |
| 观察字段与跨节关系 | `scorelib/src/observation/schema.rs` → `observation.rs`；Rust 类型见 `types.rs` |
| M 文本与导出 | `scorelib/src/score.rs` → `pitch.rs` → `music_exports.rs`；前端调号与时值工具在 `ui/workbench/score-theory.js` |

前端使用原生 ES Module，修改后强制刷新页面即可查看。Node.js 22 可运行前端测试；Rust 测试覆盖音乐解析、组装、导出和后端接口：

```bash
npm test --prefix ui
cargo test --locked -p scorelib -p guitarocr-engine -p guitarocr-backend
cargo fmt --all -- --check
```

文档分工：README 介绍产品和快速上手；算法文档说明模型与评测；数据规范提供可运行的语法示例；本页集中放开发与部署命令。代码注释说明单位、音乐语义和模块接口，名称能表达的内容交给名称。

### 试听链路

`engrave()` 把项目与当前草稿转换成同一套 alphaTab Score，供绘谱与试听使用。试听由 alphaTab 的 [MidiFileGenerator](https://github.com/CoderLine/alphaTab/blob/v1.8.4/packages/alphatab/src/midi/MidiFileGenerator.ts) 解释奏法，再交给 [AlphaSynth](https://github.com/CoderLine/alphaTab/blob/v1.8.4/packages/alphatab/src/synth/AlphaSynth.ts) 和内置 SoundFont 合成声音。

`playback-plan.js` 保留相邻小节的延音、滑音关系及全谱速度，只为当前谱表生成 MIDI。单节试听按谱面顺序展开一次，再截取所选小节：跨入的延音在开头发声，跨出的音符在节末收束，反复与跳转不改变选节。`audio-preview.js` 负责加载音色、播放与停止，切换小节或离开项目会结束试听。音色版本和许可证见 [`vendor/soundfont/README.md`](../ui/workbench/vendor/soundfont/README.md)。

## 2. 构建桌面应用（Tauri）

桌面构建需要 **Node.js 22**、**Rust 1.90+**、**CMake**、系统 C/C++ 工具链及 [Tauri v2 系统依赖](https://v2.tauri.app/start/prerequisites/)。完整 CI 流程可参考 [`.github/workflows/desktop.yml`](../.github/workflows/desktop.yml)。

1. **构建或下载 `llama.cpp` 运行时**：
   运行 [`scripts/build_llamacpp_runtime.py`](../scripts/build_llamacpp_runtime.py) 编译 CPU / CUDA / Metal 推理组件，或直接复用 [`scripts/llamacpp-runtime.json`](../scripts/llamacpp-runtime.json) 中的预编译发布包。
2. **收集原生依赖**：
   运行 [`scripts/collect_native_assets.py`](../scripts/collect_native_assets.py) 收集 `llama.cpp`、ONNX Runtime 和 PDFium 并生成组件清单 `manifest.json`。
3. **打包桌面端**：

```bash
npm ci --prefix desktop
npm run native:build --prefix desktop
node scripts/prepare_native_desktop.mjs --manifest /path/to/manifest.json --require-inference
python scripts/smoke_desktop.py
npm run build --prefix desktop
```

检查脚本使用暂存的桌面资源，验证示例项目的导入、保存、GP5 / MusicXML 导出和 ZIP 往返。生成的安装包位于 `target/release/bundle/`。

## 3. 自托管服务器部署

将准备好的 `desktop/src-tauri/resources/` 放置到服务器目录，创建管理员账户后以 `--server` 模式启动，并在前端配置 HTTPS 反向代理（参见 [`deploy/guitarocr.service`](../deploy/guitarocr.service) 与 [`deploy/Caddyfile`](../deploy/Caddyfile)）：

```bash
# 以下密码读取语法适用于 Bash。
read -rsp '管理员密码: ' GUITAROCR_ADMIN_PASSWORD
export GUITAROCR_ADMIN_PASSWORD
./native/guitarocr-backend --projects /var/lib/guitarocr/projects --create-admin admin
unset GUITAROCR_ADMIN_PASSWORD

./native/guitarocr-backend --server --bind 127.0.0.1 --port 8080 \
  --public-url https://ocr.example.com --assets ui/workbench --resources native \
  --projects /var/lib/guitarocr/projects --models /var/lib/guitarocr/models --slots 4
```

## 4. 数据生产与模型训练

### 4.1 训练环境准备

研究与训练使用 **Python 3.11**、`uv` 和 **Rust**；Python 通过 PyO3 调用与产品共用的 `scorelib`：

```bash
git lfs pull --include='weights/**'
uv sync --locked --package guitarocr-research --python 3.11 --extra glm-ocr --extra training --extra dev
uv run --no-sync python -m playwright install chromium
source .venv/bin/activate
```

VLM 微调另需安装 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 与对应 CUDA 版本的 FlashAttention 2。

### 4.2 小节观察模型训练与导出

1. **生成多谱型观察数据**：

```bash
for mode in tab notation tab-chords both standard drums; do
  python -m research.data.observations --mode "$mode" --output "database/observations-$mode"
done
python -m research.data.observations \
  --merge database/observations-tab database/observations-notation database/observations-tab-chords \
    database/observations-both database/observations-standard database/observations-drums \
  --output database/observations
```

2. **微调 GLM-OCR 小节模型**（配置见 [`research/configs/measures/train.yaml`](../research/configs/measures/train.yaml)）：

```bash
python -m research.training.measures --tokenize-only
python -m research.training.measures --check
CUDA_VISIBLE_DEVICES=0 python -m research.training.measures
```

3. **合并 LoRA、评测与导出 GGUF**：

```bash
python -m research.export.export_glm --base weights/score_ocr/merged \
  --adapter output/observation_ocr --output output/observation_ocr/merged
python -m research.evaluation.measures --manifest database/observations/manifest_test.jsonl \
  --model output/observation_ocr/merged --output output/observation-test
python -m research.export.export_gguf --llama-cpp /path/to/llama.cpp \
  --model output/observation_ocr/merged --output output/gguf/score_ocr
```

### 4.3 其他训练与数据入口速查

| 任务 | Python 模块入口 |
| --- | --- |
| GP 源谱渲染与标签生成 | `research.data.run`、`research.data.build_measure_data`（GP8 原生桥见 [`gpbridge/native/`](../gpbridge/native)） |
| 信息读取数据生成 | `research.data.build_info_data` |
| 版面检测与候选质量 | `research.training.layout`、`research.training.layout_quality` |
| 拍号、调号、谱号与谱表 | `research.training.state`、`calibrate_state`、`clef_spatial`、`staff_evidence` |
| 谱行、名称与音轨关系 | `research.training.structure_relation`、`part_continuation`、`staff_label`、`instrument_text` |
| 和弦名与指法图 | `research.training.chord_text_roles`、`chord_evidence`、`chord_calibration` |
| 辅助模型 ONNX 导出 | `research.export.export_auxiliary` |

裁图评测使用单独的 [vLLM 环境](../research/configs/runtime-vllm.txt)，默认路径为 `tools/vllm-venv`，可通过 `GUITAROCR_VLLM_PYTHON` 指定。整谱评测入口是 `research.evaluation.pipeline`，它调用实际后端完成识别。

GP 源谱数据生产使用 Guitar Pro 8.1.2.37；Linux 通过 Wine 运行，工作目录由 `research.data.run` 的参数指定。模型、数据集、候选权重和工具分别放在 `weights/`、`database/`、`output/`、`tools/`。各入口的 `--help` 列出所需参数。
