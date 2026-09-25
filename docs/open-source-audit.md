# 开源审核与代码整理记录

审核日期：2026-09-25。对象是工作目录中的重构版本，审核基线提交为 `84645ea`。审核时已有的大量未提交迁移得到保留；后续按处理流程与工作台、模型、安装与 CI、文档四组整理为本地提交。没有改写既有历史或公开发布。

结论：按处理阶段组织代码的方向合理，核心流程可运行；完成下列授权与发布待办后再作为有明确许可证的开源项目发布。测试通过不等于模型转录准确，也不等于许可证或供应链审核已完成。

代码简化参考 Anthropic 官方 Claude Code 插件 [code-simplifier](https://github.com/anthropics/claude-plugins-official/tree/main/plugins/code-simplifier)，其原始规则位于 [agents/code-simplifier.md](https://github.com/anthropics/claude-plugins-official/blob/ceb9b72b4c4c20ad39efce780edd0aabe80ebce3/plugins/code-simplifier/agents/code-simplifier.md)。初次本机搜索没有找到，随后从官方仓库读取规则并据此复核本轮改动：保持行为、消除重复与无效嵌套、避免嵌套三元表达式、优先可读性。本轮应用了公开规则，没有安装或运行 Claude Code 插件本身；同时使用人工审阅、AST 分析、Ruff 和行为回归。

## 公开发布前仍需处理

| 优先级 | 发现与证据 | 建议处理 |
| --- | --- | --- |
| 高 | 根目录没有 `LICENSE`，`pyproject.toml` 没有项目许可声明。PyMuPDF 的安装元数据明确为 AGPL-3.0 / 商业双许可；PyGuitarPro 为 LGPL-3.0-only。 | 由项目所有者确认源码、示例与模型的许可，补完整文本和包元数据；结合现有依赖确认兼容性。参见 [第三方说明](../THIRD_PARTY_NOTICES.md)。本轮未替所有者授予许可。 |
| 高 | 三个发布模型的模型卡没有权重许可证，训练曲谱授权清单不完整；小节模型的历史训练快照也未完整保留。 | 补模型授权、数据来源和可再分发依据；无法确认授权的资产应从公开发行中移出或用可授权数据重训。单纯不发布原始曲谱不能解决全部授权问题。 |
| 高 | `transformers==5.8.0` 命中 [GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)，别名 `CVE-2026-9856` / `PYSEC-2026-3929`；公告修复版本为 5.10.0。 | 触发条件是对恶意模型的 `chat_template` 配置调用 tokenizer / processor 的 `save_pretrained()`，可能越界写文件。当前推理源码没有该调用，默认模型固定 revision 并验证哈希，降低了这一路径的暴露；外部模型与训练流程仍须核对。当前文档固定的 LLaMA-Factory `97b32d3…` 明确要求 Transformers ≤5.8.0，因此需要协调升级训练框架与 Transformers，重新验证训练、推理、保存和恢复后再解除该待办。 |
| 中 | 两个原生 DLL 虽有对应源码和来源哈希，但本轮未执行 Windows MSVC 编译、目标 GP8 运行验收，也没有证明当前 DLL 来自当前源码。`gp_stubs.h` 和 `.def` 涉及专有软件内部接口。 | 确认代码及声明来源，按 `docs/native-build.md` 生成构建记录并验收；如暂时无法验证，可先提供经授权的源码并明确二进制状态。 |
| 中 | Windows 安装和原生构建只有 CI 配置，本轮没有 Windows 实机结果；完整 Paddle 环境的传递依赖未锁定。 | 在最终提交上执行 Windows 安装及 Native workflow；保存 Paddle 环境的完整版本与许可证清单。Linux 验证不能代替 Windows 验证。 |
| 中 | 本轮真实样例中，PDF 小节完全匹配 0/4，图片 3/4，两者均能通过结构校验并写出 GP5。 | 保留人工校对定位；不要宣传无损转录或由 GP5 导出成功推断准确率。补独立测试集及消费级硬件指标。 |
| 低 | 当前模块使用 `shared`、`layout`、`pipeline` 等顶层包名；适合独立项目环境，但作为通用库安装时可能与其他项目同名包冲突。 | 后续稳定公共 API 时可统一到 `guitarocr.*` 命名空间；本轮没有在既有大规模迁移上再做目录搬迁。 |

审核基线只有一个提交，其中包含约 30 MiB 的旧 `.pt` 权重和旧架构；后续提交中的删除不会移除这段历史。公开历史前应确认这些旧权重也可再分发。所有新模块、文档、LFS 权重及删除记录均应纳入提交；公开发布前仍需从最终提交的新检出运行 CI。

## 本轮已修复

| 文件 | 修改及原因 |
| --- | --- |
| `webapp/app.py` | 原先接受任意 Host，仅比较 Origin 与 Host；同域重绑定请求可读取 API。现在首先限制明确的主机名，再验证写操作的 Origin 协议与主机。默认支持 localhost、IPv4 / IPv6 回环地址；额外主机需显式配置。 |
| `webapp/projects.py` | JSON 根值为数组或空值的畸形项目包原先抛出未处理异常。现在检查清单类型，拒绝非规范路径、大小写别名和保留文件名，导入失败清理残留目录。已有越界路径、符号链接、体积和 SHA-256 检查保留。 |
| `pyproject.toml`、`uv.lock`、安装脚本 | Pillow 11.3.0 命中 18 个去重后的公告，包括 PSD 解码内存破坏。最低版本提高为 12.3，更新锁文件及独立 Paddle 安装命令。 |
| `layout/pages.py`、`webapp/app.py` | 图片扩展名和可用解码器集中定义，只允许 PNG / JPEG / BMP / TIFF。扩展名检查之外增加实际格式限制，防止改名文件进入未支持的解码器。 |
| `scripts/package_release.py`、`.gitignore` | 发布包自动收录现有的 `LICENSE*`、`NOTICE*` 及第三方说明，排除 `.env`、私钥、嵌套环境与 Git 目录。此前仅添加许可证文件不会进入发布 ZIP，Git 忽略规则也不会约束自行遍历目录的打包脚本。 |
| `webapp/app.py`、`layout/run.py` | 明确以 UTF-8 读取包含中文标题或路径的 JSON，避免 Windows 系统默认编码造成错误。 |

WebUI 仍是单用户本地应用，没有身份认证。Host 限制不等于访问控制，不应直接作为公网服务；显式启用网络访问时需要独立的认证与部署方案。

## 冗余与结构整理

- `layout/tab_geometry.py`：两条定位路径原先分别实现了相同的小节线尾部过滤、重复线合并、最小宽度和密度检查。统一到 `_detect_tab_boundaries()`，文件从 340 行降至 267 行，检测阈值保持原值。
- `shared/glm_backend.py`：去掉方法内临时定义的类及其多层闭包，将适配器加载、选择和生成改为普通方法与轻量句柄。仍然懒加载一个基座，并在同一锁内完成适配器切换与生成；支持先用基座再加载 LoRA，以及暂时禁用 LoRA。
- `webapp/projects.py`：JSON / JSONL 的路径转换统一为 `transform_file()`，导入导出使用同一逻辑，减少双份实现漂移。
- 删除导出器里无效的 `import` / `del`、未使用的循环变量、重复状态赋值；合并信息识别中执行相同操作的分支，将带副作用的条件表达式改为明确的 `if`。
- 按官方 `code-simplifier` 规则复核后，将项目列表状态的嵌套三元表达式改为 `if/elif/else`，发布包文件筛选改为逐项跳过的明确条件，保留原筛选和状态优先级。
- 静态导入检查没有发现当前阶段之间的循环依赖。`layout` / `document_info` / `measure_ocr` / `gp5_export` 依赖 `shared`，`pipeline` 和 `webapp` 组织各阶段，责任划分合理。仅有数行的训练入口提供明确 CLI，不应为了减少文件数量删掉。
- `datagen/native-source/dllmain.cpp` 约 3574 行，`datagen/native/score_official_score.py` 约 1450 行，仍是维护成本较高的部分。后续可按协议、绘制钩子、布局导出及模型校验职责拆分；需先补齐真实 GP8 回归材料，本轮未修改这些平台敏感算法。

Ruff 的额外简化规则输出也经过人工筛选；循环中立即调用的局部函数、延迟加载重依赖和 API 工厂闭包不自动认定为错误，没有逐条机械改写。

## 实际验证

| 检查 | 本轮结果 |
| --- | --- |
| 标准 Ruff 检查 | 通过 |
| 原有核心 / API / 数据 / 安装器测试 | 基线 37 项通过；新增回归后共 44 项通过 |
| 独立轻量环境 | 按更新后的 uv.lock 新建，仅安装 webui / dev，确认没有 Torch，44 项全部通过 |
| Chromium | 两项真实浏览器场景通过，覆盖上传、翻页、编辑、冲突处理和 GP5 下载 |
| TAB 算法对比 | 80 组合成配置及 `examples/demo.png` 与整理前结果一致；另有短音符杆与跨谱表连接线的回归测试 |
| 真实模型 | Linux / Python 3.11 / H100，主环境与 Paddle 环境使用 Pillow 12.3；PDF、图片分别成功定位 4 个小节并导出 GP5，耗时约 16.0 / 17.3 秒；完全匹配分别为 0/4、3/4，与既有示例记录一致 |
| wheel / ZIP | wheel 构建与资源检查通过；实际权重经哈希检查后成功打包。打包回归额外验证许可文件、私密文件排除、CRLF 和包内校验清单 |
| 依赖公告 | pip-audit 2.10.1 检查核心、webui 与 glm-ocr 锁定依赖；更新后剩余上述 1 条 Transformers 公告，扫描退出码为 1，不能记为全部通过 |
| 凭据与本机路径 | 当前可发布文本和可见历史中，未匹配常见私钥、HF / GitHub / API / AWS 凭据格式或所查的个人目录模式。属于有限模式扫描，不保证不存在其他敏感信息；未把模型张量内容作为凭据进行解释 |

新增回归覆盖 Host 与 Origin、畸形 ZIP 与路径别名、伪装图片、基座与 LoRA 切换、小节线行为和发布包内容。没有运行完整训练、Windows / GP8 验收或全部第三方二进制的安全分析。

复查命令：

```bash
uv run --no-sync ruff check datagen layout document_info measure_ocr gp5_export pipeline shared webapp scripts examples tests
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m unittest discover -s tests -p 'browser_*.py' -v
uv export --locked --no-dev --extra webui --extra glm-ocr --no-emit-project --no-hashes --format requirements-txt --output-file output/audit-requirements.txt
uv tool run pip-audit==2.10.1 --no-deps --disable-pip -r output/audit-requirements.txt
```

扫描只查询依赖名称与版本；Paddle 独立环境、开发依赖、Windows 条件依赖和原生组件没有被这次主环境公告扫描完整覆盖。正式发布时应分别扫描实际安装环境。
