# 发布审核

最初审核于 2026-09-25，基线提交为 `84645ea`。2026-09-26 更新模型训练和原生导出验收状态。软件验证见[验证记录](validation.md)，授权材料见[第三方说明](../THIRD_PARTY_NOTICES.md)。

## 待处理事项

| 事项 | 当前证据与后续工作 |
| --- | --- |
| 项目许可证 | 根目录尚无 LICENSE，包元数据未声明许可。项目所有者需确定源码、示例和模型的授权，并处理 PyMuPDF 的 AGPL／商业许可及其他依赖的要求。 |
| 模型与曲谱授权 | 权重许可和完整训练素材授权清单仍待补齐。历史 Git 中也有旧权重，需要一并确认再分发依据。 |
| Transformers 公告 | 2026-09-25 扫描发现 5.8.0 命中 [GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)，别名 CVE-2026-9856 / PYSEC-2026-3929，公告修复版本为 5.10.0。固定的 LLaMA-Factory 提交要求 Transformers ≤5.8.0，需协调升级并验证训练、推理、保存和恢复。 |
| 原生工具授权 | GP8 内部接口声明、自定义代码和 Qt 链接方式的授权仍需确认。Linux 交叉编译及 GP8＋Wine 已验收，Windows MSVC 实机运行尚待验证。 |
| 平台与依赖记录 | 完整 Windows 安装需实机验收，Paddle 独立环境还需完整的依赖版本与许可清单。 |
| 硬件与识别范围 | 消费级显卡耗时、最低内存和显存未测定；真实扫描件的代表性整谱评测仍不足。当前独立裁图指标与少量整谱验收见[训练报告](training-v3-report.md)。 |

上述 Transformers 问题在恶意模型的 `chat_template` 被 tokenizer／processor 的 `save_pretrained()` 保存时可能造成越界写入。项目推理代码未调用该方法，默认基座固定 revision 并校验哈希；外部模型和训练流程仍需核查。

## 已完成的修复

- Web 请求先限制 Host，再核对写操作的 Origin，阻止 DNS 重绑定访问。工作台仍按本地单用户使用设计。
- 任务响应在锁内复制状态，避免后台导入完成时产生“任务完成但缺少页面”的不一致快照。
- 项目 ZIP 校验清单类型、路径、符号链接、大小和 SHA-256，拒绝非规范路径及大小写别名，失败后清理残留目录。
- Pillow 最低版本升至 12.3；图片上传同时检查扩展名和实际解码格式。
- 发布包收录 LICENSE、NOTICE 和第三方说明，排除环境文件、私钥、本地环境及 Git 目录。
- JSON 明确使用 UTF-8，统一重复的小节线处理、模型池及项目路径转换逻辑。

原生导出器此后已由仓库源码交叉编译，在 Guitar Pro 8.1.2.37＋Wine 中完成三种排版和中文谱头验收。构建工具、DLL 哈希及复现方法见[原生构建](native-build.md)。

## 检查与复现

核心测试、浏览器测试和打包命令见[参与开发](../CONTRIBUTING.md)。依赖公告扫描命令为：

```bash
uv export --locked --no-dev --extra webui --extra glm-ocr --no-emit-project --no-hashes --format requirements-txt --output-file output/audit-requirements.txt
uv tool run pip-audit==2.10.1 --no-deps --disable-pip -r output/audit-requirements.txt
```

2026-09-25 的扫描覆盖核心、webui、glm-ocr 锁定依赖，更新 Pillow 后仍有上述一条 Transformers 公告，退出码为 1。Paddle、开发依赖、Windows 条件依赖和原生组件未被完整覆盖，发布前应分别扫描实际安装环境。

当日 45 项核心测试、2 项 Chromium 场景、wheel 和 ZIP 检查通过，80 组合成 TAB 几何配置与整理前一致。后续训练和界面变更的验证记录单独更新，避免混用不同版本的测试结果。
