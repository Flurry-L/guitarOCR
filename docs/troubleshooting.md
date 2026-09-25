# 故障排查

优先保留启动窗口里的第一条失败信息。日志自动写入 `output/logs/`；修复后重跑原来的安装或启动脚本，已下载的包和模型会复用。

| 现象 | 处理 |
| --- | --- |
| 权重是很小的文本文件 / 提示 LFS 指针 | 在 Git 检出目录执行 `git lfs install`、`git lfs pull`；若使用了 GitHub 的源码 ZIP，请按 [获取源码步骤](../README.md#获取源码) 重新克隆，ZIP 不含安装器补下载权重所需的 Git 信息 |
| Windows 下载 `adapter_config.json` 时出现 HTTP 404 | 旧安装器误用 LFS 下载地址，且 Git 自动换行可能导致配置校验失败。在仓库目录执行 `git pull --ff-only` 后重跑 `start.bat`，更新后的安装器会从正确地址修复配置，无需删除已下载环境 |
| 安装时提示 `uv.lock needs to be updated` | 在仓库目录执行 `git pull --ff-only` 后重跑启动脚本。更新后的安装器直接导出仓库的固定依赖，使用已创建的 Python 3.11；本机 uv 镜像配置不会触发锁文件重新解析 |
| Hugging Face 下载中断 | 重跑安装脚本。支持标准 `HTTPS_PROXY`；已有可信镜像时可自行设置 `HF_ENDPOINT`，安装后仍会校验 SHA-256 |
| uv 下载失败 | 检查对 astral.sh / GitHub 的访问与系统时间，再重跑。不要关闭证书验证 |
| GPU / CUDA 不可用 | 更新 NVIDIA 驱动；一键 CUDA 13 环境需要 580 或更新驱动。也可运行 `install.bat --device cpu` / `bash install.sh --device cpu` |
| 显存不足 | 关闭占用显存的程序、减小不必要的大框，或使用 `start.bat --device cpu` / `bash start.sh --device cpu` |
| Windows 缺 DLL / C++ Runtime | 重跑启动脚本完成 Microsoft C++ 运行库安装；也可执行 `winget install --id Microsoft.VCRedist.2015+.x64 --exact`，安装后按提示重启 |
| Linux 缺 libGL.so.1 或 libglib | `sudo apt-get install libgl1 libglib2.0-0` |
| 端口 7860 已占用 | 打开已有工作台，或加 `--port 7861` 启动 |
| 似乎只有一页 / 打开了旧谱 | 查看顶部文件名和总页数。选择新文件后还需点击「导入乐谱」。用页面下拉框、缩略图或下一页查看其他页 |
| 自动框不正确 | 选择图片模型重新定位，或手动移动、缩放、补画。保存后再处理后续步骤 |
| 一个小节识别错误 | 校对当前小节，或点击「重新识别此小节」。其余小节保留；跨小节连线仍需检查邻近小节 |
| 识别中断 / 关闭了窗口 | 重新启动并打开原项目，点击「继续上次识别」。停止在当前模型调用结束后生效 |
| 提示其他页面已修改 | 本页未覆盖新结果。先复制需要保留的编辑文本，再刷新载入最新项目 |
| GP5 暂时不能导出 | 检查所有待确认小节并保存。失败占位必须明确确认 |
| 换电脑后项目路径失效 | 使用「下载完整项目 ZIP」并在新电脑的导入页恢复；不要只复制 session.json |

一键环境检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1 check
```

```bash
bash scripts/bootstrap.sh check
```

手动环境检查：`uv run --no-sync guitarocr-check --hashes --device cuda --layout-python tools/paddlex-venv/bin/python`。不含模型的开发环境可使用 `--core`。检查失败时会给出具体缺失项与修复命令。

自动恢复保留最后已保存的小节；浏览器里尚未保存的编辑不会写入项目。项目 ZIP 包含原始上传内容，分享前请确认内容适合公开。
