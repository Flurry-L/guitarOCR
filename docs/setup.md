# 安装与启动

从[发布页](https://github.com/Flurry-L/guitarOCR/releases/tag/v0.1.0)下载 Apple 芯片 Mac 或 Windows 安装包。打开应用后点击「打开乐谱」。

![应用启动页](assets/desktop-0.1.webp)

可以先打开内置示例练习编辑，或恢复项目 ZIP。导入 PDF／图片并开始识别时，应用会提示下载约 1.8 GB 模型。建议至少 8 GB 内存、6 GB 可用磁盘；大谱面需要更多内存。

## 设备选择

| 设备 | 本机识别 |
| --- | --- |
| Windows x64 | 自动选择兼容的 NVIDIA GPU，否则使用 CPU |
| macOS Apple Silicon | Apple GPU |
| Linux、Intel Mac | [从源码构建](native-packaging.md) |
| 手机 | 浏览器连接[自托管服务](server.md) |

应用自动选择可用设备，可在「设置」查看。Windows 只下载当前设备需要的加速组件，CPU 与 GPU 共用模型；下载失败可使用 CPU。NVIDIA 加速要求 CUDA 12.8 兼容驱动、计算能力至少 7.5，并有足够显存；无需安装 CUDA Toolkit。其他 Windows 显卡使用 CPU。Windows 需要 WebView2，macOS 要求 13.4 或更新版本。

已有服务器时，在启动页展开「连接服务器」，填写服务地址。乐谱在服务器处理，本机无需下载模型；关闭网页后任务仍会继续。

## macOS 首次打开

M 系列 Mac 下载 `GuitarOCR_0.1.0_aarch64.dmg`。

当前安装包未使用 Apple Developer ID 签名和公证。若从本项目发布页下载后提示「已损坏」或无法打开，先把 `GuitarOCR.app` 拖到「应用程序」，再在终端执行：

```bash
xattr -dr com.apple.quarantine /Applications/GuitarOCR.app
```

重新打开应用即可。该命令解除这份应用的下载隔离标记。

## 保存与更新

编辑保存到运行识别的设备上：本机模式保存在这台电脑，远程模式保存在服务器。启动页的「打开数据文件夹」可查看本机项目、模型和日志。

更新应用会继续使用已有项目和模型。移动到另一台设备时，在「导出」下载项目 ZIP，再从新设备的「恢复项目备份」导入。

退出桌面应用会停止本机识别；已保存的结果保留，可从「项目」继续处理。

[操作说明](webui.md) · [自托管部署](server.md) · [从源码构建](native-packaging.md)
