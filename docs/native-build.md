# 构建 GP8 原生导出 DLL

`datagen/native-source/` 包含 C++ 源码、导出符号、接口声明和构建脚本。代码源自 GPOMR 的原生导出器，当前构建所需文件均在本仓库中。

当前 DLL 已增加 `display_mode=tab|notation|both`，由本目录源码在 Linux 上使用 clang-cl 18、xwin 的 MSVC/Windows SDK 和 Qt 5.15.2 MSVC SDK 交叉编译。已在 GP8 8.1.2.37 + Wine 中实测三种排版及小节框，SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `gpomr_native_export.dll` | `b2d53af39af3336ffa1fd755b105d378fbcf63d250fdebc1551cf6ebb0390927` |
| `gpomr_amprof_preload.dll` | `c4aeb8de615bfd70cfbbc266d6cd356761503c5f882a4d9dc2131a554ab60b33` |

前者由原生会话加载，后者是 GP8 预加载代理。它们只用于数据生产，PDF / 图片识别及 GP5 导出不依赖 Guitar Pro、Wine 或这些 DLL。

## Windows x64 构建

在 Windows PowerShell 中安装 Python、Visual Studio C++ 构建工具和 Qt 5 开发包：

```powershell
winget install --id Python.Python.3.11 --exact
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
py -3.11 -m pip install aqtinstall
py -3.11 -m aqt install-qt windows desktop 5.15.2 win64_msvc2019_64 -O C:\Qt
```

如果刚安装 Python，请重新打开 PowerShell。在仓库根目录执行：

```powershell
$vsRoot = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
powershell -ExecutionPolicy Bypass -File datagen/native-source/build.ps1 -QtRoot C:\Qt\5.15.2\msvc2019_64 -VisualStudioRoot $vsRoot
```

默认生成：

```text
datagen/native-bin/gpomr_native_export.dll
datagen/native-bin/gpomr_amprof_preload.dll
```

可通过 `-OutputDll`、`-OutputPreloadDll` 指定其他输出位置。脚本在临时目录生成导入库和中间文件，使用 MSVC x64 编译，然后自动清理。Qt 头文件和链接库使用传入的 `QtRoot`；GP8 导入库由仓库中的 `.def` 生成。

来源环境使用 Guitar Pro **8.1.2.37**，其 Qt5Core.dll 文件版本为 **5.15.3.0**。上面的 aqt 命令安装可公开获取的 Qt 5.15.2 开发包；Qt 5 的补丁版本通常保持二进制兼容，仍应在目标 GP8 运行时验证。

这些接口与 GP8 的内部 ABI 有关。更换 GP8 或 Qt 主版本后，需要重新验证接口声明、导出符号和运行结果。当前提供 Windows MSVC 与 Linux clang-cl 两条构建路径；不同编译器的二进制不保证逐字节相同。

构建后按 [数据导出环境](setup.md#数据导出环境) 的命令导出一份 GP 源谱，检查生成的 PDF、`layout.json` 和 `official-score.json`。Guitar Pro 安装程序及其运行库不属于本项目的开源内容。

构建成功后会在 DLL 旁生成 `build-manifest.json`，记录源文件 SHA-256、编译器版本、Qt 路径和 DLL 哈希。`runtime_validation` 初始为 `not_run`，只有完成目标 GP8 导出验收后才能另行记录通过。

## Linux 交叉编译

准备 clang-cl、lld-link、llvm-lib、[xwin](https://github.com/Jake-Shadle/xwin) 的 x86_64 SDK，以及 Qt 5.15.2 的 Windows MSVC 64 位头文件和导入库。无需改动 GP8 运行库。

```bash
xwin --accept-license --arch x86_64 splat --output tools/native-build/msvc
uvx --from aqtinstall aqt install-qt windows desktop 5.15.2 win64_msvc2019_64 --archives qtbase -O tools/native-build/qt
uv run --no-sync python datagen/native-source/build_linux.py \
  --sdk tools/native-build/msvc --qt tools/native-build/qt/5.15.2/msvc2019_64 \
  --clang clang-cl-18 --linker tools/native-build/llvm/usr/lib/llvm-18/bin/lld-link --lib /usr/bin/llvm-lib-18 \
  --output datagen/native-bin
```

三种模式都通过 GP8 的 `TrackViewGroup` 设置显示方式，随后从真实 `BarView` 取得坐标；混合谱同一小节的两个谱表合并成一个框。`tab_only` 字段保留用于兼容，旧布局缺少 `display_mode` 时按 TAB 解释。音符品位的 glyph 标注仍仅在 TAB 模式导出，另外两种模式提供 PDF、小节／速度框和官方 score 标签。

`--linker` 应指向本机实际的 `lld-link`，且保留该文件名以选择 Windows 链接模式。上例是当前环境安装路径。

## 中文元数据与字体

旧 GP3/4/5 字符串没有可靠的编码声明。生成多语言谱头时，可在源文件旁放置 UTF-8 JSON，例如 `source.gp5.metadata.json`：

```json
{"title":"海风小品", "subtitle":"进阶课程", "artist":"青竹音乐教室"}
```

导出器在 Guitar Pro 加载乐谱后调用原生 `Score::setProperty` 设置字段，由 Guitar Pro 自己排版。支持字符串字段 `title`、`subtitle`、`artist`、`album`、`words`、`music`、`copyright`、`tabber`、`instructions`、`notice`；未知字段或非字符串会报错。数据准备和导出阶段都会复制这个附属文件。复用已有导出前会比较 GP 源文件、附属文件内容以及实际谱面类型；发生变化时要求使用新的输出目录，防止复用旧标签或旧页面。新导出也会核对实际类型与请求类型是否一致。

Wine 导出环境还需可显示中文的字体。当前使用 `tools/native-build/fonts/NotoSerifCJKsc-Regular.otf`（同目录保留 OFL 许可证），并设置 `FONTCONFIG_FILE` 为同目录 `fonts.conf` 的绝对路径。字体和编码分别验收；仅 PDF 文字标签正确不能证明图像没有方框。当前 DLL 已导出并审核 180 个中英文谱头变体 × 三种排版，540 份原生元数据全部匹配，记录在 `database/headers/native_header_audit.json`。
