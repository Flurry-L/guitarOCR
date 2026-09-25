# 构建 GP8 原生导出 DLL

源码已从同一工作区的 `GPOMR/datagen/native-source/` 收入 `datagen/native-source/`，包含 `dllmain.cpp`、`score_dump.cpp/.h`、完整的导出符号 `.def`、接口声明 `gp_stubs.h` 和 `build.ps1`。不再引用相邻仓库。原项目的副本保留，避免破坏它的构建。

两个预编译 DLL 与来源目录中的文件 SHA-256 一致：

| 文件 | SHA-256 |
| --- | --- |
| `gpomr_native_export.dll` | `56f301cf9b501507c01e3e531b0b98c4fada0a1338baa75a7f48754e28f3b18b` |
| `gpomr_amprof_preload.dll` | `45cc6d08f63d8a7c6cc70043d09d39f68da967c6b363e6d4365d78890daf5f50` |

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

这些接口与 GP8 的内部 ABI 有关。更换 GP8 或 Qt 主版本后，需要重新验证接口声明、导出符号和运行结果。当前迁入的构建脚本来自已有导出链；本次 Linux 环境未执行 Windows MSVC 编译，不能据此宣称重新构建后的二进制与预编译文件一致。

构建后按 [数据导出环境](setup.md#数据导出环境) 的命令导出一份 GP 源谱，检查生成的 PDF、`layout.json` 和 `official-score.json`。Guitar Pro 安装程序及其运行库不属于本项目的开源内容。


构建成功后会在 DLL 旁生成 `build-manifest.json`，记录源文件 SHA-256、编译器版本、Qt 路径和 DLL 哈希。`runtime_validation` 初始为 `not_run`，只有完成目标 GP8 导出验收后才能另行记录通过。GitHub Actions 的 Build native exporter 可手动执行 Windows 编译并保存构件；这不替代实际 GP8 运行验证。
