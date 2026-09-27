param(
    [string]$QtRoot = $env:GPOMR_QT_ROOT,
    [string]$VisualStudioRoot = $env:GPOMR_VS_ROOT,
    [string]$OutputDll,
    [string]$OutputPreloadDll
)

$ErrorActionPreference = "Stop"
$originalLocation = Get-Location
$build = $null
try {
$native = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$build = Join-Path (
    [System.IO.Path]::GetTempPath()
) ("gpomr-native-build-" + [guid]::NewGuid().ToString("N"))
$imports = Join-Path $build "import_libs"
$preloadBuild = Join-Path $build "preload"
$output = Join-Path $native "native-bin"
if (-not $OutputDll) {
    $OutputDll = Join-Path $output "gpomr_native_export.dll"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputDll)) {
    $OutputDll = [System.IO.Path]::GetFullPath(
        (Join-Path $originalLocation.Path $OutputDll)
    )
}

if (-not $QtRoot -or -not (Test-Path (Join-Path $QtRoot "include/QtCore"))) {
    throw "Qt headers are required; pass -QtRoot or set GPOMR_QT_ROOT."
}
if (-not $OutputPreloadDll) {
    $OutputPreloadDll = Join-Path $output "gpomr_amprof_preload.dll"
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPreloadDll)) {
    $OutputPreloadDll = [System.IO.Path]::GetFullPath(
        (Join-Path $originalLocation.Path $OutputPreloadDll)
    )
}
$qt = (Resolve-Path -LiteralPath $QtRoot).Path

if (-not $VisualStudioRoot) {
    throw "Visual Studio is required; pass -VisualStudioRoot or set GPOMR_VS_ROOT."
}
$VisualStudioRoot = (Resolve-Path -LiteralPath $VisualStudioRoot).Path
$vsDevCmd = Join-Path $VisualStudioRoot "Common7/Tools/VsDevCmd.bat"
if (-not (Test-Path $vsDevCmd)) {
    throw "Visual Studio developer command file is missing: $vsDevCmd"
}
$devEnvironment = & $env:COMSPEC /d /s /c `
    "`"$vsDevCmd`" -no_logo -arch=x64 >nul && set"
if ($LASTEXITCODE -ne 0) {
    throw "Visual Studio x64 developer environment initialization failed"
}
$developerPath = $null
foreach ($line in $devEnvironment) {
    $separator = $line.IndexOf('=')
    if ($separator -le 0) { continue }
    $name = $line.Substring(0, $separator)
    $value = $line.Substring($separator + 1)
    if ($name -ieq "Path") {
        $developerPath = $value
        continue
    }
    Set-Item -LiteralPath "Env:$name" -Value $value
}
if (-not $developerPath) {
    throw "Visual Studio developer environment did not provide PATH"
}
$env:Path = $developerPath

New-Item -ItemType Directory -Force -Path `
    $build,$imports,$preloadBuild,$output,(Split-Path -Parent $OutputDll),(
        Split-Path -Parent $OutputPreloadDll
    ) | Out-Null
foreach ($library in @("GPCore", "AMUtils", "AMPainting")) {
    & lib.exe /nologo /machine:x64 `
        "/def:$(Join-Path $PSScriptRoot "toolchain/$library.def")" `
        "/out:$(Join-Path $imports "$library.lib")"
    if ($LASTEXITCODE -ne 0) { throw "Failed to generate $library.lib" }
}
& lib.exe /nologo /machine:x64 `
    "/def:$(Join-Path $PSScriptRoot 'toolchain/AMProfProxy.def')" `
    "/out:$(Join-Path $imports 'AMProfOriginal.lib')"
if ($LASTEXITCODE -ne 0) { throw "Failed to generate AMProfOriginal.lib" }

& cl.exe /nologo /O2 /EHsc /std:c++17 /utf-8 /MD /LD `
    "/I$qt/include" `
    "/I$qt/include/QtCore" `
    "/I$qt/include/QtGui" `
    "/I$qt/include/QtWidgets" `
    "/I$(Join-Path $PSScriptRoot 'toolchain')" `
    (Join-Path $PSScriptRoot "dllmain.cpp") `
    (Join-Path $PSScriptRoot "score_dump.cpp") `
    "/Fe:$OutputDll" `
    "/Fo:$build/" `
    /link "/LIBPATH:$qt/lib" "/LIBPATH:$imports" /DLL `
    "/IMPLIB:$(Join-Path $build 'gpomr_native_export.lib')" `
    Qt5Core.lib Qt5Gui.lib Qt5Widgets.lib `
    GPCore.lib AMUtils.lib AMPainting.lib
if ($LASTEXITCODE -ne 0) { throw "Guitar Pro native hook build failed" }

& cl.exe /nologo /O2 /EHsc /std:c++17 /utf-8 /MD /LD `
    "/I$qt/include" `
    "/I$qt/include/QtCore" `
    "/I$qt/include/QtGui" `
    "/I$qt/include/QtWidgets" `
    "/I$(Join-Path $PSScriptRoot 'toolchain')" `
    (Join-Path $PSScriptRoot "dllmain.cpp") `
    (Join-Path $PSScriptRoot "score_dump.cpp") `
    "/Fe:$OutputPreloadDll" `
    "/Fo:$preloadBuild/" `
    /link "/LIBPATH:$qt/lib" "/LIBPATH:$imports" /DLL `
    "/DEF:$(Join-Path $PSScriptRoot 'toolchain/AMProfProxy.def')" `
    "/IMPLIB:$(Join-Path $build 'gpomr_amprof_preload.lib')" `
    Qt5Core.lib Qt5Gui.lib Qt5Widgets.lib `
    GPCore.lib AMUtils.lib AMPainting.lib AMProfOriginal.lib
if ($LASTEXITCODE -ne 0) { throw "Guitar Pro native preload build failed" }

Write-Output "Built $OutputDll"
Write-Output "Built $OutputPreloadDll"
$sourceHashes = @{}
Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($PSScriptRoot.Length + 1).Replace('\', '/')
    $sourceHashes[$relative] = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLower()
}
$buildRecord = @{
    built_at_utc = [DateTime]::UtcNow.ToString('o')
    compiler = (Get-Command cl.exe).Source
    compiler_version = (Get-Item (Get-Command cl.exe).Source).VersionInfo.FileVersion
    qt_root = $qt
    source_sha256 = $sourceHashes
    outputs = @(
        @{ name = [IO.Path]::GetFileName($OutputDll); sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputDll).Hash.ToLower() },
        @{ name = [IO.Path]::GetFileName($OutputPreloadDll); sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $OutputPreloadDll).Hash.ToLower() }
    )
    runtime_validation = 'not_run'
}
$buildRecord | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path (Split-Path -Parent $OutputDll) 'build-manifest.json')
} finally {
    Set-Location -LiteralPath $originalLocation.Path
    if ($build -and (Test-Path -LiteralPath $build)) {
        Remove-Item -LiteralPath $build -Recurse -Force
    }
}
