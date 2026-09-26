param(
    [Parameter(Position=0)][ValidateSet('start', 'install', 'check')][string]$Command = 'start',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$LauncherArgs
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot '..')
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
New-Item -ItemType Directory -Force -Path 'tools/uv','output/logs' | Out-Null
$logFile = Join-Path (Get-Location) ('output/logs/launcher-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
function Get-FileWithRetry([string]$Url, [string]$Destination) {
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            Invoke-WebRequest $Url -UseBasicParsing -TimeoutSec 120 -OutFile $Destination
            return
        } catch {
            if ($attempt -eq 3) { throw }
            Write-Host 'Download interrupted. Retrying...'
            Start-Sleep -Seconds 1
        }
    }
}
try {
    Start-Transcript -Path $logFile | Out-Null
    Write-Host 'GuitarOCR - install and launch'
    Write-Host "Log: $logFile"
    if (-not (Test-Path "$env:WINDIR/System32/msvcp140.dll") -or -not (Test-Path "$env:WINDIR/System32/vcruntime140_1.dll")) {
        Write-Host 'Microsoft C++ Runtime is required. Windows may ask for permission to install it.'
        $runtimeInstaller = Join-Path (Get-Location) 'tools/uv/vc_redist.x64.exe'
        Get-FileWithRetry 'https://aka.ms/vs/17/release/vc_redist.x64.exe' $runtimeInstaller
        $runtimeProcess = Start-Process -FilePath $runtimeInstaller -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -Wait -PassThru
        if ($runtimeProcess.ExitCode -eq 3010) { throw 'Microsoft C++ Runtime requires a Windows restart. Restart and run start.bat again.' }
        if ($runtimeProcess.ExitCode -notin @(0,1638)) { throw "Microsoft C++ Runtime installation failed: $($runtimeProcess.ExitCode)" }
    }
    $uvBinary = Join-Path (Get-Location) 'tools/uv/uv.exe'
    if (-not (Test-Path -LiteralPath $uvBinary)) {
        Get-FileWithRetry 'https://astral.sh/uv/0.12.17/install.ps1' 'tools/uv/install.ps1'
        $env:UV_UNMANAGED_INSTALL = Join-Path (Get-Location) 'tools/uv'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'tools/uv/install.ps1'
        if ($LASTEXITCODE -ne 0) { throw 'uv installation failed. Check network access and retry.' }
    }
    $env:GUITAROCR_UV = $uvBinary
    if (-not $env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT = '30' }
    if (-not $env:UV_HTTP_RETRIES) { $env:UV_HTTP_RETRIES = '2' }
    # A failed native probe is expected when Python is not installed yet.
    $ErrorActionPreference = 'Continue'
    $bootstrapPython = & $uvBinary --config-file scripts/bootstrap-uv.toml python find --no-python-downloads 3.11 2>$null
    $pythonFound = $LASTEXITCODE -eq 0
    $ErrorActionPreference = 'Stop'
    if (-not $pythonFound) {
        $pythonMirror = $env:UV_PYTHON_INSTALL_MIRROR
        if (-not $pythonMirror) { $pythonMirror = 'https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone' }
        Write-Host 'Downloading Python from the configured mirror (Nanjing University by default).'
        & $uvBinary --config-file scripts/bootstrap-uv.toml python install 3.11 --no-bin --no-registry --mirror $pythonMirror
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Python mirror failed. Retrying the official source.'
            & $uvBinary --config-file scripts/bootstrap-uv.toml python install 3.11 --no-bin --no-registry --mirror 'https://github.com/astral-sh/python-build-standalone/releases/download'
            if ($LASTEXITCODE -ne 0) { throw 'Python download failed. Check the network and run start.bat again.' }
        }
        $bootstrapPython = & $uvBinary --config-file scripts/bootstrap-uv.toml python find --no-python-downloads 3.11
        if ($LASTEXITCODE -ne 0) { throw 'Python installation could not be located.' }
    }
    & $uvBinary run --config-file scripts/bootstrap-uv.toml --no-project --no-python-downloads --python $bootstrapPython scripts/launcher.py $Command @LauncherArgs
    $result = $LASTEXITCODE
    if ($result -ne 0) { throw "GuitarOCR exited with code $result. See the log above and docs/troubleshooting.md." }
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
