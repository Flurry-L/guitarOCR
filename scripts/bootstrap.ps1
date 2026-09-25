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
try {
    Start-Transcript -Path $logFile | Out-Null
    Write-Host 'GuitarOCR - install and launch'
    Write-Host "Log: $logFile"
    if (-not (Test-Path "$env:WINDIR/System32/msvcp140.dll") -or -not (Test-Path "$env:WINDIR/System32/vcruntime140_1.dll")) {
        Write-Host 'Microsoft C++ Runtime is required. Windows may ask for permission to install it.'
        $runtimeInstaller = Join-Path (Get-Location) 'tools/uv/vc_redist.x64.exe'
        Invoke-WebRequest 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -UseBasicParsing -OutFile $runtimeInstaller
        $runtimeProcess = Start-Process -FilePath $runtimeInstaller -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -Wait -PassThru
        if ($runtimeProcess.ExitCode -eq 3010) { throw 'Microsoft C++ Runtime requires a Windows restart. Restart and run start.bat again.' }
        if ($runtimeProcess.ExitCode -notin @(0,1638)) { throw "Microsoft C++ Runtime installation failed: $($runtimeProcess.ExitCode)" }
    }
    $uvBinary = Join-Path (Get-Location) 'tools/uv/uv.exe'
    if (-not (Test-Path -LiteralPath $uvBinary)) {
        Invoke-WebRequest 'https://astral.sh/uv/0.12.17/install.ps1' -UseBasicParsing -OutFile 'tools/uv/install.ps1'
        $env:UV_UNMANAGED_INSTALL = Join-Path (Get-Location) 'tools/uv'
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File 'tools/uv/install.ps1'
        if ($LASTEXITCODE -ne 0) { throw 'uv installation failed. Check network access and retry.' }
    }
    $env:GUITAROCR_UV = $uvBinary
    & $uvBinary run --config-file scripts/bootstrap-uv.toml --no-project --python 3.11 scripts/launcher.py $Command @LauncherArgs
    $result = $LASTEXITCODE
    if ($result -ne 0) { throw "GuitarOCR exited with code $result. See the log above and docs/troubleshooting.md." }
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
