@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" install %*
set "guitarocr_exit=%errorlevel%"
pause
exit /b %guitarocr_exit%
