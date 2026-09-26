@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1" start %*
set "guitarocr_exit=%errorlevel%"
if not "%guitarocr_exit%"=="0" pause
exit /b %guitarocr_exit%
