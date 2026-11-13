# Launcher script for PowerShell
$RootPath = Split-Path -Parent $PSScriptRoot
Set-Location $RootPath

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "  KHỞI ĐỘNG HỆ THỐNG WEB NHẬN DIỆN KÝ HIỆU & GHÉP CÂU GEMINI LLM" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Cyan

if (Test-Path ".\venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Warning "Không tìm thấy virtual environment tại .\venv! Sử dụng python mặc định..."
}

python web_app/server.py $args

