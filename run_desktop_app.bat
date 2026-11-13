@echo off
cd /d "%~dp0"

echo ===================================================================
echo   KHOI DONG UNG DUNG NHAN DIEN KY HIEU DESKTOP (OPENCV GUI)
echo ===================================================================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [CANH BAO] Khong tim thay thu muc venv! Dang su dung python he thong...
)

python app.py %*
pause
