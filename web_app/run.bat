@echo off
cd /d "%~dp0.."

echo ===================================================================
echo   KHOI DONG HE THONG NHAN DIEN KY HIEU VA GHEP CAU GEMINI LLM
echo ===================================================================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [CANH BAO] Khong tim thay thu muc venv! Dang su dung python he thong...
)

echo.
echo [!] Dia chi truy cap: http://localhost:8000/
echo [!] Dang mo trinh duyet...
start http://localhost:8000/ 2>nul

echo [!] Dang khoi dong Web Server...
echo ===================================================================
python web_app\server.py %*
pause
