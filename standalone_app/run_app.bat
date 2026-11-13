@echo off
cd /d "%~dp0"

echo ===================================================================
echo   PHAN MEM NHAN DIEN KY HIEU VA GHEP CAU GEMINI LLM (STANDALONE)
echo ===================================================================

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
) else (
    echo [CANH BAO] Khong tim thay moi truong venv. Dang dung python he thong...
)

echo.
echo [*] Dang khoi dong ung dung va mo cua so Desktop...
python app.py %*
pause

