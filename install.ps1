@echo off
REM One-shot LOCAL install (Windows): Python + MedPsy GGUF + QVAC sidecar deps.
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ============================================================
echo   QVAC vs Cloud - Health Test - local install (one time)
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo Python 3 not found. Install from https://www.python.org/downloads/
  exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js ^>= 22.17 required. Install from https://nodejs.org/
  exit /b 1
)

echo ==^> 1/3 Python virtualenv + deps...
if not exist ".venv" (
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q huggingface_hub

if not exist ".env" (
  copy /Y .env.example .env >nul
  echo     Created .env - paste your full OpenRouter key.
)

echo ==^> 2/3 MedPsy GGUF from Hugging Face (~2.5 GB)...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\download_medpsy_gguf.ps1"
if errorlevel 1 exit /b 1

echo ==^> 3/3 QVAC SDK sidecar npm install...
cd sidecar
call npm install
if errorlevel 1 exit /b 1
cd ..

echo.
echo Installation complete.
echo.
echo   Terminal A:  cd sidecar ^&^& npm start
echo   Terminal B:  .venv\Scripts\activate ^&^& streamlit run app.py
echo                -^> Automated Benchmark
echo.
echo   Edit .env with OPENROUTER_API_KEY=sk-or-v1-... (full key).
echo.
endlocal
