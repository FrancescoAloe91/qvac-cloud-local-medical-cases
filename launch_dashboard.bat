@echo off
REM Start QVAC SDK sidecar + Streamlit dashboard (Windows). No Ollama.
setlocal EnableExtensions
cd /d "%~dp0"

set PORT=8501
set URL=http://localhost:%PORT%
set SIDECAR_URL=http://127.0.0.1:8787
set SIDECAR_LOG=%TEMP%\qvac-sidecar.log
set SIDECAR_PID=%TEMP%\qvac-sidecar.pid

if not exist ".venv" (
  echo First run: creating virtualenv...
  python -m venv .venv
  call ".venv\Scripts\activate.bat"
  pip install -q -r requirements.txt
) else (
  call ".venv\Scripts\activate.bat"
)

if exist ".env" (
  for /f "usebackq tokens=1* delims==" %%a in (".env") do (
    if not "%%a"=="" if not "%%a"=="#" set "%%a=%%b"
  )
)

if not defined QVAC_DEVICE set QVAC_DEVICE=gpu
if not defined QVAC_GPU_LAYERS set QVAC_GPU_LAYERS=99
if not defined QVAC_WARM_LOAD set QVAC_WARM_LOAD=1

if not defined QVAC_MODEL_PATH (
  if exist "models\medpsy-4b-q4_k_m-imat.gguf" (
    set "QVAC_MODEL_PATH=%CD%\models\medpsy-4b-q4_k_m-imat.gguf"
  )
)

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js is required for the QVAC sidecar. Install Node 22+ then re-run.
  goto start_streamlit
)

if not exist "sidecar\qvac_server.mjs" (
  echo Sidecar missing. Run install.bat once.
  goto start_streamlit
)

if not exist "sidecar\node_modules\@qvac\sdk" (
  echo @qvac/sdk missing. Run: cd sidecar ^&^& npm ci
  goto start_streamlit
)

curl -sf "%SIDECAR_URL%/health" >nul 2>&1
if errorlevel 1 (
  echo Starting QVAC sidecar on :8787 ...
  start "" /B cmd /c "cd /d "%CD%\sidecar" && node qvac_server.mjs >> "%SIDECAR_LOG%" 2>&1"
  for /l %%i in (1,1,60) do (
    curl -sf "%SIDECAR_URL%/health" >nul 2>&1 && goto sidecar_ok
    timeout /t 1 /nobreak >nul
  )
  echo WARNING: sidecar health not ready. Log: %SIDECAR_LOG%
) else (
  echo QVAC sidecar already healthy.
)
:sidecar_ok

:start_streamlit
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set STREAMLIT_SERVER_SHOW_EMAIL_PROMPT=false

netstat -ano | findstr ":%PORT% " | findstr LISTENING >nul
if not errorlevel 1 (
  start "" "%URL%"
  exit /b 0
)

start "" /B streamlit run app.py --server.port=%PORT% --server.headless=true --server.showEmailPrompt=false
for /l %%i in (1,1,30) do (
  curl -sf "%URL%" >nul 2>&1 && (
    start "" "%URL%"
    exit /b 0
  )
  timeout /t 1 /nobreak >nul
)
echo Streamlit did not start. Check the terminal output.
exit /b 1
