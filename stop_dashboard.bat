@echo off
REM Stop Streamlit dashboard + QVAC sidecar (Windows). Does not touch Ollama.
setlocal EnableExtensions
cd /d "%~dp0"

set PORT=8501
set SIDECAR_PORT=8787

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
  taskkill /F /PID %%p >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%SIDECAR_PORT% " ^| findstr LISTENING') do (
  taskkill /F /PID %%p >nul 2>&1
)

echo Dashboard and QVAC sidecar stopped.
exit /b 0
