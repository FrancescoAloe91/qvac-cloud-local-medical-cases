@echo off
REM Prefer launch_dashboard.bat — it starts the QVAC sidecar + Streamlit.
REM This file redirects so double-clicking run.bat still gets local GGUF support.
cd /d "%~dp0"
echo.
echo run.bat now launches launch_dashboard.bat (QVAC sidecar + Streamlit).
echo For Streamlit-only without sidecar, run: streamlit run app.py
echo.
call "%~dp0launch_dashboard.bat"
