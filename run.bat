@echo off
rem Launch Speech-2-Text (no console window). Creates the venv on first run.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Setting up virtual environment...
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)
start "" ".venv\Scripts\pythonw.exe" app.py
exit /b 0

:error
echo Setup failed.
pause
