@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Project environment missing. See README.md for first-time setup.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" "src\ui.py"
