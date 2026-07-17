@echo off
setlocal
cd /d "%~dp0"

call "Verify.cmd"
if errorlevel 1 (
  echo.
  echo Bundle verification failed. No program was started.
  exit /b 1
)

echo.
"runtime\python.exe" "battle_runner.py" --config "battle.toml"
set "RUN_STATUS=%ERRORLEVEL%"

echo.
if not "%RUN_STATUS%"=="0" echo Battle runner exited with status %RUN_STATUS%.
pause
exit /b %RUN_STATUS%
