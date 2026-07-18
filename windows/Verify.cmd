@echo off
setlocal
cd /d "%~dp0"

echo Verifying portable bundle checksums...
"runtime\python.exe" "verify_bundle.py" "SHA256SUMS.txt"
exit /b %ERRORLEVEL%
