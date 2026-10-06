@echo off
cd /d "%~dp0"
echo Starting Lucidscapes production. Press Ctrl+C to stop everything cleanly.
echo.
python resume_production.py
echo.
echo Window will stay open so you can see the final status. Press any key to close.
pause >nul
