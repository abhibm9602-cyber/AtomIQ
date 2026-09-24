@echo off
title AtomIQ App Launcher
echo ===================================================
echo   Launching AtomIQ AI Dashboard...
echo ===================================================
echo.
cd /d "C:\Users\Abhi\.gemini\antigravity\scratch\AtomIQ"
"C:\Users\Abhi\.gemini\antigravity\scratch\AtomIQ\venv\Scripts\streamlit.exe" run app.py
pause
