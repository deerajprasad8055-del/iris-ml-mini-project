@echo off
title Iris ML Web App Launcher
cd /d "%~dp0"

echo ========================================================
echo          🌸 Starting Iris ML Web Frontend App           
echo ========================================================
echo.
echo Launching Streamlit web app in your default browser...
echo Press Ctrl+C in this window anytime to stop the server.
echo.

python -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start Streamlit app. Please check Python installation.
    pause
)
