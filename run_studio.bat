@echo off
title Voice Cloning Studio
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please run install_deps.bat first.
    pause
    exit /b 1
)

python voice_studio.py
if errorlevel 1 (
    echo.
    echo App exited with an error. Run install_deps.bat if dependencies are missing.
    pause
)
