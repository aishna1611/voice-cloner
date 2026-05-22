@echo off
title Voice Cloning Studio — Dependency Installer
color 0A

echo ============================================================
echo   Voice Cloning Studio — Dependency Installer
echo ============================================================
echo.
echo This will install all required Python packages.
echo No internet connection needed after this step.
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.9 or later from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

:: Upgrade pip silently
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo Installing packages...
echo.

echo [1/4] Installing numpy (audio processing)...
pip install numpy --quiet
if errorlevel 1 (echo [WARN] numpy install failed — some features limited) else (echo       Done.)

echo [2/4] Installing scipy (audio DSP + WAV files)...
pip install scipy --quiet
if errorlevel 1 (echo [WARN] scipy install failed — some features limited) else (echo       Done.)

echo [3/4] Installing sounddevice (microphone recording)...
pip install sounddevice --quiet
if errorlevel 1 (
    echo [WARN] sounddevice failed. Trying alternate method...
    pip install sounddevice --pre --quiet
)
echo       Done.

echo [4/4] Installing pyttsx3 (offline text-to-speech engine)...
pip install pyttsx3 --quiet
if errorlevel 1 (echo [WARN] pyttsx3 install failed — TTS will not work) else (echo       Done.)

echo.
echo ============================================================
echo   Installation complete!
echo ============================================================
echo.
echo You can now run the app by double-clicking:
echo   run_studio.bat
echo.
pause
