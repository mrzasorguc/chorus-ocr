@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Chorus OCR Setup

echo.
echo =====================================================
echo   Chorus OCR - Multi-engine one-click installation
echo =====================================================
echo.

echo Chorus always installs and uses:
echo   - EasyOCR
 echo  - PaddleOCR
echo   - Tesseract
 echo.
echo Optional: GOT-OCR 2.0
 echo NOTE: GOT-OCR is required for Maximum Performance mode.
echo It needs more disk space, memory and installation time.
echo.
choice /C YN /N /M "Install GOT-OCR for Maximum Performance? [Y/N]: "
if errorlevel 2 (
    set "INSTALL_GOT=0"
) else (
    set "INSTALL_GOT=1"
)

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo.
        echo Python was not found.
        echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
        echo During installation, select "Add Python to PATH".
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

where tesseract >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" goto :tesseract_ready
    echo.
    echo Tesseract was not found. Attempting automatic installation...
    where winget >nul 2>nul
    if errorlevel 1 (
        echo Automatic Tesseract installation is unavailable.
        echo Install it from https://github.com/UB-Mannheim/tesseract/wiki
        pause
        exit /b 1
    )
    winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo Tesseract installation failed.
        echo Install it from https://github.com/UB-Mannheim/tesseract/wiki
        pause
        exit /b 1
    )
)

:tesseract_ready
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [1/3] Creating a private Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
)

set "VENV_PY=.venv\Scripts\python.exe"
echo [2/3] Installing the three-engine Chorus core...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :error

if "%INSTALL_GOT%"=="1" (
    echo Installing Chorus with GOT-OCR Maximum Performance support...
    "%VENV_PY%" -m pip install -e ".[demo,got]"
) else (
    echo Installing Chorus Standard Fusion without GOT-OCR...
    "%VENV_PY%" -m pip install -e ".[demo]"
)
if errorlevel 1 goto :error

echo [3/3] Opening Chorus in your browser...
"%VENV_PY%" -m chorus.web
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Installation could not be completed.
echo Check your internet connection and the messages above.
pause
exit /b 1
