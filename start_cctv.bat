@echo off
title GATEWATCH CONTROL CENTER

cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo Virtual environment tidak ditemukan.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python process_manager.py

pause