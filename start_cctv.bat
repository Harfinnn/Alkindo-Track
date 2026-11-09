@echo off
rem Jalankan dari folder yang sama dengan dual_cctv_counting.py
cd /d "%~dp0"

set VIEWER_PORT=8080
set INFER_DEVICE=intel:gpu

if not exist venv\Scripts\activate.bat (
    echo Membuat virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install ultralytics openvino opencv-python numpy torch
) else (
    call venv\Scripts\activate.bat
)

echo Web viewer: http://localhost:%VIEWER_PORT%
:loop
python human_tracking_dual_cctv.py --headless
echo Script berhenti. Mulai ulang dalam 5 detik... (Ctrl+C untuk batal)
timeout /t 5 /nobreak >nul
goto loop