
@echo off
cd /d "%~dp0"

set VIEWER_PORT=8080

rem =================================================
rem NVIDIA CUDA CONFIGURATION
rem =================================================
set INFER_DEVICE=0

rem Pastikan tidak menggunakan Intel OpenVINO
echo ====================================
echo ALKINDO CCTV TRACKING SYSTEM
echo Backend : NVIDIA CUDA
echo Device  : GPU %INFER_DEVICE%
echo Viewer  : http://localhost:%VIEWER_PORT%
echo ====================================

if not exist venv\Scripts\activate.bat (
    echo Membuat virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat

    pip install ultralytics opencv-python numpy
) else (
    call venv\Scripts\activate.bat
)

rem Periksa CUDA sebelum menjalankan aplikasi
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Tidak tersedia')"

if errorlevel 1 (
    echo Gagal memeriksa PyTorch.
    pause
    exit /b 1
)

:loop
python human_tracking_dual_cctv_cuda.py --headless

echo.
echo Script berhenti. Mulai ulang dalam 5 detik...
timeout /t 5 /nobreak >nul
goto loop