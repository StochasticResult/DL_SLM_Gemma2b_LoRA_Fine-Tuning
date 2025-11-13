@echo off
chcp 65001 >nul
echo ========================================
echo Gemma-2 LoRA Fine-Tuning - Local GPU Workflow
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python environment...
python --version
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)
echo.

echo [2/3] Installing dependencies...
echo This may take a few minutes. Please stand by...
python -m pip install --upgrade pip

echo Installing PyTorch (CUDA 12.1 build)...
python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ⚠️ CUDA build failed; attempting CPU-only installation...
    python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
)

echo Installing remaining packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies!
    pause
    exit /b 1
)
echo.

echo [3/3] Launching training...
echo.
python train_local.py
if errorlevel 1 (
    echo.
    echo ❌ An error occurred during training!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Training complete!
echo ========================================
pause

