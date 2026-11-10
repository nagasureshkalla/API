@echo off
SETLOCAL

echo ======================================
echo Python Virtual Environment Setup
echo ======================================

REM Ensure virtualenv is available
pip install virtualenv

REM Create venv only if it does not exist
IF NOT EXIST venv (
    echo Virtual environment not found. Creating venv...
    virtualenv venv
) ELSE (
    echo Virtual environment already exists. Skipping creation.
)

echo ======================================
echo Activating Virtual Environment
echo ======================================
call venv\Scripts\activate

echo ======================================
echo Installing Dependencies
echo ======================================

pip install Flask
pip install torch
pip install git+https://github.com/huggingface/diffusers.git
pip install rembg
pip install onnxruntime
pip install transformers
pip install accelerate sentencepiece accelerator
pip install gpustat

echo ======================================
echo SetUp complete. Run Your Application using following command : python app.py
echo ======================================


echo ======================================
echo Done
echo ======================================
pause
ENDLOCAL
