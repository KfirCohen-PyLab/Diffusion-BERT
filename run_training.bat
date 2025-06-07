@echo off
echo =====================================================
echo    DIFFUSION BERT CONDITIONAL TRAINING LAUNCHER
echo =====================================================
echo.
echo This will start the interactive training launcher
echo.
echo Make sure you have:
echo - Python 3.8+ installed
echo - PyTorch with CUDA (if using GPU)
echo - All required dependencies
echo.
pause

python run_conditional_training.py

echo.
echo =====================================================
echo Training session completed. Press any key to exit.
pause 