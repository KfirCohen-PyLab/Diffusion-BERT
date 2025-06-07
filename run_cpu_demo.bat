@echo off
echo =====================================================
echo     DIFFUSION BERT - CPU CLASSROOM DEMONSTRATION
echo =====================================================
echo.
echo 💻 This demo is optimized for laptops without GPU
echo 🎓 Perfect for classroom presentations
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Set default model path - you can change this
set MODEL_PATH=model_name_bert-base-uncased_lr_0.0005_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_timestep_none_ckpts\final.th

REM Check if user provided model path as argument
if not "%~1"=="" (
    set MODEL_PATH=%~1
)

echo 🔍 Using model: %MODEL_PATH%
echo.

REM Check if model exists
if not exist "%MODEL_PATH%" (
    echo ❌ Model file not found: %MODEL_PATH%
    echo.
    echo 💡 Usage: run_cpu_demo.bat [path_to_model]
    echo 📝 Example: run_cpu_demo.bat "models\my_model\final.th"
    echo.
    pause
    exit /b 1
)

echo ✅ Model file found!
echo 🚀 Starting CPU demonstration...
echo.

REM Run the demonstration
python cpu_classroom_demo.py --checkpoint "%MODEL_PATH%" --mode full

echo.
echo 🎉 Demonstration completed!
echo 💻 All processing was done on CPU
pause 