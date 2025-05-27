@echo off
setlocal EnableDelayedExpansion

:: Batch file to run DiffusionBERT sampling scripts (predict.py and predict_downstream_condition.py)
:: Usage: diffusionbert_sample.bat <model_name> <topk> <step_size>
:: Example: diffusionbert_sample.bat D3PM 30 2

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not found in PATH.
    exit /b 1
)

:: Check if required arguments are provided
if "%1"=="" (
    echo Error: Model name not provided.
    echo Usage: %0 ^<model_name^> ^<topk^> ^<step_size^>
    exit /b 1
)
if "%2"=="" (
    echo Error: topk value not provided.
    echo Usage: %0 ^<model_name^> ^<topk^> ^<step_size^>
    exit /b 1
)
if "%3"=="" (
    echo Error: step_size value not provided.
    echo Usage: %0 ^<model_name^> ^<topk^> ^<step_size^>
    exit /b 1
)

:: Set variables from arguments
set MODEL_NAME=%1
set TOPK=%2
set STEP_SIZE=%3

:: Define paths
set PREDICT_SCRIPT=predict.py
set CONDITIONAL_SCRIPT=predict_downstream_condition.py
set OUTPUT_DIR=generation_results
set TEMP_FILE=temp.txt

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to create output directory %OUTPUT_DIR%.
        exit /b 1
    )
)

:: Check if predict.py exists
if not exist "%PREDICT_SCRIPT%" (
    echo Error: %PREDICT_SCRIPT% not found in the current directory.
    exit /b 1
)

:: Check if predict_downstream_condition.py exists
if not exist "%CONDITIONAL_SCRIPT%" (
    echo Warning: %CONDITIONAL_SCRIPT% not found. Skipping conditional sampling.
    set RUN_CONDITIONAL=0
) else (
    set RUN_CONDITIONAL=1
)

:: Run unconditional sampling (predict.py)
echo Running unconditional sampling with %PREDICT_SCRIPT%...
python %PREDICT_SCRIPT% --topk %TOPK% --step_size %STEP_SIZE% --name %MODEL_NAME%
if %ERRORLEVEL% neq 0 (
    echo Error: Unconditional sampling failed.
    exit /b 1
)
echo Unconditional sampling completed. Outputs saved to %TEMP_FILE% and %OUTPUT_DIR%\%MODEL_NAME%_step_curve.txt

:: Run conditional sampling (predict_downstream_condition.py) if the script exists
if %RUN_CONDITIONAL%==1 (
    echo Running conditional sampling with %CONDITIONAL_SCRIPT%...
    python %CONDITIONAL_SCRIPT% --topk %TOPK% --step_size %STEP_SIZE% --name %MODEL_NAME%
    if %ERRORLEVEL% neq 0 (
        echo Error: Conditional sampling failed.
        exit /b 1
    )
    echo Conditional sampling completed. Outputs saved to %TEMP_FILE% and %OUTPUT_DIR%\%MODEL_NAME%_step_curve.txt
)

:: Verify output files
if not exist "%TEMP_FILE%" (
    echo Warning: %TEMP_FILE% was not created. Check the script execution.
)
if not exist "%OUTPUT_DIR%\%MODEL_NAME%_step_curve.txt" (
    echo Warning: %OUTPUT_DIR%\%MODEL_NAME%_step_curve.txt was not created. Check the script execution.
)

echo Done.
exit /b 0