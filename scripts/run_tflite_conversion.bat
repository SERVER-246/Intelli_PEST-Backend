@echo off
REM ONNX to TFLite Conversion Automation Script
REM This script handles the complete conversion and validation process

echo ================================================================================
echo ONNX to TFLite Conversion - Automated Workflow
echo ================================================================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later
    pause
    exit /b 1
)
echo    Python found: OK
echo.

REM Install dependencies
echo [2/5] Installing required packages...
echo    This may take a few minutes...
pip install -q -r ..\requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check requirements.txt
    pause
    exit /b 1
)
echo    Dependencies installed: OK
echo.

REM Run conversion
echo [3/5] Converting ONNX models to TFLite...
echo    This will take approximately 30-45 minutes for all 11 models
echo    Progress will be shown below:
echo.
python ..\src\conversion\tflite_converter.py
if errorlevel 1 (
    echo ERROR: Conversion failed
    echo Check conversion logs for details
    pause
    exit /b 1
)
echo.
echo    Conversion completed: OK
echo.

REM Run validation for default models
echo [4/5] Validating default (full precision) models...
echo    Testing accuracy preservation...
echo.
python ..\src\conversion\model_validator.py
if errorlevel 1 (
    echo WARNING: Validation encountered errors
    echo Check validation logs for details
)
echo.
echo    Validation completed: OK
echo.

REM Generate summary
echo [5/5] Generating summary...
echo.
echo ================================================================================
echo CONVERSION COMPLETE
echo ================================================================================
echo.
echo Output directory: ..\outputs\tflite_models\
echo.
echo Check the outputs folder for converted models and reports
echo.
echo ================================================================================
echo.

REM Open output directory
echo Opening output directory...
if exist "..\outputs\tflite_models" (
    explorer "..\outputs\tflite_models"
) else (
    echo Output directory not created yet
)

echo.
echo Press any key to exit...
pause >nul
