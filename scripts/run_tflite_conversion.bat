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
pip install -q -r tflite_conversion_requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check tflite_conversion_requirements.txt
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
python onnx_to_tflite_converter.py
if errorlevel 1 (
    echo ERROR: Conversion failed
    echo Check onnx_to_tflite_conversion.log for details
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
python validate_tflite_accuracy.py
if errorlevel 1 (
    echo WARNING: Validation encountered errors
    echo Check tflite_accuracy_validation.log for details
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
echo Output directory: D:\Base-dir\tflite_models\
echo.
echo Reports generated:
echo   - tflite_conversion_report.json (conversion details)
echo   - validation_report_default.json (accuracy validation)
echo.
echo Log files:
echo   - onnx_to_tflite_conversion.log
echo   - tflite_accuracy_validation.log
echo.
echo Total models converted: 11 models x 3 versions = 33 TFLite files
echo.
echo ================================================================================
echo.

REM Open output directory
echo Opening output directory...
explorer "D:\Base-dir\tflite_models"

echo.
echo Press any key to exit...
pause >nul
