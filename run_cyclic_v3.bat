@echo off
echo ============================================
echo V3: Cyclic Correction Reconstruction
echo ============================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 1: Running cyclic correction reconstruction...
python src\approach_resnet_tf_refinement\v3\reconstruct_cyclic_correction.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Reconstruction failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Constructing video...
python src\approach_resnet_tf_refinement\v3\construct_video.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Video construction failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo V3 pipeline completed successfully!
echo ============================================
echo.
echo Output video: src\approach_resnet_tf_refinement\v3\output\reconstructed_cyclic_v3.mp4
echo.

pause
