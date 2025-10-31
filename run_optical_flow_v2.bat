@echo off
echo ========================================
echo OPTICAL FLOW RECONSTRUCTION - v2
echo ========================================
echo.

cd /d "C:\Users\Asus\Desktop\vs code files\Tecdia project\jumbled-frames-reconstruction"

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Step 1: Reconstructing frames with Optical Flow...
echo ------------------------------------------------
python src\approach_resnet_tf_refinement\v2\reconstruct_optical_flow.py
if %errorlevel% neq 0 (
    echo ERROR: Reconstruction failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Creating video from reconstructed frames...
echo ------------------------------------------------
python src\approach_resnet_tf_refinement\v2\construct_video.py
if %errorlevel% neq 0 (
    echo ERROR: Video creation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo OPTICAL FLOW RECONSTRUCTION COMPLETE!
echo ========================================
echo Output video: output\reconstructed_optical_flow_v2.mp4
echo.
pause
