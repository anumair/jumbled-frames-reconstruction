@echo off
echo ========================================
echo ResNet50 + Refinement Pipeline
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 1/5: Extracting ResNet50 features (PyTorch)...
python src\approach_resnet_tf_refinement\1_extract_resnet_features.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in ResNet50 extraction!
    pause
    exit /b 1
)

echo.
echo Step 2/5: Computing similarity matrix...
python src\approach_resnet_tf_refinement\2_compute_similarity.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in similarity computation!
    pause
    exit /b 1
)

echo.
echo Step 3/5: Initial frame ordering (smart start detection)...
python src\approach_resnet_tf_refinement\3_order_frames.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in initial ordering!
    pause
    exit /b 1
)

echo.
echo Step 4/5: Refining ordering (2-opt + sliding window)...
python src\approach_resnet_tf_refinement\4_refine_ordering.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in refinement!
    pause
    exit /b 1
)

echo.
echo Step 5/5: Reconstructing video...
python src\approach_resnet_tf_refinement\5_reconstruct_video.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in video reconstruction!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Pipeline completed successfully!
echo Output video: output\reconstructed_resnet_tf_refined.mp4
echo ========================================
pause
