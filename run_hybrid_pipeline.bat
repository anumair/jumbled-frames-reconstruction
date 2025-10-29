@echo off
echo ========================================
echo Hybrid Approach Pipeline (ResNet50 + YOLOv11)
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 1/5: Extracting ResNet50 features...
python src\approach_resnet_yolo\1_extract_resnet_features.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in ResNet50 extraction!
    pause
    exit /b 1
)

echo.
echo Step 2/5: Extracting YOLO features...
python src\approach_resnet_yolo\2_extract_yolo_features.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in YOLO extraction!
    pause
    exit /b 1
)

echo.
echo Step 3/5: Computing combined similarity...
python src\approach_resnet_yolo\3_compute_similarity.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in similarity computation!
    pause
    exit /b 1
)

echo.
echo Step 4/5: Ordering frames...
python src\approach_resnet_yolo\4_order_frames.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in frame ordering!
    pause
    exit /b 1
)

echo.
echo Step 5/5: Reconstructing video...
python src\approach_resnet_yolo\5_reconstruct_video.py
if %ERRORLEVEL% NEQ 0 (
    echo Error in video reconstruction!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Pipeline completed successfully!
echo Output video: output\reconstructed_resnet_yolo.mp4
echo ========================================
pause
