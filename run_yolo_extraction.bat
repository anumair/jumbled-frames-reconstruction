@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Running YOLO feature extraction...
python src\approach_resnet_yolo\2_extract_yolo_features.py

echo.
echo Done!
pause
