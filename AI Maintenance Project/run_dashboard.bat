@echo off
cd /d "%~dp0"
echo Installing required packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installation failed. Check your Python/pip setup.
    pause
    exit /b 1
)
echo.
echo Starting AI Maintenance Dashboard...
python -m streamlit run dashboard.py
pause
