@echo off
echo ========================================
echo Amharic XTTS Fine-Tuning WebUI Installer
echo ========================================
echo.
echo This will install to your current Python environment.
echo No virtual environment will be created.
echo.
pause

REM Run smart installer
python smart_install.py

echo.
echo Installation complete!
echo Run "launch.bat" or "python xtts_demo.py" to start.
pause
