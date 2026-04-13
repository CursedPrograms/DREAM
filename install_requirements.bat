@echo off
echo Updating global pip...
python.exe -m pip install --upgrade pip

echo Creating Virtual Environment (Python 3.11)...
py -3.11 -m venv venv311

echo Activating environment...
call venv311\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete! Current Python version:
python --version

pause