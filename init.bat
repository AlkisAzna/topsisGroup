@echo off

echo Creating python environments and downloading necessary packages...
echo.
echo Enter password when prompted and press enter
echo Don't worry if nothing appears on the screen while typing. That's the way it works ;)
echo.

python -m venv %USERPROFILE%\topsisGroup
cd %USERPROFILE%\CustomerSystem
git clone https://github.com/AlkisAzna/topsisGroup.git

call %USERPROFILE%\topsisGroup\Scripts\activate.bat
%USERPROFILE%\topsisGroup\Scripts\python.exe -m pip install --upgrade pip
pip install -r %USERPROFILE%\topsisGroup\topsisGroup\requirements.txt
call %USERPROFILE%\CustomerSystem\Scripts\deactivate.bat

echo.
pause
