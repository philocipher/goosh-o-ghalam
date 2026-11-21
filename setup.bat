@echo off
setlocal
cd /d "%~dp0"
rem Sets up everything inside this folder, so the folder can be copied to any Windows PC:
rem   python\   a portable Python 3.12 (from python.org) with the packages this program needs
rem The models are downloaded by the review page itself (with progress) the first time.
rem Usage: setup.bat           (run by review.bat / transcribe.bat on a new computer)

set "PYVER=3.12.10"
set "PY=%~dp0python\python.exe"
echo === Persian transcriber: setup ===
echo.

if exist "%PY%" goto packages

echo [1/3] Downloading portable Python %PYVER% from python.org (11 MB)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip' -OutFile 'python.zip'"
if errorlevel 1 goto failed
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Force 'python.zip' 'python'"
if errorlevel 1 goto failed
del python.zip
rem Let this Python use installed packages ("import site") and find this program's files ("..").
powershell -NoProfile -ExecutionPolicy Bypass -Command "$f = Get-ChildItem 'python\python3*._pth' | Select-Object -First 1; $t = (Get-Content $f.FullName) -replace '^#\s*import site', 'import site'; if ($t -notcontains '..') { $t += '..' }; Set-Content -Path $f.FullName -Value $t -Encoding ascii"
if errorlevel 1 goto failed

echo [2/3] Installing pip...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
if errorlevel 1 goto failed
"%PY%" get-pip.py --no-warn-script-location --disable-pip-version-check
if errorlevel 1 goto failed
del get-pip.py

:packages
echo [3/3] Installing the packages (about 300 MB the first time; a few seconds later on)...
"%PY%" -m pip install --no-warn-script-location --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Setup complete. Everything is inside this folder: you can copy it to another computer.
echo Double-click review.bat to open the review page (the models download there, first time only).
if /i not "%~1"=="nopause" pause
exit /b 0

:failed
echo.
echo Setup failed. Check the internet connection and run setup.bat again.
if /i not "%~1"=="nopause" pause
exit /b 1
