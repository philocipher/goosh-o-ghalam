@echo off
setlocal
rem Transcribe live, as it is spoken: from a microphone, or what the computer plays (a call,
rem a live stream, a video). Opens the review page with the live panel.
rem On a new computer it first sets everything up (portable Python and packages).
set "HERE=%~dp0"
set "PY=%HERE%python\python.exe"
if not exist "%PY%" set "PY=%HERE%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo First run on this computer: setting up...
  call "%HERE%setup.bat" nopause
  if errorlevel 1 ( pause & exit /b 1 )
  set "PY=%HERE%python\python.exe"
)
rem An older setup may not have the sound-capture package yet.
"%PY%" -c "import soundcard" 2>nul || "%PY%" -m pip install --no-warn-script-location --disable-pip-version-check -q -r "%HERE%requirements.txt"
"%PY%" "%HERE%transcribe.py" --review --live %*
rem (the window closes by itself when the program closes normally; on an error it stays to show it)
if errorlevel 1 pause
