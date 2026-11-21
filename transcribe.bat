@echo off
setlocal
rem Transcribe audio/video files to .srt without reviewing. Drag files or a folder onto it.
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
if "%~1"=="" (
  set /p "TARGET=Drag a file or folder here and press Enter: "
  call "%PY%" "%HERE%transcribe.py" %%TARGET%%
) else (
  "%PY%" "%HERE%transcribe.py" %*
)
pause
