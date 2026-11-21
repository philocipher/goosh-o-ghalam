@echo off
setlocal
rem Transcribe and review in the browser. Double-click to pick recordings on the page,
rem or drag a recording (or a folder) onto this file to start with it.
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
"%PY%" "%HERE%transcribe.py" --review %*
rem (the window closes by itself when the program closes normally; on an error it stays to show it)
if errorlevel 1 pause
