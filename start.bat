@echo off
rem Double-click entry point. The real logic lives in start.py -- this is just a
rem shell for people who prefer not to use a terminal, plus it switches the
rem console to UTF-8 so the launcher's output is not mojibake.
rem
rem KEEP THIS FILE PURE ASCII. cmd.exe parses .bat bytes using the OEM codepage
rem (936 on a Chinese Windows), NOT UTF-8 -- Chinese comments here get decoded as
rem garbage that swallows the line break, and the next command gets glued onto
rem the junk. Symptom: "'ckend-port' is not recognized as an internal command",
rem i.e. a comment leaking out as a command. Explanations belong in start.py.
rem
rem Arguments pass through: start.bat --no-browser / --ofs / --backend-port 5001

chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Prefer python from PATH, but it may be the Microsoft Store placeholder
rem (which only prints "not installed"). So actually run it before trusting it.
set "PY=python"
python -c "import sys" >nul 2>nul
if errorlevel 1 (
  if exist "C:\Python312\python.exe" (
    set "PY=C:\Python312\python.exe"
  ) else (
    echo No usable python found. Install Python 3.10+ and put it on PATH.
    pause
    exit /b 1
  )
)

set PYTHONIOENCODING=utf-8
"%PY%" "%~dp0start.py" %*

rem Keep the window open when it fails, so a double-click does not just flash.
rem A normal Ctrl+C exit does not come through here.
if errorlevel 1 pause
endlocal
