@echo off
REM ===================================================================
REM  One-click build launcher for the OCR tool.
REM
REM  KEEP THIS FILE PURE ASCII.
REM  cmd.exe reads .bat files using the system OEM code page (GBK on
REM  Chinese Windows) while this project's sources are UTF-8.  Putting
REM  Chinese text here gets mis-decoded, and the mangled bytes can even
REM  break line parsing and execute plain text as a command (observed:
REM  "dist\OCR-CLI.exe" was run by accident).  All user-facing messages
REM  therefore live in build.py, which handles encoding properly.
REM ===================================================================

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto nopython

python "build.py" %*
if errorlevel 1 goto failed

echo.
pause
exit /b 0

:nopython
echo.
echo   [ERROR] Python was not found in PATH.
echo.
echo   Please install Python 3.10+ and tick "Add python.exe to PATH":
echo     https://www.python.org/downloads/
echo.
pause
exit /b 1

:failed
echo.
echo   [ERROR] Build failed. See the log above for details.
echo.
pause
exit /b 1
