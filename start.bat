@echo off
cd /d "%~dp0"
if not exist node_modules call npm install
if errorlevel 1 goto :error
call npm run build
if errorlevel 1 goto :error
start "" /b node_modules\electron\dist\electron.exe .
exit /b 0
:error
echo Kite could not start. See the error above.
pause
exit /b 1
