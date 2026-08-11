@echo off
if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo Applio does not require administrator permissions and should be run as a regular user.
    echo.
    pause
    exit /b 1
)

setlocal
for %%F in ("%~dp0.") do set "folder_name=%%~nF"
title %folder_name%

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

set HIP_VISIBLE_DEVICES="0"
SET DISABLE_ADDMM_CUDA_LT=1
set CUDA_VISIBLE_DEVICES=-1

:: Added CPU flag here
env\python.exe app.py --open --device cpu
echo.
pause