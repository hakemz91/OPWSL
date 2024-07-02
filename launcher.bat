@echo off
set /p confirm=You sure want to run OPWSL? (y/n): 
if /i "%confirm%"=="y" (
    wsl -d Ubuntu-22.04
) else (
    echo Operation canceled by user.
)
