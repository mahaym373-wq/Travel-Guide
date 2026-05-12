@echo off
title Smart India Travel Guide - Starting...
echo ============================================================
echo   Smart India Travel Guide - Semantic Network Chatbot
echo ============================================================
echo.
echo   Starting the server... please wait.
echo.

:: Try Python 3.11 first (has all dependencies), fallback to default python
where py >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   Using Python Launcher...
    start "Travel Guide Server" py -3.11 -m uvicorn main:app --port 8000
) else (
    echo   Using default Python...
    start "Travel Guide Server" python -m uvicorn main:app --port 8000
)

:: Wait for server to start
echo   Waiting for server to initialize...
timeout /t 4 /nobreak >nul

:: Open browser automatically
echo   Opening browser...
start http://localhost:8000

echo.
echo ============================================================
echo   Server is running at: http://localhost:8000
echo   
echo   To STOP the server, close the other command window
echo   or press Ctrl+C in it.
echo ============================================================
echo.
pause
