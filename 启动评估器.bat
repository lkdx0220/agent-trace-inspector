@echo off
setlocal
cd /d "%~dp0"

set "PORT=8000"
set "URL=http://127.0.0.1:%PORT%/"

echo ==========================================
echo   Agent Trace Inspector Launcher
echo ==========================================
echo.

rem --- If the server is already running, just open the browser ---
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/status' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Server already running.
    echo [INFO] Opening browser: %URL%
    start "" "%URL%"
    exit /b 0
)

rem --- Find Python ---
set "PYTHON=python"
where python >nul 2>&1
if errorlevel 1 (
    set "PYTHON=py"
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Please install Python 3 and add it to PATH.
        pause
        exit /b 1
    )
)

echo [INFO] Starting FastAPI server with %PYTHON% ...
start "Agent Trace Inspector" cmd /k "%PYTHON% -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%"

rem --- Wait until the server responds ---
set /a tries=0
:wait
set /a tries+=1
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8000/api/status' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto open
if %tries% geq 30 (
    echo [ERROR] Timed out waiting for the server. Check the server window for logs.
    start "" "%URL%"
    pause
    exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto wait

:open
echo [INFO] Server is ready, opening browser: %URL%
start "" "%URL%"
echo.
echo Server is running in the separate "Agent Trace Inspector" window.
echo Close that window to stop the server.
echo This launcher window can be closed now.
ping -n 2 127.0.0.1 >nul
exit /b 0
