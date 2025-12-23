@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Starting Skin Classifier App
echo ========================================

:: Get current IP address - try multiple methods
set IP=
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set TEMP_IP=%%a
    set TEMP_IP=!TEMP_IP: =!
    echo Checking IP: !TEMP_IP!
    echo !TEMP_IP! | findstr /r "^192\.168\." >nul && set IP=!TEMP_IP! && goto :found
    echo !TEMP_IP! | findstr /r "^10\." >nul && set IP=!TEMP_IP! && goto :found
    echo !TEMP_IP! | findstr /r "^172\." >nul && set IP=!TEMP_IP! && goto :found
)

:: Fallback: use localhost if no private IP found
if "%IP%"=="" (
    echo Warning: Could not detect private IP, using localhost
    set IP=127.0.0.1
)

:found
echo.
echo Detected IP: %IP%
echo.
echo NOTE: Frontend uses dynamic IP detection via Expo Constants
echo The app will automatically connect to: http://%IP%:8000
echo.

:: Kill any existing Python processes
echo Stopping any running servers...
taskkill /F /IM python.exe /T >nul 2>&1

echo.
echo ========================================
echo Starting Backend Server
echo ========================================
start "Backend Server" cmd /k "cd /d "%~dp0backend" && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo Waiting for backend to start and load ML models...
echo This may take 15-20 seconds...
timeout /t 5 /nobreak >nul

:: Wait for backend health check to respond (with retry limit)
set RETRY_COUNT=0
set MAX_RETRIES=10

:wait_backend
set /a RETRY_COUNT+=1
echo Checking if backend is ready (attempt %RETRY_COUNT%/%MAX_RETRIES%)...

:: Try localhost first (always works if backend is running locally)
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2 -UseBasicParsing; Write-Host '  Backend is responding on localhost'; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo Backend is ready on localhost!
    goto :backend_ready
)

:: If localhost failed, try the detected IP
if not "%IP%"=="127.0.0.1" (
    powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://%IP%:8000/health' -TimeoutSec 2 -UseBasicParsing; Write-Host '  Backend is responding on %IP%'; exit 0 } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        echo Backend is ready on %IP%!
        goto :backend_ready
    )
)

:: Check if we've exceeded max retries
if %RETRY_COUNT% GEQ %MAX_RETRIES% (
    echo.
    echo ERROR: Backend failed to respond after %MAX_RETRIES% attempts
    echo The backend server may have started but is not responding to health checks.
    echo.
    echo Troubleshooting steps:
    echo 1. Check the Backend Server window for errors
    echo 2. Try accessing http://127.0.0.1:8000/health in your browser
    echo 3. Run troubleshoot_connection.bat for detailed diagnostics
    echo 4. Check if Windows Firewall is blocking port 8000
    echo.
    echo Continuing anyway... Frontend will start but may not connect initially.
    timeout /t 5
    goto :backend_ready
)

echo   Backend not ready yet, waiting 3 more seconds...
timeout /t 3 /nobreak >nul
goto :wait_backend

:backend_ready
echo Backend is ready!
echo.
echo ========================================
echo Starting Frontend with Expo Go mode
echo ========================================

:: Get Wi-Fi IP using PowerShell (more reliable)
for /f "tokens=*" %%i in ('powershell -NoProfile -Command "(Get-NetIPAddress -InterfaceAlias 'Wi-Fi' -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress"') do set WIFI_IP=%%i

:: Fallback to Ethernet
if "%WIFI_IP%"=="" (
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command "(Get-NetIPAddress -InterfaceAlias 'Ethernet' -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress"') do set WIFI_IP=%%i
)

:: Use detected IP or fallback to the earlier detection
if not "%WIFI_IP%"=="" set IP=%WIFI_IP%

echo Frontend will use IP: %IP%
start "Frontend" cmd /k "cd /d "%~dp0frontend" && rmdir /s /q .expo 2>nul & set REACT_NATIVE_PACKAGER_HOSTNAME=%IP% && npx expo start --go --clear"

echo.
echo ========================================
echo Both servers started!
echo Backend: http://%IP%:8000
echo Frontend: LAN mode (scan QR code in terminal)
echo ========================================
