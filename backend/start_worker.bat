@echo off
REM Celery Worker Startup Script for Windows
REM
REM This script starts a Celery worker for processing async analysis jobs.
REM Make sure Redis is running before starting the worker.
REM
REM Usage: start_worker.bat [queue_name]
REM   queue_name: Optional queue to listen on (default: all queues)
REM
REM Examples:
REM   start_worker.bat                    - Start worker for all queues
REM   start_worker.bat analysis           - Start worker for analysis queue only
REM   start_worker.bat batch              - Start worker for batch queue only

echo ==========================================
echo   Skin Classifier - Celery Worker
echo ==========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if Redis is running
echo Checking Redis connection...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Redis is not running!
    echo Please start Redis first:
    echo   - Download Redis for Windows from: https://github.com/tporadowski/redis/releases
    echo   - Or use Docker: docker run -d -p 6379:6379 redis:latest
    echo.
    pause
    exit /b 1
)
echo Redis is running.
echo.

REM Set queue parameter
set QUEUE=%1
if "%QUEUE%"=="" (
    set QUEUES=default,analysis,batch
    echo Starting worker for ALL queues: default, analysis, batch
) else (
    set QUEUES=%QUEUE%
    echo Starting worker for queue: %QUEUE%
)

echo.
echo Worker Configuration:
echo   - Concurrency: 2 (optimized for GPU memory)
echo   - Queues: %QUEUES%
echo   - Log Level: INFO
echo.
echo Press Ctrl+C to stop the worker.
echo ==========================================
echo.

REM Start Celery worker
REM Note: Using -P solo on Windows as eventlet/gevent have issues
celery -A celery_app worker --loglevel=info --concurrency=2 -Q %QUEUES% -P solo

pause
