@echo off
REM Celery Flower Dashboard Startup Script
REM
REM Flower is a real-time web-based monitor for Celery.
REM Access the dashboard at: http://localhost:5555
REM
REM Features:
REM   - Real-time task monitoring
REM   - Worker status and statistics
REM   - Task history and results
REM   - Rate limiting controls

echo ==========================================
echo   Skin Classifier - Celery Flower Monitor
echo ==========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting Flower monitoring dashboard...
echo.
echo Access the dashboard at: http://localhost:5555
echo.
echo Press Ctrl+C to stop Flower.
echo ==========================================
echo.

celery -A celery_app flower --port=5555

pause
