#!/bin/bash
# Celery Worker Startup Script for Unix/Linux/MacOS
#
# This script starts a Celery worker for processing async analysis jobs.
# Make sure Redis is running before starting the worker.
#
# Usage: ./start_worker.sh [queue_name]
#   queue_name: Optional queue to listen on (default: all queues)
#
# Examples:
#   ./start_worker.sh                    - Start worker for all queues
#   ./start_worker.sh analysis           - Start worker for analysis queue only
#   ./start_worker.sh batch              - Start worker for batch queue only

echo "=========================================="
echo "  Skin Classifier - Celery Worker"
echo "=========================================="
echo ""

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Redis is running
echo "Checking Redis connection..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo ""
    echo "ERROR: Redis is not running!"
    echo "Please start Redis first:"
    echo "  - MacOS: brew services start redis"
    echo "  - Ubuntu: sudo systemctl start redis"
    echo "  - Docker: docker run -d -p 6379:6379 redis:latest"
    echo ""
    exit 1
fi
echo "Redis is running."
echo ""

# Set queue parameter
QUEUE=$1
if [ -z "$QUEUE" ]; then
    QUEUES="default,analysis,batch"
    echo "Starting worker for ALL queues: default, analysis, batch"
else
    QUEUES=$QUEUE
    echo "Starting worker for queue: $QUEUE"
fi

echo ""
echo "Worker Configuration:"
echo "  - Concurrency: 2 (optimized for GPU memory)"
echo "  - Queues: $QUEUES"
echo "  - Log Level: INFO"
echo ""
echo "Press Ctrl+C to stop the worker."
echo "=========================================="
echo ""

# Start Celery worker
celery -A celery_app worker --loglevel=info --concurrency=2 -Q $QUEUES
