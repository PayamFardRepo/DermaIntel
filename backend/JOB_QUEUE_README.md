# Job Queue System (Celery/Redis)

This document describes the asynchronous job queue system for the Skin Classifier application, which handles long-running ML analysis operations without blocking API requests.

## Overview

The job queue uses:
- **Celery** - Distributed task queue
- **Redis** - Message broker and result backend
- **Flower** (optional) - Real-time monitoring dashboard

## Why Use the Job Queue?

The synchronous analysis endpoints can timeout on slow connections because:
- Full classification takes 2-5 seconds with multiple ML models
- Batch skin checks process 20-50 images (20-60 seconds)
- Histopathology analysis is computationally intensive

The job queue allows:
- Immediate response with job ID
- Non-blocking analysis on slow/mobile connections
- Progress tracking during processing
- Automatic retries on failures
- Better resource utilization

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install celery redis flower kombu
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Start Redis

**Windows (using Docker):**
```bash
docker run -d -p 6379:6379 redis:latest
```

**MacOS:**
```bash
brew install redis
brew services start redis
```

**Ubuntu:**
```bash
sudo apt install redis-server
sudo systemctl start redis
```

### 3. Start the Celery Worker

**Windows:**
```bash
start_worker.bat
```

**Unix/Linux/MacOS:**
```bash
chmod +x start_worker.sh
./start_worker.sh
```

### 4. Start the API Server

```bash
python -m uvicorn main:app --reload
```

### 5. (Optional) Start Flower Dashboard

```bash
start_flower.bat
# Access at http://localhost:5555
```

## API Endpoints

### Submit Jobs

| Endpoint | Description | Estimated Time |
|----------|-------------|----------------|
| `POST /jobs/submit/binary-classify` | Binary lesion detection | ~5 seconds |
| `POST /jobs/submit/full-classify` | Full multi-model classification | ~15 seconds |
| `POST /jobs/submit/dermoscopy` | Dermoscopy feature analysis | ~10 seconds |
| `POST /jobs/submit/burn-classify` | Burn severity classification | ~8 seconds |
| `POST /jobs/submit/histopathology` | Tissue analysis | ~20 seconds |
| `POST /jobs/submit/batch-skin-check` | Multiple image batch | ~10s per image |

### Check Status

| Endpoint | Description |
|----------|-------------|
| `GET /jobs/status/{job_id}` | Check job status and progress |
| `GET /jobs/result/{job_id}` | Get job result (with optional wait) |
| `DELETE /jobs/cancel/{job_id}` | Cancel pending/running job |
| `GET /jobs/list` | List recent jobs |
| `GET /jobs/health` | Check queue system health |
| `GET /jobs/stats` | Get processing statistics |

## Usage Examples

### Submit a Full Classification Job

```python
import requests

# Submit job
response = requests.post(
    "http://localhost:8000/jobs/submit/full-classify",
    files={"file": open("skin_image.jpg", "rb")},
    data={
        "body_location": "arm",
        "enable_multimodal": True
    },
    headers={"Authorization": f"Bearer {token}"}
)

job_data = response.json()
job_id = job_data["job_id"]
print(f"Job submitted: {job_id}")
print(f"Poll URL: {job_data['poll_url']}")
```

### Poll for Results

```python
import time

while True:
    status_response = requests.get(
        f"http://localhost:8000/jobs/status/{job_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    status = status_response.json()

    if status["status"] == "SUCCESS":
        print("Analysis complete!")
        print(status["result"])
        break
    elif status["status"] == "FAILURE":
        print(f"Analysis failed: {status['error']}")
        break
    elif status["status"] == "PROGRESS":
        progress = status.get("progress", {})
        print(f"Progress: {progress.get('percent', 0)}% - {progress.get('status', '')}")

    time.sleep(2)  # Poll every 2 seconds
```

### Wait for Result (Blocking)

```python
# Wait up to 30 seconds for result
result_response = requests.get(
    f"http://localhost:8000/jobs/result/{job_id}?wait=true&timeout=30",
    headers={"Authorization": f"Bearer {token}"}
)
print(result_response.json())
```

## Configuration

Environment variables in `.env`:

```bash
# Redis connection
REDIS_URL=redis://localhost:6379/0

# Celery settings
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Task timeouts (seconds)
CELERY_TASK_SOFT_TIME_LIMIT=120    # Warning at 2 minutes
CELERY_TASK_TIME_LIMIT=180         # Hard kill at 3 minutes

# Worker settings
CELERY_WORKER_CONCURRENCY=2        # Limit for GPU memory
CELERY_WORKER_PREFETCH_MULTIPLIER=1
```

## Task Queues

The system uses three queues with different priorities:

| Queue | Priority | Tasks |
|-------|----------|-------|
| `default` | Normal | General tasks |
| `analysis` | High | Classification tasks |
| `batch` | Low | Batch processing |

Start workers for specific queues:
```bash
# Analysis queue only (for dedicated ML worker)
start_worker.bat analysis

# Batch queue only
start_worker.bat batch
```

## Monitoring

### Flower Dashboard

Access at `http://localhost:5555` to see:
- Active/completed tasks
- Worker status
- Task execution times
- Error rates

### Health Check

```bash
curl http://localhost:8000/jobs/health
```

Response:
```json
{
  "redis_connected": true,
  "workers_available": 2,
  "workers": ["celery@worker1"],
  "queues": {
    "default": 0,
    "analysis": 3,
    "batch": 1
  },
  "status": "healthy"
}
```

## Troubleshooting

### Redis Connection Failed

```
ERROR: Redis is not running!
```

Solution: Start Redis server (see Quick Start section).

### No Workers Available

```json
{"status": "degraded", "warning": "No workers available"}
```

Solution: Start at least one Celery worker.

### Task Timeout

Tasks have a 3-minute hard timeout. For batch processing with many images:
- Split into smaller batches (max 50 images)
- Increase `CELERY_TASK_TIME_LIMIT`

### Memory Issues

ML models use significant GPU memory. If workers crash:
- Reduce `CELERY_WORKER_CONCURRENCY` to 1
- Enable `worker_max_tasks_per_child=50` (auto-restart workers)

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│   Redis     │
│  (Mobile)   │     │   Server    │     │  (Broker)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                           ┌───────────────────┼───────────────────┐
                           │                   │                   │
                    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
                    │   Worker 1  │     │   Worker 2  │     │   Worker N  │
                    │  (GPU/CPU)  │     │  (GPU/CPU)  │     │  (GPU/CPU)  │
                    └─────────────┘     └─────────────┘     └─────────────┘
                           │                   │                   │
                           └───────────────────┼───────────────────┘
                                               │
                                        ┌──────▼──────┐
                                        │   Redis     │
                                        │  (Results)  │
                                        └─────────────┘
```

## Files

| File | Description |
|------|-------------|
| `celery_app.py` | Celery configuration |
| `tasks.py` | Task definitions |
| `routers/jobs_router.py` | API endpoints |
| `start_worker.bat` | Windows worker script |
| `start_worker.sh` | Unix worker script |
| `start_flower.bat` | Monitoring dashboard |

## Comparison: Sync vs Async

| Feature | Sync (`/full_classify/`) | Async (`/jobs/submit/full-classify`) |
|---------|--------------------------|--------------------------------------|
| Response | Wait for completion | Immediate with job_id |
| Timeout risk | High on slow connections | None |
| Progress tracking | No | Yes |
| Retry on failure | No | Automatic |
| Cancel support | No | Yes |
| Best for | Fast connections | Slow/mobile connections |

Both endpoints produce identical results - choose based on your use case.
