"""
Celery Application Configuration

This module configures Celery for asynchronous task processing.
Used primarily for long-running ML analysis operations.

Usage:
    # Start worker
    celery -A celery_app worker --loglevel=info --concurrency=2

    # Start Flower monitoring (optional)
    celery -A celery_app flower --port=5555
"""

from celery import Celery
import config

# Create Celery app
celery_app = Celery(
    "skin_classifier",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
    include=["tasks"]  # Include tasks module
)

# Configure Celery
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task result expiration
    result_expires=config.CELERY_TASK_RESULT_EXPIRES,

    # Task time limits
    task_soft_time_limit=config.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=config.CELERY_TASK_TIME_LIMIT,

    # Worker settings
    worker_concurrency=config.CELERY_WORKER_CONCURRENCY,
    worker_prefetch_multiplier=config.CELERY_WORKER_PREFETCH_MULTIPLIER,

    # Task routing for priority queues
    task_queues={
        "default": {
            "exchange": "default",
            "routing_key": "default",
        },
        "analysis": {
            "exchange": "analysis",
            "routing_key": "analysis.#",
        },
        "batch": {
            "exchange": "batch",
            "routing_key": "batch.#",
        },
    },

    # Default queue
    task_default_queue="default",

    # Route tasks to appropriate queues
    task_routes={
        "tasks.full_classify_task": {"queue": "analysis"},
        "tasks.binary_classify_task": {"queue": "analysis"},
        "tasks.multimodal_analyze_task": {"queue": "analysis"},
        "tasks.dermoscopy_analyze_task": {"queue": "analysis"},
        "tasks.burn_classify_task": {"queue": "analysis"},
        "tasks.histopathology_analyze_task": {"queue": "analysis"},
        "tasks.batch_skin_check_task": {"queue": "batch"},
    },

    # Track task state
    task_track_started=True,

    # Retry settings
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,

    # Result backend settings for Redis
    result_backend_transport_options={
        "retry_policy": {
            "timeout": 5.0
        }
    },

    # Worker will restart after processing N tasks (memory management for ML models)
    worker_max_tasks_per_child=50,
)


# Task state constants
class TaskStatus:
    """Task status constants."""
    PENDING = "PENDING"
    STARTED = "STARTED"
    PROGRESS = "PROGRESS"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    REVOKED = "REVOKED"


def get_task_info(task_id: str) -> dict:
    """
    Get information about a task by ID.

    Args:
        task_id: The Celery task ID

    Returns:
        dict with task status and result
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    info = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
    }

    if result.ready():
        if result.successful():
            info["result"] = result.result
        else:
            info["error"] = str(result.result)
    elif result.status == "PROGRESS":
        info["progress"] = result.info

    return info


def revoke_task(task_id: str, terminate: bool = False) -> bool:
    """
    Cancel a pending or running task.

    Args:
        task_id: The Celery task ID
        terminate: If True, forcefully terminate the task

    Returns:
        True if revocation was sent
    """
    celery_app.control.revoke(task_id, terminate=terminate)
    return True


# Autodiscover tasks if needed
# celery_app.autodiscover_tasks(['routers'])

if __name__ == "__main__":
    celery_app.start()
