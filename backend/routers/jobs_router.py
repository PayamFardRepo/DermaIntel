"""
Jobs Router - Asynchronous Job Queue Endpoints

Provides endpoints for:
- Submitting analysis jobs to the queue
- Checking job status and progress
- Retrieving job results
- Canceling pending jobs

This router enables non-blocking analysis for slow connections
by offloading ML processing to Celery workers.
"""

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
import uuid
import json

from database import get_db, User, AnalysisHistory
from auth import get_current_active_user
import config

router = APIRouter(prefix="/jobs", tags=["Jobs"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class JobSubmitResponse(BaseModel):
    """Response when a job is submitted."""
    job_id: str
    status: str
    message: str
    poll_url: str
    estimated_time_seconds: Optional[int] = None


class JobStatusResponse(BaseModel):
    """Response for job status check."""
    job_id: str
    status: str
    progress: Optional[dict] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    """Response for listing jobs."""
    jobs: List[dict]
    total: int


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_uploaded_image(contents: bytes, filename: str, user_id: int) -> str:
    """Save uploaded image to disk and return path."""
    uploads_dir = Path(config.UPLOAD_DIR)
    uploads_dir.mkdir(exist_ok=True)

    ext = Path(filename).suffix if filename else ".jpg"
    unique_filename = f"{user_id}_{uuid.uuid4().hex}{ext}"
    filepath = uploads_dir / unique_filename

    with open(filepath, "wb") as f:
        f.write(contents)

    return str(filepath)


def get_celery_task_status(task_id: str) -> dict:
    """Get status of a Celery task."""
    from celery.result import AsyncResult
    from celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    status_info = {
        "job_id": task_id,
        "status": result.status,
        "ready": result.ready(),
    }

    if result.status == "PROGRESS":
        status_info["progress"] = result.info
    elif result.ready():
        if result.successful():
            status_info["result"] = result.result
            status_info["status"] = "SUCCESS"
        else:
            status_info["error"] = str(result.result)
            status_info["status"] = "FAILURE"

    return status_info


# =============================================================================
# JOB SUBMISSION ENDPOINTS
# =============================================================================

@router.post("/submit/binary-classify", response_model=JobSubmitResponse)
async def submit_binary_classify_job(
    file: UploadFile = File(...),
    save_to_db: bool = Form(True),
    body_location: str = Form(None),
    body_sublocation: str = Form(None),
    body_side: str = Form(None),
    body_map_x: float = Form(None),
    body_map_y: float = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Submit a binary classification job to the queue.

    This is the async version of /upload/ endpoint.
    Returns immediately with a job_id that can be polled for results.
    """
    from tasks import binary_classify_task

    # Read and save image
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    image_path = save_uploaded_image(contents, file.filename, current_user.id)

    # Submit task to Celery
    task = binary_classify_task.delay(
        image_path=image_path,
        user_id=current_user.id,
        filename=file.filename,
        save_to_db=save_to_db,
        body_location=body_location,
        body_sublocation=body_sublocation,
        body_side=body_side,
        body_map_x=body_map_x,
        body_map_y=body_map_y,
    )

    return JobSubmitResponse(
        job_id=task.id,
        status="PENDING",
        message="Job submitted successfully",
        poll_url=f"/jobs/status/{task.id}",
        estimated_time_seconds=5
    )


@router.post("/submit/full-classify", response_model=JobSubmitResponse)
async def submit_full_classify_job(
    file: UploadFile = File(...),
    save_to_db: bool = Form(True),
    body_location: str = Form(None),
    body_sublocation: str = Form(None),
    body_side: str = Form(None),
    body_map_x: float = Form(None),
    body_map_y: float = Form(None),
    condition_hint: str = Form(None),
    clinical_context_json: str = Form(None),
    patient_age: int = Form(None),
    fitzpatrick_skin_type: str = Form(None),
    lesion_duration: str = Form(None),
    has_changed_recently: bool = Form(None),
    is_new_lesion: bool = Form(None),
    symptoms_itching: bool = Form(None),
    symptoms_bleeding: bool = Form(None),
    symptoms_pain: bool = Form(None),
    personal_history_melanoma: bool = Form(None),
    personal_history_skin_cancer: bool = Form(None),
    family_history_melanoma: bool = Form(None),
    family_history_skin_cancer: bool = Form(None),
    history_severe_sunburns: bool = Form(None),
    uses_tanning_beds: bool = Form(None),
    immunosuppressed: bool = Form(None),
    many_moles: bool = Form(None),
    enable_multimodal: bool = Form(True),
    include_labs: bool = Form(True),
    include_history: bool = Form(True),
    lesion_group_id: int = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Submit a full classification job to the queue.

    This is the async version of /full_classify/ endpoint.
    Performs comprehensive analysis using multiple models.
    Returns immediately with a job_id that can be polled for results.
    """
    from tasks import full_classify_task

    # Read and save image
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    image_path = save_uploaded_image(contents, file.filename, current_user.id)

    # Build clinical context
    clinical_context = {}
    if clinical_context_json:
        try:
            clinical_context = json.loads(clinical_context_json)
        except:
            pass

    if patient_age is not None:
        clinical_context["patient_age"] = patient_age
    if fitzpatrick_skin_type is not None:
        clinical_context["fitzpatrick_skin_type"] = fitzpatrick_skin_type
    if lesion_duration is not None:
        clinical_context["lesion_duration"] = lesion_duration
    if has_changed_recently is not None:
        clinical_context["has_changed_recently"] = has_changed_recently
    if body_location is not None:
        clinical_context["body_location"] = body_location

    symptoms = {}
    if symptoms_itching is not None:
        symptoms["itching"] = symptoms_itching
    if symptoms_bleeding is not None:
        symptoms["bleeding"] = symptoms_bleeding
    if symptoms_pain is not None:
        symptoms["pain"] = symptoms_pain
    if symptoms:
        clinical_context["symptoms"] = symptoms

    if personal_history_melanoma is not None:
        clinical_context["personal_history_melanoma"] = personal_history_melanoma
    if personal_history_skin_cancer is not None:
        clinical_context["personal_history_skin_cancer"] = personal_history_skin_cancer
    if family_history_melanoma is not None:
        clinical_context["family_history_melanoma"] = family_history_melanoma
    if family_history_skin_cancer is not None:
        clinical_context["family_history_skin_cancer"] = family_history_skin_cancer
    if history_severe_sunburns is not None:
        clinical_context["history_severe_sunburns"] = history_severe_sunburns
    if uses_tanning_beds is not None:
        clinical_context["uses_tanning_beds"] = uses_tanning_beds
    if immunosuppressed is not None:
        clinical_context["immunosuppressed"] = immunosuppressed
    if many_moles is not None:
        clinical_context["many_moles"] = many_moles
    if is_new_lesion is not None:
        clinical_context["is_new_lesion"] = is_new_lesion

    # Submit task to Celery
    task = full_classify_task.delay(
        image_path=image_path,
        user_id=current_user.id,
        filename=file.filename,
        save_to_db=save_to_db,
        body_location=body_location,
        body_sublocation=body_sublocation,
        body_side=body_side,
        body_map_x=body_map_x,
        body_map_y=body_map_y,
        clinical_context=clinical_context,
        enable_multimodal=enable_multimodal,
        include_labs=include_labs,
        include_history=include_history,
        lesion_group_id=lesion_group_id,
    )

    return JobSubmitResponse(
        job_id=task.id,
        status="PENDING",
        message="Full classification job submitted",
        poll_url=f"/jobs/status/{task.id}",
        estimated_time_seconds=15
    )


@router.post("/submit/dermoscopy", response_model=JobSubmitResponse)
async def submit_dermoscopy_job(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Submit a dermoscopy analysis job to the queue.

    Analyzes dermoscopic features and calculates 7-Point Checklist scores.
    """
    from tasks import dermoscopy_analyze_task

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    image_path = save_uploaded_image(contents, file.filename, current_user.id)

    task = dermoscopy_analyze_task.delay(
        image_path=image_path,
        user_id=current_user.id,
        filename=file.filename,
    )

    return JobSubmitResponse(
        job_id=task.id,
        status="PENDING",
        message="Dermoscopy analysis job submitted",
        poll_url=f"/jobs/status/{task.id}",
        estimated_time_seconds=10
    )


@router.post("/submit/burn-classify", response_model=JobSubmitResponse)
async def submit_burn_classify_job(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Submit a burn classification job to the queue.

    Classifies burn severity and provides treatment recommendations.
    """
    from tasks import burn_classify_task

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    image_path = save_uploaded_image(contents, file.filename, current_user.id)

    task = burn_classify_task.delay(
        image_path=image_path,
        user_id=current_user.id,
        filename=file.filename,
    )

    return JobSubmitResponse(
        job_id=task.id,
        status="PENDING",
        message="Burn classification job submitted",
        poll_url=f"/jobs/status/{task.id}",
        estimated_time_seconds=8
    )


@router.post("/submit/histopathology", response_model=JobSubmitResponse)
async def submit_histopathology_job(
    file: UploadFile = File(...),
    tissue_type: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Submit a histopathology analysis job to the queue.

    Analyzes biopsy slide images for tissue classification and malignancy assessment.
    """
    from tasks import histopathology_analyze_task

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file received")

    image_path = save_uploaded_image(contents, file.filename, current_user.id)

    task = histopathology_analyze_task.delay(
        image_path=image_path,
        user_id=current_user.id,
        filename=file.filename,
        tissue_type=tissue_type,
    )

    return JobSubmitResponse(
        job_id=task.id,
        status="PENDING",
        message="Histopathology analysis job submitted",
        poll_url=f"/jobs/status/{task.id}",
        estimated_time_seconds=20
    )


@router.post("/submit/batch-skin-check", response_model=JobSubmitResponse)
async def submit_batch_skin_check_job(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Submit a batch skin check job to the queue.

    Processes multiple images for full-body skin examination.
    This is ideal for periodic skin monitoring.
    """
    from tasks import batch_skin_check_task
    from database import BatchSkinCheck

    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")

    # Create batch check record
    batch_record = BatchSkinCheck(
        user_id=current_user.id,
        status="processing",
        total_images=len(files),
        created_at=datetime.utcnow()
    )
    db.add(batch_record)
    db.commit()
    db.refresh(batch_record)

    # Save all images
    image_paths = []
    for file in files:
        contents = await file.read()
        if len(contents) > 0:
            image_path = save_uploaded_image(contents, file.filename, current_user.id)
            image_paths.append(image_path)

    # Submit batch task
    task = batch_skin_check_task.delay(
        image_paths=image_paths,
        user_id=current_user.id,
        check_id=batch_record.id,
    )

    # Update batch record with task ID
    batch_record.task_id = task.id
    db.commit()

    return JobSubmitResponse(
        job_id=task.id,
        status="PENDING",
        message=f"Batch skin check submitted ({len(image_paths)} images)",
        poll_url=f"/jobs/status/{task.id}",
        estimated_time_seconds=len(image_paths) * 10
    )


# =============================================================================
# JOB STATUS AND MANAGEMENT ENDPOINTS
# =============================================================================

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get the status of a submitted job.

    Poll this endpoint to check if your job has completed.
    Returns progress information for running jobs.
    """
    status_info = get_celery_task_status(job_id)

    return JobStatusResponse(
        job_id=job_id,
        status=status_info["status"],
        progress=status_info.get("progress"),
        result=status_info.get("result"),
        error=status_info.get("error"),
    )


@router.get("/result/{job_id}")
async def get_job_result(
    job_id: str,
    wait: bool = False,
    timeout: int = 30,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get the result of a completed job.

    Args:
        job_id: The job ID returned when submitting
        wait: If True, wait for the job to complete (up to timeout)
        timeout: Maximum seconds to wait if wait=True

    Returns:
        The job result if completed, or current status if still running
    """
    from celery.result import AsyncResult
    from celery_app import celery_app

    result = AsyncResult(job_id, app=celery_app)

    if wait and not result.ready():
        try:
            # Wait for result with timeout
            result.get(timeout=timeout)
        except Exception:
            pass

    if result.ready():
        if result.successful():
            return {
                "job_id": job_id,
                "status": "SUCCESS",
                "result": result.result
            }
        else:
            return {
                "job_id": job_id,
                "status": "FAILURE",
                "error": str(result.result)
            }
    else:
        status_info = get_celery_task_status(job_id)
        return {
            "job_id": job_id,
            "status": status_info["status"],
            "progress": status_info.get("progress"),
            "message": "Job is still processing"
        }


@router.delete("/cancel/{job_id}")
async def cancel_job(
    job_id: str,
    terminate: bool = False,
    current_user: User = Depends(get_current_active_user),
):
    """
    Cancel a pending or running job.

    Args:
        job_id: The job ID to cancel
        terminate: If True, forcefully terminate running task

    Note: Already completed jobs cannot be canceled.
    """
    from celery_app import revoke_task, celery_app
    from celery.result import AsyncResult

    result = AsyncResult(job_id, app=celery_app)

    if result.ready():
        raise HTTPException(
            status_code=400,
            detail="Cannot cancel a completed job"
        )

    revoke_task(job_id, terminate=terminate)

    return {
        "job_id": job_id,
        "status": "REVOKED",
        "message": "Job cancellation requested"
    }


@router.get("/list", response_model=JobListResponse)
async def list_recent_jobs(
    limit: int = 20,
    status: str = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List recent jobs for the current user.

    This queries analysis records that have task IDs associated.
    """
    # Get recent analyses that may have been created via jobs
    query = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id
    ).order_by(AnalysisHistory.created_at.desc())

    if limit:
        query = query.limit(limit)

    analyses = query.all()

    jobs = []
    for analysis in analyses:
        jobs.append({
            "analysis_id": analysis.id,
            "analysis_type": analysis.analysis_type,
            "predicted_class": analysis.predicted_class,
            "confidence": analysis.lesion_confidence or analysis.binary_confidence,
            "risk_level": analysis.risk_level,
            "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            "processing_time": analysis.processing_time_seconds,
            "status": "completed"
        })

    return JobListResponse(
        jobs=jobs,
        total=len(jobs)
    )


# =============================================================================
# QUEUE HEALTH AND STATISTICS
# =============================================================================

@router.get("/health")
async def get_queue_health(
    current_user: User = Depends(get_current_active_user),
):
    """
    Check the health of the job queue system.

    Returns information about:
    - Redis connection status
    - Worker availability
    - Queue lengths
    """
    from celery_app import celery_app
    import redis

    health_info = {
        "redis_connected": False,
        "workers_available": 0,
        "queues": {},
        "status": "unhealthy"
    }

    # Check Redis connection
    try:
        redis_client = redis.from_url(config.REDIS_URL)
        redis_client.ping()
        health_info["redis_connected"] = True
    except Exception as e:
        health_info["redis_error"] = str(e)
        return health_info

    # Check worker availability
    try:
        inspect = celery_app.control.inspect()
        active_workers = inspect.active()
        if active_workers:
            health_info["workers_available"] = len(active_workers)
            health_info["workers"] = list(active_workers.keys())
    except Exception as e:
        health_info["worker_error"] = str(e)

    # Check queue lengths
    try:
        for queue_name in ["default", "analysis", "batch"]:
            queue_length = redis_client.llen(queue_name)
            health_info["queues"][queue_name] = queue_length
    except Exception:
        pass

    # Determine overall status
    if health_info["redis_connected"] and health_info["workers_available"] > 0:
        health_info["status"] = "healthy"
    elif health_info["redis_connected"]:
        health_info["status"] = "degraded"
        health_info["warning"] = "No workers available"

    return health_info


@router.get("/stats")
async def get_queue_stats(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get statistics about job processing.

    Returns metrics like:
    - Jobs processed today
    - Average processing time
    - Success/failure rates
    """
    from celery_app import celery_app
    from database import get_db, AnalysisHistory
    from datetime import timedelta

    db = next(get_db())
    try:
        today = datetime.utcnow().date()
        today_start = datetime.combine(today, datetime.min.time())

        # Count today's analyses
        total_today = db.query(AnalysisHistory).filter(
            AnalysisHistory.created_at >= today_start
        ).count()

        # Average processing time
        from sqlalchemy import func
        avg_time = db.query(func.avg(AnalysisHistory.processing_time_seconds)).filter(
            AnalysisHistory.created_at >= today_start,
            AnalysisHistory.processing_time_seconds.isnot(None)
        ).scalar() or 0

        # Count by risk level
        high_risk_today = db.query(AnalysisHistory).filter(
            AnalysisHistory.created_at >= today_start,
            AnalysisHistory.risk_level.in_(["high", "very_high"])
        ).count()

        return {
            "jobs_today": total_today,
            "average_processing_time_seconds": round(avg_time, 2),
            "high_risk_cases_today": high_risk_today,
            "date": today.isoformat()
        }
    finally:
        db.close()
