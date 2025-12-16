"""
Monitoring Router - Model Health and Alert Management

Provides admin endpoints for:
- Viewing ML model health metrics
- Managing system alerts
- Configuring alert thresholds
- Exporting monitoring data

Requires admin/professional user access for most endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel

from database import get_db, User, SystemAlert
from auth import get_current_active_user
from model_monitoring import (
    monitor,
    AlertSeverity,
    AlertType,
    ModelHealthMetrics,
    ModelAlert
)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ModelHealthResponse(BaseModel):
    """Response model for model health."""
    model_name: str
    status: str
    total_inferences: int
    successful_inferences: int
    failed_inferences: int
    error_rate: float
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    avg_confidence: float
    min_confidence: float
    inferences_last_hour: int
    errors_last_hour: int
    last_inference_at: Optional[str]
    last_error_at: Optional[str]
    last_error_message: Optional[str]
    uptime_percentage: float
    active_alerts: List[str]


class AlertResponse(BaseModel):
    """Response model for alerts."""
    id: str
    timestamp: str
    alert_type: str
    severity: str
    model_name: str
    title: str
    message: str
    metrics: dict
    resolved: bool
    acknowledged: bool


class SystemSummaryResponse(BaseModel):
    """Response model for system summary."""
    overall_status: str
    total_models: int
    models_by_status: dict
    active_alerts: int
    alerts_by_severity: dict
    critical_alerts: int
    timestamp: str


class AcknowledgeRequest(BaseModel):
    """Request to acknowledge an alert."""
    notes: Optional[str] = None


class ResolveRequest(BaseModel):
    """Request to resolve an alert."""
    resolution_notes: Optional[str] = None


class ThresholdUpdateRequest(BaseModel):
    """Request to update alert thresholds."""
    error_rate_warning: Optional[float] = None
    error_rate_error: Optional[float] = None
    error_rate_critical: Optional[float] = None
    inference_time_warning: Optional[int] = None
    inference_time_error: Optional[int] = None
    inference_time_critical: Optional[int] = None
    low_confidence_warning: Optional[float] = None
    low_confidence_error: Optional[float] = None
    consecutive_failures_warning: Optional[int] = None
    consecutive_failures_error: Optional[int] = None
    consecutive_failures_critical: Optional[int] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def require_admin_or_professional(user: User) -> None:
    """Ensure user has admin or professional access."""
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    # Allow if user is admin or professional account type
    is_admin = getattr(user, 'is_superuser', False) or getattr(user, 'is_admin', False)
    is_professional = getattr(user, 'account_type', '') == 'professional'

    if not (is_admin or is_professional):
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires admin or professional access"
        )


def health_to_response(health: ModelHealthMetrics) -> ModelHealthResponse:
    """Convert ModelHealthMetrics to response model."""
    return ModelHealthResponse(
        model_name=health.model_name,
        status=health.status,
        total_inferences=health.total_inferences,
        successful_inferences=health.successful_inferences,
        failed_inferences=health.failed_inferences,
        error_rate=health.error_rate,
        avg_inference_time_ms=health.avg_inference_time_ms,
        p95_inference_time_ms=health.p95_inference_time_ms,
        p99_inference_time_ms=health.p99_inference_time_ms,
        avg_confidence=health.avg_confidence,
        min_confidence=health.min_confidence,
        inferences_last_hour=health.inferences_last_hour,
        errors_last_hour=health.errors_last_hour,
        last_inference_at=health.last_inference_at.isoformat() if health.last_inference_at else None,
        last_error_at=health.last_error_at.isoformat() if health.last_error_at else None,
        last_error_message=health.last_error_message,
        uptime_percentage=health.uptime_percentage,
        active_alerts=health.active_alerts
    )


def alert_to_response(alert: ModelAlert) -> AlertResponse:
    """Convert ModelAlert to response model."""
    return AlertResponse(
        id=alert.id,
        timestamp=alert.timestamp.isoformat(),
        alert_type=alert.alert_type.value,
        severity=alert.severity.value,
        model_name=alert.model_name,
        title=alert.title,
        message=alert.message,
        metrics=alert.metrics,
        resolved=alert.resolved,
        acknowledged=alert.acknowledged
    )


def persist_alert_to_db(alert: ModelAlert, db: Session) -> SystemAlert:
    """Persist an in-memory alert to the database."""
    import hashlib

    # Create fingerprint for deduplication
    fingerprint = hashlib.md5(
        f"{alert.alert_type.value}:{alert.model_name}".encode()
    ).hexdigest()

    # Check for existing alert with same fingerprint
    existing = db.query(SystemAlert).filter(
        SystemAlert.fingerprint == fingerprint,
        SystemAlert.status == "active"
    ).first()

    if existing:
        # Update existing alert
        existing.occurrence_count += 1
        existing.last_occurrence = datetime.utcnow()
        existing.severity = alert.severity.value
        existing.title = alert.title
        existing.message = alert.message
        existing.metrics = alert.metrics
        db.commit()
        return existing

    # Create new alert
    db_alert = SystemAlert(
        alert_id=alert.id,
        alert_type=alert.alert_type.value,
        severity=alert.severity.value,
        model_name=alert.model_name,
        title=alert.title,
        message=alert.message,
        metrics=alert.metrics,
        fingerprint=fingerprint,
        status="active"
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert


# =============================================================================
# MODEL HEALTH ENDPOINTS
# =============================================================================

@router.get("/health/summary", response_model=SystemSummaryResponse)
async def get_system_summary(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get overall system health summary.

    Returns status of all models and alert counts.
    Available to all authenticated users.
    """
    summary = monitor.get_system_summary()
    return SystemSummaryResponse(**summary)


@router.get("/health/models")
async def get_all_models_health(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get health metrics for all monitored models.

    Returns detailed metrics for each ML model including
    error rates, inference times, and confidence levels.
    """
    require_admin_or_professional(current_user)

    all_health = monitor.get_all_models_health()
    return {
        name: health_to_response(health) if health else None
        for name, health in all_health.items()
    }


@router.get("/health/models/{model_name}", response_model=ModelHealthResponse)
async def get_model_health(
    model_name: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get health metrics for a specific model.

    Args:
        model_name: Name of the model (binary_model, lesion_model, etc.)
    """
    require_admin_or_professional(current_user)

    health = monitor.get_model_health(model_name)
    if not health:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found or has no data"
        )

    return health_to_response(health)


@router.get("/metrics/export")
async def export_metrics(
    current_user: User = Depends(get_current_active_user),
):
    """
    Export all monitoring metrics.

    Returns metrics in a format suitable for external
    monitoring systems (Prometheus, Grafana, etc.).
    """
    require_admin_or_professional(current_user)
    return monitor.export_metrics()


# =============================================================================
# ALERT MANAGEMENT ENDPOINTS
# =============================================================================

@router.get("/alerts/active")
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get all active (unresolved) alerts.

    Args:
        severity: Filter by severity (info, warning, error, critical)
        model: Filter by model name
    """
    require_admin_or_professional(current_user)

    alerts = monitor.get_active_alerts()

    # Apply filters
    if severity:
        try:
            sev = AlertSeverity(severity)
            alerts = [a for a in alerts if a.severity == sev]
        except ValueError:
            pass

    if model:
        alerts = [a for a in alerts if a.model_name == model]

    return {
        "total": len(alerts),
        "alerts": [alert_to_response(a) for a in alerts]
    }


@router.get("/alerts/history")
async def get_alert_history(
    limit: int = Query(50, le=200),
    severity: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    days: int = Query(7, description="Number of days of history"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get alert history from the database.

    Args:
        limit: Maximum number of alerts to return
        severity: Filter by severity
        model: Filter by model name
        days: Number of days of history to retrieve
    """
    require_admin_or_professional(current_user)

    cutoff = datetime.utcnow() - timedelta(days=days)

    query = db.query(SystemAlert).filter(
        SystemAlert.created_at >= cutoff
    )

    if severity:
        query = query.filter(SystemAlert.severity == severity)

    if model:
        query = query.filter(SystemAlert.model_name == model)

    alerts = query.order_by(desc(SystemAlert.created_at)).limit(limit).all()

    return {
        "total": len(alerts),
        "alerts": [
            {
                "id": a.id,
                "alert_id": a.alert_id,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "model_name": a.model_name,
                "title": a.title,
                "message": a.message,
                "status": a.status,
                "acknowledged": a.acknowledged,
                "resolved": a.resolved,
                "occurrence_count": a.occurrence_count,
                "first_occurrence": a.first_occurrence.isoformat() if a.first_occurrence else None,
                "last_occurrence": a.last_occurrence.isoformat() if a.last_occurrence else None,
                "created_at": a.created_at.isoformat() if a.created_at else None
            }
            for a in alerts
        ]
    }


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    request: AcknowledgeRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Acknowledge an alert.

    Acknowledging an alert indicates that an admin is aware of
    the issue and is working on it.
    """
    require_admin_or_professional(current_user)

    # Update in-memory alert
    success = monitor.acknowledge_alert(alert_id, current_user.username)

    # Update database alert
    db_alert = db.query(SystemAlert).filter(
        SystemAlert.alert_id == alert_id
    ).first()

    if db_alert:
        db_alert.acknowledged = True
        db_alert.acknowledged_by = current_user.username
        db_alert.acknowledged_at = datetime.utcnow()
        db.commit()
    elif not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {
        "message": "Alert acknowledged",
        "alert_id": alert_id,
        "acknowledged_by": current_user.username
    }


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    request: ResolveRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Resolve an alert.

    Resolving an alert indicates that the issue has been fixed
    and the alert should no longer be active.
    """
    require_admin_or_professional(current_user)

    # Update in-memory alert
    success = monitor.resolve_alert(alert_id)

    # Update database alert
    db_alert = db.query(SystemAlert).filter(
        SystemAlert.alert_id == alert_id
    ).first()

    if db_alert:
        db_alert.status = "resolved"
        db_alert.resolved = True
        db_alert.resolved_at = datetime.utcnow()
        db_alert.resolved_by = current_user.username
        db_alert.resolution_notes = request.resolution_notes
        db.commit()
    elif not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {
        "message": "Alert resolved",
        "alert_id": alert_id,
        "resolved_by": current_user.username
    }


@router.get("/alerts/stats")
async def get_alert_stats(
    days: int = Query(30, description="Number of days for statistics"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get alert statistics.

    Returns counts of alerts by severity, type, and model
    over the specified time period.
    """
    require_admin_or_professional(current_user)

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Count by severity
    severity_counts = db.query(
        SystemAlert.severity,
        func.count(SystemAlert.id)
    ).filter(
        SystemAlert.created_at >= cutoff
    ).group_by(SystemAlert.severity).all()

    # Count by type
    type_counts = db.query(
        SystemAlert.alert_type,
        func.count(SystemAlert.id)
    ).filter(
        SystemAlert.created_at >= cutoff
    ).group_by(SystemAlert.alert_type).all()

    # Count by model
    model_counts = db.query(
        SystemAlert.model_name,
        func.count(SystemAlert.id)
    ).filter(
        SystemAlert.created_at >= cutoff,
        SystemAlert.model_name.isnot(None)
    ).group_by(SystemAlert.model_name).all()

    # Resolution stats
    total_alerts = db.query(SystemAlert).filter(
        SystemAlert.created_at >= cutoff
    ).count()

    resolved_alerts = db.query(SystemAlert).filter(
        SystemAlert.created_at >= cutoff,
        SystemAlert.resolved == True
    ).count()

    return {
        "period_days": days,
        "total_alerts": total_alerts,
        "resolved_alerts": resolved_alerts,
        "resolution_rate": round(resolved_alerts / total_alerts * 100, 1) if total_alerts > 0 else 0,
        "by_severity": {s: c for s, c in severity_counts},
        "by_type": {t: c for t, c in type_counts},
        "by_model": {m: c for m, c in model_counts}
    }


# =============================================================================
# THRESHOLD CONFIGURATION
# =============================================================================

@router.get("/thresholds")
async def get_thresholds(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get current alert thresholds.

    Returns the current configuration for when alerts
    are triggered.
    """
    require_admin_or_professional(current_user)

    t = monitor.thresholds
    return {
        "error_rate": {
            "warning": t.ERROR_RATE_WARNING,
            "error": t.ERROR_RATE_ERROR,
            "critical": t.ERROR_RATE_CRITICAL
        },
        "inference_time_ms": {
            "warning": t.INFERENCE_TIME_WARNING,
            "error": t.INFERENCE_TIME_ERROR,
            "critical": t.INFERENCE_TIME_CRITICAL
        },
        "low_confidence": {
            "warning": t.LOW_CONFIDENCE_WARNING,
            "error": t.LOW_CONFIDENCE_ERROR
        },
        "consecutive_failures": {
            "warning": t.CONSECUTIVE_FAILURES_WARNING,
            "error": t.CONSECUTIVE_FAILURES_ERROR,
            "critical": t.CONSECUTIVE_FAILURES_CRITICAL
        },
        "rate_window_seconds": t.RATE_WINDOW_SECONDS,
        "min_samples_for_alert": t.MIN_SAMPLES_FOR_ALERT
    }


@router.put("/thresholds")
async def update_thresholds(
    request: ThresholdUpdateRequest,
    current_user: User = Depends(get_current_active_user),
):
    """
    Update alert thresholds.

    Allows customizing when alerts are triggered.
    Requires admin access.
    """
    # Require admin for threshold changes
    is_admin = getattr(current_user, 'is_superuser', False) or getattr(current_user, 'is_admin', False)
    if not is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only admins can modify alert thresholds"
        )

    t = monitor.thresholds

    # Update only provided values
    if request.error_rate_warning is not None:
        t.ERROR_RATE_WARNING = request.error_rate_warning
    if request.error_rate_error is not None:
        t.ERROR_RATE_ERROR = request.error_rate_error
    if request.error_rate_critical is not None:
        t.ERROR_RATE_CRITICAL = request.error_rate_critical

    if request.inference_time_warning is not None:
        t.INFERENCE_TIME_WARNING = request.inference_time_warning
    if request.inference_time_error is not None:
        t.INFERENCE_TIME_ERROR = request.inference_time_error
    if request.inference_time_critical is not None:
        t.INFERENCE_TIME_CRITICAL = request.inference_time_critical

    if request.low_confidence_warning is not None:
        t.LOW_CONFIDENCE_WARNING = request.low_confidence_warning
    if request.low_confidence_error is not None:
        t.LOW_CONFIDENCE_ERROR = request.low_confidence_error

    if request.consecutive_failures_warning is not None:
        t.CONSECUTIVE_FAILURES_WARNING = request.consecutive_failures_warning
    if request.consecutive_failures_error is not None:
        t.CONSECUTIVE_FAILURES_ERROR = request.consecutive_failures_error
    if request.consecutive_failures_critical is not None:
        t.CONSECUTIVE_FAILURES_CRITICAL = request.consecutive_failures_critical

    return {
        "message": "Thresholds updated",
        "thresholds": await get_thresholds(current_user)
    }


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@router.post("/test-alert")
async def create_test_alert(
    severity: str = Query("warning", description="Alert severity"),
    model: str = Query("test_model", description="Model name"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a test alert for testing the alerting system.

    This creates a real alert that will appear in the monitoring
    dashboard. Useful for testing notifications and workflows.
    """
    require_admin_or_professional(current_user)

    # Validate severity
    try:
        sev = AlertSeverity(severity)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid severity. Must be one of: info, warning, error, critical"
        )

    # Create test alert through monitor
    monitor.register_model(model)

    # Record a failure to trigger alert
    for _ in range(monitor.thresholds.CONSECUTIVE_FAILURES_WARNING):
        alert = monitor.record_inference(
            model_name=model,
            inference_time_ms=0,
            success=False,
            error=f"Test failure triggered by {current_user.username}"
        )

    if alert:
        # Persist to database
        persist_alert_to_db(alert, db)

        return {
            "message": "Test alert created",
            "alert": alert_to_response(alert)
        }

    return {
        "message": "Alert thresholds not met",
        "note": "Try increasing consecutive failures or check current thresholds"
    }


@router.delete("/alerts/clear-test")
async def clear_test_alerts(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Clear test alerts and data.

    Removes test model data from monitoring.
    Useful for cleaning up after testing.
    """
    require_admin_or_professional(current_user)

    # Clear test model history
    monitor.clear_history("test_model")

    # Remove test alerts from database
    deleted = db.query(SystemAlert).filter(
        SystemAlert.model_name == "test_model"
    ).delete()
    db.commit()

    return {
        "message": "Test data cleared",
        "alerts_deleted": deleted
    }
