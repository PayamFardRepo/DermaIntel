"""
Audit Logging Helper for Quality Assurance and Legal Documentation
Provides utility functions for logging AI predictions and system events
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from database import AuditLog
import psutil
import platform


def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        return {
            "cpu_usage_percent": cpu_percent,
            "memory_total_mb": memory.total / (1024 * 1024),
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_percent": memory.percent,
            "platform": platform.system(),
            "python_version": platform.python_version()
        }
    except Exception as e:
        return {"error": str(e)}


def calculate_image_hash(image_bytes: bytes) -> str:
    """Calculate SHA-256 hash of image for integrity verification"""
    return hashlib.sha256(image_bytes).hexdigest()


def log_prediction(
    db: Session,
    user_id: Optional[int],
    username: Optional[str],
    session_id: Optional[str],
    model_name: str,
    model_version: str,
    prediction_result: str,
    confidence_score: float,
    prediction_probabilities: Dict[str, float],
    analysis_id: Optional[int] = None,
    image_bytes: Optional[bytes] = None,
    image_metadata: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[float] = None,
    uncertainty_metrics: Optional[Dict[str, Any]] = None,
    quality_passed: Optional[bool] = None,
    quality_issues: Optional[List[str]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    gpu_used: bool = False,
    reliability_score: Optional[float] = None,
    flags: Optional[List[str]] = None,
    consent_obtained: bool = True,
    endpoint: Optional[str] = None,
    http_method: Optional[str] = None,
    request_params: Optional[Dict[str, Any]] = None,
    response_status: int = 200,
    error_occurred: bool = False,
    error_message: Optional[str] = None,
    error_stack_trace: Optional[str] = None
) -> AuditLog:
    """
    Log an AI prediction to the audit trail

    Args:
        db: Database session
        user_id: ID of the user making the request
        username: Username of the user
        session_id: Session identifier
        model_name: Name of the AI model used
        model_version: Version/checkpoint of the model
        prediction_result: Primary predicted class
        confidence_score: Confidence of prediction (0-1)
        prediction_probabilities: Full probability distribution
        analysis_id: ID of the associated analysis record
        image_bytes: Raw image bytes for hash calculation
        image_metadata: Image size, format, quality metrics
        processing_time_ms: Time taken for inference
        uncertainty_metrics: Monte Carlo Dropout uncertainty metrics
        quality_passed: Whether image quality checks passed
        quality_issues: List of detected quality issues
        ip_address: User's IP address
        user_agent: Browser/device information
        gpu_used: Whether GPU was used for inference
        reliability_score: Overall reliability score (0-1)
        flags: Any warnings or flags raised
        consent_obtained: User consent for data processing
        endpoint: API endpoint called
        http_method: GET, POST, etc.
        request_params: Request parameters (sanitized)
        response_status: HTTP response code
        error_occurred: Whether an error occurred
        error_message: Error details if any
        error_stack_trace: Stack trace for debugging

    Returns:
        AuditLog: The created audit log entry
    """

    # Calculate image hash if image bytes provided
    input_data_hash = None
    if image_bytes:
        input_data_hash = calculate_image_hash(image_bytes)

    # Get system resources
    system_resources = get_system_resources()

    # Determine severity based on confidence and errors
    severity = "info"
    if error_occurred:
        severity = "error"
    elif confidence_score < 0.6:
        severity = "warning"
    elif quality_passed is False:
        severity = "warning"

    # Create audit log entry
    audit_log = AuditLog(
        # User and session
        user_id=user_id,
        username=username,
        session_id=session_id,

        # Event classification
        event_type="prediction",
        event_category="ai_inference",
        severity=severity,

        # AI Prediction
        analysis_id=analysis_id,
        model_name=model_name,
        model_version=model_version,
        prediction_result=prediction_result,
        confidence_score=confidence_score,
        prediction_probabilities=prediction_probabilities,

        # Input data tracking
        input_data_hash=input_data_hash,
        input_metadata=image_metadata,

        # Processing metadata
        processing_time_ms=processing_time_ms,
        gpu_used=gpu_used,
        system_resources=system_resources,

        # Quality assurance
        quality_passed=quality_passed,
        uncertainty_metrics=uncertainty_metrics,
        reliability_score=reliability_score,
        flags=flags or [],

        # Legal and compliance
        consent_obtained=consent_obtained,
        data_retention_days=2555,  # 7 years for medical records
        anonymized=False,

        # IP and location
        ip_address=ip_address,
        user_agent=user_agent,
        geo_location=None,  # Could be added with geo-IP lookup

        # Action details
        action="classify_image",
        endpoint=endpoint,
        http_method=http_method,
        request_params=request_params,
        response_status=response_status,

        # Error tracking
        error_occurred=error_occurred,
        error_message=error_message,
        error_stack_trace=error_stack_trace,

        # Audit metadata
        created_at=datetime.utcnow(),
        created_by="system",
        audit_reviewed=False
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return audit_log


def log_user_action(
    db: Session,
    user_id: Optional[int],
    username: Optional[str],
    session_id: Optional[str],
    event_type: str,
    action: str,
    endpoint: Optional[str] = None,
    http_method: Optional[str] = None,
    request_params: Optional[Dict[str, Any]] = None,
    response_status: int = 200,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    error_occurred: bool = False,
    error_message: Optional[str] = None
) -> AuditLog:
    """
    Log a user action (login, export, share, etc.)

    Args:
        db: Database session
        user_id: ID of the user
        username: Username
        session_id: Session identifier
        event_type: Type of event (login, export, share, etc.)
        action: Specific action taken
        endpoint: API endpoint called
        http_method: GET, POST, etc.
        request_params: Request parameters (sanitized)
        response_status: HTTP response code
        ip_address: User's IP address
        user_agent: Browser/device information
        error_occurred: Whether an error occurred
        error_message: Error details if any

    Returns:
        AuditLog: The created audit log entry
    """

    severity = "error" if error_occurred else "info"

    audit_log = AuditLog(
        # User and session
        user_id=user_id,
        username=username,
        session_id=session_id,

        # Event classification
        event_type=event_type,
        event_category="user_action",
        severity=severity,

        # Action details
        action=action,
        endpoint=endpoint,
        http_method=http_method,
        request_params=request_params,
        response_status=response_status,

        # IP and location
        ip_address=ip_address,
        user_agent=user_agent,

        # Error tracking
        error_occurred=error_occurred,
        error_message=error_message,

        # Audit metadata
        created_at=datetime.utcnow(),
        created_by="system",
        audit_reviewed=False,

        # Legal compliance
        consent_obtained=True,
        data_retention_days=2555
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return audit_log


def log_system_event(
    db: Session,
    event_type: str,
    action: str,
    severity: str = "info",
    details: Optional[Dict[str, Any]] = None,
    error_occurred: bool = False,
    error_message: Optional[str] = None,
    error_stack_trace: Optional[str] = None
) -> AuditLog:
    """
    Log a system event (startup, shutdown, error, etc.)

    Args:
        db: Database session
        event_type: Type of event
        action: Specific action/event
        severity: info, warning, error, critical
        details: Additional event details
        error_occurred: Whether an error occurred
        error_message: Error details if any
        error_stack_trace: Stack trace for debugging

    Returns:
        AuditLog: The created audit log entry
    """

    audit_log = AuditLog(
        # Event classification
        event_type=event_type,
        event_category="system_event",
        severity=severity,

        # Action details
        action=action,
        request_params=details,

        # Error tracking
        error_occurred=error_occurred,
        error_message=error_message,
        error_stack_trace=error_stack_trace,

        # Audit metadata
        created_at=datetime.utcnow(),
        created_by="system",
        audit_reviewed=False,

        # System resources
        system_resources=get_system_resources()
    )

    db.add(audit_log)
    db.commit()
    db.refresh(audit_log)

    return audit_log
