"""
Model Monitoring Service

Tracks ML model performance metrics and generates alerts for:
- Model failures (crashes, timeouts, exceptions)
- Degraded performance (slow inference, low confidence)
- Anomaly detection (unusual error patterns)

Usage:
    from model_monitoring import ModelMonitor, monitor

    # Record a successful inference
    monitor.record_inference(
        model_name="binary_model",
        inference_time_ms=150,
        confidence=0.92,
        success=True
    )

    # Record a failure
    monitor.record_inference(
        model_name="lesion_model",
        inference_time_ms=0,
        success=False,
        error="CUDA out of memory"
    )

    # Check health
    health = monitor.get_model_health("binary_model")
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

# Import structured logging
from structured_logging import get_logger, log_model_inference

# Create logger for monitoring
logger = get_logger("ml.monitoring")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of model alerts."""
    MODEL_FAILURE = "model_failure"
    HIGH_ERROR_RATE = "high_error_rate"
    SLOW_INFERENCE = "slow_inference"
    LOW_CONFIDENCE = "low_confidence"
    MODEL_UNAVAILABLE = "model_unavailable"
    MEMORY_WARNING = "memory_warning"
    THROUGHPUT_DEGRADATION = "throughput_degradation"
    ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class InferenceRecord:
    """Record of a single model inference."""
    timestamp: datetime
    model_name: str
    inference_time_ms: float
    confidence: Optional[float]
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAlert:
    """A model performance alert."""
    id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    model_name: str
    title: str
    message: str
    metrics: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None


@dataclass
class ModelHealthMetrics:
    """Health metrics for a single model."""
    model_name: str
    status: str  # healthy, degraded, unhealthy, unavailable
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
    last_inference_at: Optional[datetime]
    last_error_at: Optional[datetime]
    last_error_message: Optional[str]
    uptime_percentage: float
    active_alerts: List[str]


class AlertThresholds:
    """Configurable alert thresholds."""

    # Error rate thresholds (percentage)
    ERROR_RATE_WARNING = 5.0  # 5% error rate
    ERROR_RATE_ERROR = 10.0  # 10% error rate
    ERROR_RATE_CRITICAL = 25.0  # 25% error rate

    # Inference time thresholds (milliseconds)
    INFERENCE_TIME_WARNING = 2000  # 2 seconds
    INFERENCE_TIME_ERROR = 5000  # 5 seconds
    INFERENCE_TIME_CRITICAL = 10000  # 10 seconds

    # Confidence thresholds
    LOW_CONFIDENCE_WARNING = 0.3  # Average confidence below 30%
    LOW_CONFIDENCE_ERROR = 0.2  # Average confidence below 20%

    # Consecutive failures before alert
    CONSECUTIVE_FAILURES_WARNING = 3
    CONSECUTIVE_FAILURES_ERROR = 5
    CONSECUTIVE_FAILURES_CRITICAL = 10

    # Time window for rate calculations (seconds)
    RATE_WINDOW_SECONDS = 300  # 5 minutes

    # Minimum samples before alerting
    MIN_SAMPLES_FOR_ALERT = 10

    # Throughput degradation (inferences per minute)
    THROUGHPUT_DEGRADATION_PERCENT = 50  # Alert if throughput drops 50%


class ModelMonitor:
    """
    Central model monitoring service.

    Tracks performance metrics for all ML models and generates
    alerts when thresholds are exceeded.
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize the monitor.

        Args:
            max_history: Maximum inference records to keep per model
        """
        self.max_history = max_history
        self._lock = threading.Lock()

        # Inference history per model
        self._history: Dict[str, deque] = {}

        # Active alerts
        self._alerts: Dict[str, ModelAlert] = {}

        # Alert history (resolved alerts)
        self._alert_history: deque = deque(maxlen=1000)

        # Consecutive failure counters
        self._consecutive_failures: Dict[str, int] = {}

        # Model availability tracking
        self._model_last_success: Dict[str, datetime] = {}
        self._model_registered: Dict[str, datetime] = {}

        # Baseline metrics for anomaly detection
        self._baselines: Dict[str, Dict[str, float]] = {}

        # Alert callbacks
        self._alert_callbacks: List[callable] = []

        # Thresholds (can be customized)
        self.thresholds = AlertThresholds()

        # Alert counter for IDs
        self._alert_counter = 0

    def register_model(self, model_name: str) -> None:
        """Register a model for monitoring."""
        with self._lock:
            if model_name not in self._history:
                self._history[model_name] = deque(maxlen=self.max_history)
                self._consecutive_failures[model_name] = 0
                self._model_registered[model_name] = datetime.utcnow()

    def record_inference(
        self,
        model_name: str,
        inference_time_ms: float,
        success: bool,
        confidence: Optional[float] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelAlert]:
        """
        Record a model inference result.

        Args:
            model_name: Name of the model
            inference_time_ms: Time taken for inference in milliseconds
            success: Whether the inference succeeded
            confidence: Model confidence score (0-1)
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            ModelAlert if an alert was triggered, None otherwise
        """
        with self._lock:
            # Ensure model is registered
            if model_name not in self._history:
                self.register_model(model_name)

            # Create record
            record = InferenceRecord(
                timestamp=datetime.utcnow(),
                model_name=model_name,
                inference_time_ms=inference_time_ms,
                confidence=confidence,
                success=success,
                error=error,
                metadata=metadata or {}
            )

            # Add to history
            self._history[model_name].append(record)

            # Update tracking
            if success:
                self._consecutive_failures[model_name] = 0
                self._model_last_success[model_name] = record.timestamp
            else:
                self._consecutive_failures[model_name] += 1

            # Check for alerts
            alert = self._check_alerts(model_name, record)

            return alert

    def _check_alerts(self, model_name: str, record: InferenceRecord) -> Optional[ModelAlert]:
        """Check if the latest inference triggers any alerts."""
        alerts_triggered = []

        # Check consecutive failures
        consecutive = self._consecutive_failures.get(model_name, 0)
        if consecutive >= self.thresholds.CONSECUTIVE_FAILURES_CRITICAL:
            alert = self._create_alert(
                AlertType.MODEL_FAILURE,
                AlertSeverity.CRITICAL,
                model_name,
                f"Critical: {model_name} has {consecutive} consecutive failures",
                f"Model {model_name} has failed {consecutive} times in a row. "
                f"Last error: {record.error or 'Unknown'}",
                {"consecutive_failures": consecutive, "last_error": record.error}
            )
            alerts_triggered.append(alert)
        elif consecutive >= self.thresholds.CONSECUTIVE_FAILURES_ERROR:
            alert = self._create_alert(
                AlertType.MODEL_FAILURE,
                AlertSeverity.ERROR,
                model_name,
                f"Error: {model_name} has {consecutive} consecutive failures",
                f"Model {model_name} has failed {consecutive} times in a row.",
                {"consecutive_failures": consecutive, "last_error": record.error}
            )
            alerts_triggered.append(alert)
        elif consecutive >= self.thresholds.CONSECUTIVE_FAILURES_WARNING:
            alert = self._create_alert(
                AlertType.MODEL_FAILURE,
                AlertSeverity.WARNING,
                model_name,
                f"Warning: {model_name} has {consecutive} consecutive failures",
                f"Model {model_name} has failed {consecutive} times in a row.",
                {"consecutive_failures": consecutive, "last_error": record.error}
            )
            alerts_triggered.append(alert)

        # Check error rate (need enough samples)
        history = list(self._history[model_name])
        recent = self._get_recent_records(history, self.thresholds.RATE_WINDOW_SECONDS)

        if len(recent) >= self.thresholds.MIN_SAMPLES_FOR_ALERT:
            error_rate = (sum(1 for r in recent if not r.success) / len(recent)) * 100

            if error_rate >= self.thresholds.ERROR_RATE_CRITICAL:
                alert = self._create_alert(
                    AlertType.HIGH_ERROR_RATE,
                    AlertSeverity.CRITICAL,
                    model_name,
                    f"Critical: {model_name} error rate at {error_rate:.1f}%",
                    f"Model {model_name} has a {error_rate:.1f}% error rate over the last "
                    f"{self.thresholds.RATE_WINDOW_SECONDS} seconds ({len(recent)} samples).",
                    {"error_rate": error_rate, "samples": len(recent)}
                )
                alerts_triggered.append(alert)
            elif error_rate >= self.thresholds.ERROR_RATE_ERROR:
                alert = self._create_alert(
                    AlertType.HIGH_ERROR_RATE,
                    AlertSeverity.ERROR,
                    model_name,
                    f"Error: {model_name} error rate at {error_rate:.1f}%",
                    f"Model {model_name} has an elevated error rate.",
                    {"error_rate": error_rate, "samples": len(recent)}
                )
                alerts_triggered.append(alert)
            elif error_rate >= self.thresholds.ERROR_RATE_WARNING:
                alert = self._create_alert(
                    AlertType.HIGH_ERROR_RATE,
                    AlertSeverity.WARNING,
                    model_name,
                    f"Warning: {model_name} error rate at {error_rate:.1f}%",
                    f"Model {model_name} error rate is above normal.",
                    {"error_rate": error_rate, "samples": len(recent)}
                )
                alerts_triggered.append(alert)

        # Check slow inference
        if record.success and record.inference_time_ms > 0:
            if record.inference_time_ms >= self.thresholds.INFERENCE_TIME_CRITICAL:
                alert = self._create_alert(
                    AlertType.SLOW_INFERENCE,
                    AlertSeverity.CRITICAL,
                    model_name,
                    f"Critical: {model_name} inference took {record.inference_time_ms:.0f}ms",
                    f"Model {model_name} took {record.inference_time_ms:.0f}ms for inference, "
                    f"exceeding critical threshold of {self.thresholds.INFERENCE_TIME_CRITICAL}ms.",
                    {"inference_time_ms": record.inference_time_ms}
                )
                alerts_triggered.append(alert)
            elif record.inference_time_ms >= self.thresholds.INFERENCE_TIME_ERROR:
                alert = self._create_alert(
                    AlertType.SLOW_INFERENCE,
                    AlertSeverity.ERROR,
                    model_name,
                    f"Error: {model_name} inference slow ({record.inference_time_ms:.0f}ms)",
                    f"Model {model_name} inference time is degraded.",
                    {"inference_time_ms": record.inference_time_ms}
                )
                alerts_triggered.append(alert)
            elif record.inference_time_ms >= self.thresholds.INFERENCE_TIME_WARNING:
                alert = self._create_alert(
                    AlertType.SLOW_INFERENCE,
                    AlertSeverity.WARNING,
                    model_name,
                    f"Warning: {model_name} inference slow ({record.inference_time_ms:.0f}ms)",
                    f"Model {model_name} inference time is above normal.",
                    {"inference_time_ms": record.inference_time_ms}
                )
                alerts_triggered.append(alert)

        # Check low confidence
        if record.success and record.confidence is not None:
            successful_recent = [r for r in recent if r.success and r.confidence is not None]
            if len(successful_recent) >= self.thresholds.MIN_SAMPLES_FOR_ALERT:
                avg_confidence = statistics.mean(r.confidence for r in successful_recent)

                if avg_confidence < self.thresholds.LOW_CONFIDENCE_ERROR:
                    alert = self._create_alert(
                        AlertType.LOW_CONFIDENCE,
                        AlertSeverity.ERROR,
                        model_name,
                        f"Error: {model_name} avg confidence at {avg_confidence:.1%}",
                        f"Model {model_name} average confidence is very low, indicating "
                        f"potential model degradation or data distribution shift.",
                        {"avg_confidence": avg_confidence, "samples": len(successful_recent)}
                    )
                    alerts_triggered.append(alert)
                elif avg_confidence < self.thresholds.LOW_CONFIDENCE_WARNING:
                    alert = self._create_alert(
                        AlertType.LOW_CONFIDENCE,
                        AlertSeverity.WARNING,
                        model_name,
                        f"Warning: {model_name} avg confidence at {avg_confidence:.1%}",
                        f"Model {model_name} average confidence is below normal.",
                        {"avg_confidence": avg_confidence, "samples": len(successful_recent)}
                    )
                    alerts_triggered.append(alert)

        # Return most severe alert
        if alerts_triggered:
            # Sort by severity
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.ERROR: 1,
                AlertSeverity.WARNING: 2,
                AlertSeverity.INFO: 3
            }
            alerts_triggered.sort(key=lambda a: severity_order[a.severity])
            return alerts_triggered[0]

        return None

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        model_name: str,
        title: str,
        message: str,
        metrics: Dict[str, Any]
    ) -> ModelAlert:
        """Create and store a new alert."""
        # Check for existing active alert of same type
        alert_key = f"{model_name}:{alert_type.value}"

        if alert_key in self._alerts:
            existing = self._alerts[alert_key]
            # Update existing alert if severity increased
            if severity.value < existing.severity.value:
                existing.severity = severity
                existing.title = title
                existing.message = message
                existing.metrics = metrics
                existing.timestamp = datetime.utcnow()
            return existing

        # Create new alert
        self._alert_counter += 1
        alert = ModelAlert(
            id=f"alert-{self._alert_counter:06d}",
            timestamp=datetime.utcnow(),
            alert_type=alert_type,
            severity=severity,
            model_name=model_name,
            title=title,
            message=message,
            metrics=metrics
        )

        self._alerts[alert_key] = alert

        # Trigger callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Alert callback error", extra={"error": str(e), "alert_id": alert.id})

        return alert

    def _get_recent_records(
        self,
        history: List[InferenceRecord],
        seconds: int
    ) -> List[InferenceRecord]:
        """Get records from the last N seconds."""
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        return [r for r in history if r.timestamp >= cutoff]

    def get_model_health(self, model_name: str) -> Optional[ModelHealthMetrics]:
        """Get health metrics for a specific model."""
        with self._lock:
            if model_name not in self._history:
                return None

            history = list(self._history[model_name])
            if not history:
                return ModelHealthMetrics(
                    model_name=model_name,
                    status="no_data",
                    total_inferences=0,
                    successful_inferences=0,
                    failed_inferences=0,
                    error_rate=0.0,
                    avg_inference_time_ms=0.0,
                    p95_inference_time_ms=0.0,
                    p99_inference_time_ms=0.0,
                    avg_confidence=0.0,
                    min_confidence=0.0,
                    inferences_last_hour=0,
                    errors_last_hour=0,
                    last_inference_at=None,
                    last_error_at=None,
                    last_error_message=None,
                    uptime_percentage=100.0,
                    active_alerts=[]
                )

            # Calculate metrics
            successful = [r for r in history if r.success]
            failed = [r for r in history if not r.success]

            # Inference times
            times = [r.inference_time_ms for r in successful if r.inference_time_ms > 0]
            avg_time = statistics.mean(times) if times else 0.0
            p95_time = self._percentile(times, 95) if times else 0.0
            p99_time = self._percentile(times, 99) if times else 0.0

            # Confidence
            confidences = [r.confidence for r in successful if r.confidence is not None]
            avg_conf = statistics.mean(confidences) if confidences else 0.0
            min_conf = min(confidences) if confidences else 0.0

            # Last hour metrics
            hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent = [r for r in history if r.timestamp >= hour_ago]
            errors_last_hour = sum(1 for r in recent if not r.success)

            # Error rate
            error_rate = (len(failed) / len(history)) * 100 if history else 0.0

            # Last error
            last_error = failed[-1] if failed else None

            # Status determination
            status = "healthy"
            if error_rate >= self.thresholds.ERROR_RATE_CRITICAL:
                status = "unhealthy"
            elif error_rate >= self.thresholds.ERROR_RATE_ERROR:
                status = "degraded"
            elif error_rate >= self.thresholds.ERROR_RATE_WARNING:
                status = "warning"

            # Active alerts for this model
            active_alerts = [
                a.id for key, a in self._alerts.items()
                if a.model_name == model_name and not a.resolved
            ]

            # Uptime (percentage of successful inferences)
            uptime = (len(successful) / len(history)) * 100 if history else 100.0

            return ModelHealthMetrics(
                model_name=model_name,
                status=status,
                total_inferences=len(history),
                successful_inferences=len(successful),
                failed_inferences=len(failed),
                error_rate=round(error_rate, 2),
                avg_inference_time_ms=round(avg_time, 2),
                p95_inference_time_ms=round(p95_time, 2),
                p99_inference_time_ms=round(p99_time, 2),
                avg_confidence=round(avg_conf, 4),
                min_confidence=round(min_conf, 4),
                inferences_last_hour=len(recent),
                errors_last_hour=errors_last_hour,
                last_inference_at=history[-1].timestamp if history else None,
                last_error_at=last_error.timestamp if last_error else None,
                last_error_message=last_error.error if last_error else None,
                uptime_percentage=round(uptime, 2),
                active_alerts=active_alerts
            )

    def get_all_models_health(self) -> Dict[str, ModelHealthMetrics]:
        """Get health metrics for all monitored models."""
        with self._lock:
            return {
                name: self.get_model_health(name)
                for name in self._history.keys()
            }

    def get_active_alerts(self) -> List[ModelAlert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [a for a in self._alerts.values() if not a.resolved]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[ModelAlert]:
        """Get alerts filtered by severity."""
        with self._lock:
            return [a for a in self._alerts.values() if a.severity == severity]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts.values():
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_by = acknowledged_by
                    return True
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        with self._lock:
            for key, alert in list(self._alerts.items()):
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    self._alert_history.append(alert)
                    del self._alerts[key]
                    return True
            return False

    def add_alert_callback(self, callback: callable) -> None:
        """Add a callback to be called when alerts are triggered."""
        self._alert_callbacks.append(callback)

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire monitoring system."""
        with self._lock:
            all_health = self.get_all_models_health()
            active_alerts = self.get_active_alerts()

            # Count by status
            status_counts = {}
            for health in all_health.values():
                status = health.status if health else "unknown"
                status_counts[status] = status_counts.get(status, 0) + 1

            # Count alerts by severity
            alert_counts = {}
            for alert in active_alerts:
                sev = alert.severity.value
                alert_counts[sev] = alert_counts.get(sev, 0) + 1

            # Overall status
            if any(h.status == "unhealthy" for h in all_health.values() if h):
                overall_status = "unhealthy"
            elif any(h.status == "degraded" for h in all_health.values() if h):
                overall_status = "degraded"
            elif any(h.status == "warning" for h in all_health.values() if h):
                overall_status = "warning"
            else:
                overall_status = "healthy"

            return {
                "overall_status": overall_status,
                "total_models": len(all_health),
                "models_by_status": status_counts,
                "active_alerts": len(active_alerts),
                "alerts_by_severity": alert_counts,
                "critical_alerts": alert_counts.get("critical", 0),
                "timestamp": datetime.utcnow().isoformat()
            }

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a list."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_data):
            return sorted_data[-1]
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics in a format suitable for external monitoring."""
        with self._lock:
            models = {}
            for name in self._history.keys():
                health = self.get_model_health(name)
                if health:
                    models[name] = {
                        "status": health.status,
                        "error_rate": health.error_rate,
                        "avg_latency_ms": health.avg_inference_time_ms,
                        "p95_latency_ms": health.p95_inference_time_ms,
                        "avg_confidence": health.avg_confidence,
                        "total_inferences": health.total_inferences,
                        "uptime_percent": health.uptime_percentage
                    }

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "summary": self.get_system_summary(),
                "models": models,
                "alerts": [
                    {
                        "id": a.id,
                        "type": a.alert_type.value,
                        "severity": a.severity.value,
                        "model": a.model_name,
                        "title": a.title,
                        "timestamp": a.timestamp.isoformat()
                    }
                    for a in self.get_active_alerts()
                ]
            }

    def clear_history(self, model_name: Optional[str] = None) -> None:
        """Clear inference history (for testing or reset)."""
        with self._lock:
            if model_name:
                if model_name in self._history:
                    self._history[model_name].clear()
            else:
                for deq in self._history.values():
                    deq.clear()


# Global monitor instance
monitor = ModelMonitor()


# Convenience functions
def record_inference(
    model_name: str,
    inference_time_ms: float,
    success: bool,
    confidence: Optional[float] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[ModelAlert]:
    """Record an inference to the global monitor with structured logging."""
    # Log using structured logging
    log_model_inference(
        model_name=model_name,
        inference_time_ms=inference_time_ms,
        success=success,
        confidence=confidence,
        error=error,
        **(metadata or {})
    )

    # Record in monitor for alerting
    return monitor.record_inference(
        model_name=model_name,
        inference_time_ms=inference_time_ms,
        success=success,
        confidence=confidence,
        error=error,
        metadata=metadata
    )


def get_model_health(model_name: str) -> Optional[ModelHealthMetrics]:
    """Get health metrics from the global monitor."""
    return monitor.get_model_health(model_name)


def get_system_summary() -> Dict[str, Any]:
    """Get system summary from the global monitor."""
    return monitor.get_system_summary()


# =============================================================================
# ALERT NOTIFICATION DISPATCH
# =============================================================================

def dispatch_alert_notification(alert: ModelAlert) -> None:
    """
    Dispatch alert notification to database and relevant users.

    This is registered as a callback with the monitor to be called
    whenever an alert is triggered.
    """
    import hashlib

    # Create fingerprint for deduplication
    fingerprint = hashlib.md5(
        f"{alert.alert_type.value}:{alert.model_name}:{alert.severity.value}".encode()
    ).hexdigest()

    try:
        # Import database components
        from database import SessionLocal, SystemAlert as DBSystemAlert

        db = SessionLocal()
        try:
            # Check for existing active alert with same fingerprint
            existing = db.query(DBSystemAlert).filter(
                DBSystemAlert.fingerprint == fingerprint,
                DBSystemAlert.status == "active"
            ).first()

            if existing:
                # Update occurrence count
                existing.occurrence_count += 1
                existing.last_occurrence = datetime.utcnow()
                existing.metrics = alert.metrics
                db.commit()
            else:
                # Create new alert record
                db_alert = DBSystemAlert(
                    alert_id=alert.id,
                    alert_type=alert.alert_type.value,
                    severity=alert.severity.value,
                    model_name=alert.model_name,
                    title=alert.title,
                    message=alert.message,
                    metrics=alert.metrics,
                    status="active",
                    fingerprint=fingerprint,
                    occurrence_count=1,
                    created_at=alert.timestamp,
                    last_occurrence=alert.timestamp
                )
                db.add(db_alert)
                db.commit()

            # Dispatch notification to admin users for critical/error alerts
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
                _notify_admins(db, alert)

        finally:
            db.close()

    except Exception as e:
        # Log error but don't fail the inference
        logger.error("Failed to dispatch alert", extra={"error": str(e), "alert_id": alert.id})


def _notify_admins(db, alert: ModelAlert) -> None:
    """Send notification to admin users about the alert."""
    try:
        from database import User, Notification

        # Find admin/professional users
        admins = db.query(User).filter(
            User.role.in_(["admin", "professional", "dermatologist"])
        ).all()

        for admin in admins:
            notification = Notification(
                user_id=admin.id,
                title=f"[{alert.severity.value.upper()}] {alert.title}",
                message=alert.message,
                type="system_alert",
                priority="high" if alert.severity == AlertSeverity.CRITICAL else "medium",
                metadata={
                    "alert_id": alert.id,
                    "alert_type": alert.alert_type.value,
                    "model_name": alert.model_name,
                    "severity": alert.severity.value
                }
            )
            db.add(notification)

        db.commit()

    except Exception as e:
        logger.error("Failed to notify admins", extra={"error": str(e), "alert_id": alert.id})


def initialize_alert_dispatch():
    """Initialize the alert dispatch callback."""
    monitor.add_alert_callback(dispatch_alert_notification)
    logger.info("Alert dispatch initialized")


# Auto-initialize when module is imported
try:
    initialize_alert_dispatch()
except Exception as e:
    logger.warning("Failed to initialize alert dispatch", extra={"error": str(e)})
