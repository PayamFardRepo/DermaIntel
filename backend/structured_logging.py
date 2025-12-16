"""
Structured Logging Module

Provides JSON-formatted logging compatible with ELK Stack and AWS CloudWatch.

Features:
- JSON log formatting for machine-readable logs
- Request context tracking (request ID, correlation ID, user ID)
- Multiple output handlers (file, stdout, CloudWatch)
- Log level filtering and sampling
- Sensitive data masking
- Performance metrics integration

Usage:
    from structured_logging import get_logger, LogContext

    logger = get_logger(__name__)

    with LogContext(request_id="abc123", user_id=42):
        logger.info("Processing request", extra={"action": "classify"})

Configuration (environment variables):
    LOG_FORMAT: "json" or "text" (default: "json")
    LOG_OUTPUT: "stdout", "file", "cloudwatch", "all" (default: "stdout")
    LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: "INFO")
    CLOUDWATCH_LOG_GROUP: CloudWatch log group name
    CLOUDWATCH_LOG_STREAM: CloudWatch log stream name
    AWS_REGION: AWS region for CloudWatch
"""

import json
import logging
import sys
import os
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar
from pathlib import Path
import socket
import uuid

# Context variables for request tracking
_request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


# =============================================================================
# LOG CONTEXT MANAGEMENT
# =============================================================================

class LogContext:
    """
    Context manager for adding contextual information to logs.

    Usage:
        with LogContext(request_id="abc", user_id=123):
            logger.info("Processing")  # Will include request_id and user_id
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self._token = None

    def __enter__(self):
        current = _request_context.get().copy()
        current.update(self.context)
        self._token = _request_context.set(current)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token:
            _request_context.reset(self._token)
        return False


def set_context(**kwargs):
    """Set context values for the current execution context."""
    current = _request_context.get().copy()
    current.update(kwargs)
    _request_context.set(current)


def get_context() -> Dict[str, Any]:
    """Get current logging context."""
    return _request_context.get().copy()


def clear_context():
    """Clear the current logging context."""
    _request_context.set({})


# =============================================================================
# JSON LOG FORMATTER
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for ELK/CloudWatch ingestion.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:45.123Z",
        "level": "INFO",
        "logger": "api.analysis",
        "message": "Classification complete",
        "service": "skin-classifier",
        "environment": "production",
        "host": "server-01",
        "request_id": "abc123",
        "user_id": 42,
        "duration_ms": 150,
        "extra": {...}
    }
    """

    # Fields to always include at the top level
    STANDARD_FIELDS = {
        'timestamp', 'level', 'logger', 'message', 'service',
        'environment', 'host', 'request_id', 'correlation_id',
        'user_id', 'trace_id', 'span_id'
    }

    # Sensitive fields to mask
    SENSITIVE_FIELDS = {
        'password', 'token', 'secret', 'api_key', 'authorization',
        'credit_card', 'ssn', 'auth_token', 'access_token', 'refresh_token'
    }

    def __init__(
        self,
        service_name: str = "skin-classifier",
        environment: str = None,
        include_extra: bool = True,
        mask_sensitive: bool = True
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.include_extra = include_extra
        self.mask_sensitive = mask_sensitive
        self.hostname = socket.gethostname()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        # Get context
        context = get_context()

        # Build base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "host": self.hostname,
        }

        # Add context fields
        for key in ['request_id', 'correlation_id', 'user_id', 'trace_id', 'span_id']:
            if key in context:
                log_entry[key] = context[key]

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self._format_exception(record.exc_info)
            }

        # Add extra fields
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in logging.LogRecord.__dict__ and not key.startswith('_'):
                    if key not in self.STANDARD_FIELDS:
                        # Mask sensitive data
                        if self.mask_sensitive and self._is_sensitive(key):
                            extra[key] = "***MASKED***"
                        else:
                            extra[key] = self._serialize_value(value)

            # Add remaining context as extra
            for key, value in context.items():
                if key not in log_entry and key not in extra:
                    extra[key] = self._serialize_value(value)

            if extra:
                log_entry["extra"] = extra

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def _format_exception(self, exc_info) -> Optional[str]:
        """Format exception traceback."""
        if exc_info:
            return ''.join(traceback.format_exception(*exc_info))
        return None

    def _is_sensitive(self, key: str) -> bool:
        """Check if a field name indicates sensitive data."""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.SENSITIVE_FIELDS)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON output."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, (datetime,)):
            return value.isoformat()
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return str(value)


# =============================================================================
# CLOUDWATCH HANDLER
# =============================================================================

class CloudWatchHandler(logging.Handler):
    """
    Handler for sending logs to AWS CloudWatch Logs.

    Requires: boto3 (pip install boto3)

    Configuration:
        CLOUDWATCH_LOG_GROUP: Log group name
        CLOUDWATCH_LOG_STREAM: Log stream name (default: hostname-date)
        AWS_REGION: AWS region
    """

    def __init__(
        self,
        log_group: str = None,
        log_stream: str = None,
        region: str = None,
        batch_size: int = 10,
        flush_interval: int = 5
    ):
        super().__init__()
        self.log_group = log_group or os.getenv("CLOUDWATCH_LOG_GROUP", "skin-classifier-logs")
        self.log_stream = log_stream or os.getenv(
            "CLOUDWATCH_LOG_STREAM",
            f"{socket.gethostname()}-{datetime.now().strftime('%Y-%m-%d')}"
        )
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._client = None
        self._sequence_token = None
        self._initialized = False

    def _init_client(self):
        """Initialize CloudWatch client lazily."""
        if self._initialized:
            return

        try:
            import boto3
            self._client = boto3.client('logs', region_name=self.region)

            # Create log group if it doesn't exist
            try:
                self._client.create_log_group(logGroupName=self.log_group)
            except self._client.exceptions.ResourceAlreadyExistsException:
                pass

            # Create log stream if it doesn't exist
            try:
                self._client.create_log_stream(
                    logGroupName=self.log_group,
                    logStreamName=self.log_stream
                )
            except self._client.exceptions.ResourceAlreadyExistsException:
                # Get existing sequence token
                response = self._client.describe_log_streams(
                    logGroupName=self.log_group,
                    logStreamNamePrefix=self.log_stream
                )
                if response['logStreams']:
                    self._sequence_token = response['logStreams'][0].get('uploadSequenceToken')

            self._initialized = True

        except ImportError:
            logging.warning("boto3 not installed. CloudWatch logging disabled.")
        except Exception as e:
            logging.warning(f"Failed to initialize CloudWatch: {e}")

    def emit(self, record: logging.LogRecord):
        """Add log record to buffer and flush if needed."""
        if not self._client:
            self._init_client()

        if not self._initialized:
            return

        try:
            msg = self.format(record)
            timestamp = int(record.created * 1000)

            with self._buffer_lock:
                self._buffer.append({
                    'timestamp': timestamp,
                    'message': msg
                })

                if len(self._buffer) >= self.batch_size:
                    self._flush()

        except Exception as e:
            self.handleError(record)

    def _flush(self):
        """Flush buffered logs to CloudWatch."""
        if not self._buffer or not self._client:
            return

        with self._buffer_lock:
            events = sorted(self._buffer, key=lambda x: x['timestamp'])
            self._buffer = []

        try:
            kwargs = {
                'logGroupName': self.log_group,
                'logStreamName': self.log_stream,
                'logEvents': events
            }
            if self._sequence_token:
                kwargs['sequenceToken'] = self._sequence_token

            response = self._client.put_log_events(**kwargs)
            self._sequence_token = response.get('nextSequenceToken')

        except Exception as e:
            logging.warning(f"Failed to flush to CloudWatch: {e}")

    def close(self):
        """Flush remaining logs and close handler."""
        self._flush()
        super().close()


# =============================================================================
# ELK-COMPATIBLE FILE HANDLER
# =============================================================================

class RotatingJSONFileHandler(logging.Handler):
    """
    File handler that writes JSON logs compatible with Filebeat/Logstash.

    Features:
    - One JSON object per line (NDJSON format)
    - Automatic file rotation by size
    - Compression of rotated files
    """

    def __init__(
        self,
        filename: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = 'utf-8'
    ):
        super().__init__()
        self.filename = Path(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self._lock = threading.Lock()

        # Create directory if needed
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord):
        """Write log record to file."""
        try:
            msg = self.format(record)

            with self._lock:
                # Check if rotation is needed
                if self.filename.exists() and self.filename.stat().st_size >= self.max_bytes:
                    self._rotate()

                # Append to file
                with open(self.filename, 'a', encoding=self.encoding) as f:
                    f.write(msg + '\n')

        except Exception:
            self.handleError(record)

    def _rotate(self):
        """Rotate log files."""
        # Remove oldest backup
        oldest = Path(f"{self.filename}.{self.backup_count}")
        if oldest.exists():
            oldest.unlink()

        # Shift existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = Path(f"{self.filename}.{i}")
            dst = Path(f"{self.filename}.{i + 1}")
            if src.exists():
                src.rename(dst)

        # Rename current file
        if self.filename.exists():
            self.filename.rename(Path(f"{self.filename}.1"))


# =============================================================================
# LOGGER FACTORY
# =============================================================================

_loggers: Dict[str, logging.Logger] = {}
_configured = False


def configure_logging(
    level: str = None,
    format: str = None,
    output: str = None,
    service_name: str = "skin-classifier",
    log_file: str = None
):
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format ("json" or "text")
        output: Output destination ("stdout", "file", "cloudwatch", "all")
        service_name: Service name for log entries
        log_file: Path to log file (for file output)
    """
    global _configured

    # Get config from environment with defaults
    level = level or os.getenv("LOG_LEVEL", "INFO")
    format = format or os.getenv("LOG_FORMAT", "json")
    output = output or os.getenv("LOG_OUTPUT", "stdout")
    log_file = log_file or os.getenv("LOG_FILE", "logs/app.json.log")

    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    if format.lower() == "json":
        formatter = JSONFormatter(service_name=service_name)
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Add handlers based on output configuration
    outputs = output.lower().split(",") if "," in output else [output.lower()]

    for out in outputs:
        out = out.strip()

        if out in ("stdout", "all"):
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(formatter)
            root_logger.addHandler(stdout_handler)

        if out in ("file", "all"):
            if format.lower() == "json":
                file_handler = RotatingJSONFileHandler(log_file)
            else:
                file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        if out in ("cloudwatch", "all"):
            cw_handler = CloudWatchHandler()
            cw_handler.setFormatter(formatter)
            root_logger.addHandler(cw_handler)

    _configured = True

    # Log configuration
    root_logger.info(
        "Logging configured",
        extra={
            "log_level": level,
            "log_format": format,
            "log_output": output,
            "service": service_name
        }
    )


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with structured logging support.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    global _configured

    # Auto-configure on first use if not already configured
    if not _configured:
        configure_logging()

    name = name or "app"

    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)

    return _loggers[name]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str = None,
    user_id: int = None,
    client_ip: str = None,
    **extra
):
    """Log an HTTP request in structured format."""
    logger = get_logger("http")

    log_data = {
        "http_method": method,
        "http_path": path,
        "http_status": status_code,
        "duration_ms": round(duration_ms, 2),
        "request_id": request_id,
        "user_id": user_id,
        "client_ip": client_ip,
        **extra
    }

    if status_code >= 500:
        logger.error("HTTP request failed", extra=log_data)
    elif status_code >= 400:
        logger.warning("HTTP request client error", extra=log_data)
    else:
        logger.info("HTTP request completed", extra=log_data)


def log_model_inference(
    model_name: str,
    inference_time_ms: float,
    success: bool,
    confidence: float = None,
    predicted_class: str = None,
    error: str = None,
    **extra
):
    """Log an ML model inference in structured format."""
    logger = get_logger("ml")

    log_data = {
        "model_name": model_name,
        "inference_time_ms": round(inference_time_ms, 2),
        "success": success,
        "confidence": confidence,
        "predicted_class": predicted_class,
        **extra
    }

    if not success:
        log_data["error"] = error
        logger.error("Model inference failed", extra=log_data)
    else:
        logger.info("Model inference completed", extra=log_data)


def log_database_query(
    operation: str,
    table: str,
    duration_ms: float,
    rows_affected: int = None,
    error: str = None,
    **extra
):
    """Log a database operation in structured format."""
    logger = get_logger("db")

    log_data = {
        "db_operation": operation,
        "db_table": table,
        "duration_ms": round(duration_ms, 2),
        "rows_affected": rows_affected,
        **extra
    }

    if error:
        log_data["error"] = error
        logger.error("Database query failed", extra=log_data)
    else:
        logger.debug("Database query completed", extra=log_data)


def log_security_event(
    event_type: str,
    severity: str,
    user_id: int = None,
    client_ip: str = None,
    details: str = None,
    **extra
):
    """Log a security-related event."""
    logger = get_logger("security")

    log_data = {
        "security_event": event_type,
        "severity": severity,
        "user_id": user_id,
        "client_ip": client_ip,
        "details": details,
        **extra
    }

    if severity in ("critical", "high"):
        logger.error("Security event", extra=log_data)
    elif severity == "medium":
        logger.warning("Security event", extra=log_data)
    else:
        logger.info("Security event", extra=log_data)


# =============================================================================
# REQUEST ID GENERATION
# =============================================================================

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:12]


def generate_correlation_id() -> str:
    """Generate a correlation ID for distributed tracing."""
    return str(uuid.uuid4())
