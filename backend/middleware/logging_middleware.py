"""
Request Logging Middleware

Captures all API calls with structured JSON logging for:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- AWS CloudWatch
- Local JSON log files

Features:
- Request/response logging with timing
- Request ID tracking for distributed tracing
- Correlation ID propagation
- User context tracking
- Sensitive data masking
- Performance metrics
"""

import time
import uuid
from datetime import datetime
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import LOG_LEVEL, DEBUG

# Import structured logging
from structured_logging import (
    get_logger,
    LogContext,
    set_context,
    clear_context,
    log_request,
    generate_request_id,
)

# Create logger
logger = get_logger("api.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all incoming requests and outgoing responses
    using structured JSON logging.
    """

    # Paths to exclude from detailed logging (to reduce noise)
    EXCLUDE_PATHS = {"/health", "/favicon.ico", "/docs", "/redoc", "/openapi.json"}

    # Paths with sensitive data (don't log request body)
    SENSITIVE_PATHS = {"/login", "/register", "/token", "/reset-password"}

    def __init__(self, app: ASGIApp, log_headers: bool = False, log_body: bool = False):
        super().__init__(app)
        self.log_headers = log_headers
        self.log_body = log_body

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID or use incoming correlation ID
        request_id = request.headers.get("X-Request-ID") or generate_request_id()
        correlation_id = request.headers.get("X-Correlation-ID") or request_id

        # Get request details
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else None
        client_ip = self._get_client_ip(request)

        # Skip detailed logging for excluded paths
        if path in self.EXCLUDE_PATHS:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Set logging context for this request
        with LogContext(
            request_id=request_id,
            correlation_id=correlation_id,
            client_ip=client_ip,
            http_method=method,
            http_path=path
        ):
            # Start timer
            start_time = time.time()

            # Log incoming request
            request_log_data = {
                "event": "request_started",
                "query_params": query_params,
                "user_agent": request.headers.get("user-agent"),
                "content_type": request.headers.get("content-type"),
                "content_length": request.headers.get("content-length"),
            }

            if self.log_headers and DEBUG:
                request_log_data["headers"] = self._get_safe_headers(request)

            logger.info("Incoming request", extra=request_log_data)

            # Process request
            try:
                response = await call_next(request)

                # Calculate processing time
                duration_ms = (time.time() - start_time) * 1000

                # Get response details
                status_code = response.status_code
                content_length = response.headers.get("content-length")

                # Log response using structured format
                log_request(
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    request_id=request_id,
                    client_ip=client_ip,
                    query_params=query_params,
                    content_length=int(content_length) if content_length else None,
                    user_agent=request.headers.get("user-agent"),
                )

                # Add tracing headers to response
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Correlation-ID"] = correlation_id
                response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"

                return response

            except Exception as e:
                # Log error
                duration_ms = (time.time() - start_time) * 1000

                logger.error(
                    "Request failed with exception",
                    extra={
                        "event": "request_error",
                        "duration_ms": round(duration_ms, 2),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    exc_info=True
                )
                raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, handling proxies."""
        # Check for forwarded headers (when behind proxy/load balancer)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client
        if request.client:
            return request.client.host

        return "unknown"

    def _get_safe_headers(self, request: Request) -> dict:
        """Get headers excluding sensitive ones."""
        sensitive_headers = {"authorization", "cookie", "x-api-key", "x-auth-token"}
        return {
            k: v for k, v in request.headers.items()
            if k.lower() not in sensitive_headers
        }


class RequestStatsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request statistics for monitoring.
    Exports metrics in a format compatible with Prometheus/CloudWatch.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.stats = {
            "total_requests": 0,
            "total_errors": 0,
            "requests_by_method": {},
            "requests_by_status": {},
            "requests_by_path": {},
            "response_times": [],  # Last N response times for percentile calculation
            "avg_response_time_ms": 0,
            "total_response_time_ms": 0,
            "started_at": datetime.utcnow().isoformat(),
        }
        self._max_response_times = 1000  # Keep last 1000 for percentiles

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Track request
        self.stats["total_requests"] += 1

        method = request.method
        self.stats["requests_by_method"][method] = \
            self.stats["requests_by_method"].get(method, 0) + 1

        # Process request
        response = await call_next(request)

        # Track response
        process_time_ms = (time.time() - start_time) * 1000

        # Update response times list (for percentiles)
        self.stats["response_times"].append(process_time_ms)
        if len(self.stats["response_times"]) > self._max_response_times:
            self.stats["response_times"].pop(0)

        self.stats["total_response_time_ms"] += process_time_ms
        self.stats["avg_response_time_ms"] = \
            self.stats["total_response_time_ms"] / self.stats["total_requests"]

        status = response.status_code
        self.stats["requests_by_status"][status] = \
            self.stats["requests_by_status"].get(status, 0) + 1

        if status >= 500:
            self.stats["total_errors"] += 1

        # Track path (simplified)
        path = request.url.path.split("/")[1] if request.url.path != "/" else "root"
        self.stats["requests_by_path"][path] = \
            self.stats["requests_by_path"].get(path, 0) + 1

        return response

    def get_stats(self) -> dict:
        """Get current statistics in CloudWatch-compatible format."""
        # Calculate percentiles
        response_times = sorted(self.stats["response_times"])
        p50 = self._percentile(response_times, 50)
        p95 = self._percentile(response_times, 95)
        p99 = self._percentile(response_times, 99)

        return {
            "total_requests": self.stats["total_requests"],
            "total_errors": self.stats["total_errors"],
            "error_rate_percent": round(
                self.stats["total_errors"] / max(self.stats["total_requests"], 1) * 100, 2
            ),
            "avg_response_time_ms": round(self.stats["avg_response_time_ms"], 2),
            "p50_response_time_ms": round(p50, 2),
            "p95_response_time_ms": round(p95, 2),
            "p99_response_time_ms": round(p99, 2),
            "requests_by_method": self.stats["requests_by_method"],
            "requests_by_status": self.stats["requests_by_status"],
            "top_paths": dict(
                sorted(
                    self.stats["requests_by_path"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ),
            "started_at": self.stats["started_at"],
            "uptime_seconds": (
                datetime.utcnow() - datetime.fromisoformat(self.stats["started_at"])
            ).total_seconds(),
        }

    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile of sorted data."""
        if not data:
            return 0.0
        index = (percentile / 100) * (len(data) - 1)
        lower = int(index)
        upper = min(lower + 1, len(data) - 1)
        weight = index - lower
        return data[lower] * (1 - weight) + data[upper] * weight

    def export_cloudwatch_metrics(self) -> list:
        """Export metrics in CloudWatch PutMetricData format."""
        stats = self.get_stats()
        timestamp = datetime.utcnow()

        return [
            {
                "MetricName": "RequestCount",
                "Value": stats["total_requests"],
                "Unit": "Count",
                "Timestamp": timestamp
            },
            {
                "MetricName": "ErrorCount",
                "Value": stats["total_errors"],
                "Unit": "Count",
                "Timestamp": timestamp
            },
            {
                "MetricName": "ErrorRate",
                "Value": stats["error_rate_percent"],
                "Unit": "Percent",
                "Timestamp": timestamp
            },
            {
                "MetricName": "AverageLatency",
                "Value": stats["avg_response_time_ms"],
                "Unit": "Milliseconds",
                "Timestamp": timestamp
            },
            {
                "MetricName": "P95Latency",
                "Value": stats["p95_response_time_ms"],
                "Unit": "Milliseconds",
                "Timestamp": timestamp
            },
            {
                "MetricName": "P99Latency",
                "Value": stats["p99_response_time_ms"],
                "Unit": "Milliseconds",
                "Timestamp": timestamp
            },
        ]


# Singleton instance for stats (accessible from endpoints)
_stats_middleware = None


def get_stats_middleware() -> RequestStatsMiddleware:
    """Get the stats middleware instance."""
    global _stats_middleware
    return _stats_middleware


def set_stats_middleware(middleware: RequestStatsMiddleware):
    """Set the stats middleware instance."""
    global _stats_middleware
    _stats_middleware = middleware
