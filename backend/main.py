"""
Skin Disease Analysis API - Refactored Main Application

This is the refactored version of main.py that uses FastAPI routers
for better code organization and maintainability.

ALL ROUTERS COMPLETED:
- auth_router.py - Authentication, profile, user settings
- notifications_router.py - Notifications and alerts
- treatment_router.py - Treatment management
- billing_router.py - Billing and payments
- education_router.py - Patient education
- appointments_router.py - Appointments and providers
- tracking_router.py - Lesion tracking and sun exposure
- analysis_router.py - Core analysis, upload, classification
- clinical_router.py - Biopsy, dermoscopy, burns, photography
- analytics_router.py - Population health, risk calculator
- teledermatology_router.py - Dermatologists, consultations, referrals
- batch_router.py - Batch skin checks
"""

# Fix Windows console encoding for Unicode characters (must be before other imports)
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Database setup
from database import create_tables, migrate_inflammatory_fields, migrate_user_roles, migrate_consensus_tables, migrate_clinical_trials_tables

# Request logging middleware
from middleware import (
    RequestLoggingMiddleware,
    RequestStatsMiddleware,
    get_stats_middleware,
    set_stats_middleware,
)

# =============================================================================
# IMPORT ROUTERS
# =============================================================================

# Authentication & User Management
from routers.auth_router import router as auth_router

# Notifications & Alerts
from routers.notifications_router import router as notifications_router
from routers.notifications_router import alerts_router

# Treatment Management
from routers.treatment_router import router as treatment_router

# Billing & Payments
from routers.billing_router import router as billing_router

# Patient Education
from routers.education_router import router as education_router

# Appointments & Providers
from routers.appointments_router import router as appointments_router

# Lesion Tracking & Sun Exposure
from routers.tracking_router import router as tracking_router

# Core Analysis & Classification
from routers.analysis_router import router as analysis_router

# Clinical Analysis (Biopsy, Dermoscopy, Burns, Photography)
from routers.clinical_router import router as clinical_router

# Analytics & Risk Calculator
from routers.analytics_router import router as analytics_router

# Teledermatology (Consultations, Referrals, Second Opinions)
from routers.teledermatology_router import router as teledermatology_router

# Batch Skin Checks
from routers.batch_router import router as batch_router

# Genetics (Genetic Risk Factors)
from routers.genetics_router import router as genetics_router

# Medications (Interaction Checking, Safety)
from routers.medications_router import router as medications_router

# Lab Results
from routers.lab_results_router import router as lab_results_router

# AI Chat (LLM Integration)
from routers.ai_chat_router import router as ai_chat_router

# Data Augmentation (Synthetic Data Generation)
from routers.data_augmentation_router import router as data_augmentation_router

# Advanced Teledermatology (Video, Consensus, Triage)
from advanced_teledermatology import router as advanced_telederm_router

# Enhanced ML (Segmentation, Temporal Prediction, Federated Learning)
from routers.ml_enhanced_router import router as ml_enhanced_router

# Job Queue (Async Analysis)
from routers.jobs_router import router as jobs_router

# Model Monitoring and Alerts
from routers.monitoring_router import router as monitoring_router

# Clinical Trials (Recruitment Matching)
from routers.clinical_trials_router import router as clinical_trials_router

# Publication Reports
from routers.report_router import router as report_router

# Wearable Integration (Apple Watch, Fitbit, Garmin UV tracking)
from routers.wearable_router import router as wearable_router

# Cost Transparency (price estimates, provider comparison, GoodRx)
from routers.cost_transparency_router import router as cost_transparency_router

# Existing router (already was separate)
from clinic_management import router as clinic_router

# Import shared module (loads all ML models)
import shared

# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="Skin Disease Analysis API",
    description="AI-powered dermatology analysis platform with comprehensive skin lesion classification, dermoscopy analysis, teledermatology, and more.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files for serving uploaded images
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Create database tables on startup
create_tables()
migrate_inflammatory_fields()
migrate_user_roles()
migrate_consensus_tables()
migrate_clinical_trials_tables()

# Wearable integration tables
from database import migrate_wearable_tables
migrate_wearable_tables()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware (logs all API calls)
app.add_middleware(RequestLoggingMiddleware, log_headers=False, log_body=False)

# Add request stats middleware (tracks request statistics)
stats_middleware = RequestStatsMiddleware(app)
set_stats_middleware(stats_middleware)

# =============================================================================
# REGISTER ROUTERS
# =============================================================================

# Authentication & User Management
app.include_router(auth_router)

# Notifications & Alerts
app.include_router(notifications_router)
app.include_router(alerts_router)

# Treatment Management
app.include_router(treatment_router)

# Billing & Payments
app.include_router(billing_router)

# Patient Education
app.include_router(education_router)

# Appointments & Providers
app.include_router(appointments_router)

# Lesion Tracking & Sun Exposure
app.include_router(tracking_router)

# Core Analysis & Classification
app.include_router(analysis_router)

# Clinical Analysis (Biopsy, Dermoscopy, Burns, Photography)
app.include_router(clinical_router)

# Analytics & Risk Calculator
app.include_router(analytics_router)

# Teledermatology
app.include_router(teledermatology_router)

# Batch Skin Checks
app.include_router(batch_router)

# Clinic Management (existing router)
app.include_router(clinic_router)

# Genetics (Genetic Risk Factors)
app.include_router(genetics_router)

# Medications (Interaction Checking, Safety)
app.include_router(medications_router)

# Lab Results
app.include_router(lab_results_router)

# AI Chat (LLM Integration)
app.include_router(ai_chat_router)

# Data Augmentation (Synthetic Data Generation)
app.include_router(data_augmentation_router)

# Advanced Teledermatology (Video, Consensus, Triage)
app.include_router(advanced_telederm_router)

# Enhanced ML (Segmentation, Temporal Prediction, Federated Learning)
app.include_router(ml_enhanced_router)

# Job Queue (Async Analysis)
app.include_router(jobs_router)

# Model Monitoring and Alerts
app.include_router(monitoring_router)

# Clinical Trials (Recruitment Matching)
app.include_router(clinical_trials_router)

# Publication Reports
app.include_router(report_router)

# Wearable Integration (UV Tracking)
app.include_router(wearable_router)

# Cost Transparency (price estimates, provider comparison, GoodRx)
app.include_router(cost_transparency_router)

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/")
def read_root():
    return {
        "message": "Skin Disease Analysis API v2.0 - Refactored",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """
    Basic health check endpoint.
    Returns system health status including database, memory, disk, and uptime.
    """
    import psutil
    import time
    from datetime import datetime

    # Check database connectivity
    db_status = "healthy"
    try:
        from database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    # Memory info
    memory = psutil.virtual_memory()

    # Disk info for uploads directory
    disk = psutil.disk_usage(str(uploads_dir.absolute()))

    # CPU info
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Uptime (process start time)
    process = psutil.Process()
    uptime_seconds = time.time() - process.create_time()
    uptime_str = _format_uptime(uptime_seconds)

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_percent": memory.percent,
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_percent": round(disk.percent, 1),
        },
        "cpu": {
            "percent": cpu_percent,
            "cores": psutil.cpu_count(),
        },
        "uptime": uptime_str,
        "uptime_seconds": round(uptime_seconds),
    }


def _format_uptime(seconds: float) -> str:
    """Format uptime seconds to human readable string."""
    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


@app.get("/health/models")
def model_health_check():
    """
    Check status of all loaded ML models.
    Returns model availability, device info, and memory usage.
    """
    import psutil
    import torch

    # GPU memory info if CUDA available
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        }

    # Count loaded models
    models_status = {
        "binary_model": shared.binary_model is not None,
        "lesion_model": shared.lesion_model is not None,
        "isic_model": shared.isic_model is not None,
        "isic_2020_binary_model": shared.isic_2020_binary_model is not None,
        "inflammatory_model": shared.inflammatory_model is not None,
        "infectious_model": shared.infectious_model is not None,
    }

    loaded_count = sum(1 for v in models_status.values() if v)
    total_count = len(models_status)

    return {
        "status": "healthy" if loaded_count == total_count else "degraded",
        "models": models_status,
        "summary": {
            "loaded": loaded_count,
            "total": total_count,
            "all_loaded": loaded_count == total_count,
        },
        "device": {
            "type": str(shared.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu_info,
        },
        "process_memory_mb": round(psutil.Process().memory_info().rss / (1024**2), 1),
    }


@app.get("/health/db")
def database_health_check():
    """
    Detailed database health check.
    Returns connection status, table counts, and database size.
    """
    from database import SessionLocal, User, AnalysisHistory
    from sqlalchemy import text

    try:
        db = SessionLocal()

        # Test query
        db.execute(text("SELECT 1"))

        # Get table counts
        user_count = db.query(User).count()
        analysis_count = db.query(AnalysisHistory).count()

        # Get database file size (for SQLite)
        db_size = None
        try:
            import os
            from config import DATABASE_URL
            if "sqlite" in DATABASE_URL:
                db_path = DATABASE_URL.replace("sqlite:///", "").replace("./", "")
                if os.path.exists(db_path):
                    db_size = round(os.path.getsize(db_path) / (1024**2), 2)
        except:
            pass

        db.close()

        return {
            "status": "healthy",
            "connection": "ok",
            "tables": {
                "users": user_count,
                "analyses": analysis_count,
            },
            "database_size_mb": db_size,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e),
        }


@app.get("/api/stats")
def get_request_stats():
    """Get API request statistics for monitoring."""
    stats_mw = get_stats_middleware()
    if stats_mw:
        return stats_mw.get_stats()
    return {"error": "Stats middleware not initialized"}


@app.get("/api/routers")
def list_routers():
    """List all registered routers and their status."""
    return {
        "routers": [
            {"name": "auth", "prefix": "/", "description": "Authentication, profile, settings", "endpoints": 10},
            {"name": "notifications", "prefix": "/notifications", "description": "Notifications and alerts", "endpoints": 5},
            {"name": "alerts", "prefix": "/alerts", "description": "System alerts", "endpoints": 1},
            {"name": "treatment", "prefix": "/", "description": "Treatment management", "endpoints": 6},
            {"name": "billing", "prefix": "/billing, /payments", "description": "Billing and payments", "endpoints": 7},
            {"name": "education", "prefix": "/education", "description": "Patient education", "endpoints": 5},
            {"name": "appointments", "prefix": "/appointments, /providers", "description": "Appointments and providers", "endpoints": 10},
            {"name": "tracking", "prefix": "/lesion_groups, /sun-exposure", "description": "Lesion tracking and sun exposure", "endpoints": 8},
            {"name": "analysis", "prefix": "/upload, /full_classify, /analysis", "description": "Core ML classification", "endpoints": 15},
            {"name": "clinical", "prefix": "/classify-burn, /dermoscopy, /biopsy, /photography", "description": "Clinical analysis tools", "endpoints": 12},
            {"name": "analytics", "prefix": "/population-health, /risk-calculator", "description": "Population health and risk calculator", "endpoints": 10},
            {"name": "teledermatology", "prefix": "/dermatologists, /consultations, /referrals, /second-opinions", "description": "Teledermatology services", "endpoints": 18},
            {"name": "batch", "prefix": "/batch", "description": "Batch skin checks", "endpoints": 8},
            {"name": "clinic_management", "prefix": "/api/clinics", "description": "Clinic management", "endpoints": 25}
        ],
        "total_endpoints": 140,
        "status": "all_routers_active"
    }


# =============================================================================
# MIGRATION COMPLETE
# =============================================================================
#
# All routers have been migrated from main.py to individual router files:
#
# routers/
# ├── auth_router.py          - Authentication, profile, user settings
# ├── notifications_router.py  - Notifications and alerts
# ├── treatment_router.py      - Treatment management
# ├── billing_router.py        - Billing and payments
# ├── education_router.py      - Patient education
# ├── appointments_router.py   - Appointments and providers
# ├── tracking_router.py       - Lesion tracking and sun exposure
# ├── analysis_router.py       - Core ML analysis, upload, classification
# ├── clinical_router.py       - Biopsy, dermoscopy, burns, photography
# ├── analytics_router.py      - Population health, risk calculator
# ├── teledermatology_router.py- Dermatologists, consultations, referrals
# └── batch_router.py          - Batch skin checks
#
# To run this refactored application:
#   uvicorn main_refactored:app --reload --port 8000
#
# Once verified, replace main.py with this file:
#   1. mv main.py main_legacy.py
#   2. mv main_refactored.py main.py
#
# =============================================================================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
