"""
Application Configuration

Loads configuration from environment variables with sensible defaults.
All hardcoded paths and settings should be defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (override=True means .env takes precedence over system env vars)
load_dotenv(override=True)

# =============================================================================
# BASE PATHS
# =============================================================================

# Base directory (where this config file is located)
BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skin_classifier.db")

# =============================================================================
# AUTHENTICATION
# =============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# =============================================================================
# FILE STORAGE
# =============================================================================

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
UPLOAD_DIR.mkdir(exist_ok=True)

# Maximum file upload size (in bytes) - default 10MB
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", str(10 * 1024 * 1024)))

# =============================================================================
# MODEL PATHS
# =============================================================================

# Binary lesion classifier (ResNet18)
BINARY_CLASSIFIER_PATH = Path(os.getenv(
    "BINARY_CLASSIFIER_PATH",
    str(BASE_DIR / "binary_classifier_model" / "best_resnet18_binary.pth")
))

# Lesion classification model (HuggingFace checkpoint)
LESION_MODEL_PATH = Path(os.getenv(
    "LESION_MODEL_PATH",
    str(BASE_DIR / "checkpoint-6762")
))

# ISIC 8-class model checkpoint
ISIC_MODEL_PATH = Path(os.getenv(
    "ISIC_MODEL_PATH",
    str(BASE_DIR / "checkpoints" / "isic_old" / "best_model.pth")
))

# ISIC 8-class model fallback path
ISIC_MODEL_FALLBACK_PATH = Path(os.getenv(
    "ISIC_MODEL_FALLBACK_PATH",
    str(BASE_DIR / "checkpoints" / "isic" / "best_model.pth")
))

# ISIC 2020 binary classification model
ISIC_2020_BINARY_PATH = Path(os.getenv(
    "ISIC_2020_BINARY_PATH",
    str(BASE_DIR / "checkpoints" / "isic" / "best_model.pth")
))

# Infectious disease model directory
INFECTIOUS_MODEL_DIR = Path(os.getenv(
    "INFECTIOUS_MODEL_DIR",
    str(BASE_DIR / "infectious_disease_model")
))

# DinoV2 inflammatory model (HuggingFace model ID)
INFLAMMATORY_MODEL_ID = os.getenv(
    "INFLAMMATORY_MODEL_ID",
    "Jayanth2002/dinov2-base-finetuned-SkinDisease"
)

# Pigmentation model directory (optional)
PIGMENTATION_MODEL_DIR = Path(os.getenv(
    "PIGMENTATION_MODEL_DIR",
    str(BASE_DIR / "pigmentation_model")
))

# =============================================================================
# CHECKPOINTS DIRECTORY
# =============================================================================

CHECKPOINTS_DIR = Path(os.getenv(
    "CHECKPOINTS_DIR",
    str(BASE_DIR / "checkpoints")
))

# =============================================================================
# MODEL INFERENCE SETTINGS
# =============================================================================

# Monte Carlo Dropout samples for uncertainty quantification
MC_DROPOUT_SAMPLES = int(os.getenv("MC_DROPOUT_SAMPLES", "25"))

# Image size for model inference
MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "224"))

# Confidence thresholds
HIGH_RISK_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_RISK_CONFIDENCE_THRESHOLD", "0.8"))
MEDIUM_RISK_CONFIDENCE_THRESHOLD = float(os.getenv("MEDIUM_RISK_CONFIDENCE_THRESHOLD", "0.5"))

# =============================================================================
# CORS SETTINGS
# =============================================================================

# Comma-separated list of allowed origins, or "*" for all
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() in ("true", "1", "yes")

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "server.log"))

# =============================================================================
# CELERY / REDIS CONFIGURATION
# =============================================================================

# Redis connection URL
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery broker and result backend
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Task settings
CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "120"))  # 2 minutes
CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "180"))  # 3 minutes hard limit
CELERY_TASK_RESULT_EXPIRES = int(os.getenv("CELERY_TASK_RESULT_EXPIRES", "3600"))  # 1 hour

# Worker settings
CELERY_WORKER_CONCURRENCY = int(os.getenv("CELERY_WORKER_CONCURRENCY", "2"))  # Limit for GPU memory
CELERY_WORKER_PREFETCH_MULTIPLIER = int(os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER", "1"))

# =============================================================================
# EXTERNAL SERVICES (Optional)
# =============================================================================

# Google Places API for specialist finder
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# Push notification service
PUSH_NOTIFICATION_KEY = os.getenv("PUSH_NOTIFICATION_KEY", "")

# Email service (for notifications)
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@skinclassifier.app")


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def get_config_summary():
    """Returns a summary of current configuration (for debugging)."""
    return {
        "database_url": DATABASE_URL[:20] + "..." if len(DATABASE_URL) > 20 else DATABASE_URL,
        "debug": DEBUG,
        "host": HOST,
        "port": PORT,
        "upload_dir": str(UPLOAD_DIR),
        "binary_classifier_exists": BINARY_CLASSIFIER_PATH.exists(),
        "lesion_model_exists": LESION_MODEL_PATH.exists(),
        "isic_model_exists": ISIC_MODEL_PATH.exists() or ISIC_MODEL_FALLBACK_PATH.exists(),
        "infectious_model_exists": INFECTIOUS_MODEL_DIR.exists(),
        "checkpoints_dir": str(CHECKPOINTS_DIR),
    }
