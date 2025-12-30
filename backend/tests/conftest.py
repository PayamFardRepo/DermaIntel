"""
Pytest configuration and shared fixtures for Skin Classifier tests.

This module provides:
- Test database setup (SQLite in-memory)
- FastAPI TestClient configuration
- Authentication fixtures (test users, tokens)
- Mock ML model fixtures
- Sample image fixtures

Performance Optimizations:
- TESTING=1 environment variable enables mock models in shared.py
- Session-scoped app fixture (loads models once per test session)
- Tests run 10-50x faster with mock models
"""

import pytest
import os
import sys
from datetime import datetime, timedelta
from typing import Generator
from unittest.mock import patch, MagicMock, Mock
import base64
from io import BytesIO

# =============================================================================
# ENABLE TEST MODE BEFORE ANY IMPORTS
# =============================================================================
# This must be set BEFORE importing shared.py to skip ML model loading
os.environ["TESTING"] = "1"

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from database import Base, User, AnalysisHistory
from auth import get_password_hash, create_access_token, SECRET_KEY, ALGORITHM


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine using SQLite in-memory."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_db(test_engine) -> Generator[Session, None, None]:
    """Create a test database session with automatic rollback."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def override_get_db(test_db):
    """Override the get_db dependency to use test database."""
    def _override_get_db():
        try:
            yield test_db
        finally:
            pass
    return _override_get_db


# =============================================================================
# APPLICATION FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def app_instance():
    """
    Create a FastAPI app instance ONCE per test session.
    This dramatically speeds up tests by avoiding repeated model loading.
    """
    from main import app as fastapi_app
    return fastapi_app


@pytest.fixture(scope="function")
def app(app_instance, override_get_db):
    """Configure the app with test database for each test."""
    from database import get_db

    app_instance.dependency_overrides[get_db] = override_get_db
    yield app_instance
    app_instance.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(app) -> Generator[TestClient, None, None]:
    """Create a TestClient for making requests to the app."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


# =============================================================================
# USER FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def test_user_data():
    """Test user data for registration."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "TestPassword123!",
        "full_name": "Test User",
        "age": 30,
        "gender": "male"
    }


@pytest.fixture(scope="function")
def test_user(test_db, test_user_data) -> User:
    """Create a test user in the database."""
    user = User(
        username=test_user_data["username"],
        email=test_user_data["email"],
        hashed_password=get_password_hash(test_user_data["password"]),
        full_name=test_user_data["full_name"],
        age=test_user_data["age"],
        gender=test_user_data["gender"],
        is_active=True,
        role="patient",
        display_mode="simple",
        created_at=datetime.utcnow()
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture(scope="function")
def professional_user(test_db) -> User:
    """Create a professional/dermatologist user for testing professional features."""
    user = User(
        username="drsmith",
        email="dr.smith@clinic.com",
        hashed_password=get_password_hash("DoctorPass123!"),
        full_name="Dr. Jane Smith",
        age=45,
        gender="female",
        is_active=True,
        role="dermatologist",
        display_mode="professional",
        is_verified_professional=True,
        professional_license_number="MD123456",
        professional_license_state="CA",
        npi_number="1234567890",
        verification_date=datetime.utcnow(),
        created_at=datetime.utcnow()
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture(scope="function")
def admin_user(test_db) -> User:
    """Create an admin user for testing admin features."""
    user = User(
        username="admin",
        email="admin@skinclassifier.com",
        hashed_password=get_password_hash("AdminPass123!"),
        full_name="Admin User",
        age=35,
        gender="other",
        is_active=True,
        role="admin",
        display_mode="professional",
        created_at=datetime.utcnow()
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture(scope="function")
def inactive_user(test_db) -> User:
    """Create an inactive user for testing inactive user handling."""
    user = User(
        username="inactive_user",
        email="inactive@example.com",
        hashed_password=get_password_hash("InactivePass123!"),
        full_name="Inactive User",
        is_active=False,
        role="patient",
        created_at=datetime.utcnow()
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


# =============================================================================
# AUTHENTICATION FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def user_token(test_user) -> str:
    """Generate a valid JWT token for the test user."""
    access_token = create_access_token(
        data={"sub": test_user.username},
        expires_delta=timedelta(minutes=30)
    )
    return access_token


@pytest.fixture(scope="function")
def professional_token(professional_user) -> str:
    """Generate a valid JWT token for the professional user."""
    access_token = create_access_token(
        data={"sub": professional_user.username},
        expires_delta=timedelta(minutes=30)
    )
    return access_token


@pytest.fixture(scope="function")
def admin_token(admin_user) -> str:
    """Generate a valid JWT token for the admin user."""
    access_token = create_access_token(
        data={"sub": admin_user.username},
        expires_delta=timedelta(minutes=30)
    )
    return access_token


@pytest.fixture(scope="function")
def expired_token(test_user) -> str:
    """Generate an expired JWT token for testing expiration."""
    access_token = create_access_token(
        data={"sub": test_user.username},
        expires_delta=timedelta(minutes=-10)  # Expired 10 minutes ago
    )
    return access_token


@pytest.fixture(scope="function")
def auth_headers(user_token) -> dict:
    """Get authorization headers for the test user."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture(scope="function")
def professional_auth_headers(professional_token) -> dict:
    """Get authorization headers for the professional user."""
    return {"Authorization": f"Bearer {professional_token}"}


@pytest.fixture(scope="function")
def admin_auth_headers(admin_token) -> dict:
    """Get authorization headers for the admin user."""
    return {"Authorization": f"Bearer {admin_token}"}


# =============================================================================
# IMAGE FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def sample_image_bytes():
    """Generate a simple test image as bytes (1x1 red pixel JPEG)."""
    # Minimal valid JPEG (1x1 red pixel)
    # This is a base64-encoded minimal JPEG image
    jpeg_base64 = (
        "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
        "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
        "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
        "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/"
        "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAB//2Q=="
    )
    return base64.b64decode(jpeg_base64)


@pytest.fixture(scope="function")
def sample_image_file(sample_image_bytes):
    """Create a file-like object from sample image bytes."""
    return BytesIO(sample_image_bytes)


@pytest.fixture(scope="function")
def sample_upload_image(sample_image_bytes):
    """Create a tuple suitable for file upload in tests."""
    return ("test_image.jpg", BytesIO(sample_image_bytes), "image/jpeg")


@pytest.fixture(scope="function")
def invalid_file_bytes():
    """Generate invalid file bytes (not an image)."""
    return b"This is not a valid image file"


# =============================================================================
# ML MODEL MOCK FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def mock_classification_result():
    """Mock result for skin lesion classification."""
    return {
        "predicted_class": "melanocytic_nevus",
        "confidence": 0.87,
        "probabilities": {
            "nv": 0.87,
            "mel": 0.05,
            "bkl": 0.03,
            "bcc": 0.02,
            "akiec": 0.02,
            "df": 0.005,
            "vasc": 0.005
        },
        "risk_level": "low",
        "risk_recommendation": "Monitor for changes. No immediate concern."
    }


@pytest.fixture(scope="function")
def mock_binary_result():
    """Mock result for binary (lesion vs non-lesion) classification."""
    return {
        "is_lesion": True,
        "confidence": 0.92,
        "probabilities": {
            "non_lesion": 0.08,
            "lesion": 0.92
        }
    }


@pytest.fixture(scope="function")
def mock_inflammatory_result():
    """Mock result for inflammatory condition classification."""
    return {
        "inflammatory_condition": "eczema",
        "confidence": 0.78,
        "probabilities": {
            "eczema": 0.78,
            "psoriasis": 0.12,
            "contact_dermatitis": 0.05,
            "seborrheic_dermatitis": 0.05
        }
    }


@pytest.fixture(scope="function")
def mock_burn_result():
    """Mock result for burn severity classification."""
    return {
        "burn_severity": "first_degree",
        "confidence": 0.85,
        "burn_severity_level": 1,
        "is_burn_detected": True,
        "burn_urgency": "Low urgency - home care appropriate",
        "burn_treatment_advice": "Cool water, aloe vera, over-the-counter pain relief",
        "burn_medical_attention_required": False
    }


@pytest.fixture(scope="function")
def mock_dermoscopy_result():
    """Mock result for dermoscopy analysis."""
    return {
        "seven_point_score": 2,
        "abcd_score": 3.5,
        "features": {
            "pigment_network": {"present": True, "atypical": False},
            "dots_globules": {"present": True, "irregular": False},
            "streaks": {"present": False},
            "blue_whitish_veil": {"present": False},
            "regression_structures": {"present": False}
        },
        "recommendation": "Benign features. Routine monitoring recommended."
    }


@pytest.fixture(scope="function")
def mock_quality_check_result():
    """Mock result for image quality check."""
    return {
        "quality_score": 0.85,
        "quality_passed": True,
        "issues": []
    }


@pytest.fixture(scope="function")
def mock_quality_check_failed():
    """Mock result for failed image quality check."""
    return {
        "quality_score": 0.35,
        "quality_passed": False,
        "issues": ["Image is too blurry", "Insufficient lighting"]
    }


@pytest.fixture(autouse=False)
def mock_ml_models(
    monkeypatch,
    mock_classification_result,
    mock_binary_result,
    mock_inflammatory_result,
    mock_quality_check_result
):
    """
    Auto-mock all ML inference calls for faster tests.

    Use this fixture to skip actual model inference in integration tests.
    Set autouse=True if you want all tests to use mocked models.
    """
    # Mock the shared classify_image function if it exists
    try:
        from shared import classify_image
        monkeypatch.setattr("shared.classify_image", lambda *args, **kwargs: mock_classification_result)
    except ImportError:
        pass

    # Return mocks for manual usage
    return {
        "classification": mock_classification_result,
        "binary": mock_binary_result,
        "inflammatory": mock_inflammatory_result,
        "quality": mock_quality_check_result
    }


# =============================================================================
# ANALYSIS HISTORY FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def sample_analysis(test_db, test_user) -> AnalysisHistory:
    """Create a sample analysis record for testing."""
    analysis = AnalysisHistory(
        user_id=test_user.id,
        image_filename="test_image.jpg",
        analysis_type="full",
        is_lesion=True,
        binary_confidence=0.92,
        predicted_class="melanocytic_nevus",
        lesion_confidence=0.87,
        lesion_probabilities={
            "nv": 0.87,
            "mel": 0.05,
            "bkl": 0.03,
            "bcc": 0.02,
            "akiec": 0.02,
            "df": 0.005,
            "vasc": 0.005
        },
        risk_level="low",
        risk_recommendation="Monitor for changes",
        image_quality_score=0.85,
        image_quality_passed=True,
        body_location="back",
        body_sublocation="upper_back",
        body_side="center"
    )
    test_db.add(analysis)
    test_db.commit()
    test_db.refresh(analysis)
    return analysis


@pytest.fixture(scope="function")
def multiple_analyses(test_db, test_user) -> list:
    """Create multiple analysis records for testing history/pagination."""
    analyses = []
    conditions = [
        ("melanocytic_nevus", "low", 0.87),
        ("basal_cell_carcinoma", "high", 0.72),
        ("actinic_keratosis", "medium", 0.65),
        ("seborrheic_keratosis", "low", 0.89),
        ("melanoma", "high", 0.58)
    ]

    for i, (condition, risk, confidence) in enumerate(conditions):
        analysis = AnalysisHistory(
            user_id=test_user.id,
            image_filename=f"test_image_{i}.jpg",
            analysis_type="full",
            is_lesion=True,
            binary_confidence=0.9,
            predicted_class=condition,
            lesion_confidence=confidence,
            risk_level=risk,
            image_quality_score=0.8,
            image_quality_passed=True
        )
        test_db.add(analysis)
        analyses.append(analysis)

    test_db.commit()
    for analysis in analyses:
        test_db.refresh(analysis)

    return analyses


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def temp_upload_dir(tmp_path):
    """Create a temporary upload directory for testing."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture(scope="session")
def test_images_dir():
    """Path to test images directory."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures",
        "test_images"
    )


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_uploads():
    """Clean up any uploaded files after each test."""
    yield
    # Cleanup logic here if needed
    pass
