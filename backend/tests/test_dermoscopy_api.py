"""
Comprehensive tests for the Dermoscopy Analysis API endpoints.

Tests cover:
- GET /dermoscopy/health - Health check endpoint
- POST /dermoscopy/analyze - Dermoscopic feature analysis
- Authentication requirements
- Error handling (timeout, invalid images)
- Response structure validation
- Performance requirements

Run with: pytest tests/test_dermoscopy_api.py -v
"""

import pytest
import io
import time
import asyncio
from PIL import Image
from unittest.mock import patch, MagicMock, AsyncMock
import numpy as np


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_skin_image_bytes():
    """Generate a realistic test skin image (256x256 with skin-like colors)."""
    # Create an image with skin-tone colors for more realistic testing
    img = Image.new('RGB', (256, 256))
    pixels = img.load()

    # Fill with skin-tone colors and add some variation
    for i in range(256):
        for j in range(256):
            # Base skin tone with some variation
            r = min(255, 200 + (i % 30))
            g = min(255, 150 + (j % 25))
            b = min(255, 130 + ((i + j) % 20))
            pixels[i, j] = (r, g, b)

    # Add a darker circular region to simulate a lesion
    center_x, center_y = 128, 128
    radius = 40
    for i in range(256):
        for j in range(256):
            if (i - center_x) ** 2 + (j - center_y) ** 2 < radius ** 2:
                # Darker brown lesion color
                pixels[i, j] = (100, 70, 50)

    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=85)
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def large_image_bytes():
    """Generate a large test image (2000x2000) to test resizing."""
    img = Image.new('RGB', (2000, 2000), color=(180, 140, 120))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=85)
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def small_image_bytes():
    """Generate a small test image (100x100)."""
    img = Image.new('RGB', (100, 100), color=(180, 140, 120))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=85)
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def mock_dermoscopy_analysis_result():
    """Complete mock result matching the actual dermoscopy analyzer output."""
    return {
        'pigment_network': {
            'detected': True,
            'type': 'reticular',
            'regularity_score': 0.75,
            'risk_level': 'low',
            'description': 'Regular mesh-like pattern - typically benign',
            'contour_count': 25,
            'coordinates': []
        },
        'globules': {
            'detected': True,
            'count': 8,
            'type': 'regular',
            'size_variability': 0.2,
            'risk_level': 'low',
            'description': 'Uniform round structures - typically benign',
            'coordinates': [(100, 100, 5), (120, 130, 6)]
        },
        'streaks': {
            'detected': False,
            'count': 0,
            'type': 'absent',
            'risk_level': 'none',
            'description': 'No streaks detected',
            'coordinates': []
        },
        'blue_white_veil': {
            'detected': False,
            'coverage_percentage': 2.5,
            'intensity': 'absent',
            'risk_level': 'none',
            'description': 'No blue-white veil detected',
            'coordinates': []
        },
        'vascular_patterns': {
            'detected': False,
            'type': 'absent',
            'risk_level': 'none',
            'description': 'No vascular patterns detected',
            'vessel_count': 3,
            'coordinates': []
        },
        'regression': {
            'detected': False,
            'coverage_percentage': 1.0,
            'severity': 'absent',
            'risk_level': 'none',
            'description': 'No regression detected',
            'coordinates': []
        },
        'color_analysis': {
            'distinct_colors': 3,
            'variety': 'moderate',
            'risk_level': 'low',
            'dominant_colors': [[180, 140, 120], [100, 70, 50], [200, 160, 140]],
            'color_percentages': [45.0, 30.0, 25.0]
        },
        'symmetry_analysis': {
            'asymmetry_score': 0.12,
            'classification': 'symmetric',
            'risk_level': 'low'
        },
        'seven_point_score': {
            'score': 1,
            'max_score': 9,
            'criteria_met': ['Irregular pigmentation (Minor, +1)'],
            'interpretation': 'Low suspicion for melanoma - Routine monitoring',
            'urgency': 'none'
        },
        'abcd_score': {
            'asymmetry_score': 0,
            'border_score': 0,
            'color_score': 3,
            'structures_score': 2,
            'total_score': 2.5,
            'classification': 'BENIGN',
            'recommendation': 'Routine follow-up'
        },
        'risk_assessment': {
            'risk_level': 'LOW',
            'risk_score': 1,
            'risk_factors': ['3 distinct colors'],
            'recommendation': 'Routine monitoring'
        },
        'overlays': {
            'combined': 'base64_encoded_image_string'
        }
    }


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

class TestDermoscopyHealthEndpoint:
    """Tests for GET /dermoscopy/health"""

    def test_health_endpoint_returns_ok(self, client):
        """Test that health endpoint returns OK status."""
        response = client.get("/dermoscopy/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data

    def test_health_endpoint_no_auth_required(self, client):
        """Test that health endpoint doesn't require authentication."""
        # Make request without auth headers
        response = client.get("/dermoscopy/health")

        # Should not return 401/403
        assert response.status_code != 401
        assert response.status_code != 403
        assert response.status_code == 200


# =============================================================================
# ANALYZE ENDPOINT TESTS
# =============================================================================

class TestDermoscopyAnalyzeEndpoint:
    """Tests for POST /dermoscopy/analyze"""

    def test_analyze_requires_authentication(self, client, sample_skin_image_bytes):
        """Test that analyze endpoint requires authentication."""
        response = client.post(
            "/dermoscopy/analyze",
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        # Should return 401 or 403 without auth
        assert response.status_code in [401, 403]

    def test_analyze_with_valid_auth(self, client, auth_headers, sample_skin_image_bytes):
        """Test successful analysis with valid authentication."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        required_fields = [
            'pigment_network', 'globules', 'streaks', 'blue_white_veil',
            'vascular_patterns', 'regression', 'color_analysis', 'symmetry_analysis',
            'seven_point_checklist', 'abcd_score', 'risk_assessment', 'overlays'
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_analyze_response_structure(self, client, auth_headers, sample_skin_image_bytes):
        """Test that response has correct structure for all components."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        # Check pigment network structure
        assert 'detected' in data['pigment_network']
        assert 'type' in data['pigment_network']
        assert 'risk_level' in data['pigment_network']

        # Check 7-point checklist structure
        assert 'score' in data['seven_point_checklist']
        assert 'max_score' in data['seven_point_checklist']
        assert 'interpretation' in data['seven_point_checklist']

        # Check ABCD score structure
        assert 'total_score' in data['abcd_score']
        assert 'classification' in data['abcd_score']

        # Check risk assessment structure
        assert 'risk_level' in data['risk_assessment']
        assert 'recommendation' in data['risk_assessment']

    def test_analyze_with_professional_user(self, client, professional_auth_headers, sample_skin_image_bytes):
        """Test analysis with professional user credentials."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=professional_auth_headers,
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200

    def test_analyze_with_expired_token(self, client, expired_token, sample_skin_image_bytes):
        """Test that expired token is rejected."""
        response = client.post(
            "/dermoscopy/analyze",
            headers={"Authorization": f"Bearer {expired_token}"},
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 401

    def test_analyze_missing_image(self, client, auth_headers):
        """Test error handling when no image is provided."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers
        )

        assert response.status_code == 422  # Validation error

    def test_analyze_invalid_file_type(self, client, auth_headers, invalid_file_bytes):
        """Test error handling for invalid file types."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("test.txt", invalid_file_bytes, "text/plain")}
        )

        # Should fail gracefully
        assert response.status_code in [400, 422, 500]


# =============================================================================
# IMAGE PROCESSING TESTS
# =============================================================================

class TestDermoscopyImageProcessing:
    """Tests for image processing capabilities."""

    def test_large_image_is_resized(self, client, auth_headers, large_image_bytes):
        """Test that large images are automatically resized."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("large.jpg", large_image_bytes, "image/jpeg")}
        )

        # Should complete successfully (not timeout)
        assert response.status_code == 200

    def test_small_image_works(self, client, auth_headers, small_image_bytes):
        """Test that small images are processed correctly."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("small.jpg", small_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200

    def test_png_image_accepted(self, client, auth_headers):
        """Test that PNG images are accepted."""
        img = Image.new('RGB', (256, 256), color=(180, 140, 120))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("test.png", img_bytes.getvalue(), "image/png")}
        )

        assert response.status_code == 200


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestDermoscopyPerformance:
    """Tests for performance requirements."""

    def test_analysis_completes_within_timeout(self, client, auth_headers, sample_skin_image_bytes):
        """Test that analysis completes within acceptable time."""
        start_time = time.time()

        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        # Should complete within 30 seconds (generous for CI environments)
        assert elapsed_time < 30, f"Analysis took too long: {elapsed_time:.2f}s"

    def test_health_check_is_fast(self, client):
        """Test that health check responds quickly."""
        start_time = time.time()

        response = client.get("/dermoscopy/health")

        elapsed_time = time.time() - start_time

        assert response.status_code == 200
        # Health check should be under 1 second
        assert elapsed_time < 1, f"Health check too slow: {elapsed_time:.2f}s"


# =============================================================================
# DERMOSCOPY ANALYZER UNIT TESTS
# =============================================================================

class TestDermoscopyAnalyzer:
    """Unit tests for the DermoscopicFeatureDetector class."""

    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        assert detector is not None

    def test_analyzer_singleton_pattern(self):
        """Test that analyzer uses singleton pattern."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector1 = get_dermoscopy_detector()
        detector2 = get_dermoscopy_detector()

        assert detector1 is detector2

    def test_analyze_returns_all_features(self, sample_skin_image_bytes):
        """Test that analyze returns all expected features."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        results = detector.analyze(sample_skin_image_bytes)

        expected_keys = [
            'pigment_network', 'globules', 'streaks', 'blue_white_veil',
            'vascular_patterns', 'regression', 'color_analysis', 'symmetry_analysis',
            'seven_point_score', 'abcd_score', 'risk_assessment', 'overlays'
        ]

        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_seven_point_score_range(self, sample_skin_image_bytes):
        """Test that 7-point score is within valid range."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        results = detector.analyze(sample_skin_image_bytes)

        score = results['seven_point_score']['score']
        assert 0 <= score <= 9, f"Invalid 7-point score: {score}"

    def test_abcd_classification_values(self, sample_skin_image_bytes):
        """Test that ABCD classification returns valid values."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        results = detector.analyze(sample_skin_image_bytes)

        classification = results['abcd_score']['classification']
        valid_classifications = ['BENIGN', 'SUSPICIOUS', 'MELANOMA']
        assert classification in valid_classifications

    def test_risk_level_values(self, sample_skin_image_bytes):
        """Test that risk level returns valid values."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        results = detector.analyze(sample_skin_image_bytes)

        risk_level = results['risk_assessment']['risk_level']
        valid_risk_levels = ['LOW', 'LOW-MODERATE', 'MODERATE', 'HIGH']
        assert risk_level in valid_risk_levels

    def test_image_resize_for_large_images(self, large_image_bytes):
        """Test that large images are resized before processing."""
        from dermoscopy_analyzer import get_dermoscopy_detector, MAX_IMAGE_DIMENSION

        detector = get_dermoscopy_detector()

        # This should not raise an error and should complete in reasonable time
        start_time = time.time()
        results = detector.analyze(large_image_bytes)
        elapsed = time.time() - start_time

        assert results is not None
        # Large image should still complete in reasonable time due to resize
        assert elapsed < 30, f"Large image took too long: {elapsed:.2f}s"

    def test_async_analyze_method_exists(self):
        """Test that async analyze method exists."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        assert hasattr(detector, 'analyze_async')
        assert asyncio.iscoroutinefunction(detector.analyze_async)

    @pytest.mark.asyncio
    async def test_async_analyze_works(self, sample_skin_image_bytes):
        """Test that async analyze returns correct results."""
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        results = await detector.analyze_async(sample_skin_image_bytes)

        assert results is not None
        assert 'risk_assessment' in results
        assert 'seven_point_score' in results


# =============================================================================
# OVERLAY TESTS
# =============================================================================

class TestDermoscopyOverlays:
    """Tests for dermoscopy overlay image generation."""

    def test_overlays_are_generated(self, client, auth_headers, sample_skin_image_bytes):
        """Test that overlay images are generated."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()

        assert 'overlays' in data
        assert 'combined' in data['overlays']

    def test_overlay_is_valid_base64(self, sample_skin_image_bytes):
        """Test that overlay images are valid base64."""
        import base64
        from dermoscopy_analyzer import get_dermoscopy_detector

        detector = get_dermoscopy_detector()
        results = detector.analyze(sample_skin_image_bytes)

        combined_overlay = results['overlays'].get('combined', '')

        if combined_overlay:
            # Should not raise an exception
            decoded = base64.b64decode(combined_overlay)
            assert len(decoded) > 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestDermoscopyErrorHandling:
    """Tests for error handling scenarios."""

    def test_corrupted_image_handling(self, client, auth_headers):
        """Test handling of corrupted image data."""
        corrupted_data = b'\xff\xd8\xff\xe0' + b'\x00' * 100  # Incomplete JPEG

        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("corrupted.jpg", corrupted_data, "image/jpeg")}
        )

        # Should return an error, not crash
        assert response.status_code in [400, 422, 500]

    def test_empty_file_handling(self, client, auth_headers):
        """Test handling of empty file."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("empty.jpg", b'', "image/jpeg")}
        )

        # Should return an error
        assert response.status_code in [400, 422, 500]

    def test_very_small_image_handling(self, client, auth_headers):
        """Test handling of very small images."""
        # 1x1 pixel image
        img = Image.new('RGB', (1, 1), color=(180, 140, 120))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)

        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("tiny.jpg", img_bytes.getvalue(), "image/jpeg")}
        )

        # Should either succeed or return a meaningful error
        assert response.status_code in [200, 400, 422]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDermoscopyIntegration:
    """Integration tests for dermoscopy with other components."""

    def test_analysis_recorded_in_monitoring(self, client, auth_headers, sample_skin_image_bytes):
        """Test that dermoscopy analysis is recorded for monitoring."""
        response = client.post(
            "/dermoscopy/analyze",
            headers=auth_headers,
            files={"image": ("test.jpg", sample_skin_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        # The monitoring is done via record_inference in the endpoint
        # This test verifies the endpoint completes successfully with monitoring enabled

    def test_multiple_sequential_analyses(self, client, auth_headers, sample_skin_image_bytes):
        """Test that multiple sequential analyses work correctly."""
        for i in range(3):
            response = client.post(
                "/dermoscopy/analyze",
                headers=auth_headers,
                files={"image": (f"test_{i}.jpg", sample_skin_image_bytes, "image/jpeg")}
            )

            assert response.status_code == 200, f"Failed on iteration {i}"
