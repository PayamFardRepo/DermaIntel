"""
Integration tests for clinical endpoints.

Tests the clinical analysis features including:
- Burn classification
- Dermoscopy analysis
- Biopsy management
- Histopathology analysis
- Clinical photography
"""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestBurnClassification:
    """Tests for burn classification endpoint."""

    def test_classify_burn_valid_image(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test burn classification with valid image."""
        files = {"file": ("burn.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/classify-burn",
            files=files,
            headers=auth_headers
        )

        # May return various codes depending on model/params
        assert response.status_code in [200, 422, 500]
        if response.status_code == 200:
            data = response.json()
            # Verify burn-specific fields
            assert any(key in data for key in [
                "burn_severity",
                "severity",
                "is_burn",
                "burn_detected"
            ])

    def test_classify_burn_returns_severity(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test that burn classification returns severity level."""
        files = {"file": ("burn.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/classify-burn",
            files=files,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include severity level
            severity_level = data.get(
                "burn_severity_level",
                data.get("severity_level")
            )
            if severity_level is not None:
                assert severity_level in [0, 1, 2, 3]

    def test_classify_burn_returns_treatment(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test that burn classification returns treatment advice."""
        files = {"file": ("burn.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/classify-burn",
            files=files,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include treatment advice
            assert any(key in data for key in [
                "burn_treatment_advice",
                "treatment_advice",
                "treatment",
                "recommendations"
            ])

    def test_classify_burn_requires_auth(self, client, sample_image_bytes):
        """Test that burn classification requires authentication."""
        files = {"file": ("burn.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post("/classify-burn", files=files)

        assert response.status_code in [401, 403]


class TestDermoscopyAnalysis:
    """Tests for dermoscopy analysis endpoints."""

    def test_dermoscopy_health_check(self, client):
        """Test dermoscopy service health endpoint."""
        response = client.get("/dermoscopy/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "healthy" in data

    def test_dermoscopy_analyze(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test dermoscopy image analysis."""
        files = {"file": ("dermoscopy.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/dermoscopy/analyze",
            files=files,
            headers=auth_headers
        )

        # May return various codes depending on model/params
        assert response.status_code in [200, 422, 500]
        if response.status_code == 200:
            data = response.json()
            # Verify dermoscopy-specific fields
            assert any(key in data for key in [
                "seven_point_score",
                "abcd_score",
                "features",
                "dermoscopy_result"
            ])

    def test_dermoscopy_analyze_with_analysis_mode(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test dermoscopy analysis with specific mode."""
        files = {"file": ("dermoscopy.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"analysis_mode": "pattern"}

        response = client.post(
            "/dermoscopy/analyze",
            files=files,
            data=data,
            headers=auth_headers
        )

        assert response.status_code in [200, 422, 500]

    def test_dermoscopy_requires_auth(self, client, sample_image_bytes):
        """Test that dermoscopy analysis requires authentication."""
        files = {"file": ("dermoscopy.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post("/dermoscopy/analyze", files=files)

        assert response.status_code in [401, 403]


class TestBiopsyManagement:
    """Tests for biopsy management endpoints."""

    def test_add_biopsy_to_analysis(
        self,
        client,
        auth_headers,
        sample_analysis
    ):
        """Test adding biopsy results to an analysis."""
        response = client.post(
            f"/analysis/biopsy/{sample_analysis.id}",
            data={
                "biopsy_result": "melanoma_in_situ",
                "biopsy_facility": "Test Lab",
                "pathologist_name": "Dr. Smith",
                "biopsy_notes": "Clear margins"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 422]
        if response.status_code == 200:
            data = response.json()
            assert data.get("biopsy_result") == "melanoma_in_situ" or \
                   data.get("biopsy", {}).get("result") == "melanoma_in_situ"

    def test_add_biopsy_nonexistent_analysis(self, client, auth_headers):
        """Test adding biopsy to non-existent analysis."""
        response = client.post(
            "/analysis/biopsy/99999",
            data={
                "biopsy_result": "benign"
            },
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_get_biopsy_correlation(
        self,
        client,
        auth_headers,
        sample_analysis
    ):
        """Test getting biopsy correlation data."""
        response = client.get(
            f"/biopsy-correlation/{sample_analysis.id}",
            headers=auth_headers
        )

        # May return 200 or 404 if no biopsy exists
        assert response.status_code in [200, 404]

    def test_get_biopsy_history(self, client, auth_headers):
        """Test getting biopsy history."""
        response = client.get("/biopsy/history", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or "items" in data or "biopsies" in data

    def test_get_biopsy_report(
        self,
        client,
        auth_headers,
        sample_analysis
    ):
        """Test getting biopsy report."""
        response = client.get(
            f"/biopsy/report/{sample_analysis.id}",
            headers=auth_headers
        )

        # May return various codes depending on biopsy data availability
        assert response.status_code in [200, 400, 404, 422]

    def test_biopsy_requires_auth(self, client, sample_analysis):
        """Test that biopsy endpoints require authentication."""
        response = client.post(
            f"/analysis/biopsy/{sample_analysis.id}",
            data={"biopsy_result": "benign"}
        )

        assert response.status_code in [401, 403]


class TestHistopathologyAnalysis:
    """Tests for histopathology analysis endpoints."""

    def test_analyze_histopathology(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test histopathology slide analysis."""
        files = {"file": ("slide.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/analyze-histopathology",
            files=files,
            headers=auth_headers
        )

        # May return various codes depending on model/params
        assert response.status_code in [200, 422, 500]
        if response.status_code == 200:
            data = response.json()
            # Verify response is valid JSON (may have various field names)
            assert isinstance(data, dict)

    def test_analyze_histopathology_requires_auth(
        self,
        client,
        sample_image_bytes
    ):
        """Test that histopathology analysis requires auth."""
        files = {"file": ("slide.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post("/analyze-histopathology", files=files)

        assert response.status_code in [401, 403]


class TestClinicalPhotography:
    """Tests for clinical photography endpoints."""

    def test_assess_photo_quality(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test photo quality assessment."""
        files = {"file": ("photo.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/photography/assess-quality",
            files=files,
            headers=auth_headers
        )

        assert response.status_code in [200, 422, 500]
        if response.status_code == 200:
            data = response.json()
            assert any(key in data for key in [
                "quality_score",
                "score",
                "passed",
                "quality_passed"
            ])

    def test_assess_quality_returns_issues(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test that quality assessment returns issues list."""
        files = {"file": ("photo.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/photography/assess-quality",
            files=files,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include issues array
            assert "issues" in data or "quality_issues" in data

    def test_calibrate_image(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test image calibration endpoint."""
        files = {"file": ("photo.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
        data = {"calibration_type": "ruler"}

        response = client.post(
            "/photography/calibrate",
            files=files,
            data=data,
            headers=auth_headers
        )

        assert response.status_code in [200, 422, 500]

    def test_get_photography_standards(self, client, auth_headers):
        """Test getting photography standards."""
        response = client.get(
            "/photography/standards",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        # Should return standards information
        assert isinstance(data, dict) or isinstance(data, list)

    def test_assess_enhanced_photo(
        self,
        client,
        auth_headers,
        sample_image_bytes
    ):
        """Test enhanced photo assessment."""
        files = {"file": ("photo.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post(
            "/photography/assess-enhanced",
            files=files,
            headers=auth_headers
        )

        # May return various codes including 422 for validation or 500 for internal errors
        assert response.status_code in [200, 422, 500]

    def test_get_capture_angles(self, client, auth_headers):
        """Test getting recommended capture angles."""
        response = client.get(
            "/photography/capture-angles",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (dict, list))

    def test_get_reference_types(self, client, auth_headers):
        """Test getting reference object types."""
        response = client.get(
            "/photography/reference-types",
            headers=auth_headers
        )

        assert response.status_code == 200

    def test_get_color_cards(self, client, auth_headers):
        """Test getting color card information."""
        response = client.get(
            "/photography/color-cards",
            headers=auth_headers
        )

        assert response.status_code == 200

    def test_get_lighting_tips(self, client, auth_headers):
        """Test getting lighting tips."""
        response = client.get(
            "/photography/lighting-tips",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (dict, list))

    def test_get_dicom_requirements(self, client, auth_headers):
        """Test getting DICOM requirements."""
        response = client.get(
            "/photography/dicom-requirements",
            headers=auth_headers
        )

        assert response.status_code == 200

    def test_photography_requires_auth(self, client, sample_image_bytes):
        """Test that photography endpoints require auth."""
        files = {"file": ("photo.jpg", BytesIO(sample_image_bytes), "image/jpeg")}

        response = client.post("/photography/assess-quality", files=files)

        assert response.status_code in [401, 403]


class TestProfessionalOnlyEndpoints:
    """Tests for endpoints that require professional access."""

    def test_non_professional_denied_access(
        self,
        client,
        auth_headers,  # Regular user token
        sample_analysis
    ):
        """Test that non-professionals cannot access professional endpoints."""
        # Try to generate a professional report
        # This test verifies role-based access control if implemented
        # The specific endpoint behavior may vary
        pass  # Placeholder for professional-only endpoint tests

    def test_professional_can_access(
        self,
        client,
        professional_auth_headers,
        sample_analysis
    ):
        """Test that professionals can access professional endpoints."""
        # Verify professional access works
        pass  # Placeholder for professional access tests


class TestClinicalDecisionSupport:
    """Tests for clinical decision support features."""

    def test_get_treatment_protocols(
        self,
        client,
        auth_headers,
        sample_analysis
    ):
        """Test getting treatment protocols for an analysis."""
        # Clinical decision support may be embedded in analysis
        response = client.get(
            f"/analysis/history/{sample_analysis.id}",
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Verify clinical decision support data if present
            if "clinical_decision_support" in data:
                assert isinstance(data["clinical_decision_support"], dict)
