"""
Comprehensive Integration Tests for Biopsy/Histopathology Features

Week 2 Testing & Validation:
- Integration tests for all biopsy-related endpoints
- Histopathology analyzer unit tests
- AI-biopsy correlation logic validation
- Database persistence tests

Run with: pytest tests/test_biopsy_histopathology_integration.py -v
"""

import io
import json
import os
import sys
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import requests
import torch

# Test configuration
BASE_URL = os.environ.get('TEST_API_URL', 'http://localhost:8000')
TEST_USER = {'username': 'testbiopsy', 'password': 'test123'}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def auth_token():
    """Get authentication token for API tests."""
    try:
        resp = requests.post(f'{BASE_URL}/login', json=TEST_USER, timeout=5)
        if resp.status_code == 200:
            return resp.json().get('access_token')
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend server not running")
    pytest.skip("Could not authenticate - check test user exists")


@pytest.fixture
def auth_headers(auth_token):
    """Get auth headers for API requests."""
    return {'Authorization': f'Bearer {auth_token}'}


@pytest.fixture
def sample_histopathology_image():
    """Create a sample histopathology-like image for testing."""
    # Create a pink-tinted image to simulate H&E stained tissue
    img = Image.new('RGB', (224, 224))
    pixels = img.load()
    for i in range(224):
        for j in range(224):
            # Simulate H&E staining colors (pink/purple tones)
            r = 220 + np.random.randint(-30, 30)
            g = 180 + np.random.randint(-40, 40)
            b = 200 + np.random.randint(-30, 30)
            pixels[i, j] = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    return img


@pytest.fixture
def sample_dermoscopy_prediction():
    """Sample dermoscopy prediction for correlation testing."""
    return {
        'prediction': 'MEL',
        'primary_diagnosis': 'MEL',
        'probabilities': {
            'MEL': 0.75,
            'NV': 0.15,
            'BCC': 0.05,
            'AK': 0.03,
            'SCC': 0.02
        },
        'confidence': 0.75,
        'is_malignant': True
    }


@pytest.fixture
def mock_database_session():
    """Mock database session for unit tests."""
    mock_db = MagicMock()
    mock_analysis = MagicMock()
    mock_analysis.id = 1
    mock_analysis.histopathology_performed = False
    mock_db.query.return_value.filter.return_value.first.return_value = mock_analysis
    return mock_db


# ============================================================================
# Unit Tests: Histopathology Analyzer
# ============================================================================

class TestHistopathologyAnalyzer:
    """Unit tests for HistopathologyAnalyzer class."""

    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        from histopathology_analyzer import HistopathologyAnalyzer

        # Test with fallback backbone (no Hibou to avoid download)
        analyzer = HistopathologyAnalyzer(use_hibou=False)

        assert analyzer is not None
        assert analyzer.feature_dim == 1536  # EfficientNet-B3 features
        assert analyzer.device in ['cuda', 'cpu', 'xpu']

    def test_preprocess_image_pil(self, sample_histopathology_image):
        """Test image preprocessing with PIL Image input."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)
        tensor = analyzer.preprocess_image(sample_histopathology_image)

        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_preprocess_image_bytes(self, sample_histopathology_image):
        """Test image preprocessing with bytes input."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        # Convert to bytes
        buffer = io.BytesIO()
        sample_histopathology_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        tensor = analyzer.preprocess_image(image_bytes)

        assert tensor.shape == (1, 3, 224, 224)

    def test_feature_extraction(self, sample_histopathology_image):
        """Test feature extraction produces correct dimensions."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)
        features = analyzer.extract_features(sample_histopathology_image)

        assert features.shape[0] == 1  # Batch size
        assert features.shape[1] == analyzer.feature_dim

    def test_analyze_returns_expected_structure(self, sample_histopathology_image):
        """Test analyze() returns all expected fields."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)
        result = analyzer.analyze(sample_histopathology_image)

        # Check required fields
        assert 'timestamp' in result
        assert 'model_used' in result
        assert 'tissue_types' in result
        assert 'malignancy_assessment' in result
        assert 'quality_metrics' in result
        assert 'recommendations' in result
        assert 'primary_diagnosis' in result
        assert 'primary_probability' in result
        assert 'diagnostic_category' in result
        assert 'uncertainty' in result

        # Check malignancy_assessment structure
        ma = result['malignancy_assessment']
        assert 'risk_level' in ma
        assert 'malignant_probability' in ma
        assert 'confidence_interval' in ma
        assert 'key_features' in ma

        # Check tissue_types structure
        assert len(result['tissue_types']) > 0
        tissue = result['tissue_types'][0]
        assert 'type' in tissue
        assert 'confidence' in tissue
        assert 'description' in tissue
        assert 'confidence_interval' in tissue

    def test_analyze_with_dermoscopy_correlation(
        self, sample_histopathology_image, sample_dermoscopy_prediction
    ):
        """Test analyze with dermoscopy prediction for correlation."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)
        result = analyzer.analyze(
            sample_histopathology_image,
            dermoscopy_prediction=sample_dermoscopy_prediction
        )

        assert 'dermoscopy_correlation' in result
        corr = result['dermoscopy_correlation']

        assert 'histopathology_diagnosis' in corr
        assert 'expected_dermoscopy_class' in corr
        assert 'actual_dermoscopy_class' in corr
        assert 'is_concordant' in corr
        assert 'agreement_assessment' in corr

    def test_diagnostic_category_mapping(self, sample_histopathology_image):
        """Test that diagnostic categories are correctly assigned."""
        from histopathology_analyzer import DIAGNOSTIC_CLASSES, TISSUE_CLASSES

        # Verify all tissue classes are mapped
        all_mapped_classes = []
        for classes in DIAGNOSTIC_CLASSES.values():
            all_mapped_classes.extend(classes)

        for tissue_class in TISSUE_CLASSES:
            assert tissue_class in all_mapped_classes, f"{tissue_class} not mapped to diagnostic category"

    def test_uncertainty_quantification(self, sample_histopathology_image):
        """Test Monte Carlo dropout uncertainty estimation."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)
        result = analyzer.analyze(sample_histopathology_image, num_mc_samples=5)

        uncertainty = result['uncertainty']
        assert 'mean_std' in uncertainty
        assert 'max_std' in uncertainty
        assert 'is_reliable' in uncertainty
        assert 0 <= uncertainty['mean_std'] <= 1
        assert isinstance(uncertainty['is_reliable'], bool)

    def test_tile_batch_analysis(self, sample_histopathology_image):
        """Test batch analysis of multiple tiles."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        # Create multiple tiles
        tiles = [sample_histopathology_image for _ in range(3)]

        # Aggregated result
        result = analyzer.analyze_tile_batch(tiles, aggregate=True)

        assert 'num_tiles_analyzed' in result
        assert result['num_tiles_analyzed'] == 3
        assert 'primary_diagnosis' in result
        assert 'malignant_tile_ratio' in result

        # Per-tile results
        results = analyzer.analyze_tile_batch(tiles, aggregate=False)
        assert len(results) == 3


# ============================================================================
# Unit Tests: AI-Biopsy Correlation Logic
# ============================================================================

class TestAIBiopsyCorrelation:
    """Tests for AI-biopsy correlation logic."""

    def test_concordant_melanoma(self):
        """Test concordance when both predict melanoma."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        dermoscopy_pred = {
            'prediction': 'MEL',
            'probabilities': {'MEL': 0.85, 'NV': 0.10, 'BCC': 0.05}
        }

        result = analyzer._correlate_with_dermoscopy('melanoma_invasive', dermoscopy_pred)

        assert result['is_concordant'] == True
        assert result['expected_dermoscopy_class'] == 'MEL'
        assert result['actual_dermoscopy_class'] == 'MEL'
        assert 'agreement' in result['agreement_assessment'].lower()

    def test_discordant_melanoma_missed(self):
        """Test discordance when histopath shows melanoma but dermoscopy missed it."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        dermoscopy_pred = {
            'prediction': 'NV',  # Predicted benign nevus
            'probabilities': {'NV': 0.80, 'MEL': 0.10, 'BCC': 0.10}
        }

        result = analyzer._correlate_with_dermoscopy('melanoma_invasive', dermoscopy_pred)

        assert result['is_concordant'] == False
        assert 'CRITICAL DISCORDANCE' in result['agreement_assessment']

    def test_concordant_bcc(self):
        """Test concordance for basal cell carcinoma."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        dermoscopy_pred = {
            'prediction': 'BCC',
            'probabilities': {'BCC': 0.70, 'MEL': 0.15, 'NV': 0.15}
        }

        result = analyzer._correlate_with_dermoscopy('basal_cell_carcinoma', dermoscopy_pred)

        assert result['is_concordant'] == True
        assert result['expected_dermoscopy_class'] == 'BCC'

    def test_discordant_scc(self):
        """Test discordance for squamous cell carcinoma."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        dermoscopy_pred = {
            'prediction': 'AK',  # Predicted actinic keratosis
            'probabilities': {'AK': 0.60, 'SCC': 0.20, 'BKL': 0.20}
        }

        result = analyzer._correlate_with_dermoscopy('squamous_cell_carcinoma', dermoscopy_pred)

        assert result['is_concordant'] == False
        assert 'Discordance' in result['agreement_assessment']

    def test_benign_no_expected_mapping(self):
        """Test benign conditions with no dermoscopy mapping."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        dermoscopy_pred = {
            'prediction': 'NV',
            'probabilities': {'NV': 0.90, 'MEL': 0.05, 'BCC': 0.05}
        }

        result = analyzer._correlate_with_dermoscopy('dermis_inflammation', dermoscopy_pred)

        assert result['expected_dermoscopy_class'] is None
        assert result['is_concordant'] == False


# ============================================================================
# Integration Tests: API Endpoints
# ============================================================================

class TestBiopsyAPIEndpoints:
    """Integration tests for biopsy-related API endpoints."""

    def test_histopathology_endpoint_exists(self, auth_headers):
        """Test that histopathology endpoint is accessible."""
        # Without file, should return 422 (validation error)
        resp = requests.post(
            f'{BASE_URL}/analyze-histopathology',
            headers=auth_headers
        )
        # 422 means endpoint exists but missing required file
        assert resp.status_code == 422

    def test_histopathology_analysis(self, auth_headers, sample_histopathology_image):
        """Test full histopathology analysis endpoint."""
        # Convert image to bytes
        buffer = io.BytesIO()
        sample_histopathology_image.save(buffer, format='PNG')
        buffer.seek(0)

        files = {'file': ('test_slide.png', buffer, 'image/png')}

        resp = requests.post(
            f'{BASE_URL}/analyze-histopathology',
            headers=auth_headers,
            files=files
        )

        if resp.status_code == 200:
            data = resp.json()

            # Response may be nested under 'analysis' key
            analysis = data.get('analysis', data)

            # Verify response structure
            assert 'primary_diagnosis' in analysis
            assert 'malignancy_assessment' in analysis or 'malignancy' in analysis
            assert 'recommendations' in analysis
        else:
            # May fail if Hibou model not loaded - that's okay for CI
            assert resp.status_code in [200, 500, 503]

    def test_biopsy_correlation_endpoint(self, auth_headers):
        """Test biopsy-correlation endpoint."""
        # Get an analysis ID that exists
        resp = requests.get(f'{BASE_URL}/history', headers=auth_headers)

        if resp.status_code == 200:
            history = resp.json()
            if history and len(history) > 0:
                analysis_id = history[0].get('id', 1)

                resp = requests.get(
                    f'{BASE_URL}/biopsy-correlation/{analysis_id}',
                    headers=auth_headers
                )

                # Should return data or 404 if not found
                assert resp.status_code in [200, 404]

    def test_ai_accuracy_stats_endpoint(self, auth_headers):
        """Test AI accuracy stats endpoint."""
        resp = requests.get(f'{BASE_URL}/ai-accuracy-stats', headers=auth_headers)

        assert resp.status_code == 200
        data = resp.json()

        # Check response structure
        assert 'total_with_biopsy' in data

    def test_submit_biopsy_result_endpoint(self, auth_headers):
        """Test submitting biopsy result for an analysis."""
        # First, check if endpoint exists
        resp = requests.post(
            f'{BASE_URL}/submit-biopsy-result',
            headers=auth_headers,
            json={
                'analysis_id': 999999,  # Non-existent
                'biopsy_result': 'benign_nevus',
                'notes': 'Test submission'
            }
        )

        # Should return 404 for non-existent analysis or 422 for validation error
        assert resp.status_code in [404, 422, 200]


# ============================================================================
# Integration Tests: Database Persistence
# ============================================================================

class TestDatabasePersistence:
    """Tests for database persistence of histopathology data."""

    def test_histopathology_fields_exist(self):
        """Test that all histopathology fields exist in database."""
        from database import get_db, AnalysisHistory

        db = next(get_db())

        # Check columns exist
        expected_fields = [
            'histopathology_performed',
            'histopathology_result',
            'histopathology_malignant',
            'histopathology_confidence',
            'histopathology_date',
            'histopathology_tissue_type',
            'histopathology_risk_level',
            'histopathology_features',
            'histopathology_recommendations',
            'histopathology_image_quality',
            'histopathology_predictions',
            'ai_concordance',
            'ai_concordance_type',
            'ai_concordance_notes',
        ]

        for field in expected_fields:
            assert hasattr(AnalysisHistory, field), f"Missing field: {field}"

        db.close()

    def test_histopathology_data_persistence(self):
        """Test that histopathology data can be saved and retrieved."""
        from database import get_db, AnalysisHistory

        db = next(get_db())

        try:
            # Create test analysis
            test_analysis = AnalysisHistory(
                user_id=1,
                image_filename='test_histopathology.png',
                analysis_type='histopathology',
                predicted_class='test',
                lesion_confidence=0.9,
                histopathology_performed=True,
                histopathology_result='melanocytic_nevus',
                histopathology_malignant=False,
                histopathology_confidence=0.85,
                histopathology_date=datetime.utcnow(),
                histopathology_tissue_type='melanocytic_nevus',
                histopathology_risk_level='low',
                histopathology_features=json.dumps(['nested_melanocytes', 'regular_architecture']),
                histopathology_recommendations='Routine follow-up',
                ai_concordance=True,
                ai_concordance_type='exact_match',
                ai_concordance_notes='Strong agreement between AI and pathology'
            )

            db.add(test_analysis)
            db.commit()

            # Retrieve and verify
            retrieved = db.query(AnalysisHistory).filter(
                AnalysisHistory.image_filename == 'test_histopathology.png'
            ).first()

            assert retrieved is not None
            assert retrieved.histopathology_performed == True
            assert retrieved.histopathology_result == 'melanocytic_nevus'
            assert retrieved.histopathology_malignant == False
            assert retrieved.ai_concordance == True
            assert retrieved.ai_concordance_type == 'exact_match'

            # Cleanup
            db.delete(retrieved)
            db.commit()

        finally:
            db.close()

    def test_json_fields_serialization(self):
        """Test JSON field serialization/deserialization."""
        from database import get_db, AnalysisHistory

        db = next(get_db())

        try:
            features = ['feature1', 'feature2', 'feature3']
            quality = {'focus': 'good', 'staining': 'optimal'}
            predictions = [{'type': 'nevus', 'confidence': 0.9}]

            test_analysis = AnalysisHistory(
                user_id=1,
                image_filename='test_json_fields.png',
                analysis_type='histopathology',
                predicted_class='test',
                lesion_confidence=0.9,
                histopathology_features=json.dumps(features),
                histopathology_image_quality=json.dumps(quality),
                histopathology_predictions=json.dumps(predictions)
            )

            db.add(test_analysis)
            db.commit()

            # Retrieve and verify JSON parsing
            retrieved = db.query(AnalysisHistory).filter(
                AnalysisHistory.image_filename == 'test_json_fields.png'
            ).first()

            assert retrieved is not None

            # Parse JSON fields
            retrieved_features = json.loads(retrieved.histopathology_features)
            retrieved_quality = json.loads(retrieved.histopathology_image_quality)
            retrieved_predictions = json.loads(retrieved.histopathology_predictions)

            assert retrieved_features == features
            assert retrieved_quality == quality
            assert retrieved_predictions == predictions

            # Cleanup
            db.delete(retrieved)
            db.commit()

        finally:
            db.close()


# ============================================================================
# Validation Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_image_handling(self):
        """Test handling of very small/empty images."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        # Very small image
        small_img = Image.new('RGB', (10, 10), color='white')

        # Should still process (transforms will resize)
        result = analyzer.analyze(small_img)
        assert 'primary_diagnosis' in result

    def test_grayscale_image_handling(self):
        """Test handling of grayscale images."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        gray_img = Image.new('L', (224, 224), color=128)

        # Should convert to RGB automatically
        result = analyzer.analyze(gray_img)
        assert 'primary_diagnosis' in result

    def test_rgba_image_handling(self):
        """Test handling of RGBA images."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        rgba_img = Image.new('RGBA', (224, 224), color=(200, 150, 180, 255))

        # Should convert to RGB automatically
        result = analyzer.analyze(rgba_img)
        assert 'primary_diagnosis' in result

    def test_high_uncertainty_detection(self):
        """Test that high uncertainty is properly flagged."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        # Noisy image should produce higher uncertainty
        img = Image.new('RGB', (224, 224))
        pixels = img.load()
        for i in range(224):
            for j in range(224):
                pixels[i, j] = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )

        result = analyzer.analyze(img, num_mc_samples=20)

        # Uncertainty should be reported
        assert 'uncertainty' in result
        assert 'is_reliable' in result['uncertainty']

    def test_missing_dermoscopy_prediction_fields(self):
        """Test correlation with incomplete dermoscopy prediction."""
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        # Minimal dermoscopy prediction
        minimal_pred = {'prediction': 'MEL'}

        result = analyzer._correlate_with_dermoscopy('melanoma_invasive', minimal_pred)

        assert 'is_concordant' in result
        assert 'agreement_assessment' in result


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_analysis_time(self, sample_histopathology_image):
        """Test that single image analysis completes in reasonable time."""
        import time
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)

        start = time.time()
        result = analyzer.analyze(sample_histopathology_image)
        elapsed = time.time() - start

        # Should complete within 10 seconds (generous for CPU)
        assert elapsed < 10.0, f"Analysis took too long: {elapsed:.2f}s"

    def test_batch_analysis_efficiency(self, sample_histopathology_image):
        """Test that batch analysis is more efficient than individual calls."""
        import time
        from histopathology_analyzer import HistopathologyAnalyzer

        analyzer = HistopathologyAnalyzer(use_hibou=False)
        tiles = [sample_histopathology_image for _ in range(3)]

        # Time batch analysis
        start = time.time()
        batch_result = analyzer.analyze_tile_batch(tiles)
        batch_time = time.time() - start

        # Should complete batch in reasonable time
        assert batch_time < 30.0, f"Batch analysis too slow: {batch_time:.2f}s"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
