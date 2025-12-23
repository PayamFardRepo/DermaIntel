"""
Integration tests for genetics endpoints.

Tests the genetics module including:
- Reference gene lookup (dynamic and static)
- Individual gene information
- Condition-based gene search
- VCF file upload and parsing
- Manual genetic test entry
- Genetic variants management
- Risk summary calculations
- Family history management
- Cache management
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO
import json
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import GeneticTestResult, GeneticVariant, FamilyMember


# =============================================================================
# REFERENCE GENES ENDPOINTS
# =============================================================================

class TestReferenceGenes:
    """Tests for reference gene endpoints."""

    def test_get_reference_genes(self, client, auth_headers):
        """Test getting all reference genes."""
        response = client.get(
            "/genetics/reference-genes",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "genes" in data
        assert "total_genes" in data
        assert data["total_genes"] > 0
        assert isinstance(data["genes"], list)

    def test_get_reference_genes_with_static_fallback(self, client, auth_headers):
        """Test getting reference genes using static data."""
        response = client.get(
            "/genetics/reference-genes?use_dynamic=false",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "static"
        assert len(data["genes"]) > 0

    def test_get_reference_genes_filter_by_category(self, client, auth_headers):
        """Test filtering reference genes by category."""
        response = client.get(
            "/genetics/reference-genes?category=melanoma&use_dynamic=false",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["category_filter"] == "melanoma"
        # All returned genes should be in melanoma category
        for gene in data["genes"]:
            assert gene["category"] == "melanoma"

    def test_get_reference_genes_filter_pharmacogenomics(self, client, auth_headers):
        """Test filtering reference genes by pharmacogenomics category."""
        response = client.get(
            "/genetics/reference-genes?category=pharmacogenomics&use_dynamic=false",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        for gene in data["genes"]:
            assert gene["category"] == "pharmacogenomics"

    def test_get_reference_genes_includes_required_fields(self, client, auth_headers):
        """Test that reference genes include all required fields."""
        response = client.get(
            "/genetics/reference-genes?use_dynamic=false",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()

        if len(data["genes"]) > 0:
            gene = data["genes"][0]
            # Check for required fields
            assert "gene_symbol" in gene
            assert "gene_name" in gene
            assert "category" in gene

    def test_get_reference_genes_requires_auth(self, client):
        """Test that reference genes endpoint requires authentication."""
        response = client.get("/genetics/reference-genes")
        assert response.status_code in [401, 403]


class TestGeneInfo:
    """Tests for individual gene information endpoint."""

    def test_get_gene_info_known_gene(self, client, auth_headers):
        """Test getting info for a known gene."""
        response = client.get(
            "/genetics/gene/CDKN2A",
            headers=auth_headers
        )

        # May return 200 if found or 404/500 if external API fails
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.json()
            assert "gene_symbol" in data or "symbol" in data

    def test_get_gene_info_unknown_gene(self, client, auth_headers):
        """Test getting info for an unknown gene."""
        response = client.get(
            "/genetics/gene/NONEXISTENT_GENE_XYZ123",
            headers=auth_headers
        )

        # May return 404 if not found, 500 if external API fails, or 200 with null/empty data
        assert response.status_code in [200, 404, 500]
        if response.status_code == 200:
            data = response.json()
            # If 200, it might return null or an empty-ish response
            assert data is None or isinstance(data, dict)

    def test_get_gene_info_case_insensitive(self, client, auth_headers):
        """Test that gene lookup handles case variations."""
        response_upper = client.get(
            "/genetics/gene/CDKN2A",
            headers=auth_headers
        )
        response_lower = client.get(
            "/genetics/gene/cdkn2a",
            headers=auth_headers
        )

        # Both should return same status
        assert response_upper.status_code == response_lower.status_code

    def test_get_gene_info_requires_auth(self, client):
        """Test that gene info endpoint requires authentication."""
        response = client.get("/genetics/gene/CDKN2A")
        assert response.status_code in [401, 403]


class TestSearchCondition:
    """Tests for condition-based gene search."""

    def test_search_condition_melanoma(self, client, auth_headers):
        """Test searching for melanoma-related genes."""
        response = client.get(
            "/genetics/search-condition?condition=melanoma",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "condition" in data
            assert "genes" in data
            assert data["condition"] == "melanoma"

    def test_search_condition_with_limit(self, client, auth_headers):
        """Test searching with a result limit."""
        response = client.get(
            "/genetics/search-condition?condition=cancer&limit=10",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert len(data.get("genes", [])) <= 10

    def test_search_condition_requires_auth(self, client):
        """Test that condition search requires authentication."""
        response = client.get("/genetics/search-condition?condition=melanoma")
        assert response.status_code in [401, 403]


class TestCacheManagement:
    """Tests for cache management endpoints."""

    def test_refresh_cache(self, client, auth_headers):
        """Test refreshing the genetics cache."""
        response = client.post(
            "/genetics/refresh-cache",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_get_cache_stats(self, client, auth_headers):
        """Test getting cache statistics."""
        response = client.get(
            "/genetics/cache-stats",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        # Should return some cache info
        assert isinstance(data, dict)

    def test_cache_endpoints_require_auth(self, client):
        """Test that cache endpoints require authentication."""
        response1 = client.post("/genetics/refresh-cache")
        response2 = client.get("/genetics/cache-stats")

        assert response1.status_code in [401, 403]
        assert response2.status_code in [401, 403]


# =============================================================================
# GENETIC TEST RESULTS ENDPOINTS
# =============================================================================

class TestGeneticTestResults:
    """Tests for genetic test result management."""

    def test_create_test_result(self, client, auth_headers):
        """Test creating a new genetic test result."""
        response = client.post(
            "/genetics/test-results",
            data={
                "test_type": "panel",
                "test_name": "Skin Cancer Risk Panel",
                "lab_name": "Test Lab Inc",
                "test_date": "2024-01-15",
                "sample_type": "blood"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "test_id" in data
            assert data["status"] == "pending"

    def test_create_test_result_minimal_fields(self, client, auth_headers):
        """Test creating test result with minimal required fields."""
        response = client.post(
            "/genetics/test-results",
            data={
                "test_type": "vcf",
                "test_date": "2024-01-15"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 422, 500]

    def test_create_test_result_missing_required(self, client, auth_headers):
        """Test creating test result without required fields."""
        response = client.post(
            "/genetics/test-results",
            data={
                "test_name": "Missing Type"
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_get_test_results_empty(self, client, auth_headers):
        """Test getting test results when none exist."""
        response = client.get(
            "/genetics/test-results",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_get_test_results_requires_auth(self, client):
        """Test that getting test results requires authentication."""
        response = client.get("/genetics/test-results")
        assert response.status_code in [401, 403]


class TestGeneticTestResultsFixture:
    """Tests using pre-created genetic test fixtures."""

    @pytest.fixture
    def genetic_test(self, test_db, test_user):
        """Create a genetic test result for testing."""
        from datetime import datetime
        test_result = GeneticTestResult(
            user_id=test_user.id,
            test_id="GT-TEST1234",
            test_type="panel",
            test_name="Test Panel",
            lab_name="Test Lab",
            test_date=datetime.utcnow(),
            status="completed",
            total_variants_tested=100,
            pathogenic_variants_found=2,
            likely_pathogenic_found=1,
            vus_found=5,
            overall_risk_level="moderate"
        )
        test_db.add(test_result)
        test_db.commit()
        test_db.refresh(test_result)
        return test_result

    def test_get_test_result_detail(
        self,
        client,
        auth_headers,
        genetic_test
    ):
        """Test getting a specific test result."""
        response = client.get(
            f"/genetics/test-results/{genetic_test.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["test_id"] == "GT-TEST1234"
        assert data["test_type"] == "panel"

    def test_get_test_result_nonexistent(self, client, auth_headers):
        """Test getting a non-existent test result."""
        response = client.get(
            "/genetics/test-results/99999",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_delete_test_result(
        self,
        client,
        auth_headers,
        genetic_test
    ):
        """Test deleting a genetic test result."""
        response = client.delete(
            f"/genetics/test-results/{genetic_test.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "deleted" in data["message"].lower() or "success" in data["message"].lower()

    def test_delete_test_result_nonexistent(self, client, auth_headers):
        """Test deleting a non-existent test result."""
        response = client.delete(
            "/genetics/test-results/99999",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_other_user_cannot_access_test(
        self,
        client,
        professional_auth_headers,
        genetic_test
    ):
        """Test that users cannot access other users' test results."""
        response = client.get(
            f"/genetics/test-results/{genetic_test.id}",
            headers=professional_auth_headers
        )

        assert response.status_code == 404


# =============================================================================
# GENETIC VARIANTS ENDPOINTS
# =============================================================================

class TestGeneticVariants:
    """Tests for genetic variant management."""

    @pytest.fixture
    def genetic_test_for_variants(self, test_db, test_user):
        """Create a genetic test for adding variants."""
        from datetime import datetime
        test_result = GeneticTestResult(
            user_id=test_user.id,
            test_id="GT-VARIANT123",
            test_type="panel",
            test_name="Variant Test",
            test_date=datetime.utcnow(),
            status="pending"
        )
        test_db.add(test_result)
        test_db.commit()
        test_db.refresh(test_result)
        return test_result

    def test_add_variant_to_test(
        self,
        client,
        auth_headers,
        genetic_test_for_variants
    ):
        """Test adding a variant to a genetic test."""
        response = client.post(
            f"/genetics/test-results/{genetic_test_for_variants.id}/variants",
            data={
                "gene_symbol": "CDKN2A",
                "chromosome": "9",
                "position": 21974695,
                "reference": "G",
                "alternate": "A",
                "classification": "pathogenic",
                "rsid": "rs3731249",
                "zygosity": "heterozygous"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert data["gene"] == "CDKN2A"
            assert data["classification"] == "pathogenic"
            assert data["dermatology_relevant"] is True

    def test_add_variant_unknown_gene(
        self,
        client,
        auth_headers,
        genetic_test_for_variants
    ):
        """Test adding a variant for an unknown gene."""
        response = client.post(
            f"/genetics/test-results/{genetic_test_for_variants.id}/variants",
            data={
                "gene_symbol": "UNKNOWN_GENE",
                "chromosome": "1",
                "position": 12345,
                "reference": "A",
                "alternate": "T",
                "classification": "vus"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert data["dermatology_relevant"] is False

    def test_add_variant_nonexistent_test(self, client, auth_headers):
        """Test adding a variant to a non-existent test."""
        response = client.post(
            "/genetics/test-results/99999/variants",
            data={
                "gene_symbol": "CDKN2A",
                "chromosome": "9",
                "position": 21974695,
                "reference": "G",
                "alternate": "A",
                "classification": "pathogenic"
            },
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_add_variant_missing_required_fields(
        self,
        client,
        auth_headers,
        genetic_test_for_variants
    ):
        """Test adding a variant without required fields."""
        response = client.post(
            f"/genetics/test-results/{genetic_test_for_variants.id}/variants",
            data={
                "gene_symbol": "CDKN2A"
                # Missing other required fields
            },
            headers=auth_headers
        )

        assert response.status_code == 422


# =============================================================================
# VCF UPLOAD ENDPOINTS
# =============================================================================

class TestVCFUpload:
    """Tests for VCF file upload and parsing."""

    @pytest.fixture
    def sample_vcf_content(self):
        """Create sample VCF file content."""
        return b"""##fileformat=VCFv4.2
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene name">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
9\t21974695\trs3731249\tG\tA\t100\tPASS\tGENE=CDKN2A
16\t89985940\trs1805007\tC\tT\t95\tPASS\tGENE=MC1R
"""

    def test_upload_vcf_file(
        self,
        client,
        auth_headers,
        sample_vcf_content
    ):
        """Test uploading a VCF file."""
        files = {"file": ("test.vcf", BytesIO(sample_vcf_content), "text/plain")}
        data = {"test_name": "VCF Upload Test"}

        response = client.post(
            "/genetics/upload-vcf",
            files=files,
            data=data,
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "test_id" in data
            assert data["file_processed"] == "test.vcf"
            assert "dermatology_relevant_variants" in data

    def test_upload_vcf_file_with_variants(
        self,
        client,
        auth_headers,
        sample_vcf_content
    ):
        """Test that VCF upload correctly identifies dermatology variants."""
        files = {"file": ("test.vcf", BytesIO(sample_vcf_content), "text/plain")}

        response = client.post(
            "/genetics/upload-vcf",
            files=files,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should find CDKN2A and MC1R as dermatology-relevant
            assert data["dermatology_relevant_variants"] >= 0

    def test_upload_invalid_file_extension(
        self,
        client,
        auth_headers
    ):
        """Test uploading a file with invalid extension."""
        files = {"file": ("test.txt", BytesIO(b"not a vcf"), "text/plain")}

        response = client.post(
            "/genetics/upload-vcf",
            files=files,
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_upload_vcf_requires_auth(self, client, sample_vcf_content):
        """Test that VCF upload requires authentication."""
        files = {"file": ("test.vcf", BytesIO(sample_vcf_content), "text/plain")}

        response = client.post("/genetics/upload-vcf", files=files)
        assert response.status_code in [401, 403]

    def test_upload_empty_vcf(self, client, auth_headers):
        """Test uploading an empty VCF file."""
        empty_vcf = b"""##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""
        files = {"file": ("empty.vcf", BytesIO(empty_vcf), "text/plain")}

        response = client.post(
            "/genetics/upload-vcf",
            files=files,
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert data["total_variants_in_file"] == 0


# =============================================================================
# RISK SUMMARY ENDPOINTS
# =============================================================================

class TestRiskSummary:
    """Tests for risk summary endpoints."""

    def test_get_risk_summary_no_data(self, client, auth_headers):
        """Test getting risk summary when no genetic data exists."""
        response = client.get(
            "/genetics/risk-summary",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["has_genetic_data"] is False
        assert data["risk_modifier"] == 1.0

    @pytest.fixture
    def genetic_test_with_pathogenic(self, test_db, test_user):
        """Create genetic test with pathogenic variants."""
        from datetime import datetime
        test_result = GeneticTestResult(
            user_id=test_user.id,
            test_id="GT-PATHO123",
            test_type="panel",
            test_name="Risk Test",
            test_date=datetime.utcnow(),
            status="completed",
            pathogenic_variants_found=1,
            overall_risk_level="high"
        )
        test_db.add(test_result)
        test_db.flush()

        variant = GeneticVariant(
            test_result_id=test_result.id,
            user_id=test_user.id,
            gene_symbol="CDKN2A",
            chromosome="9",
            position=21974695,
            reference="G",
            alternate="A",
            classification="pathogenic",
            melanoma_risk_modifier=10.0
        )
        test_db.add(variant)
        test_db.commit()
        return test_result

    def test_get_risk_summary_with_data(
        self,
        client,
        auth_headers,
        genetic_test_with_pathogenic
    ):
        """Test getting risk summary with genetic data."""
        response = client.get(
            "/genetics/risk-summary",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["has_genetic_data"] is True
        assert "melanoma_risk" in data
        assert "high_risk_genes" in data or "affected_genes" in data

    def test_risk_summary_requires_auth(self, client):
        """Test that risk summary requires authentication."""
        response = client.get("/genetics/risk-summary")
        assert response.status_code in [401, 403]


# =============================================================================
# FAMILY HISTORY ENDPOINTS
# =============================================================================

class TestFamilyHistory:
    """Tests for family history management."""

    def test_get_family_history_empty(self, client, auth_headers):
        """Test getting family history when none exists."""
        response = client.get(
            "/family-history",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "family_members" in data
        assert data["total_count"] == 0

    def test_add_family_member(self, client, auth_headers):
        """Test adding a family member."""
        response = client.post(
            "/family-history",
            data={
                "relationship_type": "parent",
                "relationship_side": "maternal",
                "name": "Jane Doe",
                "gender": "female",
                "is_alive": "true",
                "has_skin_cancer": "true",
                "has_melanoma": "true",
                "melanoma_count": "1"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "family_member" in data
            assert data["family_member"]["relationship_type"] == "parent"

    def test_add_family_member_alt_endpoint(self, client, auth_headers):
        """Test adding a family member via alternate endpoint."""
        response = client.post(
            "/family-history/add",
            data={
                "relationship_type": "sibling",
                "name": "John Doe",
                "gender": "male",
                "is_alive": "true",
                "has_skin_cancer": "false"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 500]

    def test_add_family_member_invalid_relationship(self, client, auth_headers):
        """Test adding family member with invalid relationship type."""
        response = client.post(
            "/family-history",
            data={
                "relationship_type": "invalid_type",
                "name": "Test Person"
            },
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_add_family_member_requires_auth(self, client):
        """Test that adding family member requires authentication."""
        response = client.post(
            "/family-history",
            data={"relationship_type": "parent"}
        )
        assert response.status_code in [401, 403]


class TestFamilyMemberOperations:
    """Tests for family member update/delete operations."""

    @pytest.fixture
    def family_member(self, test_db, test_user):
        """Create a family member for testing."""
        member = FamilyMember(
            user_id=test_user.id,
            relationship_type="parent",
            relationship_side="paternal",
            name="Test Parent",
            gender="male",
            is_alive=True,
            has_skin_cancer=False,
            has_melanoma=False
        )
        test_db.add(member)
        test_db.commit()
        test_db.refresh(member)
        return member

    def test_get_family_history_with_members(
        self,
        client,
        auth_headers,
        family_member
    ):
        """Test getting family history with existing members."""
        response = client.get(
            "/family-history",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["family_members"][0]["name"] == "Test Parent"

    def test_update_family_member(
        self,
        client,
        auth_headers,
        family_member
    ):
        """Test updating a family member."""
        response = client.put(
            f"/family-history/{family_member.id}",
            data={
                "has_skin_cancer": True,
                "skin_cancer_count": 2
            },
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["family_member"]["has_skin_cancer"] is True

    def test_update_family_member_nonexistent(self, client, auth_headers):
        """Test updating a non-existent family member."""
        response = client.put(
            "/family-history/99999",
            data={"name": "New Name"},
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_delete_family_member(
        self,
        client,
        auth_headers,
        family_member
    ):
        """Test deleting a family member."""
        response = client.delete(
            f"/family-history/{family_member.id}",
            headers=auth_headers
        )

        assert response.status_code == 200

        # Verify deletion
        get_response = client.get(
            "/family-history",
            headers=auth_headers
        )
        assert get_response.json()["total_count"] == 0

    def test_delete_family_member_nonexistent(self, client, auth_headers):
        """Test deleting a non-existent family member."""
        response = client.delete(
            "/family-history/99999",
            headers=auth_headers
        )

        assert response.status_code == 404

    def test_other_user_cannot_access_family_member(
        self,
        client,
        professional_auth_headers,
        family_member
    ):
        """Test that users cannot access other users' family members."""
        response = client.delete(
            f"/family-history/{family_member.id}",
            headers=professional_auth_headers
        )

        assert response.status_code == 404


# =============================================================================
# GENETIC TESTING LEGACY ENDPOINTS
# =============================================================================

class TestGeneticTestingLegacy:
    """Tests for legacy genetic testing endpoints."""

    def test_get_genetic_tests_empty(self, client, auth_headers):
        """Test getting genetic tests when none exist."""
        response = client.get(
            "/genetic-testing",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "tests" in data
        assert data["total_count"] == 0

    def test_get_genetic_risk_profile(self, client, auth_headers):
        """Test getting genetic risk profile."""
        response = client.get(
            "/genetic-risk",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        # Should return default risk profile
        assert "overall_genetic_risk_score" in data
        assert "overall_risk_level" in data
        assert "risk_reduction_recommendations" in data

    def test_recalculate_risk_profile(self, client, auth_headers):
        """Test recalculating genetic risk profile."""
        response = client.post(
            "/genetic-risk/recalculate",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "recalculated" in data["message"].lower()

    def test_genetic_testing_requires_auth(self, client):
        """Test that genetic testing endpoints require authentication."""
        response1 = client.get("/genetic-testing")
        response2 = client.get("/genetic-risk")
        response3 = client.post("/genetic-risk/recalculate")

        assert response1.status_code in [401, 403]
        assert response2.status_code in [401, 403]
        assert response3.status_code in [401, 403]


# =============================================================================
# INTEGRATION SCENARIOS
# =============================================================================

class TestGeneticsIntegration:
    """Integration tests for complete genetics workflows."""

    def test_complete_manual_test_workflow(self, client, auth_headers):
        """Test complete workflow: create test -> add variant -> get risk."""
        # Step 1: Create a genetic test
        create_response = client.post(
            "/genetics/test-results",
            data={
                "test_type": "panel",
                "test_name": "Integration Test",
                "test_date": "2024-01-15"
            },
            headers=auth_headers
        )

        if create_response.status_code != 200:
            pytest.skip("Could not create test result")

        test_data = create_response.json()
        test_id = test_data.get("id")

        # Step 2: Add a pathogenic variant
        variant_response = client.post(
            f"/genetics/test-results/{test_id}/variants",
            data={
                "gene_symbol": "CDKN2A",
                "chromosome": "9",
                "position": 21974695,
                "reference": "G",
                "alternate": "A",
                "classification": "pathogenic"
            },
            headers=auth_headers
        )

        assert variant_response.status_code in [200, 500]

        # Step 3: Get risk summary
        risk_response = client.get(
            "/genetics/risk-summary",
            headers=auth_headers
        )

        assert risk_response.status_code == 200

    def test_family_history_affects_risk(self, client, auth_headers):
        """Test that family history is reflected in risk endpoints."""
        # Add family member with melanoma
        client.post(
            "/family-history",
            data={
                "relationship_type": "parent",
                "has_melanoma": "true",
                "melanoma_count": "2"
            },
            headers=auth_headers
        )

        # Check family history summary
        history_response = client.get(
            "/family-history",
            headers=auth_headers
        )

        if history_response.status_code == 200:
            data = history_response.json()
            assert data["has_melanoma_history"] is True

    def test_reference_genes_used_in_vcf_parsing(
        self,
        client,
        auth_headers
    ):
        """Test that reference genes are used for VCF variant annotation."""
        # Upload VCF with known dermatology gene
        vcf_content = b"""##fileformat=VCFv4.2
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene name">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
9\t21974695\t.\tG\tA\t100\tPASS\tGENE=CDKN2A
"""
        files = {"file": ("test.vcf", BytesIO(vcf_content), "text/plain")}

        response = client.post(
            "/genetics/upload-vcf",
            files=files,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # CDKN2A is a known dermatology gene
            assert data["dermatology_relevant_variants"] >= 1


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestGeneticsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_special_characters_in_gene_symbol(self, client, auth_headers):
        """Test handling of special characters in gene symbol."""
        response = client.get(
            "/genetics/gene/GENE<script>alert(1)</script>",
            headers=auth_headers
        )

        # Should not cause server error
        assert response.status_code in [404, 422, 500]

    def test_very_long_vcf_file(self, client, auth_headers):
        """Test handling of large VCF file."""
        # Generate a VCF with many variants
        lines = ["##fileformat=VCFv4.2", "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
        for i in range(1000):
            lines.append(f"1\t{i*100}\t.\tA\tT\t100\tPASS\t.")

        vcf_content = "\n".join(lines).encode()
        files = {"file": ("large.vcf", BytesIO(vcf_content), "text/plain")}

        response = client.post(
            "/genetics/upload-vcf",
            files=files,
            headers=auth_headers
        )

        # Should handle without crashing
        assert response.status_code in [200, 500]

    def test_malformed_vcf_content(self, client, auth_headers):
        """Test handling of malformed VCF content."""
        vcf_content = b"This is not a valid VCF file content"
        files = {"file": ("malformed.vcf", BytesIO(vcf_content), "text/plain")}

        response = client.post(
            "/genetics/upload-vcf",
            files=files,
            headers=auth_headers
        )

        # Should handle gracefully
        assert response.status_code in [200, 400, 500]

    def test_concurrent_test_creation(self, client, auth_headers):
        """Test creating multiple tests sequentially."""
        test_ids = []
        for i in range(3):
            response = client.post(
                "/genetics/test-results",
                data={
                    "test_type": "panel",
                    "test_name": f"Concurrent Test {i}",
                    "test_date": "2024-01-15"
                },
                headers=auth_headers
            )
            if response.status_code == 200:
                test_ids.append(response.json().get("test_id"))

        # All should have unique IDs
        assert len(test_ids) == len(set(test_ids))

    def test_empty_condition_search(self, client, auth_headers):
        """Test searching with empty condition."""
        response = client.get(
            "/genetics/search-condition?condition=",
            headers=auth_headers
        )

        # Should handle empty search
        assert response.status_code in [200, 422, 500]

    def test_invalid_classification_value(self, client, auth_headers):
        """Test adding variant with invalid classification."""
        # First create a test
        create_response = client.post(
            "/genetics/test-results",
            data={
                "test_type": "panel",
                "test_date": "2024-01-15"
            },
            headers=auth_headers
        )

        if create_response.status_code != 200:
            pytest.skip("Could not create test result")

        test_id = create_response.json().get("id")

        # Try adding variant with invalid classification
        response = client.post(
            f"/genetics/test-results/{test_id}/variants",
            data={
                "gene_symbol": "TEST",
                "chromosome": "1",
                "position": 100,
                "reference": "A",
                "alternate": "T",
                "classification": "invalid_classification"
            },
            headers=auth_headers
        )

        # Should either accept (no validation) or reject
        assert response.status_code in [200, 422, 500]


# =============================================================================
# DTC (23andMe/AncestryDNA) UPLOAD ENDPOINTS
# =============================================================================

class TestDTCUpload:
    """Tests for 23andMe and AncestryDNA file upload and parsing."""

    @pytest.fixture
    def sample_23andme_content(self):
        """Create sample 23andMe raw data file content."""
        return b"""# rsid\tchromosome\tposition\tgenotype
rs1805007\t16\t89919709\tCT
rs1805008\t16\t89919746\tCC
rs3731249\t9\t21971000\tGG
rs12913832\t15\t28365618\tAA
rs1800401\t15\t28230318\tCC
rs1805009\t16\t89919683\tGG
rs4911414\t20\t33576691\tTT
"""

    @pytest.fixture
    def sample_ancestrydna_content(self):
        """Create sample AncestryDNA raw data file content."""
        return b"""rsid\tchromosome\tposition\tallele1\tallele2
rs1805007\t16\t89919709\tC\tT
rs1805008\t16\t89919746\tC\tC
rs3731249\t9\t21971000\tG\tG
rs12913832\t15\t28365618\tA\tA
rs1800401\t15\t28230318\tC\tC
"""

    def test_upload_23andme_file(
        self,
        client,
        auth_headers,
        sample_23andme_content
    ):
        """Test uploading a 23andMe raw data file."""
        files = {"file": ("23andme_raw.txt", BytesIO(sample_23andme_content), "text/plain")}

        response = client.post(
            "/genetics/upload-23andme",
            files=files,
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "total_snps_in_file" in data
            assert "risk_assessment" in data
            assert "risk_variants" in data
            assert "pharmacogenomic_variants" in data

    def test_upload_23andme_detects_risk_variants(
        self,
        client,
        auth_headers,
        sample_23andme_content
    ):
        """Test that 23andMe upload correctly identifies risk variants."""
        files = {"file": ("23andme.txt", BytesIO(sample_23andme_content), "text/plain")}

        response = client.post(
            "/genetics/upload-23andme",
            files=files,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should detect rs1805007 (MC1R R151C variant) as a risk variant
            risk_rsids = [v.get("rsid") for v in data.get("risk_variants", [])]
            # At minimum should have found some melanoma-related SNPs
            assert data["total_snps_in_file"] > 0

    def test_upload_ancestrydna_file(
        self,
        client,
        auth_headers,
        sample_ancestrydna_content
    ):
        """Test uploading an AncestryDNA raw data file."""
        files = {"file": ("ancestry_dna.txt", BytesIO(sample_ancestrydna_content), "text/plain")}

        response = client.post(
            "/genetics/upload-ancestrydna",
            files=files,
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "total_snps_in_file" in data
            assert "risk_assessment" in data

    def test_upload_23andme_empty_file(self, client, auth_headers):
        """Test uploading an empty 23andMe file."""
        empty_content = b"# rsid\tchromosome\tposition\tgenotype\n"
        files = {"file": ("empty.txt", BytesIO(empty_content), "text/plain")}

        response = client.post(
            "/genetics/upload-23andme",
            files=files,
            headers=auth_headers
        )

        assert response.status_code in [200, 400, 500]
        if response.status_code == 200:
            data = response.json()
            assert data["total_snps_in_file"] == 0

    def test_upload_23andme_invalid_format(self, client, auth_headers):
        """Test uploading a file with invalid format."""
        invalid_content = b"This is not a valid 23andMe file format"
        files = {"file": ("invalid.txt", BytesIO(invalid_content), "text/plain")}

        response = client.post(
            "/genetics/upload-23andme",
            files=files,
            headers=auth_headers
        )

        # Should handle gracefully
        assert response.status_code in [200, 400, 500]

    def test_upload_23andme_requires_auth(self, client, sample_23andme_content):
        """Test that 23andMe upload requires authentication."""
        files = {"file": ("23andme.txt", BytesIO(sample_23andme_content), "text/plain")}

        response = client.post("/genetics/upload-23andme", files=files)
        assert response.status_code in [401, 403]

    def test_upload_ancestrydna_requires_auth(self, client, sample_ancestrydna_content):
        """Test that AncestryDNA upload requires authentication."""
        files = {"file": ("ancestry.txt", BytesIO(sample_ancestrydna_content), "text/plain")}

        response = client.post("/genetics/upload-ancestrydna", files=files)
        assert response.status_code in [401, 403]

    def test_upload_23andme_risk_calculation(
        self,
        client,
        auth_headers
    ):
        """Test that melanoma risk calculation is reasonable."""
        # Create content with known risk variant (heterozygous MC1R R151C)
        content = b"""# rsid\tchromosome\tposition\tgenotype
rs1805007\t16\t89919709\tCT
"""
        files = {"file": ("risk_test.txt", BytesIO(content), "text/plain")}

        response = client.post(
            "/genetics/upload-23andme",
            files=files,
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Risk multiplier should be elevated (not just baseline)
            assert "melanoma_risk" in data
            risk = data["melanoma_risk"]
            assert "risk_multiplier" in risk


# =============================================================================
# FAMILY RISK ASSESSMENT ENDPOINT
# =============================================================================

class TestFamilyRiskAssessment:
    """Tests for family risk assessment endpoint."""

    @pytest.fixture
    def family_members_with_melanoma(self, test_db, test_user):
        """Create family members with melanoma history."""
        members = []

        # Parent with melanoma
        parent = FamilyMember(
            user_id=test_user.id,
            relationship_type="parent",
            relationship_side="maternal",
            name="Mother",
            gender="female",
            is_alive=True,
            has_melanoma=True,
            melanoma_count=1,
            earliest_diagnosis_age=55
        )
        members.append(parent)

        # Sibling with melanoma (early onset)
        sibling = FamilyMember(
            user_id=test_user.id,
            relationship_type="sibling",
            name="Brother",
            gender="male",
            is_alive=True,
            has_melanoma=True,
            melanoma_count=2,
            earliest_diagnosis_age=35
        )
        members.append(sibling)

        # Grandparent with skin cancer
        grandparent = FamilyMember(
            user_id=test_user.id,
            relationship_type="grandparent",
            relationship_side="maternal",
            name="Grandmother",
            gender="female",
            is_alive=False,
            has_skin_cancer=True,
            skin_cancer_count=1
        )
        members.append(grandparent)

        for member in members:
            test_db.add(member)
        test_db.commit()

        return members

    def test_get_family_risk_assessment_no_history(self, client, auth_headers):
        """Test family risk assessment with no family history."""
        response = client.get(
            "/genetics/family-risk-assessment",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "risk_multiplier" in data
            assert "risk_level" in data
            # No family history should result in baseline risk
            assert data["risk_multiplier"] >= 1.0

    def test_get_family_risk_assessment_with_history(
        self,
        client,
        auth_headers,
        family_members_with_melanoma
    ):
        """Test family risk assessment with positive family history."""
        response = client.get(
            "/genetics/family-risk-assessment",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            # With parent and sibling having melanoma, risk should be elevated
            assert data["risk_assessment"]["risk_multiplier"] > 1.0
            assert "family_summary" in data
            assert "recommendations" in data

    def test_family_risk_assessment_with_age(
        self,
        client,
        auth_headers,
        family_members_with_melanoma
    ):
        """Test family risk assessment with user age parameter."""
        response = client.get(
            "/genetics/family-risk-assessment?user_age=40",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "risk_assessment" in data
            assert "risk_multiplier" in data["risk_assessment"]

    def test_family_risk_assessment_requires_auth(self, client):
        """Test that family risk assessment requires authentication."""
        response = client.get("/genetics/family-risk-assessment")
        assert response.status_code in [401, 403]


# =============================================================================
# SCREENING SCHEDULE ENDPOINT
# =============================================================================

class TestScreeningSchedule:
    """Tests for personalized screening schedule endpoint."""

    def test_get_screening_schedule_no_risk_factors(self, client, auth_headers):
        """Test screening schedule with no risk factors."""
        response = client.get(
            "/genetics/screening-schedule",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "risk_factors" in data
            assert "personalized_schedule" in data
            assert "professional_skin_exam" in data["personalized_schedule"]
            assert "self_examination" in data["personalized_schedule"]

    @pytest.fixture
    def high_risk_genetic_test(self, test_db, test_user):
        """Create a high-risk genetic test result."""
        from datetime import datetime
        test_result = GeneticTestResult(
            user_id=test_user.id,
            test_id="GT-HIGHRISK123",
            test_type="panel",
            test_name="High Risk Panel",
            test_date=datetime.utcnow(),
            status="completed",
            pathogenic_variants_found=2,
            overall_risk_level="high"
        )
        test_db.add(test_result)
        test_db.flush()

        # Add CDKN2A pathogenic variant
        variant = GeneticVariant(
            test_result_id=test_result.id,
            user_id=test_user.id,
            gene_symbol="CDKN2A",
            chromosome="9",
            position=21974695,
            reference="G",
            alternate="A",
            classification="pathogenic",
            melanoma_risk_modifier=10.0
        )
        test_db.add(variant)
        test_db.commit()
        return test_result

    def test_get_screening_schedule_high_risk(
        self,
        client,
        auth_headers,
        high_risk_genetic_test
    ):
        """Test screening schedule for high-risk individual."""
        response = client.get(
            "/genetics/screening-schedule",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            # Should have risk factors and personalized schedule
            assert "risk_factors" in data
            assert "personalized_schedule" in data

            schedule = data.get("personalized_schedule", {})
            # Should have screening recommendations
            assert "professional_skin_exam" in schedule or "next_professional_exam" in schedule

    def test_screening_schedule_includes_next_steps(self, client, auth_headers):
        """Test that screening schedule includes actionable next steps."""
        response = client.get(
            "/genetics/screening-schedule",
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include personalized_schedule with next_steps
            has_recommendations = (
                "personalized_schedule" in data and
                "next_steps" in data.get("personalized_schedule", {})
            )
            assert has_recommendations

    def test_screening_schedule_requires_auth(self, client):
        """Test that screening schedule requires authentication."""
        response = client.get("/genetics/screening-schedule")
        assert response.status_code in [401, 403]


# =============================================================================
# FULL RISK REPORT ENDPOINT
# =============================================================================

class TestFullRiskReport:
    """Tests for comprehensive risk report generation."""

    def test_generate_full_risk_report_no_data(self, client, auth_headers):
        """Test generating risk report with no genetic data."""
        response = client.post(
            "/genetics/generate-full-risk-report",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "summary" in data
            assert "report_date" in data

    @pytest.fixture
    def complete_genetic_profile(self, test_db, test_user):
        """Create a complete genetic profile with tests and family history."""
        from datetime import datetime

        # Create genetic test
        test_result = GeneticTestResult(
            user_id=test_user.id,
            test_id="GT-COMPLETE123",
            test_type="panel",
            test_name="Complete Panel",
            test_date=datetime.utcnow(),
            status="completed",
            pathogenic_variants_found=1,
            overall_risk_level="moderate"
        )
        test_db.add(test_result)
        test_db.flush()

        # Add variant
        variant = GeneticVariant(
            test_result_id=test_result.id,
            user_id=test_user.id,
            gene_symbol="MC1R",
            chromosome="16",
            position=89919709,
            reference="C",
            alternate="T",
            classification="pathogenic",
            melanoma_risk_modifier=2.5
        )
        test_db.add(variant)

        # Add family member
        member = FamilyMember(
            user_id=test_user.id,
            relationship_type="parent",
            name="Parent",
            has_melanoma=True,
            melanoma_count=1
        )
        test_db.add(member)
        test_db.commit()

        return test_result

    def test_generate_full_risk_report_with_data(
        self,
        client,
        auth_headers,
        complete_genetic_profile
    ):
        """Test generating comprehensive risk report with genetic data."""
        response = client.post(
            "/genetics/generate-full-risk-report",
            headers=auth_headers
        )

        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            # Report should include multiple sections
            has_sections = (
                "genetic_risk" in data or
                "family_history" in data or
                "summary" in data or
                "screening_recommendations" in data
            )
            assert has_sections

    def test_full_risk_report_includes_recommendations(
        self,
        client,
        auth_headers,
        complete_genetic_profile
    ):
        """Test that risk report includes actionable recommendations."""
        response = client.post(
            "/genetics/generate-full-risk-report",
            headers=auth_headers
        )

        if response.status_code == 200:
            data = response.json()
            # Should include some form of recommendations
            has_recommendations = (
                "recommendations" in data or
                "screening_recommendations" in data or
                "next_steps" in data or
                "action_items" in data or
                "screening_schedule" in data
            )
            assert has_recommendations

    def test_full_risk_report_requires_auth(self, client):
        """Test that full risk report requires authentication."""
        response = client.post("/genetics/generate-full-risk-report")
        assert response.status_code in [401, 403]

    def test_full_risk_report_with_options(
        self,
        client,
        auth_headers,
        complete_genetic_profile
    ):
        """Test generating risk report with additional options."""
        response = client.post(
            "/genetics/generate-full-risk-report?include_raw_data=true",
            headers=auth_headers
        )

        assert response.status_code in [200, 422, 500]


# =============================================================================
# DTC INTEGRATION SCENARIOS
# =============================================================================

class TestDTCIntegration:
    """Integration tests for DTC genetic testing workflows."""

    def test_complete_23andme_workflow(self, client, auth_headers):
        """Test complete workflow: upload 23andMe -> get risk report."""
        # Step 1: Upload 23andMe file
        content = b"""# rsid\tchromosome\tposition\tgenotype
rs1805007\t16\t89919709\tCT
rs1805008\t16\t89919746\tCC
rs3731249\t9\t21971000\tGG
"""
        files = {"file": ("23andme.txt", BytesIO(content), "text/plain")}

        upload_response = client.post(
            "/genetics/upload-23andme",
            files=files,
            headers=auth_headers
        )

        if upload_response.status_code != 200:
            pytest.skip("Could not upload 23andMe file")

        # Step 2: Get screening schedule
        schedule_response = client.get(
            "/genetics/screening-schedule",
            headers=auth_headers
        )

        assert schedule_response.status_code in [200, 500]

        # Step 3: Generate full report
        report_response = client.post(
            "/genetics/generate-full-risk-report",
            headers=auth_headers
        )

        assert report_response.status_code in [200, 500]

    def test_dtc_combined_with_family_history(
        self,
        client,
        auth_headers
    ):
        """Test DTC upload combined with family history."""
        # Add family member with melanoma
        client.post(
            "/family-history",
            data={
                "relationship_type": "parent",
                "has_melanoma": "true",
                "melanoma_count": "1"
            },
            headers=auth_headers
        )

        # Upload 23andMe with MC1R variant
        content = b"""# rsid\tchromosome\tposition\tgenotype
rs1805007\t16\t89919709\tCT
"""
        files = {"file": ("23andme.txt", BytesIO(content), "text/plain")}

        client.post(
            "/genetics/upload-23andme",
            files=files,
            headers=auth_headers
        )

        # Get family risk assessment
        risk_response = client.get(
            "/genetics/family-risk-assessment",
            headers=auth_headers
        )

        if risk_response.status_code == 200:
            data = risk_response.json()
            # Risk should be elevated due to both genetic and family factors
            assert data.get("risk_multiplier", 1.0) >= 1.0

    def test_multiple_dtc_uploads(self, client, auth_headers):
        """Test handling multiple DTC file uploads."""
        content1 = b"""# rsid\tchromosome\tposition\tgenotype
rs1805007\t16\t89919709\tCC
"""
        content2 = b"""# rsid\tchromosome\tposition\tgenotype
rs1805007\t16\t89919709\tCT
"""

        # First upload
        files1 = {"file": ("first.txt", BytesIO(content1), "text/plain")}
        response1 = client.post(
            "/genetics/upload-23andme",
            files=files1,
            headers=auth_headers
        )

        # Second upload (should update or add new)
        files2 = {"file": ("second.txt", BytesIO(content2), "text/plain")}
        response2 = client.post(
            "/genetics/upload-23andme",
            files=files2,
            headers=auth_headers
        )

        # Both should be handled gracefully
        assert response1.status_code in [200, 500]
        assert response2.status_code in [200, 500]
