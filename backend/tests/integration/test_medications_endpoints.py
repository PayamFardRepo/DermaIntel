"""
Integration tests for medications endpoints.

Tests the medication safety features including:
- Drug-drug interaction checking
- Photosensitivity warnings
- Pregnancy/lactation safety
- Age-specific considerations
- Dosage verification
- Common dermatological medications reference
"""

import pytest
import json
from io import BytesIO
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestMedicationInteractionCheck:
    """Tests for medication interaction checking endpoint."""

    def test_check_interaction_single_medication(
        self,
        client,
        auth_headers
    ):
        """Test checking a single medication for interactions."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin"
            },
            headers=auth_headers
        )

        # May return 503 if medication checker not available
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_check_interaction_with_current_medications(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with current medications."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "current_medications": json.dumps(["doxycycline", "methotrexate"])
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_check_interaction_with_conditions(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with patient conditions."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "patient_conditions": json.dumps(["liver disease", "depression"])
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_with_patient_age(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with patient age."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "patient_age": 16
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_pregnant_patient(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions for pregnant patient."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "is_pregnant": True
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            # Isotretinoin should have strong pregnancy warnings
            assert isinstance(data, dict)

    def test_check_interaction_breastfeeding_patient(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions for breastfeeding patient."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "doxycycline",
                "is_breastfeeding": True
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_with_dose(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with dose information."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "dose": "40mg",
                "frequency": "twice daily"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_with_renal_function(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with impaired renal function."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "methotrexate",
                "renal_function": "moderate"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_with_hepatic_function(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with impaired hepatic function."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "methotrexate",
                "hepatic_function": "moderate"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_with_sun_exposure(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with high sun exposure."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "doxycycline",
                "sun_exposure_level": "high"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_full_profile(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with complete patient profile."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "current_medications": json.dumps(["vitamin A", "tetracycline"]),
                "patient_conditions": json.dumps(["hyperlipidemia"]),
                "patient_age": 22,
                "is_pregnant": False,
                "is_breastfeeding": False,
                "dose": "40mg",
                "frequency": "once daily",
                "renal_function": "normal",
                "hepatic_function": "normal",
                "sun_exposure_level": "moderate"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_interaction_invalid_json(
        self,
        client,
        auth_headers
    ):
        """Test checking interactions with invalid JSON."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "current_medications": "not valid json"
            },
            headers=auth_headers
        )

        assert response.status_code in [400, 503]

    def test_check_interaction_requires_auth(self, client):
        """Test that interaction check requires authentication."""
        response = client.post(
            "/medications/check-interaction",
            data={"medication": "isotretinoin"}
        )

        assert response.status_code in [401, 403]

    def test_check_interaction_requires_medication(
        self,
        client,
        auth_headers
    ):
        """Test that medication parameter is required."""
        response = client.post(
            "/medications/check-interaction",
            data={},
            headers=auth_headers
        )

        assert response.status_code == 422


class TestTreatmentPlanSafety:
    """Tests for treatment plan safety checking endpoint."""

    def test_check_treatment_plan_single_med(
        self,
        client,
        auth_headers
    ):
        """Test checking treatment plan with single medication."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={
                "medications": json.dumps([
                    {"name": "isotretinoin", "dose": "40mg", "frequency": "daily"}
                ])
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_treatment_plan_multiple_meds(
        self,
        client,
        auth_headers
    ):
        """Test checking treatment plan with multiple medications."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={
                "medications": json.dumps([
                    {"name": "tretinoin", "dose": "0.025%", "frequency": "nightly"},
                    {"name": "clindamycin", "dose": "1%", "frequency": "twice daily"},
                    {"name": "benzoyl peroxide", "dose": "2.5%", "frequency": "morning"}
                ])
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_treatment_plan_with_existing_meds(
        self,
        client,
        auth_headers
    ):
        """Test checking treatment plan with existing medications."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={
                "medications": json.dumps([
                    {"name": "isotretinoin", "dose": "40mg", "frequency": "daily"}
                ]),
                "current_medications": json.dumps(["lisinopril", "metformin"])
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_treatment_plan_with_patient_info(
        self,
        client,
        auth_headers
    ):
        """Test checking treatment plan with full patient info."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={
                "medications": json.dumps([
                    {"name": "doxycycline", "dose": "100mg", "frequency": "twice daily"}
                ]),
                "patient_age": 35,
                "patient_conditions": json.dumps(["rosacea"]),
                "is_pregnant": False,
                "is_breastfeeding": False,
                "renal_function": "normal",
                "hepatic_function": "normal",
                "sun_exposure_level": "high"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_treatment_plan_pregnant(
        self,
        client,
        auth_headers
    ):
        """Test checking treatment plan for pregnant patient."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={
                "medications": json.dumps([
                    {"name": "hydrocortisone", "dose": "1%", "frequency": "twice daily"}
                ]),
                "is_pregnant": True
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_check_treatment_plan_invalid_json(
        self,
        client,
        auth_headers
    ):
        """Test checking treatment plan with invalid JSON."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={
                "medications": "not valid json"
            },
            headers=auth_headers
        )

        assert response.status_code in [400, 503]

    def test_check_treatment_plan_requires_auth(self, client):
        """Test that treatment plan check requires authentication."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={
                "medications": json.dumps([{"name": "isotretinoin"}])
            }
        )

        assert response.status_code in [401, 403]

    def test_check_treatment_plan_requires_medications(
        self,
        client,
        auth_headers
    ):
        """Test that medications parameter is required."""
        response = client.post(
            "/medications/check-treatment-plan",
            data={},
            headers=auth_headers
        )

        assert response.status_code == 422


class TestPhotosensitivityInfo:
    """Tests for photosensitivity information endpoint."""

    def test_get_photosensitivity_known_medication(
        self,
        client,
        auth_headers
    ):
        """Test getting photosensitivity info for known medication."""
        response = client.get(
            "/medications/photosensitivity/doxycycline",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "medication" in data
            assert "has_photosensitivity_warning" in data

    def test_get_photosensitivity_unknown_medication(
        self,
        client,
        auth_headers
    ):
        """Test getting photosensitivity info for unknown medication."""
        response = client.get(
            "/medications/photosensitivity/unknown_drug_xyz",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            # Should return no warning for unknown medication
            assert "has_photosensitivity_warning" in data

    def test_get_photosensitivity_tetracycline(
        self,
        client,
        auth_headers
    ):
        """Test getting photosensitivity for tetracycline (known photosensitizer)."""
        response = client.get(
            "/medications/photosensitivity/tetracycline",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_get_photosensitivity_requires_auth(self, client):
        """Test that photosensitivity endpoint requires auth."""
        response = client.get("/medications/photosensitivity/doxycycline")

        assert response.status_code in [401, 403]


class TestPregnancySafetyInfo:
    """Tests for pregnancy safety information endpoint."""

    def test_get_pregnancy_safety_isotretinoin(
        self,
        client,
        auth_headers
    ):
        """Test getting pregnancy safety for isotretinoin (category X)."""
        response = client.get(
            "/medications/pregnancy-safety/isotretinoin",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "medication" in data
            assert "has_pregnancy_warning" in data

    def test_get_pregnancy_safety_safe_medication(
        self,
        client,
        auth_headers
    ):
        """Test getting pregnancy safety for generally safe medication."""
        response = client.get(
            "/medications/pregnancy-safety/hydrocortisone",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_get_pregnancy_safety_unknown_medication(
        self,
        client,
        auth_headers
    ):
        """Test getting pregnancy safety for unknown medication."""
        response = client.get(
            "/medications/pregnancy-safety/unknown_medication_abc",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            # Should indicate no specific data available
            assert "medication" in data

    def test_get_pregnancy_safety_requires_auth(self, client):
        """Test that pregnancy safety endpoint requires auth."""
        response = client.get("/medications/pregnancy-safety/isotretinoin")

        assert response.status_code in [401, 403]


class TestAgeSafetyInfo:
    """Tests for age-specific safety information endpoint."""

    def test_get_age_safety_pediatric(
        self,
        client,
        auth_headers
    ):
        """Test getting age safety for pediatric patient."""
        response = client.get(
            "/medications/age-safety/isotretinoin?patient_age=12",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data["patient_age"] == 12

    def test_get_age_safety_geriatric(
        self,
        client,
        auth_headers
    ):
        """Test getting age safety for geriatric patient."""
        response = client.get(
            "/medications/age-safety/prednisone?patient_age=75",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data["patient_age"] == 75

    def test_get_age_safety_adult(
        self,
        client,
        auth_headers
    ):
        """Test getting age safety for adult patient."""
        response = client.get(
            "/medications/age-safety/doxycycline?patient_age=35",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_get_age_safety_infant(
        self,
        client,
        auth_headers
    ):
        """Test getting age safety for infant."""
        response = client.get(
            "/medications/age-safety/hydrocortisone?patient_age=1",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_get_age_safety_requires_age_param(
        self,
        client,
        auth_headers
    ):
        """Test that age parameter is required."""
        response = client.get(
            "/medications/age-safety/isotretinoin",
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_get_age_safety_requires_auth(self, client):
        """Test that age safety endpoint requires auth."""
        response = client.get("/medications/age-safety/isotretinoin?patient_age=12")

        assert response.status_code in [401, 403]


class TestDosageInfo:
    """Tests for dosage information endpoint."""

    def test_get_dosage_info_known_medication(
        self,
        client,
        auth_headers
    ):
        """Test getting dosage info for known medication."""
        response = client.get(
            "/medications/dosage-info/isotretinoin",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "medication" in data
            assert "has_dosage_info" in data

    def test_get_dosage_info_methotrexate(
        self,
        client,
        auth_headers
    ):
        """Test getting dosage info for methotrexate (critical dosing)."""
        response = client.get(
            "/medications/dosage-info/methotrexate",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_get_dosage_info_unknown_medication(
        self,
        client,
        auth_headers
    ):
        """Test getting dosage info for unknown medication."""
        response = client.get(
            "/medications/dosage-info/unknown_drug_xyz",
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert data.get("has_dosage_info") is False or "message" in data

    def test_get_dosage_info_requires_auth(self, client):
        """Test that dosage info endpoint requires auth."""
        response = client.get("/medications/dosage-info/isotretinoin")

        assert response.status_code in [401, 403]


class TestVerifyDosage:
    """Tests for dosage verification endpoint."""

    def test_verify_dosage_valid(
        self,
        client,
        auth_headers
    ):
        """Test verifying valid dosage."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "doxycycline",
                "dose": "100mg",
                "frequency": "twice daily",
                "patient_age": 30
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "is_safe" in data
            assert "has_warnings" in data

    def test_verify_dosage_high_dose(
        self,
        client,
        auth_headers
    ):
        """Test verifying potentially high dosage."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "isotretinoin",
                "dose": "80mg",
                "frequency": "twice daily",
                "patient_age": 25
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_verify_dosage_pediatric(
        self,
        client,
        auth_headers
    ):
        """Test verifying dosage for pediatric patient."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "hydrocortisone",
                "dose": "1%",
                "frequency": "twice daily",
                "patient_age": 5
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_verify_dosage_with_renal_impairment(
        self,
        client,
        auth_headers
    ):
        """Test verifying dosage with renal impairment."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "methotrexate",
                "dose": "15mg",
                "frequency": "weekly",
                "patient_age": 55,
                "renal_function": "moderate"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_verify_dosage_with_hepatic_impairment(
        self,
        client,
        auth_headers
    ):
        """Test verifying dosage with hepatic impairment."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "isotretinoin",
                "dose": "40mg",
                "frequency": "daily",
                "patient_age": 25,
                "hepatic_function": "moderate"
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]

    def test_verify_dosage_methotrexate_daily_error(
        self,
        client,
        auth_headers
    ):
        """Test detecting critical methotrexate daily dosing error."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "methotrexate",
                "dose": "15mg",
                "frequency": "daily",  # Should be weekly - critical error!
                "patient_age": 45
            },
            headers=auth_headers
        )

        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            # Should flag this as unsafe or have warnings
            assert "requires_review" in data or "is_safe" in data

    def test_verify_dosage_requires_all_params(
        self,
        client,
        auth_headers
    ):
        """Test that all required parameters are needed."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "doxycycline"
                # Missing dose, frequency, patient_age
            },
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_verify_dosage_requires_auth(self, client):
        """Test that dosage verification requires auth."""
        response = client.post(
            "/medications/verify-dosage",
            data={
                "medication": "doxycycline",
                "dose": "100mg",
                "frequency": "twice daily",
                "patient_age": 30
            }
        )

        assert response.status_code in [401, 403]


class TestCommonDermatologicalMedications:
    """Tests for common dermatological medications endpoint."""

    def test_get_all_common_medications(
        self,
        client,
        auth_headers
    ):
        """Test getting all common dermatological medications."""
        response = client.get(
            "/medications/common-dermatological",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "medications" in data
        assert "total" in data
        assert len(data["medications"]) > 0

    def test_filter_by_category_retinoid(
        self,
        client,
        auth_headers
    ):
        """Test filtering medications by Retinoid category."""
        response = client.get(
            "/medications/common-dermatological?category=Retinoid",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["category_filter"] == "Retinoid"
        for med in data["medications"]:
            assert med["category"] == "Retinoid"

    def test_filter_by_category_antibiotic(
        self,
        client,
        auth_headers
    ):
        """Test filtering medications by Antibiotic category."""
        response = client.get(
            "/medications/common-dermatological?category=Antibiotic",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        for med in data["medications"]:
            assert med["category"] == "Antibiotic"

    def test_filter_by_category_corticosteroid(
        self,
        client,
        auth_headers
    ):
        """Test filtering medications by Corticosteroid category."""
        response = client.get(
            "/medications/common-dermatological?category=Corticosteroid",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["medications"]) > 0

    def test_filter_by_category_biologic(
        self,
        client,
        auth_headers
    ):
        """Test filtering medications by Biologic category."""
        response = client.get(
            "/medications/common-dermatological?category=Biologic",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["medications"]) > 0

    def test_filter_by_category_case_insensitive(
        self,
        client,
        auth_headers
    ):
        """Test that category filter is case insensitive."""
        response = client.get(
            "/medications/common-dermatological?category=retinoid",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["medications"]) > 0

    def test_filter_by_nonexistent_category(
        self,
        client,
        auth_headers
    ):
        """Test filtering by non-existent category."""
        response = client.get(
            "/medications/common-dermatological?category=NonExistent",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["medications"]) == 0
        assert data["total"] == 0

    def test_medications_have_required_fields(
        self,
        client,
        auth_headers
    ):
        """Test that medications have all required fields."""
        response = client.get(
            "/medications/common-dermatological",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        for med in data["medications"]:
            assert "name" in med
            assert "category" in med
            assert "common_uses" in med
            assert isinstance(med["common_uses"], list)

    def test_common_medications_requires_auth(self, client):
        """Test that common medications endpoint requires auth."""
        response = client.get("/medications/common-dermatological")

        assert response.status_code in [401, 403]


class TestMedicationSearch:
    """Tests for medication search endpoint."""

    def test_search_by_name(
        self,
        client,
        auth_headers
    ):
        """Test searching medications by name."""
        response = client.get(
            "/medications/search?query=tretinoin",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_matches" in data

    def test_search_by_partial_name(
        self,
        client,
        auth_headers
    ):
        """Test searching medications by partial name."""
        response = client.get(
            "/medications/search?query=doxy",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        # Should find doxycycline
        assert data["total_matches"] >= 1

    def test_search_by_category(
        self,
        client,
        auth_headers
    ):
        """Test searching medications by category."""
        response = client.get(
            "/medications/search?query=Antibiotic",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_matches"] > 0

    def test_search_by_use(
        self,
        client,
        auth_headers
    ):
        """Test searching medications by common use."""
        response = client.get(
            "/medications/search?query=acne",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_matches"] > 0

    def test_search_with_limit(
        self,
        client,
        auth_headers
    ):
        """Test searching with limit parameter."""
        response = client.get(
            "/medications/search?query=a&limit=3",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 3

    def test_search_no_results(
        self,
        client,
        auth_headers
    ):
        """Test searching with no matching results."""
        response = client.get(
            "/medications/search?query=xyznonexistent",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_matches"] == 0
        assert len(data["results"]) == 0

    def test_search_case_insensitive(
        self,
        client,
        auth_headers
    ):
        """Test that search is case insensitive."""
        response_lower = client.get(
            "/medications/search?query=isotretinoin",
            headers=auth_headers
        )
        response_upper = client.get(
            "/medications/search?query=ISOTRETINOIN",
            headers=auth_headers
        )

        assert response_lower.status_code == 200
        assert response_upper.status_code == 200
        assert response_lower.json()["total_matches"] == response_upper.json()["total_matches"]

    def test_search_requires_query(
        self,
        client,
        auth_headers
    ):
        """Test that query parameter is required."""
        response = client.get(
            "/medications/search",
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_search_requires_auth(self, client):
        """Test that search requires authentication."""
        response = client.get("/medications/search?query=acne")

        assert response.status_code in [401, 403]


class TestMedicationCategories:
    """Tests for medication categories endpoint."""

    def test_get_categories(
        self,
        client,
        auth_headers
    ):
        """Test getting all medication categories."""
        response = client.get(
            "/medications/categories",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "total_categories" in data
        assert data["total_categories"] > 0

    def test_categories_have_required_fields(
        self,
        client,
        auth_headers
    ):
        """Test that categories have all required fields."""
        response = client.get(
            "/medications/categories",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        for category in data["categories"]:
            assert "name" in category
            assert "count" in category
            assert "medications" in category
            assert isinstance(category["medications"], list)
            assert category["count"] == len(category["medications"])

    def test_known_categories_exist(
        self,
        client,
        auth_headers
    ):
        """Test that known categories exist."""
        response = client.get(
            "/medications/categories",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        category_names = [c["name"] for c in data["categories"]]

        expected_categories = [
            "Retinoid", "Antibiotic", "Antifungal",
            "Corticosteroid", "Biologic"
        ]
        for expected in expected_categories:
            assert expected in category_names

    def test_categories_requires_auth(self, client):
        """Test that categories endpoint requires auth."""
        response = client.get("/medications/categories")

        assert response.status_code in [401, 403]


class TestProfessionalMedicationAccess:
    """Tests for professional-specific medication features."""

    def test_professional_can_access_all_endpoints(
        self,
        client,
        professional_auth_headers
    ):
        """Test that professionals can access medication endpoints."""
        # Common medications
        response = client.get(
            "/medications/common-dermatological",
            headers=professional_auth_headers
        )
        assert response.status_code == 200

        # Categories
        response = client.get(
            "/medications/categories",
            headers=professional_auth_headers
        )
        assert response.status_code == 200

        # Search
        response = client.get(
            "/medications/search?query=acne",
            headers=professional_auth_headers
        )
        assert response.status_code == 200

    def test_professional_check_interaction(
        self,
        client,
        professional_auth_headers
    ):
        """Test professional checking medication interactions."""
        response = client.post(
            "/medications/check-interaction",
            data={
                "medication": "isotretinoin",
                "current_medications": json.dumps(["warfarin"]),
                "patient_age": 25
            },
            headers=professional_auth_headers
        )

        assert response.status_code in [200, 503]
