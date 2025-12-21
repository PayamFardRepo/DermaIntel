"""
Medications Router - Medication Interaction and Safety Checking Module

Handles medication safety checks:
- Drug-drug interactions
- Photosensitivity warnings
- Pregnancy/lactation safety
- Age-specific considerations
- Dosage verification
- Common dermatological medications reference
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Optional
import json

from database import get_db, User
from auth import get_current_active_user

router = APIRouter(tags=["Medications"])

# =============================================================================
# MEDICATION INTERACTION CHECKER INITIALIZATION
# =============================================================================

try:
    from medication_interaction_checker import medication_checker, SeverityLevel
    MEDICATION_CHECKER_AVAILABLE = True
    print("[OK] Medication interaction checker loaded in medications_router")
except Exception as e:
    print(f"Warning: Medication interaction checker not available in medications_router: {e}")
    MEDICATION_CHECKER_AVAILABLE = False


# =============================================================================
# COMMON DERMATOLOGICAL MEDICATIONS DATABASE
# =============================================================================

COMMON_DERMATOLOGICAL_MEDICATIONS = [
    # Retinoids
    {"name": "isotretinoin", "category": "Retinoid", "common_uses": ["Severe acne", "Rosacea"]},
    {"name": "tretinoin", "category": "Retinoid", "common_uses": ["Acne", "Photoaging", "Melasma"]},
    {"name": "adapalene", "category": "Retinoid", "common_uses": ["Acne"]},
    {"name": "tazarotene", "category": "Retinoid", "common_uses": ["Psoriasis", "Acne"]},

    # Antibiotics
    {"name": "doxycycline", "category": "Antibiotic", "common_uses": ["Acne", "Rosacea", "Infections"]},
    {"name": "minocycline", "category": "Antibiotic", "common_uses": ["Acne", "Rosacea"]},
    {"name": "clindamycin", "category": "Antibiotic", "common_uses": ["Acne", "Bacterial infections"]},
    {"name": "mupirocin", "category": "Antibiotic", "common_uses": ["Impetigo", "MRSA"]},

    # Antifungals
    {"name": "terbinafine", "category": "Antifungal", "common_uses": ["Onychomycosis", "Tinea"]},
    {"name": "fluconazole", "category": "Antifungal", "common_uses": ["Candidiasis", "Tinea"]},
    {"name": "itraconazole", "category": "Antifungal", "common_uses": ["Onychomycosis", "Systemic fungal"]},
    {"name": "ketoconazole", "category": "Antifungal", "common_uses": ["Seborrheic dermatitis", "Tinea"]},

    # Corticosteroids
    {"name": "prednisone", "category": "Corticosteroid", "common_uses": ["Severe eczema", "Allergic reactions"]},
    {"name": "hydrocortisone", "category": "Corticosteroid", "common_uses": ["Mild eczema", "Contact dermatitis"]},
    {"name": "triamcinolone", "category": "Corticosteroid", "common_uses": ["Eczema", "Psoriasis"]},
    {"name": "clobetasol", "category": "Corticosteroid", "common_uses": ["Severe psoriasis", "Lichen planus"]},
    {"name": "betamethasone", "category": "Corticosteroid", "common_uses": ["Eczema", "Psoriasis", "Dermatitis"]},

    # Immunosuppressants
    {"name": "methotrexate", "category": "Immunosuppressant", "common_uses": ["Psoriasis", "Eczema"]},
    {"name": "cyclosporine", "category": "Immunosuppressant", "common_uses": ["Psoriasis", "Severe eczema"]},
    {"name": "azathioprine", "category": "Immunosuppressant", "common_uses": ["Autoimmune conditions"]},
    {"name": "mycophenolate", "category": "Immunosuppressant", "common_uses": ["Autoimmune blistering diseases"]},

    # Biologics
    {"name": "adalimumab", "category": "Biologic", "common_uses": ["Psoriasis", "Hidradenitis"]},
    {"name": "secukinumab", "category": "Biologic", "common_uses": ["Psoriasis"]},
    {"name": "dupilumab", "category": "Biologic", "common_uses": ["Atopic dermatitis"]},
    {"name": "ustekinumab", "category": "Biologic", "common_uses": ["Psoriasis"]},
    {"name": "infliximab", "category": "Biologic", "common_uses": ["Psoriasis", "Hidradenitis"]},

    # Acne treatments
    {"name": "benzoyl peroxide", "category": "Acne", "common_uses": ["Acne"]},
    {"name": "azelaic acid", "category": "Acne", "common_uses": ["Acne", "Rosacea", "Melasma"]},
    {"name": "salicylic acid", "category": "Acne", "common_uses": ["Acne", "Warts"]},
    {"name": "spironolactone", "category": "Acne", "common_uses": ["Hormonal acne"]},

    # Antihistamines
    {"name": "cetirizine", "category": "Antihistamine", "common_uses": ["Urticaria", "Allergic reactions"]},
    {"name": "hydroxyzine", "category": "Antihistamine", "common_uses": ["Urticaria", "Pruritus"]},
    {"name": "fexofenadine", "category": "Antihistamine", "common_uses": ["Urticaria", "Allergic reactions"]},
    {"name": "diphenhydramine", "category": "Antihistamine", "common_uses": ["Urticaria", "Pruritus"]},

    # Antiviral
    {"name": "acyclovir", "category": "Antiviral", "common_uses": ["Herpes simplex", "Herpes zoster"]},
    {"name": "valacyclovir", "category": "Antiviral", "common_uses": ["Herpes simplex", "Herpes zoster"]},

    # Others
    {"name": "dapsone", "category": "Other", "common_uses": ["Dermatitis herpetiformis", "Acne"]},
    {"name": "finasteride", "category": "Other", "common_uses": ["Androgenetic alopecia"]},
    {"name": "minoxidil", "category": "Other", "common_uses": ["Hair loss"]},
    {"name": "colchicine", "category": "Other", "common_uses": ["Behcet's disease", "Vasculitis"]},
    {"name": "hydroxychloroquine", "category": "Other", "common_uses": ["Lupus", "Dermatomyositis"]}
]


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/medications/check-interaction")
async def check_medication_interaction(
    medication: str = Form(...),
    current_medications: str = Form(None),  # JSON array of medication names
    patient_conditions: str = Form(None),  # JSON array of conditions
    patient_age: Optional[int] = Form(None),
    is_pregnant: bool = Form(False),
    is_breastfeeding: bool = Form(False),
    dose: str = Form(None),
    frequency: str = Form(None),
    renal_function: str = Form(None),  # "normal", "mild", "moderate", "severe"
    hepatic_function: str = Form(None),
    sun_exposure_level: str = Form(None),  # "minimal", "moderate", "high", "very_high"
    current_user: User = Depends(get_current_active_user)
):
    """
    Check a medication for interactions, contraindications, and warnings.

    Returns comprehensive safety information including:
    - Drug-drug interactions with current medications
    - Contraindications based on patient conditions
    - Photosensitivity warnings
    - Age-specific warnings
    - Pregnancy/lactation warnings
    - Dosage verification
    """
    if not MEDICATION_CHECKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Medication interaction checker not available")

    try:
        # Parse JSON arrays
        current_meds = json.loads(current_medications) if current_medications else []
        conditions = json.loads(patient_conditions) if patient_conditions else []

        result = medication_checker.check_medication(
            medication=medication,
            current_medications=current_meds,
            patient_conditions=conditions,
            patient_age=patient_age,
            is_pregnant=is_pregnant,
            is_breastfeeding=is_breastfeeding,
            dose=dose,
            frequency=frequency,
            renal_function=renal_function,
            hepatic_function=hepatic_function,
            sun_exposure_level=sun_exposure_level
        )

        return medication_checker._result_to_dict(result)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        print(f"Error checking medication interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check medication: {str(e)}")


@router.post("/medications/check-treatment-plan")
async def check_treatment_plan_safety(
    medications: str = Form(...),  # JSON array of {name, dose, frequency}
    patient_age: Optional[int] = Form(None),
    patient_conditions: str = Form(None),  # JSON array
    current_medications: str = Form(None),  # JSON array of existing medications
    is_pregnant: bool = Form(False),
    is_breastfeeding: bool = Form(False),
    renal_function: str = Form(None),
    hepatic_function: str = Form(None),
    sun_exposure_level: str = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Check safety of a complete treatment plan with multiple medications.

    Checks:
    - Interactions between all medications in the plan
    - Interactions with existing patient medications
    - All individual medication warnings
    - Overall treatment plan safety score
    """
    if not MEDICATION_CHECKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Medication interaction checker not available")

    try:
        meds_list = json.loads(medications)
        conditions = json.loads(patient_conditions) if patient_conditions else []
        current_meds = json.loads(current_medications) if current_medications else []

        treatment_plan = {'medications': meds_list}
        patient_profile = {
            'age': patient_age,
            'conditions': conditions,
            'current_medications': current_meds,
            'is_pregnant': is_pregnant,
            'is_breastfeeding': is_breastfeeding,
            'renal_function': renal_function,
            'hepatic_function': hepatic_function,
            'sun_exposure_level': sun_exposure_level
        }

        result = medication_checker.check_treatment_safety(treatment_plan, patient_profile)
        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        print(f"Error checking treatment plan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check treatment plan: {str(e)}")


@router.get("/medications/photosensitivity/{medication}")
async def get_photosensitivity_info(
    medication: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get photosensitivity warning for a specific medication.

    Returns detailed sun exposure precautions including:
    - Type of photosensitivity (phototoxic vs photoallergic)
    - SPF recommendations
    - Duration of sensitivity after stopping
    - Specific precautions
    """
    if not MEDICATION_CHECKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Medication interaction checker not available")

    try:
        warning = medication_checker.database.get_photosensitivity_warning(medication)

        if not warning:
            return {
                "medication": medication,
                "has_photosensitivity_warning": False,
                "message": "No known photosensitivity warning for this medication"
            }

        return {
            "medication": medication,
            "has_photosensitivity_warning": True,
            "warning": medication_checker._photo_warning_to_dict(warning)
        }

    except Exception as e:
        print(f"Error getting photosensitivity info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get photosensitivity info: {str(e)}")


@router.get("/medications/pregnancy-safety/{medication}")
async def get_pregnancy_safety_info(
    medication: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get pregnancy and lactation safety information for a medication.

    Returns:
    - Pregnancy category
    - Risk by trimester
    - Lactation safety
    - Safe alternatives during pregnancy and lactation
    """
    if not MEDICATION_CHECKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Medication interaction checker not available")

    try:
        warning = medication_checker.database.get_pregnancy_warning(medication)

        if not warning:
            return {
                "medication": medication,
                "has_pregnancy_warning": False,
                "message": "No specific pregnancy/lactation data available. Consult healthcare provider."
            }

        return {
            "medication": medication,
            "has_pregnancy_warning": True,
            "warning": medication_checker._pregnancy_warning_to_dict(warning)
        }

    except Exception as e:
        print(f"Error getting pregnancy safety info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get pregnancy safety info: {str(e)}")


@router.get("/medications/age-safety/{medication}")
async def get_age_safety_info(
    medication: str,
    patient_age: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get age-specific safety information for a medication.

    Returns pediatric or geriatric concerns based on patient age.
    """
    if not MEDICATION_CHECKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Medication interaction checker not available")

    try:
        warning = medication_checker.database.get_age_warning(medication, patient_age)

        if not warning:
            return {
                "medication": medication,
                "patient_age": patient_age,
                "has_age_warning": False,
                "message": "No age-specific warnings for this patient"
            }

        return {
            "medication": medication,
            "patient_age": patient_age,
            "has_age_warning": True,
            "warning": medication_checker._age_warning_to_dict(warning)
        }

    except Exception as e:
        print(f"Error getting age safety info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get age safety info: {str(e)}")


@router.get("/medications/dosage-info/{medication}")
async def get_dosage_information(
    medication: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get standard dosage information for a medication.

    Returns:
    - Adult and pediatric dosing
    - Maximum daily doses
    - Renal/hepatic adjustments
    - Typical treatment duration
    """
    if not MEDICATION_CHECKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Medication interaction checker not available")

    try:
        dosage_list = medication_checker.database.get_dosage_info(medication)

        if not dosage_list:
            return {
                "medication": medication,
                "has_dosage_info": False,
                "message": "No standard dosage information available"
            }

        return {
            "medication": medication,
            "has_dosage_info": True,
            "dosage_info": [
                {
                    "formulation": d.formulation,
                    "strength": d.strength,
                    "adult_dose": d.adult_dose,
                    "adult_max_daily": d.adult_max_daily,
                    "pediatric_dose": d.pediatric_dose,
                    "pediatric_max_daily": d.pediatric_max_daily,
                    "renal_adjustment": d.renal_adjustment,
                    "hepatic_adjustment": d.hepatic_adjustment,
                    "elderly_adjustment": d.elderly_adjustment,
                    "frequency": d.application_frequency,
                    "typical_duration": d.duration_typical,
                    "max_duration": d.duration_max
                }
                for d in dosage_list
            ]
        }

    except Exception as e:
        print(f"Error getting dosage info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dosage info: {str(e)}")


@router.post("/medications/verify-dosage")
async def verify_medication_dosage(
    medication: str = Form(...),
    dose: str = Form(...),
    frequency: str = Form(...),
    patient_age: int = Form(...),
    renal_function: str = Form(None),
    hepatic_function: str = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Verify if a medication dosage is within safe/recommended range.

    Checks:
    - Standard dosage guidelines
    - Age-appropriate dosing
    - Renal/hepatic adjustments needed
    - Critical dosing errors (e.g., methotrexate daily vs weekly)
    """
    if not MEDICATION_CHECKER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Medication interaction checker not available")

    try:
        issues = medication_checker.database.verify_dosage(
            drug=medication,
            dose=dose,
            frequency=frequency,
            patient_age=patient_age,
            renal_function=renal_function,
            hepatic_function=hepatic_function
        )

        has_critical_issues = any(
            issue.get('severity') in ['contraindicated', 'critical_error']
            for issue in issues
        )

        has_warnings = any(
            issue.get('severity') in ['severe', 'moderate']
            for issue in issues
        )

        return {
            "medication": medication,
            "dose": dose,
            "frequency": frequency,
            "patient_age": patient_age,
            "is_safe": not has_critical_issues,
            "has_warnings": has_warnings,
            "issues": issues,
            "requires_review": has_critical_issues or has_warnings
        }

    except Exception as e:
        print(f"Error verifying dosage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to verify dosage: {str(e)}")


@router.get("/medications/common-dermatological")
async def get_common_dermatological_medications(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get list of common dermatological medications with basic info.

    Useful for populating medication dropdown/autocomplete.

    Optional filter by category:
    - Retinoid
    - Antibiotic
    - Antifungal
    - Corticosteroid
    - Immunosuppressant
    - Biologic
    - Acne
    - Antihistamine
    - Antiviral
    - Other
    """
    medications = COMMON_DERMATOLOGICAL_MEDICATIONS

    if category:
        medications = [m for m in medications if m["category"].lower() == category.lower()]

    return {
        "medications": medications,
        "total": len(medications),
        "category_filter": category
    }


@router.get("/medications/search")
async def search_medications(
    query: str,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    """
    Search for medications by name or category.

    Useful for autocomplete functionality.
    """
    query_lower = query.lower()

    # Search in common medications
    results = []
    for med in COMMON_DERMATOLOGICAL_MEDICATIONS:
        # Match by name
        if query_lower in med["name"].lower():
            results.append(med)
        # Match by category
        elif query_lower in med["category"].lower():
            results.append(med)
        # Match by use
        elif any(query_lower in use.lower() for use in med["common_uses"]):
            results.append(med)

    # Sort by relevance (exact name matches first)
    results.sort(key=lambda x: (
        0 if x["name"].lower().startswith(query_lower) else 1,
        x["name"]
    ))

    return {
        "query": query,
        "results": results[:limit],
        "total_matches": len(results)
    }


@router.get("/medications/categories")
async def get_medication_categories(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get list of medication categories with counts.
    """
    categories = {}
    for med in COMMON_DERMATOLOGICAL_MEDICATIONS:
        cat = med["category"]
        if cat not in categories:
            categories[cat] = {
                "name": cat,
                "count": 0,
                "medications": []
            }
        categories[cat]["count"] += 1
        categories[cat]["medications"].append(med["name"])

    return {
        "categories": list(categories.values()),
        "total_categories": len(categories)
    }
