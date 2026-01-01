"""
Lab Results Router - Manage user lab results for enhanced skin analysis.

Endpoints:
- GET /lab-results - List all lab results for user
- POST /lab-results - Create new lab result entry
- GET /lab-results/{id} - Get specific lab result
- PUT /lab-results/{id} - Update lab result
- DELETE /lab-results/{id} - Delete lab result
- POST /lab-results/parse-pdf - Parse lab results from PDF (placeholder)
"""

from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel

from database import get_db, User, LabResults
from auth import get_current_active_user

router = APIRouter(tags=["Lab Results"])


class LabResultCreate(BaseModel):
    """Schema for creating lab results."""
    test_date: date
    test_type: str = "blood"
    lab_name: Optional[str] = None
    ordering_physician: Optional[str] = None

    # CBC values
    wbc: Optional[float] = None
    rbc: Optional[float] = None
    hemoglobin: Optional[float] = None
    hematocrit: Optional[float] = None
    platelets: Optional[float] = None

    # Differential
    neutrophils: Optional[float] = None
    lymphocytes: Optional[float] = None
    monocytes: Optional[float] = None
    eosinophils: Optional[float] = None
    basophils: Optional[float] = None

    # Metabolic panel
    glucose_fasting: Optional[float] = None
    glucose_random: Optional[float] = None
    hba1c: Optional[float] = None
    bun: Optional[float] = None
    creatinine: Optional[float] = None
    egfr: Optional[float] = None
    sodium: Optional[float] = None
    potassium: Optional[float] = None
    chloride: Optional[float] = None
    co2: Optional[float] = None
    calcium: Optional[float] = None
    magnesium: Optional[float] = None
    phosphorus: Optional[float] = None

    # Liver panel
    alt: Optional[float] = None
    ast: Optional[float] = None
    alp: Optional[float] = None
    bilirubin_total: Optional[float] = None
    bilirubin_direct: Optional[float] = None
    albumin: Optional[float] = None
    total_protein: Optional[float] = None

    # Lipid panel
    cholesterol_total: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None

    # Thyroid
    tsh: Optional[float] = None
    t4_free: Optional[float] = None
    t3_free: Optional[float] = None

    # Iron studies
    iron: Optional[float] = None
    ferritin: Optional[float] = None
    tibc: Optional[float] = None

    # Vitamins
    vitamin_d: Optional[float] = None
    vitamin_b12: Optional[float] = None
    folate: Optional[float] = None

    # Inflammatory markers
    crp: Optional[float] = None
    esr: Optional[float] = None

    # Autoimmune
    ana_positive: Optional[bool] = None
    ana_titer: Optional[str] = None
    rheumatoid_factor: Optional[float] = None

    # Allergy
    ige_total: Optional[float] = None

    # Additional fields
    notes: Optional[str] = None
    is_manually_entered: Optional[bool] = True

    # Extended CBC
    mcv: Optional[float] = None
    mch: Optional[float] = None
    mchc: Optional[float] = None
    rdw: Optional[float] = None
    mpv: Optional[float] = None

    # Absolute WBC counts
    neutrophils_abs: Optional[float] = None
    lymphocytes_abs: Optional[float] = None
    monocytes_abs: Optional[float] = None
    eosinophils_abs: Optional[float] = None
    basophils_abs: Optional[float] = None

    # Extended metabolic
    eag: Optional[float] = None
    bun_creatinine_ratio: Optional[float] = None
    egfr_african_american: Optional[float] = None

    # Extended liver
    globulin: Optional[float] = None
    albumin_globulin_ratio: Optional[float] = None

    # Extended lipids
    chol_hdl_ratio: Optional[float] = None
    non_hdl_cholesterol: Optional[float] = None

    # Extended thyroid
    t3_uptake: Optional[float] = None
    t4_total: Optional[float] = None
    free_t4_index: Optional[float] = None

    # Urine
    urine_color: Optional[str] = None
    urine_appearance: Optional[str] = None
    urine_specific_gravity: Optional[float] = None
    urine_ph: Optional[float] = None
    urine_protein: Optional[str] = None
    urine_glucose: Optional[str] = None
    urine_ketones: Optional[str] = None
    urine_blood: Optional[str] = None
    urine_bilirubin: Optional[str] = None
    urine_urobilinogen: Optional[str] = None
    urine_nitrite: Optional[str] = None
    urine_leukocyte_esterase: Optional[str] = None
    urine_wbc: Optional[str] = None
    urine_rbc: Optional[str] = None
    urine_bacteria: Optional[str] = None
    urine_squamous_epithelial: Optional[str] = None
    urine_hyaline_cast: Optional[str] = None

    # Stool
    stool_color: Optional[str] = None
    stool_occult_blood: Optional[str] = None
    stool_parasites: Optional[str] = None
    stool_calprotectin: Optional[float] = None

    class Config:
        extra = "ignore"  # Ignore extra fields not in the model


class LabResultUpdate(BaseModel):
    """Schema for updating lab results. All fields optional."""
    test_date: Optional[date] = None
    test_type: Optional[str] = None
    lab_name: Optional[str] = None
    ordering_physician: Optional[str] = None
    notes: Optional[str] = None

    # All lab value fields are optional for update
    wbc: Optional[float] = None
    rbc: Optional[float] = None
    hemoglobin: Optional[float] = None
    hematocrit: Optional[float] = None
    platelets: Optional[float] = None
    mcv: Optional[float] = None
    mch: Optional[float] = None
    mchc: Optional[float] = None
    rdw: Optional[float] = None
    mpv: Optional[float] = None
    neutrophils: Optional[float] = None
    lymphocytes: Optional[float] = None
    monocytes: Optional[float] = None
    eosinophils: Optional[float] = None
    basophils: Optional[float] = None
    neutrophils_abs: Optional[float] = None
    lymphocytes_abs: Optional[float] = None
    monocytes_abs: Optional[float] = None
    eosinophils_abs: Optional[float] = None
    basophils_abs: Optional[float] = None
    glucose_fasting: Optional[float] = None
    glucose_random: Optional[float] = None
    hba1c: Optional[float] = None
    eag: Optional[float] = None
    bun: Optional[float] = None
    creatinine: Optional[float] = None
    bun_creatinine_ratio: Optional[float] = None
    egfr: Optional[float] = None
    egfr_african_american: Optional[float] = None
    sodium: Optional[float] = None
    potassium: Optional[float] = None
    chloride: Optional[float] = None
    co2: Optional[float] = None
    calcium: Optional[float] = None
    magnesium: Optional[float] = None
    phosphorus: Optional[float] = None
    alt: Optional[float] = None
    ast: Optional[float] = None
    alp: Optional[float] = None
    bilirubin_total: Optional[float] = None
    bilirubin_direct: Optional[float] = None
    albumin: Optional[float] = None
    total_protein: Optional[float] = None
    globulin: Optional[float] = None
    albumin_globulin_ratio: Optional[float] = None
    cholesterol_total: Optional[float] = None
    ldl: Optional[float] = None
    hdl: Optional[float] = None
    triglycerides: Optional[float] = None
    chol_hdl_ratio: Optional[float] = None
    non_hdl_cholesterol: Optional[float] = None
    tsh: Optional[float] = None
    t4_free: Optional[float] = None
    t3_free: Optional[float] = None
    t3_uptake: Optional[float] = None
    t4_total: Optional[float] = None
    free_t4_index: Optional[float] = None
    iron: Optional[float] = None
    ferritin: Optional[float] = None
    tibc: Optional[float] = None
    vitamin_d: Optional[float] = None
    vitamin_b12: Optional[float] = None
    folate: Optional[float] = None
    crp: Optional[float] = None
    esr: Optional[float] = None
    ana_positive: Optional[bool] = None
    ana_titer: Optional[str] = None
    rheumatoid_factor: Optional[float] = None
    ige_total: Optional[float] = None
    urine_color: Optional[str] = None
    urine_appearance: Optional[str] = None
    urine_specific_gravity: Optional[float] = None
    urine_ph: Optional[float] = None
    urine_protein: Optional[str] = None
    urine_glucose: Optional[str] = None
    urine_ketones: Optional[str] = None
    urine_blood: Optional[str] = None
    urine_bilirubin: Optional[str] = None
    urine_urobilinogen: Optional[str] = None
    urine_nitrite: Optional[str] = None
    urine_leukocyte_esterase: Optional[str] = None
    urine_wbc: Optional[str] = None
    urine_rbc: Optional[str] = None
    urine_bacteria: Optional[str] = None
    urine_squamous_epithelial: Optional[str] = None
    urine_hyaline_cast: Optional[str] = None
    stool_color: Optional[str] = None
    stool_occult_blood: Optional[str] = None
    stool_parasites: Optional[str] = None
    stool_calprotectin: Optional[float] = None

    class Config:
        extra = "ignore"


# Reference ranges for abnormality detection
REFERENCE_RANGES = {
    "wbc": (3.8, 10.8),
    "rbc": (4.0, 5.5),
    "hemoglobin": (12.0, 17.0),
    "hematocrit": (36.0, 50.0),
    "platelets": (140, 400),
    "neutrophils": (40, 70),
    "lymphocytes": (20, 40),
    "monocytes": (2, 8),
    "eosinophils": (1, 4),
    "basophils": (0, 1),
    "glucose_fasting": (70, 100),
    "hba1c": (4.0, 5.6),
    "bun": (7, 20),
    "creatinine": (0.6, 1.2),
    "egfr": (90, 120),
    "sodium": (136, 145),
    "potassium": (3.5, 5.0),
    "chloride": (98, 106),
    "calcium": (8.5, 10.5),
    "magnesium": (1.7, 2.2),
    "alt": (7, 56),
    "ast": (10, 40),
    "alp": (44, 147),
    "bilirubin_total": (0.1, 1.2),
    "albumin": (3.5, 5.0),
    "cholesterol_total": (0, 200),
    "ldl": (0, 100),
    "hdl": (40, 100),
    "triglycerides": (0, 150),
    "tsh": (0.4, 4.0),
    "t4_free": (0.8, 1.8),
    "iron": (60, 170),
    "ferritin": (20, 200),
    "vitamin_d": (30, 100),
    "vitamin_b12": (200, 900),
    "folate": (3, 20),
    "crp": (0, 3),
    "esr": (0, 20),
    "ige_total": (0, 100),
}

# Skin-relevant lab values
SKIN_RELEVANT_LABS = [
    "vitamin_d", "vitamin_b12", "folate", "iron", "ferritin",
    "eosinophils", "ige_total", "crp", "esr", "ana_positive",
    "tsh", "glucose_fasting", "hba1c", "albumin"
]


def check_abnormalities(lab_data: dict) -> dict:
    """Check for abnormal values and skin-relevant findings."""
    abnormal = []
    skin_relevant = []

    for field, value in lab_data.items():
        if value is None or field not in REFERENCE_RANGES:
            continue

        low, high = REFERENCE_RANGES[field]

        if value < low:
            status = "low"
            abnormal.append({"field": field, "value": value, "status": status, "range": f"{low}-{high}"})
        elif value > high:
            status = "high"
            abnormal.append({"field": field, "value": value, "status": status, "range": f"{low}-{high}"})

        # Check if skin-relevant
        if field in SKIN_RELEVANT_LABS and (value < low or value > high):
            skin_relevant.append({
                "field": field,
                "value": value,
                "status": "low" if value < low else "high",
                "skin_impact": get_skin_impact(field, value, low, high)
            })

    return {
        "abnormal_count": len(abnormal),
        "abnormal_values": abnormal,
        "skin_relevant_count": len(skin_relevant),
        "skin_relevant_findings": skin_relevant
    }


def get_skin_impact(field: str, value: float, low: float, high: float) -> str:
    """Get skin-related impact description for abnormal lab values."""
    impacts = {
        "vitamin_d": {
            "low": "Low vitamin D associated with psoriasis flares, poor wound healing, and increased skin infection risk",
            "high": "Rarely causes skin issues"
        },
        "vitamin_b12": {
            "low": "B12 deficiency can cause hyperpigmentation, vitiligo-like patches, and angular cheilitis",
            "high": "Rarely causes skin issues"
        },
        "iron": {
            "low": "Iron deficiency linked to pale skin, brittle nails, hair loss, and pruritus",
            "high": "Iron overload can cause bronze skin discoloration"
        },
        "ferritin": {
            "low": "Low ferritin associated with hair loss and nail changes",
            "high": "Elevated ferritin may indicate inflammation affecting skin"
        },
        "eosinophils": {
            "low": "Rarely significant for skin",
            "high": "Elevated eosinophils suggest allergic skin conditions, parasitic infections, or drug reactions"
        },
        "ige_total": {
            "low": "Rarely significant",
            "high": "High IgE associated with atopic dermatitis, urticaria, and allergic skin conditions"
        },
        "crp": {
            "low": "Normal - no inflammation",
            "high": "Elevated CRP indicates systemic inflammation that may worsen inflammatory skin conditions"
        },
        "esr": {
            "low": "Normal",
            "high": "Elevated ESR suggests inflammation - relevant for vasculitis, connective tissue diseases"
        },
        "tsh": {
            "low": "Hyperthyroidism can cause warm, moist skin, hair thinning, and pretibial myxedema",
            "high": "Hypothyroidism causes dry skin, coarse hair, brittle nails, and delayed wound healing"
        },
        "glucose_fasting": {
            "low": "Rarely affects skin",
            "high": "Elevated glucose impairs wound healing, increases infection risk, may cause acanthosis nigricans"
        },
        "hba1c": {
            "low": "Normal",
            "high": "Poor glycemic control associated with diabetic skin complications"
        },
        "albumin": {
            "low": "Low albumin causes edema and poor wound healing",
            "high": "Rarely significant for skin"
        }
    }

    status = "low" if value < low else "high"
    return impacts.get(field, {}).get(status, "May affect skin health")


@router.get("/lab-results")
async def get_lab_results(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all lab results for the current user."""
    results = db.query(LabResults).filter(
        LabResults.user_id == current_user.id
    ).order_by(LabResults.test_date.desc()).all()

    return {
        "lab_results": [
            {
                "id": r.id,
                "test_date": r.test_date.isoformat() if r.test_date else None,
                "test_type": r.test_type,
                "lab_name": r.lab_name,
                "ordering_physician": r.ordering_physician,
                "is_manually_entered": r.is_manually_entered,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                # Include key values for display
                "wbc": r.wbc,
                "hemoglobin": r.hemoglobin,
                "platelets": r.platelets,
                "glucose_fasting": r.glucose_fasting,
                "creatinine": r.creatinine,
                "vitamin_d": r.vitamin_d,
                "tsh": r.tsh,
                "crp": r.crp,
            }
            for r in results
        ],
        "total_count": len(results)
    }


@router.post("/lab-results")
async def create_lab_result(
    data: LabResultCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new lab result entry. Accepts JSON body."""

    # Convert Pydantic model to dict, excluding None values for cleaner DB insert
    lab_data_dict = data.model_dump(exclude_none=True)

    # Create the lab result with all provided fields
    lab_result = LabResults(
        user_id=current_user.id,
        **lab_data_dict
    )

    db.add(lab_result)
    db.commit()
    db.refresh(lab_result)

    # Check for abnormalities using the provided data
    analysis = check_abnormalities(lab_data_dict)

    return {
        "message": "Lab results saved successfully",
        "lab_result_id": lab_result.id,
        "abnormal_count": analysis["abnormal_count"],
        "skin_relevant_count": analysis["skin_relevant_count"],
        "abnormal_values": analysis["abnormal_values"],
        "skin_relevant_findings": analysis["skin_relevant_findings"]
    }


@router.get("/lab-results/{lab_id}")
async def get_lab_result(
    lab_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific lab result by ID."""
    result = db.query(LabResults).filter(
        LabResults.id == lab_id,
        LabResults.user_id == current_user.id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Lab result not found")

    # Return data in nested structure that frontend expects
    lab_data = {
        "id": result.id,
        "user_id": result.user_id,
        "test_date": result.test_date.isoformat() if result.test_date else None,
        "test_type": result.test_type,
        "lab_name": result.lab_name,
        "ordering_physician": result.ordering_physician,
        "notes": getattr(result, 'notes', None),
        "blood_panel": {
            "cbc": {
                "wbc": result.wbc,
                "rbc": result.rbc,
                "hemoglobin": result.hemoglobin,
                "hematocrit": result.hematocrit,
                "platelets": result.platelets,
                "mcv": result.mcv,
                "mch": result.mch,
                "mchc": result.mchc,
                "rdw": result.rdw,
                "mpv": result.mpv,
            },
            "wbc_differential": {
                "neutrophils": result.neutrophils,
                "lymphocytes": result.lymphocytes,
                "monocytes": result.monocytes,
                "eosinophils": result.eosinophils,
                "basophils": result.basophils,
            },
            "wbc_absolute": {
                "neutrophils_abs": result.neutrophils_abs,
                "lymphocytes_abs": result.lymphocytes_abs,
                "monocytes_abs": result.monocytes_abs,
                "eosinophils_abs": result.eosinophils_abs,
                "basophils_abs": result.basophils_abs,
            },
            "metabolic": {
                "glucose_fasting": result.glucose_fasting,
                "hba1c": result.hba1c,
                "eag": result.eag,
                "bun": result.bun,
                "creatinine": result.creatinine,
                "bun_creatinine_ratio": result.bun_creatinine_ratio,
                "egfr": result.egfr,
                "egfr_african_american": result.egfr_african_american,
                "sodium": result.sodium,
                "potassium": result.potassium,
                "chloride": result.chloride,
                "co2": result.co2,
                "calcium": result.calcium,
                "magnesium": result.magnesium,
            },
            "liver": {
                "alt": result.alt,
                "ast": result.ast,
                "alp": result.alp,
                "bilirubin_total": result.bilirubin_total,
                "albumin": result.albumin,
                "total_protein": result.total_protein,
                "globulin": result.globulin,
                "albumin_globulin_ratio": result.albumin_globulin_ratio,
            },
            "lipid": {
                "cholesterol_total": result.cholesterol_total,
                "ldl": result.ldl,
                "hdl": result.hdl,
                "triglycerides": result.triglycerides,
                "chol_hdl_ratio": result.chol_hdl_ratio,
                "non_hdl_cholesterol": result.non_hdl_cholesterol,
            },
            "thyroid": {
                "tsh": result.tsh,
                "t3_uptake": result.t3_uptake,
                "t4_total": result.t4_total,
                "free_t4_index": result.free_t4_index,
                "t4_free": result.t4_free,
            },
            "iron": {
                "iron": result.iron,
                "ferritin": result.ferritin,
                "tibc": result.tibc,
            },
            "vitamins": {
                "vitamin_d": result.vitamin_d,
                "vitamin_b12": result.vitamin_b12,
                "folate": result.folate,
            },
            "inflammatory": {
                "crp": result.crp,
                "esr": result.esr,
            },
            "autoimmune": {
                "ana_positive": result.ana_positive,
            },
            "allergy": {
                "ige_total": result.ige_total,
            },
        },
        "urinalysis": {
            "physical": {
                "color": result.urine_color,
                "appearance": result.urine_appearance,
                "specific_gravity": result.urine_specific_gravity,
                "ph": result.urine_ph,
            },
            "chemical": {
                "protein": result.urine_protein,
                "glucose": result.urine_glucose,
                "ketones": result.urine_ketones,
                "blood": result.urine_blood,
                "bilirubin": result.urine_bilirubin,
                "urobilinogen": result.urine_urobilinogen,
                "nitrite": result.urine_nitrite,
                "leukocyte_esterase": result.urine_leukocyte_esterase,
            },
            "microscopic": {
                "wbc": result.urine_wbc,
                "rbc": result.urine_rbc,
                "bacteria": result.urine_bacteria,
                "squamous_epithelial": result.urine_squamous_epithelial,
            },
            "casts": {
                "hyaline_cast": result.urine_hyaline_cast,
            },
        },
        "stool_test": {
            "color": result.stool_color,
            "occult_blood": result.stool_occult_blood,
            "parasites": result.stool_parasites,
            "calprotectin": result.stool_calprotectin,
        },
    }

    return lab_data


@router.put("/lab-results/{lab_id}")
async def update_lab_result(
    lab_id: int,
    data: LabResultUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update an existing lab result. Accepts JSON body."""
    result = db.query(LabResults).filter(
        LabResults.id == lab_id,
        LabResults.user_id == current_user.id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Lab result not found")

    # Update only fields that were provided (not None)
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(result, field):
            setattr(result, field, value)

    db.commit()
    db.refresh(result)

    return {
        "message": "Lab result updated successfully",
        "lab_result_id": result.id
    }


@router.delete("/lab-results/{lab_id}")
async def delete_lab_result(
    lab_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a lab result."""
    result = db.query(LabResults).filter(
        LabResults.id == lab_id,
        LabResults.user_id == current_user.id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Lab result not found")

    db.delete(result)
    db.commit()

    return {"message": "Lab result deleted successfully"}


@router.post("/lab-results/parse-pdf")
async def parse_lab_pdf_endpoint(
    file: UploadFile = File(...),
    use_ocr: bool = Form(False),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Parse lab results from an uploaded PDF.
    Uses regex patterns to extract common lab values.
    """
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Import the parser
        from lab_pdf_parser import parse_lab_pdf, validate_extracted_values

        # Read the PDF bytes
        pdf_bytes = await file.read()

        # Parse the PDF
        result = parse_lab_pdf(pdf_bytes, use_ocr=use_ocr)

        # Validate the extracted values
        validation = validate_extracted_values(result.get("extracted_values", {}))

        # Add validation warnings to parsing notes
        if validation.get("warnings"):
            result["parsing_notes"].extend(validation["warnings"])

        return result

    except Exception as e:
        print(f"Error parsing lab PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse PDF: {str(e)}. Try enabling OCR for scanned documents."
        )
