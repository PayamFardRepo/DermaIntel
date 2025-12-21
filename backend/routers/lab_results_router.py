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
    test_date: date = Form(...),
    test_type: str = Form("blood"),
    lab_name: Optional[str] = Form(None),
    ordering_physician: Optional[str] = Form(None),
    # CBC
    wbc: Optional[float] = Form(None),
    rbc: Optional[float] = Form(None),
    hemoglobin: Optional[float] = Form(None),
    hematocrit: Optional[float] = Form(None),
    platelets: Optional[float] = Form(None),
    # Differential
    neutrophils: Optional[float] = Form(None),
    lymphocytes: Optional[float] = Form(None),
    monocytes: Optional[float] = Form(None),
    eosinophils: Optional[float] = Form(None),
    basophils: Optional[float] = Form(None),
    # Metabolic
    glucose_fasting: Optional[float] = Form(None),
    glucose_random: Optional[float] = Form(None),
    hba1c: Optional[float] = Form(None),
    bun: Optional[float] = Form(None),
    creatinine: Optional[float] = Form(None),
    egfr: Optional[float] = Form(None),
    sodium: Optional[float] = Form(None),
    potassium: Optional[float] = Form(None),
    chloride: Optional[float] = Form(None),
    co2: Optional[float] = Form(None),
    calcium: Optional[float] = Form(None),
    magnesium: Optional[float] = Form(None),
    phosphorus: Optional[float] = Form(None),
    # Liver
    alt: Optional[float] = Form(None),
    ast: Optional[float] = Form(None),
    alp: Optional[float] = Form(None),
    bilirubin_total: Optional[float] = Form(None),
    bilirubin_direct: Optional[float] = Form(None),
    albumin: Optional[float] = Form(None),
    total_protein: Optional[float] = Form(None),
    # Lipid
    cholesterol_total: Optional[float] = Form(None),
    ldl: Optional[float] = Form(None),
    hdl: Optional[float] = Form(None),
    triglycerides: Optional[float] = Form(None),
    # Thyroid
    tsh: Optional[float] = Form(None),
    t4_free: Optional[float] = Form(None),
    t3_free: Optional[float] = Form(None),
    # Iron
    iron: Optional[float] = Form(None),
    ferritin: Optional[float] = Form(None),
    tibc: Optional[float] = Form(None),
    # Vitamins
    vitamin_d: Optional[float] = Form(None),
    vitamin_b12: Optional[float] = Form(None),
    folate: Optional[float] = Form(None),
    # Inflammatory
    crp: Optional[float] = Form(None),
    esr: Optional[float] = Form(None),
    # Autoimmune
    ana_positive: Optional[bool] = Form(None),
    ana_titer: Optional[str] = Form(None),
    rheumatoid_factor: Optional[float] = Form(None),
    # Allergy
    ige_total: Optional[float] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new lab result entry."""

    lab_result = LabResults(
        user_id=current_user.id,
        test_date=test_date,
        test_type=test_type,
        lab_name=lab_name,
        ordering_physician=ordering_physician,
        is_manually_entered=True,
        # CBC
        wbc=wbc,
        rbc=rbc,
        hemoglobin=hemoglobin,
        hematocrit=hematocrit,
        platelets=platelets,
        # Differential
        neutrophils=neutrophils,
        lymphocytes=lymphocytes,
        monocytes=monocytes,
        eosinophils=eosinophils,
        basophils=basophils,
        # Metabolic
        glucose_fasting=glucose_fasting,
        glucose_random=glucose_random,
        hba1c=hba1c,
        bun=bun,
        creatinine=creatinine,
        egfr=egfr,
        sodium=sodium,
        potassium=potassium,
        chloride=chloride,
        co2=co2,
        calcium=calcium,
        magnesium=magnesium,
        phosphorus=phosphorus,
        # Liver
        alt=alt,
        ast=ast,
        alp=alp,
        bilirubin_total=bilirubin_total,
        bilirubin_direct=bilirubin_direct,
        albumin=albumin,
        total_protein=total_protein,
        # Lipid
        cholesterol_total=cholesterol_total,
        ldl=ldl,
        hdl=hdl,
        triglycerides=triglycerides,
        # Thyroid
        tsh=tsh,
        t4_free=t4_free,
        t3_free=t3_free,
        # Iron
        iron=iron,
        ferritin=ferritin,
        tibc=tibc,
        # Vitamins
        vitamin_d=vitamin_d,
        vitamin_b12=vitamin_b12,
        folate=folate,
        # Inflammatory
        crp=crp,
        esr=esr,
        # Autoimmune
        ana_positive=ana_positive,
        ana_titer=ana_titer,
        rheumatoid_factor=rheumatoid_factor,
        # Allergy
        ige_total=ige_total,
    )

    db.add(lab_result)
    db.commit()
    db.refresh(lab_result)

    # Check for abnormalities
    lab_data = {
        "wbc": wbc, "rbc": rbc, "hemoglobin": hemoglobin, "hematocrit": hematocrit,
        "platelets": platelets, "neutrophils": neutrophils, "lymphocytes": lymphocytes,
        "eosinophils": eosinophils, "glucose_fasting": glucose_fasting, "hba1c": hba1c,
        "creatinine": creatinine, "sodium": sodium, "potassium": potassium,
        "alt": alt, "ast": ast, "albumin": albumin, "cholesterol_total": cholesterol_total,
        "ldl": ldl, "hdl": hdl, "triglycerides": triglycerides, "tsh": tsh,
        "iron": iron, "ferritin": ferritin, "vitamin_d": vitamin_d, "vitamin_b12": vitamin_b12,
        "crp": crp, "esr": esr, "ige_total": ige_total
    }

    analysis = check_abnormalities(lab_data)

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

    # Convert to dict, excluding SQLAlchemy internals
    lab_data = {}
    for column in LabResults.__table__.columns:
        value = getattr(result, column.name)
        if column.name == "test_date" and value:
            lab_data[column.name] = value.isoformat()
        elif column.name in ["created_at", "updated_at"] and value:
            lab_data[column.name] = value.isoformat()
        else:
            lab_data[column.name] = value

    return lab_data


@router.put("/lab-results/{lab_id}")
async def update_lab_result(
    lab_id: int,
    test_date: Optional[date] = Form(None),
    test_type: Optional[str] = Form(None),
    lab_name: Optional[str] = Form(None),
    ordering_physician: Optional[str] = Form(None),
    # CBC
    wbc: Optional[float] = Form(None),
    rbc: Optional[float] = Form(None),
    hemoglobin: Optional[float] = Form(None),
    hematocrit: Optional[float] = Form(None),
    platelets: Optional[float] = Form(None),
    # Differential
    neutrophils: Optional[float] = Form(None),
    lymphocytes: Optional[float] = Form(None),
    monocytes: Optional[float] = Form(None),
    eosinophils: Optional[float] = Form(None),
    basophils: Optional[float] = Form(None),
    # Metabolic
    glucose_fasting: Optional[float] = Form(None),
    hba1c: Optional[float] = Form(None),
    creatinine: Optional[float] = Form(None),
    sodium: Optional[float] = Form(None),
    potassium: Optional[float] = Form(None),
    # Liver
    alt: Optional[float] = Form(None),
    ast: Optional[float] = Form(None),
    albumin: Optional[float] = Form(None),
    # Lipid
    cholesterol_total: Optional[float] = Form(None),
    ldl: Optional[float] = Form(None),
    hdl: Optional[float] = Form(None),
    triglycerides: Optional[float] = Form(None),
    # Thyroid
    tsh: Optional[float] = Form(None),
    # Vitamins
    vitamin_d: Optional[float] = Form(None),
    vitamin_b12: Optional[float] = Form(None),
    # Inflammatory
    crp: Optional[float] = Form(None),
    esr: Optional[float] = Form(None),
    # Allergy
    ige_total: Optional[float] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update an existing lab result."""
    result = db.query(LabResults).filter(
        LabResults.id == lab_id,
        LabResults.user_id == current_user.id
    ).first()

    if not result:
        raise HTTPException(status_code=404, detail="Lab result not found")

    # Update provided fields
    if test_date is not None:
        result.test_date = test_date
    if test_type is not None:
        result.test_type = test_type
    if lab_name is not None:
        result.lab_name = lab_name
    if ordering_physician is not None:
        result.ordering_physician = ordering_physician

    # Update lab values if provided
    lab_fields = [
        "wbc", "rbc", "hemoglobin", "hematocrit", "platelets",
        "neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils",
        "glucose_fasting", "hba1c", "creatinine", "sodium", "potassium",
        "alt", "ast", "albumin", "cholesterol_total", "ldl", "hdl", "triglycerides",
        "tsh", "vitamin_d", "vitamin_b12", "crp", "esr", "ige_total"
    ]

    local_vars = locals()
    for field in lab_fields:
        value = local_vars.get(field)
        if value is not None:
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
async def parse_lab_pdf(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Parse lab results from an uploaded PDF.
    Note: This is a placeholder - actual PDF parsing requires OCR integration.
    """
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # For now, return a message indicating manual entry is needed
    return {
        "message": "PDF uploaded successfully. PDF parsing is not yet implemented - please enter values manually.",
        "filename": file.filename,
        "parsed_values": {},
        "requires_manual_entry": True
    }
