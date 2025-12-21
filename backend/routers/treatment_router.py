"""
Treatment Management Router

Endpoints for:
- Treatment CRUD operations
- Treatment logging (dose tracking)
- Treatment effectiveness assessments
- Treatment outcome prediction
- AR treatment simulation with image generation
"""

from fastapi import APIRouter, Depends, HTTPException, Form, Query, File, UploadFile
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from pathlib import Path
import time

from database import get_db, User, Treatment, TreatmentLog, TreatmentEffectiveness, AnalysisHistory
from auth import get_current_active_user

# Import treatment predictor for image-based simulation
from treatment_outcome_predictor import treatment_predictor

# Upload directory for treatment images
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter(tags=["Treatment Management"])


# =============================================================================
# TREATMENT CRUD
# =============================================================================

@router.post("/treatments")
async def create_treatment(
    treatment_name: str = Form(...),
    treatment_type: str = Form(...),
    start_date: str = Form(...),
    active_ingredient: Optional[str] = Form(None),
    brand_name: Optional[str] = Form(None),
    dosage: Optional[str] = Form(None),
    dosage_unit: Optional[str] = Form(None),
    route: Optional[str] = Form(None),
    frequency: Optional[str] = Form(None),
    instructions: Optional[str] = Form(None),
    planned_end_date: Optional[str] = Form(None),
    duration_weeks: Optional[int] = Form(None),
    lesion_group_id: Optional[int] = Form(None),
    target_condition: Optional[str] = Form(None),
    prescribing_physician: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new treatment record."""
    try:
        # Parse start date
        start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

        # Parse planned end date if provided
        planned_end_dt = None
        if planned_end_date:
            planned_end_dt = datetime.fromisoformat(planned_end_date.replace('Z', '+00:00'))

        # Create treatment
        treatment = Treatment(
            user_id=current_user.id,
            lesion_group_id=lesion_group_id,
            treatment_name=treatment_name,
            treatment_type=treatment_type,
            active_ingredient=active_ingredient,
            brand_name=brand_name,
            dosage=dosage,
            dosage_unit=dosage_unit,
            route=route,
            frequency=frequency,
            instructions=instructions,
            start_date=start_dt,
            planned_end_date=planned_end_dt,
            duration_weeks=duration_weeks,
            is_active=True,
            target_condition=target_condition,
            prescribing_physician=prescribing_physician,
            created_at=datetime.now()
        )

        db.add(treatment)
        db.commit()
        db.refresh(treatment)

        return {
            "id": treatment.id,
            "treatment_name": treatment.treatment_name,
            "treatment_type": treatment.treatment_type,
            "start_date": treatment.start_date.isoformat(),
            "is_active": treatment.is_active,
            "message": "Treatment created successfully"
        }

    except Exception as e:
        db.rollback()
        print(f"Error creating treatment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create treatment: {str(e)}")


@router.get("/treatments")
async def get_user_treatments(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all treatments for the current user."""
    try:
        treatments = db.query(Treatment).filter(
            Treatment.user_id == current_user.id
        ).order_by(Treatment.start_date.desc()).all()

        result = []
        for treatment in treatments:
            # Count logs for this treatment
            log_count = db.query(TreatmentLog).filter(
                TreatmentLog.treatment_id == treatment.id
            ).count()

            # Get latest effectiveness assessment
            latest_effectiveness = db.query(TreatmentEffectiveness).filter(
                TreatmentEffectiveness.treatment_id == treatment.id
            ).order_by(TreatmentEffectiveness.assessment_date.desc()).first()

            result.append({
                "id": treatment.id,
                "treatment_name": treatment.treatment_name,
                "treatment_type": treatment.treatment_type,
                "active_ingredient": treatment.active_ingredient,
                "brand_name": treatment.brand_name,
                "dosage": treatment.dosage,
                "dosage_unit": treatment.dosage_unit,
                "route": treatment.route,
                "frequency": treatment.frequency,
                "instructions": treatment.instructions,
                "start_date": treatment.start_date.isoformat(),
                "planned_end_date": treatment.planned_end_date.isoformat() if treatment.planned_end_date else None,
                "actual_end_date": treatment.actual_end_date.isoformat() if treatment.actual_end_date else None,
                "duration_weeks": treatment.duration_weeks,
                "is_active": treatment.is_active,
                "target_condition": treatment.target_condition,
                "prescribing_physician": treatment.prescribing_physician,
                "log_count": log_count,
                "latest_effectiveness": {
                    "improvement_percentage": latest_effectiveness.improvement_percentage,
                    "overall_effectiveness": latest_effectiveness.overall_effectiveness,
                    "assessment_date": latest_effectiveness.assessment_date.isoformat()
                } if latest_effectiveness else None
            })

        return result

    except Exception as e:
        print(f"Error fetching treatments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch treatments: {str(e)}")


# =============================================================================
# TREATMENT LOGGING
# =============================================================================

@router.post("/treatment-logs")
async def create_treatment_log(
    treatment_id: int = Form(...),
    administered_date: str = Form(...),
    dose_amount: Optional[float] = Form(None),
    dose_unit: Optional[str] = Form(None),
    application_area: Optional[str] = Form(None),
    application_method: Optional[str] = Form(None),
    taken_as_prescribed: bool = Form(True),
    missed_dose: bool = Form(False),
    late_dose: bool = Form(False),
    hours_late: Optional[float] = Form(None),
    immediate_reaction: Optional[str] = Form(None),
    reaction_severity: Optional[int] = Form(None),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Log a treatment dose/application."""
    try:
        # Verify treatment belongs to user
        treatment = db.query(Treatment).filter(
            Treatment.id == treatment_id,
            Treatment.user_id == current_user.id
        ).first()

        if not treatment:
            raise HTTPException(status_code=404, detail="Treatment not found")

        # Parse administered date
        admin_dt = datetime.fromisoformat(administered_date.replace('Z', '+00:00'))

        # Create log
        log = TreatmentLog(
            treatment_id=treatment_id,
            user_id=current_user.id,
            administered_date=admin_dt,
            dose_amount=dose_amount,
            dose_unit=dose_unit,
            application_area=application_area,
            application_method=application_method,
            taken_as_prescribed=taken_as_prescribed,
            missed_dose=missed_dose,
            late_dose=late_dose,
            hours_late=hours_late,
            immediate_reaction=immediate_reaction,
            reaction_severity=reaction_severity,
            notes=notes,
            created_at=datetime.now()
        )

        db.add(log)
        db.commit()
        db.refresh(log)

        return {
            "id": log.id,
            "treatment_id": log.treatment_id,
            "administered_date": log.administered_date.isoformat(),
            "taken_as_prescribed": log.taken_as_prescribed,
            "message": "Treatment log created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error creating treatment log: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create treatment log: {str(e)}")


@router.get("/treatment-logs/{treatment_id}")
async def get_treatment_logs(
    treatment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all logs for a specific treatment."""
    try:
        # Verify treatment belongs to user
        treatment = db.query(Treatment).filter(
            Treatment.id == treatment_id,
            Treatment.user_id == current_user.id
        ).first()

        if not treatment:
            raise HTTPException(status_code=404, detail="Treatment not found")

        logs = db.query(TreatmentLog).filter(
            TreatmentLog.treatment_id == treatment_id
        ).order_by(TreatmentLog.administered_date.desc()).all()

        return [{
            "id": log.id,
            "administered_date": log.administered_date.isoformat(),
            "dose_amount": log.dose_amount,
            "dose_unit": log.dose_unit,
            "application_area": log.application_area,
            "application_method": log.application_method,
            "taken_as_prescribed": log.taken_as_prescribed,
            "missed_dose": log.missed_dose,
            "late_dose": log.late_dose,
            "hours_late": log.hours_late,
            "immediate_reaction": log.immediate_reaction,
            "reaction_severity": log.reaction_severity,
            "notes": log.notes
        } for log in logs]

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching treatment logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch treatment logs: {str(e)}")


# =============================================================================
# TREATMENT EFFECTIVENESS
# =============================================================================

@router.get("/treatment-effectiveness/{treatment_id}")
async def get_treatment_effectiveness(
    treatment_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get effectiveness assessments for a treatment."""
    try:
        # Verify treatment belongs to user
        treatment = db.query(Treatment).filter(
            Treatment.id == treatment_id,
            Treatment.user_id == current_user.id
        ).first()

        if not treatment:
            raise HTTPException(status_code=404, detail="Treatment not found")

        assessments = db.query(TreatmentEffectiveness).filter(
            TreatmentEffectiveness.treatment_id == treatment_id
        ).order_by(TreatmentEffectiveness.assessment_date.asc()).all()

        return [{
            "id": a.id,
            "assessment_date": a.assessment_date.isoformat(),
            "days_into_treatment": a.days_into_treatment,
            "baseline_size_mm": a.baseline_size_mm,
            "current_size_mm": a.current_size_mm,
            "size_change_mm": a.size_change_mm,
            "size_change_percent": a.size_change_percent,
            "improvement_percentage": a.improvement_percentage,
            "overall_effectiveness": a.overall_effectiveness,
            "patient_satisfaction": a.patient_satisfaction,
            "side_effects_severity": a.side_effects_severity,
            "notes": a.notes
        } for a in assessments]

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching effectiveness data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch effectiveness data: {str(e)}")


@router.post("/treatment-effectiveness")
async def create_treatment_effectiveness(
    treatment_id: int = Form(...),
    assessment_date: str = Form(...),
    days_into_treatment: int = Form(...),
    baseline_size_mm: Optional[float] = Form(None),
    current_size_mm: Optional[float] = Form(None),
    improvement_percentage: Optional[float] = Form(None),
    overall_effectiveness: str = Form("moderate"),
    patient_satisfaction: Optional[int] = Form(None),
    side_effects_severity: Optional[int] = Form(None),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Record a treatment effectiveness assessment."""
    try:
        # Verify treatment belongs to user
        treatment = db.query(Treatment).filter(
            Treatment.id == treatment_id,
            Treatment.user_id == current_user.id
        ).first()

        if not treatment:
            raise HTTPException(status_code=404, detail="Treatment not found")

        # Parse assessment date
        assess_dt = datetime.fromisoformat(assessment_date.replace('Z', '+00:00'))

        # Calculate size change if both measurements provided
        size_change_mm = None
        size_change_percent = None
        if baseline_size_mm and current_size_mm:
            size_change_mm = current_size_mm - baseline_size_mm
            if baseline_size_mm > 0:
                size_change_percent = (size_change_mm / baseline_size_mm) * 100

        assessment = TreatmentEffectiveness(
            treatment_id=treatment_id,
            user_id=current_user.id,
            assessment_date=assess_dt,
            days_into_treatment=days_into_treatment,
            baseline_size_mm=baseline_size_mm,
            current_size_mm=current_size_mm,
            size_change_mm=size_change_mm,
            size_change_percent=size_change_percent,
            improvement_percentage=improvement_percentage,
            overall_effectiveness=overall_effectiveness,
            patient_satisfaction=patient_satisfaction,
            side_effects_severity=side_effects_severity,
            notes=notes,
            created_at=datetime.now()
        )

        db.add(assessment)
        db.commit()
        db.refresh(assessment)

        return {
            "id": assessment.id,
            "treatment_id": assessment.treatment_id,
            "assessment_date": assessment.assessment_date.isoformat(),
            "overall_effectiveness": assessment.overall_effectiveness,
            "message": "Effectiveness assessment recorded successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error recording effectiveness: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record effectiveness: {str(e)}")


# =============================================================================
# TREATMENT OUTCOME PREDICTION
# =============================================================================

@router.post("/predict-treatment-outcome")
async def predict_treatment_outcome(
    condition: str = Form(...),
    treatment_type: str = Form(...),
    patient_age: int = Form(None),
    condition_severity: str = Form(None),  # "mild", "moderate", "severe"
    treatment_duration_weeks: int = Form(12),
    previous_treatments: str = Form(None),  # JSON array of previous treatment names
    comorbidities: str = Form(None),  # JSON array
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Predict treatment outcome based on condition and patient factors.

    Uses historical data and evidence-based algorithms to predict:
    - Expected response rate
    - Time to improvement
    - Likelihood of side effects
    - Alternative recommendations
    """
    import json

    # Treatment effectiveness data (evidence-based)
    treatment_data = {
        "melanoma": {
            "surgical_excision": {"response_rate": 95, "time_to_response_weeks": 2},
            "immunotherapy": {"response_rate": 50, "time_to_response_weeks": 12},
            "targeted_therapy": {"response_rate": 60, "time_to_response_weeks": 8}
        },
        "basal_cell_carcinoma": {
            "mohs_surgery": {"response_rate": 99, "time_to_response_weeks": 2},
            "surgical_excision": {"response_rate": 95, "time_to_response_weeks": 2},
            "topical_imiquimod": {"response_rate": 80, "time_to_response_weeks": 12},
            "cryotherapy": {"response_rate": 85, "time_to_response_weeks": 4}
        },
        "psoriasis": {
            "topical_steroids": {"response_rate": 70, "time_to_response_weeks": 4},
            "phototherapy": {"response_rate": 75, "time_to_response_weeks": 8},
            "biologics": {"response_rate": 80, "time_to_response_weeks": 12},
            "methotrexate": {"response_rate": 65, "time_to_response_weeks": 8}
        },
        "eczema": {
            "topical_steroids": {"response_rate": 80, "time_to_response_weeks": 2},
            "topical_calcineurin_inhibitors": {"response_rate": 70, "time_to_response_weeks": 4},
            "dupilumab": {"response_rate": 75, "time_to_response_weeks": 8}
        },
        "acne": {
            "topical_retinoids": {"response_rate": 70, "time_to_response_weeks": 8},
            "oral_antibiotics": {"response_rate": 75, "time_to_response_weeks": 6},
            "isotretinoin": {"response_rate": 85, "time_to_response_weeks": 16}
        }
    }

    # Normalize condition name
    condition_lower = condition.lower().replace(" ", "_").replace("-", "_")
    treatment_lower = treatment_type.lower().replace(" ", "_").replace("-", "_")

    # Get base prediction
    condition_treatments = treatment_data.get(condition_lower, {})
    base_data = condition_treatments.get(treatment_lower, {
        "response_rate": 60,
        "time_to_response_weeks": 8
    })

    base_response_rate = base_data["response_rate"]
    base_time_to_response = base_data["time_to_response_weeks"]

    # Adjust for severity
    severity_multipliers = {"mild": 1.1, "moderate": 1.0, "severe": 0.85}
    severity_mult = severity_multipliers.get(condition_severity, 1.0)

    # Adjust for age
    age_multiplier = 1.0
    if patient_age:
        if patient_age > 65:
            age_multiplier = 0.95
        elif patient_age < 18:
            age_multiplier = 1.05

    # Calculate adjusted prediction
    predicted_response_rate = min(100, base_response_rate * severity_mult * age_multiplier)
    predicted_time_to_response = int(base_time_to_response / severity_mult)

    # Side effect likelihood
    side_effect_rates = {
        "topical_steroids": 15,
        "oral_steroids": 40,
        "biologics": 25,
        "immunotherapy": 60,
        "isotretinoin": 70,
        "methotrexate": 50
    }
    side_effect_likelihood = side_effect_rates.get(treatment_lower, 20)

    # Generate recommendations
    recommendations = []
    if predicted_response_rate < 60:
        recommendations.append("Consider alternative treatments with higher efficacy")
    if side_effect_likelihood > 50:
        recommendations.append("Monitor closely for side effects")
    if predicted_time_to_response > 10:
        recommendations.append("Patient education on treatment timeline expectations")

    # Find alternative treatments
    alternatives = []
    for alt_treatment, alt_data in condition_treatments.items():
        if alt_treatment != treatment_lower and alt_data["response_rate"] > base_response_rate:
            alternatives.append({
                "treatment": alt_treatment.replace("_", " ").title(),
                "expected_response_rate": alt_data["response_rate"],
                "time_to_response_weeks": alt_data["time_to_response_weeks"]
            })

    return {
        "condition": condition,
        "treatment": treatment_type,
        "prediction": {
            "expected_response_rate": round(predicted_response_rate, 1),
            "time_to_improvement_weeks": predicted_time_to_response,
            "side_effect_likelihood_percent": side_effect_likelihood,
            "confidence_level": "moderate" if condition_lower in treatment_data else "low"
        },
        "factors_considered": {
            "patient_age": patient_age,
            "condition_severity": condition_severity,
            "age_adjustment": age_multiplier,
            "severity_adjustment": severity_mult
        },
        "recommendations": recommendations,
        "alternative_treatments": alternatives[:3],
        "disclaimer": "This prediction is based on population-level data and may not reflect individual outcomes. Consult with your healthcare provider."
    }


# =============================================================================
# TREATMENT RECOMMENDATIONS
# =============================================================================

@router.get("/treatment-recommendations")
async def get_treatment_recommendations(
    condition: str = Query(...),
    severity: str = Query(None),
    patient_age: int = Query(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get evidence-based treatment recommendations for a skin condition.

    Returns first-line, second-line, and alternative treatments
    with expected efficacy and considerations.
    """
    # Treatment guidelines database
    treatment_guidelines = {
        "melanoma": {
            "first_line": [
                {"treatment": "Wide Local Excision", "efficacy": 95, "notes": "Standard of care for localized melanoma"},
                {"treatment": "Sentinel Lymph Node Biopsy", "efficacy": None, "notes": "For staging if thickness >0.8mm"}
            ],
            "second_line": [
                {"treatment": "Immunotherapy (Nivolumab/Pembrolizumab)", "efficacy": 50, "notes": "For advanced or metastatic"},
                {"treatment": "Targeted Therapy (BRAF/MEK inhibitors)", "efficacy": 60, "notes": "For BRAF-mutated tumors"}
            ],
            "adjunct": [
                {"treatment": "Regular skin surveillance", "notes": "Every 3-6 months"},
                {"treatment": "Sun protection measures", "notes": "SPF 30+ daily"}
            ]
        },
        "basal cell carcinoma": {
            "first_line": [
                {"treatment": "Mohs Micrographic Surgery", "efficacy": 99, "notes": "Best for high-risk or facial lesions"},
                {"treatment": "Standard Surgical Excision", "efficacy": 95, "notes": "For low-risk lesions"}
            ],
            "second_line": [
                {"treatment": "Cryotherapy", "efficacy": 85, "notes": "For superficial BCC"},
                {"treatment": "Topical 5-FU or Imiquimod", "efficacy": 80, "notes": "For superficial BCC on trunk/extremities"}
            ],
            "adjunct": [
                {"treatment": "Annual full-body skin exam", "notes": "High recurrence risk population"}
            ]
        },
        "psoriasis": {
            "first_line": [
                {"treatment": "Topical Corticosteroids", "efficacy": 70, "notes": "Mild-moderate disease"},
                {"treatment": "Vitamin D Analogues (Calcipotriol)", "efficacy": 65, "notes": "Maintenance therapy"}
            ],
            "second_line": [
                {"treatment": "Phototherapy (NB-UVB)", "efficacy": 75, "notes": "Moderate-severe disease"},
                {"treatment": "Methotrexate", "efficacy": 65, "notes": "Systemic option, monitor LFTs"}
            ],
            "third_line": [
                {"treatment": "Biologics (TNF/IL-17/IL-23 inhibitors)", "efficacy": 80, "notes": "Moderate-severe, failed conventional"}
            ]
        },
        "eczema": {
            "first_line": [
                {"treatment": "Emollients", "efficacy": 60, "notes": "Foundation of therapy, use liberally"},
                {"treatment": "Topical Corticosteroids", "efficacy": 80, "notes": "For flares, potency based on site"}
            ],
            "second_line": [
                {"treatment": "Topical Calcineurin Inhibitors", "efficacy": 70, "notes": "Steroid-sparing, good for face"},
                {"treatment": "Phototherapy", "efficacy": 70, "notes": "For widespread disease"}
            ],
            "third_line": [
                {"treatment": "Dupilumab", "efficacy": 75, "notes": "FDA-approved for moderate-severe AD"}
            ]
        },
        "acne": {
            "first_line": [
                {"treatment": "Topical Retinoids", "efficacy": 70, "notes": "Comedonal and inflammatory acne"},
                {"treatment": "Benzoyl Peroxide", "efficacy": 60, "notes": "Reduces P. acnes"}
            ],
            "second_line": [
                {"treatment": "Oral Antibiotics", "efficacy": 75, "notes": "Moderate inflammatory, limit to 3 months"},
                {"treatment": "Hormonal Therapy", "efficacy": 70, "notes": "For females with hormonal pattern"}
            ],
            "third_line": [
                {"treatment": "Isotretinoin", "efficacy": 85, "notes": "Severe/recalcitrant, iPLEDGE required"}
            ]
        }
    }

    # Normalize condition name
    condition_lower = condition.lower().replace("_", " ")

    guidelines = treatment_guidelines.get(condition_lower, None)

    if not guidelines:
        return {
            "condition": condition,
            "recommendations_found": False,
            "message": f"No specific guidelines found for '{condition}'. Consult a dermatologist.",
            "general_recommendations": [
                "Schedule an appointment with a dermatologist",
                "Take photos to track changes",
                "Avoid scratching or irritating the affected area",
                "Use gentle, fragrance-free skincare products"
            ]
        }

    # Adjust for severity
    if severity == "severe":
        primary_treatments = guidelines.get("second_line", guidelines["first_line"])
    else:
        primary_treatments = guidelines["first_line"]

    # Age-specific considerations
    age_notes = []
    if patient_age:
        if patient_age < 12:
            age_notes.append("Pediatric dosing may be required")
            age_notes.append("Some treatments not approved for children")
        elif patient_age > 65:
            age_notes.append("Consider drug interactions with other medications")
            age_notes.append("Monitor for increased side effect susceptibility")

    return {
        "condition": condition,
        "severity": severity,
        "recommendations_found": True,
        "treatment_recommendations": {
            "first_line": guidelines["first_line"],
            "second_line": guidelines.get("second_line", []),
            "third_line": guidelines.get("third_line", []),
            "adjunct": guidelines.get("adjunct", [])
        },
        "age_considerations": age_notes if age_notes else None,
        "recommended_approach": primary_treatments,
        "general_advice": [
            "Follow prescribed treatment regimen consistently",
            "Report any adverse effects to your healthcare provider",
            "Attend follow-up appointments as scheduled",
            "Lifestyle modifications may enhance treatment efficacy"
        ],
        "disclaimer": "These are general guidelines. Individual treatment plans should be determined by a qualified healthcare provider."
    }


@router.post("/analysis/{analysis_id}/check-treatment-interactions")
async def check_treatment_interactions_for_analysis(
    analysis_id: int,
    proposed_treatment: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Check if a proposed treatment has any interactions with the patient's
    current medications or conditions based on their analysis record.
    """
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Get patient medications from analysis
    import json
    medications = []
    if analysis.medications:
        try:
            medications = json.loads(analysis.medications) if isinstance(analysis.medications, str) else analysis.medications
        except:
            pass

    # Get medical conditions
    conditions = []
    if analysis.immunosuppression:
        conditions.append("immunosuppression")
    if analysis.previous_skin_cancers:
        conditions.append("history of skin cancer")

    # Known interactions database
    interaction_database = {
        "isotretinoin": {
            "contraindicated": ["pregnancy", "liver disease"],
            "caution": ["tetracyclines", "vitamin_a"],
            "monitoring": ["liver function", "lipid panel", "pregnancy test"]
        },
        "methotrexate": {
            "contraindicated": ["pregnancy", "liver disease", "immunosuppression"],
            "caution": ["nsaids", "trimethoprim"],
            "monitoring": ["CBC", "liver function", "renal function"]
        },
        "biologics": {
            "contraindicated": ["active infection", "tb"],
            "caution": ["immunosuppression"],
            "monitoring": ["TB screening", "infection signs"]
        },
        "topical_steroids": {
            "contraindicated": [],
            "caution": ["diabetes", "thin skin areas"],
            "monitoring": ["skin atrophy", "adrenal suppression if extensive"]
        }
    }

    treatment_lower = proposed_treatment.lower().replace(" ", "_")
    interactions = interaction_database.get(treatment_lower, {})

    # Check for medication interactions
    med_names = [m.get("name", "").lower() for m in medications if isinstance(m, dict)]

    found_interactions = []
    for med in med_names:
        if med in interactions.get("caution", []):
            found_interactions.append({
                "type": "caution",
                "medication": med,
                "note": f"Use caution when combining {proposed_treatment} with {med}"
            })

    # Check condition interactions
    condition_interactions = []
    for condition in conditions:
        if condition in interactions.get("contraindicated", []):
            condition_interactions.append({
                "type": "contraindication",
                "condition": condition,
                "note": f"{proposed_treatment} may be contraindicated with {condition}"
            })
        elif condition in interactions.get("caution", []):
            condition_interactions.append({
                "type": "caution",
                "condition": condition,
                "note": f"Use caution with {proposed_treatment} given history of {condition}"
            })

    has_issues = len(found_interactions) > 0 or len(condition_interactions) > 0

    return {
        "analysis_id": analysis_id,
        "proposed_treatment": proposed_treatment,
        "safety_check": {
            "has_interactions": has_issues,
            "medication_interactions": found_interactions,
            "condition_interactions": condition_interactions,
            "required_monitoring": interactions.get("monitoring", [])
        },
        "patient_context": {
            "current_medications": len(medications),
            "relevant_conditions": conditions
        },
        "recommendation": "Review interactions with prescriber" if has_issues else "No significant interactions identified",
        "disclaimer": "This is not a comprehensive drug interaction check. Always verify with a pharmacist or physician."
    }


# =============================================================================
# AR TREATMENT SIMULATOR (Image-based)
# =============================================================================

@router.post("/simulate-treatment-outcome")
async def simulate_treatment_outcome_with_image(
    image: UploadFile = File(...),
    treatment_type: str = Form(...),
    timeframe: str = Form(...),
    diagnosis: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """
    Simulate treatment outcome with AI-generated "before/after" images.

    This endpoint is used by the AR Treatment Simulator screen.

    Args:
        image: Image file of skin condition
        treatment_type: Type of treatment (e.g., 'mohs-surgery', 'laser-therapy')
        timeframe: '6months', '1year', or '2years'
        diagnosis: Current diagnosis (e.g., 'melanoma', 'eczema')

    Returns:
        JSON with predicted outcome including generated "after" image URLs
    """
    try:
        # Read image
        image_bytes = await image.read()

        # Calculate improvement percentage based on treatment type
        treatment_improvements = {
            # Surgical treatments
            'surgical-excision': 95,
            'mohs-surgery': 98,
            'mohs-bcc': 99,
            'excision-bcc': 95,
            'curettage': 95,
            'electrodesiccation': 92,
            'edc-bcc': 90,
            # Topical treatments
            'topical-steroid': 70,
            'topical-steroid-eczema': 80,
            'topical-steroid-psoriasis': 65,
            'topical-steroid-vitiligo': 60,
            'topical-retinoid': 75,
            'benzoyl-peroxide': 65,
            'calcineurin-inhibitor': 75,
            'calcineurin-vitiligo': 65,
            'metronidazole': 70,
            'azelaic-acid': 75,
            'vitamin-d-analog': 70,
            'salicylic-acid': 65,
            'imiquimod': 80,
            # Procedural treatments
            'laser-therapy': 85,
            'laser-ipl-rosacea': 85,
            'laser-acne': 70,
            'phototherapy': 80,
            'nbuvb-vitiligo': 75,
            'excimer-laser': 70,
            'cryotherapy': 90,
            'cryotherapy-wart': 85,
            'cryotherapy-sk': 90,
            'cantharidin': 80,
            # Systemic treatments
            'immunotherapy': 60,
            'immunotherapy-wart': 75,
            'oral-isotretinoin': 90,
            'oral-antifungal': 95,
            'oral-doxycycline': 80,
            'dupilumab': 85,
            'biologic-psoriasis': 90,
            # OTC treatments
            'prescription-cream': 75,
            'moisturizer-therapy': 60,
            'topical-antifungal': 85,
            'terbinafine-cream': 90,
        }
        base_improvement = treatment_improvements.get(treatment_type, 70)

        # Adjust for timeframe
        timeframe_adjustments = {
            '6months': 0,
            '1year': 5,
            '2years': 10
        }
        improvement_adjustment = timeframe_adjustments.get(timeframe, 5)
        projected_improvement = min(base_improvement + improvement_adjustment, 100)  # Cap at 100%

        # Generate timeline based on timeframe
        timelines = {
            '6months': [
                {'weeks': 4, 'improvement': 20, 'description': 'Early signs of improvement'},
                {'weeks': 8, 'improvement': 40, 'description': 'Noticeable reduction in symptoms'},
                {'weeks': 16, 'improvement': 65, 'description': 'Significant improvement visible'},
                {'weeks': 24, 'improvement': min(projected_improvement, 100), 'description': 'Expected 6-month outcome'}
            ],
            '1year': [
                {'weeks': 8, 'improvement': 15, 'description': 'Initial response to treatment'},
                {'weeks': 16, 'improvement': 35, 'description': 'Progressive improvement'},
                {'weeks': 32, 'improvement': 60, 'description': 'Substantial improvement'},
                {'weeks': 52, 'improvement': min(projected_improvement, 100), 'description': 'Expected 1-year outcome'}
            ],
            '2years': [
                {'weeks': 12, 'improvement': 12, 'description': 'Gradual initial improvement'},
                {'weeks': 26, 'improvement': 30, 'description': 'Steady progress'},
                {'weeks': 52, 'improvement': 55, 'description': 'Notable improvement at 1 year'},
                {'weeks': 104, 'improvement': min(projected_improvement, 100), 'description': 'Expected 2-year outcome'}
            ]
        }

        # Use enhanced prediction with all new features
        enhanced_result = treatment_predictor.predict_outcome_enhanced(
            image_bytes,
            treatment_type,
            timeframe,
            projected_improvement,
            diagnosis=diagnosis,
            compliance='typical',  # Default to typical compliance
            generate_timeline=True,  # Generate progressive images
            generate_confidence=True  # Generate best/worst/typical scenarios
        )

        # Save all images
        timestamp = int(time.time() * 1000)
        before_filename = f"treatment_before_{current_user.id}_{timestamp}.jpg"
        after_filename = f"treatment_after_{current_user.id}_{timestamp}.jpg"

        before_path = UPLOAD_DIR / before_filename
        after_path = UPLOAD_DIR / after_filename

        # Save before image
        with open(before_path, 'wb') as f:
            f.write(image_bytes)

        # Save final after image
        with open(after_path, 'wb') as f:
            f.write(enhanced_result['after_image'])

        # Save progressive timeline images
        timeline_files = []
        for idx, timeline_img in enumerate(enhanced_result.get('timeline_images', [])):
            timeline_filename = f"treatment_timeline_{current_user.id}_{timestamp}_{idx}.jpg"
            timeline_path = UPLOAD_DIR / timeline_filename
            with open(timeline_path, 'wb') as f:
                f.write(timeline_img['image_bytes'])
            timeline_files.append({
                'weeks': timeline_img['weeks'],
                'improvement': timeline_img['improvement'],
                'image_url': f'/uploads/{timeline_filename}',
                'description': f'Week {timeline_img["weeks"]}: {timeline_img["improvement"]}% improved'
            })

        # Enhanced recommendations based on severity
        severity = enhanced_result['severity']
        base_recommendations = [
            'Apply treatment exactly as prescribed by your dermatologist',
            'Use broad-spectrum SPF 30+ sunscreen daily',
            'Avoid harsh skincare products during treatment',
            'Document progress with regular photos',
            f'Schedule follow-up appointment in {"3" if timeframe == "6months" else "6" if timeframe == "1year" else "12"} months'
        ]

        # Add severity-specific recommendations
        if severity == 'severe':
            base_recommendations.insert(1, '⚠️ Severe case: Expect slower improvement, be patient')
            base_recommendations.insert(2, 'Consider combining treatments for better results')
        elif severity == 'mild':
            base_recommendations.insert(1, '✓ Mild case: Good prognosis with proper treatment')

        # Add compliance reminder
        base_recommendations.append('💊 Adherence is key: Missing treatments reduces effectiveness by up to 60%')

        return {
            'treatmentId': treatment_type,
            'projectedImprovement': min(enhanced_result['final_improvement'], 100),  # Cap at 100%
            'baseImprovement': min(projected_improvement, 100),  # Cap at 100%
            'beforeImage': f'/uploads/{before_filename}',
            'afterImage': f'/uploads/{after_filename}',
            'timeline': timeline_files,  # Progressive images with actual outcomes
            'timeframe': timeframe,
            'recommendations': base_recommendations,
            'severity': severity,  # Detected severity level
            'confidenceIntervals': enhanced_result.get('confidence_intervals', {}),  # Best/worst/typical cases
            'metadata': enhanced_result.get('metadata', {}),  # All factors applied
            'disclaimer': 'These predictions are AI-generated simulations based on clinical data and statistical models. Actual results vary significantly between patients. This tool is for educational purposes only and not a substitute for professional medical advice. Always consult a board-certified dermatologist before starting any treatment.'
        }

    except Exception as e:
        print(f"Error simulating treatment outcome: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to simulate treatment outcome: {str(e)}"
        )
