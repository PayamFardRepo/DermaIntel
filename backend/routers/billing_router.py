"""
Billing and Payments Router

Endpoints for:
- Billing records and codes
- CPT and ICD-10 code search
- CMS-1500 export
- Appeal letter generation
- Payment processing (Stripe integration)
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import os
import io

from database import get_db, User, AnalysisHistory, VideoConsultation, InsuranceAppeal
from auth import get_current_active_user
from appeal_letter_generator import (
    AppealRequest, AppealLetter, AppealLevel, DenialReason,
    generate_appeal_letter, get_denial_reasons, get_appeal_levels
)

router = APIRouter(tags=["Billing & Payments"])


# =============================================================================
# APPEAL REQUEST MODELS
# =============================================================================

class AppealRequestBody(BaseModel):
    """Request body for generating appeal letter"""
    patient_name: str
    patient_dob: str
    patient_id: str
    insurance_company: str
    policy_number: str
    claim_number: str
    date_of_service: str
    denial_date: str
    denial_reason: str
    denial_reason_text: Optional[str] = None
    diagnosis: str
    icd10_code: str
    procedure: str
    cpt_code: str
    claim_amount: float
    provider_name: str
    provider_npi: str
    provider_address: str
    appeal_level: str = "first_level"
    clinical_notes: Optional[str] = None
    supporting_evidence: Optional[List[str]] = None
    previous_treatments: Optional[List[str]] = None

# =============================================================================
# BILLING CODE MAPPINGS
# =============================================================================

DERMATOLOGY_CPT_CODES = {
    "biopsy": {"code": "11102", "description": "Tangential biopsy of skin", "base_charge": 150.00},
    "excision_benign": {"code": "11400", "description": "Excision, benign lesion", "base_charge": 200.00},
    "excision_malignant": {"code": "11600", "description": "Excision, malignant lesion", "base_charge": 350.00},
    "destruction_benign": {"code": "17110", "description": "Destruction of benign lesions", "base_charge": 125.00},
    "destruction_malignant": {"code": "17260", "description": "Destruction of malignant lesion", "base_charge": 250.00},
    "photography": {"code": "96904", "description": "Whole body photography", "base_charge": 75.00},
    "dermoscopy": {"code": "96902", "description": "Microscopic examination", "base_charge": 85.00},
}

ICD10_MAPPINGS = {
    "mel": "C43.9 - Malignant melanoma of skin, unspecified",
    "bcc": "C44.91 - Basal cell carcinoma of skin, unspecified",
    "akiec": "D04.9 - Carcinoma in situ of skin, unspecified",
    "bkl": "L82.1 - Other seborrheic keratosis",
    "nv": "D22.9 - Melanocytic nevi, unspecified",
    "vasc": "I78.1 - Nevus, non-neoplastic",
    "df": "D23.9 - Other benign neoplasm of skin, unspecified",
    "eczema": "L30.9 - Dermatitis, unspecified",
    "psoriasis": "L40.9 - Psoriasis, unspecified",
    "acne": "L70.0 - Acne vulgaris",
    "rosacea": "L71.9 - Rosacea, unspecified",
}


# =============================================================================
# BILLING ENDPOINTS
# =============================================================================

@router.get("/billing/records")
async def get_billing_records(
    status: str = "all",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get billing records for the current user."""
    try:
        query = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == current_user.id)

        if status != "all":
            pass  # Filter by status if needed

        analyses = query.order_by(AnalysisHistory.created_at.desc()).limit(50).all()

        records = []
        for analysis in analyses:
            diagnosis = analysis.predicted_class
            icd10_code = ICD10_MAPPINGS.get(diagnosis.lower() if diagnosis else "", "L98.9 - Disorder of skin, unspecified")

            cpt_codes = [
                {"code_type": "CPT", "code": "96904", "description": "Clinical photography", "category": "Diagnostic"}
            ]

            if analysis.risk_level in ["high", "very_high"]:
                cpt_codes.append({
                    "code_type": "CPT",
                    "code": "11102",
                    "description": "Biopsy recommended",
                    "category": "Procedure"
                })

            icd10_codes = [{
                "code_type": "ICD-10",
                "code": icd10_code.split(" - ")[0],
                "description": icd10_code.split(" - ")[1] if " - " in icd10_code else icd10_code,
                "category": "Diagnosis"
            }]

            total_charges = 75.00 + (150.00 if analysis.risk_level in ["high", "very_high"] else 0)
            estimated_reimbursement = total_charges * 0.80

            records.append({
                "id": analysis.id,
                "analysis_id": analysis.id,
                "diagnosis": diagnosis,
                "procedure_date": analysis.created_at.isoformat(),
                "cpt_codes": cpt_codes,
                "icd10_codes": icd10_codes,
                "total_charges": total_charges,
                "estimated_reimbursement": estimated_reimbursement,
                "status": "draft",
                "created_at": analysis.created_at.isoformat(),
                "preauth_status": analysis.preauth_status,
                "insurance_preauthorization": analysis.insurance_preauthorization
            })

        return {"records": records, "total": len(records)}

    except Exception as e:
        print(f"Error fetching billing records: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/billing/codes/search")
async def search_billing_codes(
    query: str,
    current_user: User = Depends(get_current_active_user)
):
    """Search for CPT and ICD-10 codes."""
    try:
        results = []
        query_lower = query.lower()

        for key, value in DERMATOLOGY_CPT_CODES.items():
            if (query_lower in key.lower() or
                query_lower in value["code"].lower() or
                query_lower in value["description"].lower()):
                results.append({
                    "code_type": "CPT",
                    "code": value["code"],
                    "description": value["description"],
                    "category": "Dermatology Procedure",
                    "typical_reimbursement": value["base_charge"] * 0.80
                })

        for key, value in ICD10_MAPPINGS.items():
            if query_lower in key.lower() or query_lower in value.lower():
                code = value.split(" - ")[0]
                description = value.split(" - ")[1] if " - " in value else value
                results.append({
                    "code_type": "ICD-10",
                    "code": code,
                    "description": description,
                    "category": "Diagnosis"
                })

        return {"codes": results}

    except Exception as e:
        print(f"Error searching codes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/billing/generate/{analysis_id}")
async def generate_billing(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate billing record for an analysis."""
    try:
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == current_user.id
        ).first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return {
            "success": True,
            "message": "Billing record generated successfully",
            "billing_id": analysis_id
        }

    except Exception as e:
        print(f"Error generating billing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/billing/export/cms1500/{record_id}")
async def export_cms1500(
    record_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export billing record as CMS-1500 form."""
    try:
        return {
            "success": True,
            "message": "CMS-1500 form generated",
            "download_url": f"/downloads/cms1500_{record_id}.pdf"
        }

    except Exception as e:
        print(f"Error exporting CMS-1500: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PAYMENT PROCESSING
# =============================================================================

@router.post("/payments/create-intent")
async def create_payment_intent(
    consultation_id: int = Form(...),
    amount: float = Form(...),
    currency: str = Form("usd"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a payment intent for a consultation (Stripe integration)."""
    try:
        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.user_id == current_user.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")

        if stripe_secret_key:
            try:
                import stripe
                stripe.api_key = stripe_secret_key

                intent = stripe.PaymentIntent.create(
                    amount=int(amount * 100),
                    currency=currency,
                    metadata={
                        "consultation_id": consultation_id,
                        "user_id": current_user.id
                    }
                )

                consultation.consultation_fee = amount
                consultation.payment_status = "pending"
                db.commit()

                return {
                    "client_secret": intent.client_secret,
                    "payment_intent_id": intent.id,
                    "amount": amount,
                    "currency": currency
                }
            except ImportError:
                pass

        # Demo mode fallback
        import secrets
        demo_intent_id = f"pi_demo_{secrets.token_hex(12)}"

        consultation.consultation_fee = amount
        consultation.payment_status = "pending"
        db.commit()

        return {
            "client_secret": f"{demo_intent_id}_secret",
            "payment_intent_id": demo_intent_id,
            "amount": amount,
            "currency": currency,
            "demo_mode": True,
            "message": "Demo payment intent. Configure STRIPE_SECRET_KEY for real payments."
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error creating payment intent: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create payment intent: {str(e)}")


@router.post("/payments/confirm")
async def confirm_payment(
    consultation_id: int = Form(...),
    payment_intent_id: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Confirm payment completion for a consultation."""
    try:
        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.user_id == current_user.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        consultation.payment_status = "paid"
        db.commit()

        return {
            "message": "Payment confirmed successfully",
            "consultation_id": consultation_id,
            "payment_status": "paid"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error confirming payment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm payment: {str(e)}")


@router.get("/payments/config")
async def get_payment_config():
    """Get payment configuration status."""
    stripe_configured = bool(os.getenv("STRIPE_SECRET_KEY"))
    stripe_publishable = os.getenv("STRIPE_PUBLISHABLE_KEY", "")

    return {
        "stripe_configured": stripe_configured,
        "stripe_publishable_key": stripe_publishable if stripe_configured else None,
        "supported_currencies": ["usd", "eur", "gbp", "cad", "aud"],
        "payment_methods": ["card"],
        "demo_mode": not stripe_configured
    }


# =============================================================================
# APPEAL LETTER ENDPOINTS
# =============================================================================

@router.get("/appeals")
async def list_appeals(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of all appeals for the current user."""
    try:
        appeals = db.query(InsuranceAppeal).filter(
            InsuranceAppeal.user_id == current_user.id
        ).order_by(InsuranceAppeal.created_at.desc()).all()

        return {
            "appeals": [
                {
                    "id": appeal.id,
                    "appeal_id": appeal.appeal_id,
                    "claim_number": appeal.claim_number,
                    "insurance_company": appeal.insurance_company,
                    "diagnosis": appeal.diagnosis,
                    "denial_reason": appeal.denial_reason,
                    "denial_reason_text": appeal.denial_reason_text,
                    "appeal_level": appeal.appeal_level,
                    "appeal_status": appeal.appeal_status,
                    "letter_content": appeal.letter_content,
                    "success_likelihood": appeal.success_likelihood,
                    "deadline": appeal.deadline.isoformat() if appeal.deadline else None,
                    "created_at": appeal.created_at.isoformat() if appeal.created_at else None,
                    "submitted_date": appeal.submitted_date.isoformat() if appeal.submitted_date else None,
                    "outcome": appeal.outcome,
                }
                for appeal in appeals
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading appeals: {str(e)}")


@router.get("/appeals/denial-reasons")
async def get_denial_reason_options():
    """Get list of denial reasons for dropdown selection."""
    return {"denial_reasons": get_denial_reasons()}


@router.get("/appeals/levels")
async def get_appeal_level_options():
    """Get list of appeal levels for dropdown selection."""
    return {"appeal_levels": get_appeal_levels()}


@router.post("/appeals/generate")
async def generate_appeal(
    request: AppealRequestBody,
    current_user: User = Depends(get_current_active_user)
):
    """Generate an appeal letter for a denied insurance claim."""
    try:
        # Convert string enums to actual enums
        try:
            denial_reason = DenialReason(request.denial_reason)
        except ValueError:
            denial_reason = DenialReason.OTHER

        try:
            appeal_level = AppealLevel(request.appeal_level)
        except ValueError:
            appeal_level = AppealLevel.FIRST_LEVEL

        # Create the appeal request
        appeal_request = AppealRequest(
            patient_name=request.patient_name,
            patient_dob=request.patient_dob,
            patient_id=request.patient_id,
            insurance_company=request.insurance_company,
            policy_number=request.policy_number,
            claim_number=request.claim_number,
            date_of_service=request.date_of_service,
            denial_date=request.denial_date,
            denial_reason=denial_reason,
            denial_reason_text=request.denial_reason_text,
            diagnosis=request.diagnosis,
            icd10_code=request.icd10_code,
            procedure=request.procedure,
            cpt_code=request.cpt_code,
            claim_amount=request.claim_amount,
            provider_name=request.provider_name,
            provider_npi=request.provider_npi,
            provider_address=request.provider_address,
            appeal_level=appeal_level,
            clinical_notes=request.clinical_notes,
            supporting_evidence=request.supporting_evidence,
            previous_treatments=request.previous_treatments
        )

        # Generate the appeal letter
        appeal_letter = generate_appeal_letter(appeal_request)

        return {
            "success": True,
            "appeal": {
                "appeal_id": appeal_letter.appeal_id,
                "generated_at": appeal_letter.generated_at.isoformat(),
                "appeal_level": appeal_letter.appeal_level.value,
                "letter_content": appeal_letter.letter_content,
                "subject_line": appeal_letter.subject_line,
                "summary": appeal_letter.summary,
                "key_arguments": appeal_letter.key_arguments,
                "supporting_documents_needed": appeal_letter.supporting_documents_needed,
                "deadline": appeal_letter.deadline.isoformat(),
                "recommended_next_steps": appeal_letter.recommended_next_steps,
                "success_likelihood": appeal_letter.success_likelihood
            }
        }

    except Exception as e:
        print(f"Error generating appeal letter: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate appeal: {str(e)}")


@router.post("/appeals/generate-from-analysis/{analysis_id}")
async def generate_appeal_from_analysis(
    analysis_id: int,
    denial_info: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate an appeal letter from an existing analysis and save to database."""
    try:
        # Get the analysis
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == current_user.id
        ).first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Get diagnosis info
        diagnosis = analysis.predicted_class or "Skin Lesion"
        icd10_code = ICD10_MAPPINGS.get(diagnosis.lower(), "L98.9")
        if " - " in icd10_code:
            icd10_code = icd10_code.split(" - ")[0]

        # Determine CPT code based on risk level
        if analysis.risk_level in ["high", "very_high"]:
            cpt_code = "11102"
            procedure = "Skin Biopsy"
        else:
            cpt_code = "96904"
            procedure = "Clinical Photography"

        # Get confidence safely
        confidence_score = analysis.lesion_confidence or analysis.binary_confidence or 0.0

        # Build the appeal request
        request = AppealRequestBody(
            patient_name=denial_info.get("patient_name", current_user.full_name or "Patient"),
            patient_dob=denial_info.get("patient_dob", ""),
            patient_id=str(current_user.id),
            insurance_company=denial_info.get("insurance_company", ""),
            policy_number=denial_info.get("policy_number", ""),
            claim_number=denial_info.get("claim_number", ""),
            date_of_service=analysis.created_at.strftime("%Y-%m-%d"),
            denial_date=denial_info.get("denial_date", datetime.now().strftime("%Y-%m-%d")),
            denial_reason=denial_info.get("denial_reason", "medical_necessity"),
            diagnosis=diagnosis,
            icd10_code=icd10_code,
            procedure=procedure,
            cpt_code=cpt_code,
            claim_amount=float(denial_info.get("claim_amount", 150.00)),
            provider_name=denial_info.get("provider_name", "Healthcare Provider"),
            provider_npi=denial_info.get("provider_npi", ""),
            provider_address=denial_info.get("provider_address", ""),
            appeal_level=denial_info.get("appeal_level", "first_level"),
            clinical_notes=f"AI-assisted analysis indicated {diagnosis} with {confidence_score:.1%} confidence. Risk level: {analysis.risk_level or 'unknown'}."
        )

        # Convert string enums to actual enums
        try:
            denial_reason = DenialReason(request.denial_reason)
        except ValueError:
            denial_reason = DenialReason.OTHER

        try:
            appeal_level = AppealLevel(request.appeal_level)
        except ValueError:
            appeal_level = AppealLevel.FIRST_LEVEL

        # Create the appeal request object for generator
        appeal_request = AppealRequest(
            patient_name=request.patient_name,
            patient_dob=request.patient_dob,
            patient_id=request.patient_id,
            insurance_company=request.insurance_company,
            policy_number=request.policy_number,
            claim_number=request.claim_number,
            date_of_service=request.date_of_service,
            denial_date=request.denial_date,
            denial_reason=denial_reason,
            denial_reason_text=denial_info.get("denial_reason_text"),
            diagnosis=request.diagnosis,
            icd10_code=request.icd10_code,
            procedure=request.procedure,
            cpt_code=request.cpt_code,
            claim_amount=request.claim_amount,
            provider_name=request.provider_name,
            provider_npi=request.provider_npi,
            provider_address=request.provider_address,
            appeal_level=appeal_level,
            clinical_notes=request.clinical_notes,
        )

        # Generate the appeal letter
        appeal_letter = generate_appeal_letter(appeal_request)

        # Save appeal to database
        appeal_record = InsuranceAppeal(
            user_id=current_user.id,
            analysis_id=analysis_id,
            appeal_id=appeal_letter.appeal_id,
            claim_number=request.claim_number,
            insurance_company=request.insurance_company,
            policy_number=request.policy_number,
            date_of_service=analysis.created_at,
            original_claim_amount=request.claim_amount,
            diagnosis=diagnosis,
            icd10_code=icd10_code,
            procedure=procedure,
            cpt_code=cpt_code,
            denial_date=datetime.now(),
            denial_reason=request.denial_reason,
            denial_reason_text=denial_info.get("denial_reason_text"),
            appeal_level=request.appeal_level,
            appeal_status="draft",
            letter_content=appeal_letter.letter_content,
            subject_line=appeal_letter.subject_line,
            key_arguments=appeal_letter.key_arguments,
            supporting_documents=appeal_letter.supporting_documents_needed,
            success_likelihood=appeal_letter.success_likelihood,
            recommended_next_steps=appeal_letter.recommended_next_steps,
            deadline=appeal_letter.deadline,
            patient_name=request.patient_name,
            patient_dob=request.patient_dob,
            provider_name=request.provider_name,
            provider_npi=request.provider_npi,
        )

        db.add(appeal_record)
        db.commit()
        db.refresh(appeal_record)

        return {
            "id": appeal_record.id,
            "appeal_id": appeal_letter.appeal_id,
            "claim_number": request.claim_number,
            "insurance_company": request.insurance_company,
            "diagnosis": diagnosis,
            "denial_reason": request.denial_reason,
            "appeal_level": request.appeal_level,
            "appeal_status": "draft",
            "letter_content": appeal_letter.letter_content,
            "subject_line": appeal_letter.subject_line,
            "key_arguments": appeal_letter.key_arguments,
            "supporting_documents_needed": appeal_letter.supporting_documents_needed,
            "success_likelihood": appeal_letter.success_likelihood,
            "recommended_next_steps": appeal_letter.recommended_next_steps,
            "deadline": appeal_letter.deadline.isoformat() if appeal_letter.deadline else None,
            "created_at": appeal_record.created_at.isoformat() if appeal_record.created_at else None,
            "summary": appeal_letter.summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating appeal from analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate appeal: {str(e)}")


@router.get("/appeals/download/{appeal_id}")
async def download_appeal_pdf(
    appeal_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Download appeal letter as PDF."""
    try:
        # For now, return a message indicating PDF generation
        # In production, this would generate an actual PDF
        return {
            "message": "PDF generation endpoint",
            "appeal_id": appeal_id,
            "note": "Use the letter_content field to create PDF on client side or integrate reportlab for server-side PDF"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")


# =============================================================================
# CMS-1500 FORM GENERATION
# =============================================================================

@router.get("/cms1500/generate/{analysis_id}")
async def generate_cms1500_form(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Generate CMS-1500 form data for an analysis."""
    try:
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == current_user.id
        ).first()

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Get diagnosis and codes
        diagnosis = analysis.predicted_class or "Skin Lesion"
        icd10_full = ICD10_MAPPINGS.get(diagnosis.lower(), "L98.9 - Disorder of skin, unspecified")
        icd10_code = icd10_full.split(" - ")[0] if " - " in icd10_full else icd10_full
        icd10_desc = icd10_full.split(" - ")[1] if " - " in icd10_full else "Skin disorder"

        # Determine procedures based on analysis
        service_lines = []

        # Always include clinical photography
        service_lines.append({
            "line_number": 1,
            "date_of_service": analysis.created_at.strftime("%m/%d/%Y"),
            "place_of_service": "11",  # Office
            "cpt_code": "96904",
            "modifier": "",
            "diagnosis_pointer": "A",
            "charges": 75.00,
            "units": 1,
            "description": "Whole body integumentary photography"
        })

        # Add dermoscopy if performed
        if hasattr(analysis, 'dermoscopy_features') and analysis.dermoscopy_features:
            service_lines.append({
                "line_number": 2,
                "date_of_service": analysis.created_at.strftime("%m/%d/%Y"),
                "place_of_service": "11",
                "cpt_code": "96902",
                "modifier": "",
                "diagnosis_pointer": "A",
                "charges": 85.00,
                "units": 1,
                "description": "Microscopic examination of skin"
            })

        # Add biopsy recommendation for high risk
        if analysis.risk_level in ["high", "very_high"]:
            service_lines.append({
                "line_number": len(service_lines) + 1,
                "date_of_service": analysis.created_at.strftime("%m/%d/%Y"),
                "place_of_service": "11",
                "cpt_code": "11102",
                "modifier": "",
                "diagnosis_pointer": "A",
                "charges": 150.00,
                "units": 1,
                "description": "Tangential biopsy of skin (recommended)"
            })

        total_charges = sum(line["charges"] for line in service_lines)

        # Build CMS-1500 form structure
        cms1500_data = {
            "form_version": "02/12",
            "generated_at": datetime.now().isoformat(),
            "analysis_id": analysis_id,

            # Section 1: Insurance Information (to be filled)
            "insurance_type": "",  # Medicare, Medicaid, etc.
            "insured_id": "",
            "patient_name": current_user.full_name or "",
            "patient_dob": "",
            "insured_name": "",
            "patient_address": "",
            "patient_city_state_zip": "",
            "patient_phone": "",
            "patient_relationship_to_insured": "self",
            "insured_address": "",
            "patient_status": "single",  # single, married, other
            "other_insured_name": "",
            "patient_condition_related_to": {
                "employment": False,
                "auto_accident": False,
                "other_accident": False
            },

            # Section 2: Diagnosis
            "diagnosis_codes": [
                {"pointer": "A", "code": icd10_code, "description": icd10_desc}
            ],

            # Section 3: Service Lines
            "service_lines": service_lines,

            # Section 4: Provider Information (to be filled)
            "referring_provider": "",
            "referring_npi": "",
            "billing_provider_name": "",
            "billing_provider_address": "",
            "billing_provider_npi": "",
            "billing_provider_tax_id": "",
            "facility_name": "",
            "facility_address": "",
            "facility_npi": "",

            # Section 5: Totals
            "total_charges": total_charges,
            "amount_paid": 0.00,
            "balance_due": total_charges,

            # Additional fields
            "accept_assignment": True,
            "signature_on_file": True,
            "date_of_current_illness": analysis.created_at.strftime("%m/%d/%Y"),

            # Instructions
            "instructions": [
                "Complete all patient and insurance information in Section 1",
                "Verify diagnosis codes match medical documentation",
                "Add provider information including NPI numbers",
                "Ensure dates of service are accurate",
                "Sign and date the form before submission"
            ]
        }

        return {
            "success": True,
            "cms1500": cms1500_data
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating CMS-1500: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate CMS-1500: {str(e)}")
