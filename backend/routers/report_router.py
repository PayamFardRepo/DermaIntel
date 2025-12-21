"""
Report Router - API endpoints for publication-ready report generation.

Provides endpoints for:
- Generating case reports from analysis data
- Previewing report data before generation
- Downloading generated reports
"""

import uuid
import os
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import io

from database import get_db, AnalysisHistory, User, UserProfile
from routers.auth_router import get_current_active_user
from publication_report_generator import PublicationReportGenerator
from deidentification_service import anonymize_for_publication, DeidentificationService

router = APIRouter(tags=["reports"])

# In-memory store for generated reports (in production, use Redis or database)
generated_reports: Dict[str, Dict[str, Any]] = {}


class ReportRequest(BaseModel):
    """Request body for report generation."""
    analysis_id: int
    include_images: bool = True
    include_dermoscopy: bool = True
    include_heatmap: bool = True
    include_biopsy: bool = True


class ReportPreviewResponse(BaseModel):
    """Response for report preview."""
    case_id: str
    analysis_date: str
    demographics: Dict[str, Any]
    clinical_presentation: Dict[str, Any]
    diagnosis: Dict[str, Any]
    has_images: bool
    has_dermoscopy: bool
    has_biopsy: bool
    has_abcde: bool


@router.post("/reports/case-report")
async def generate_case_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Generate a publication-ready case report.

    Returns a report ID that can be used to check status and download the PDF.
    """
    # Verify analysis exists and belongs to user
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == request.analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Generate report ID
    report_id = str(uuid.uuid4())

    # Store initial status
    generated_reports[report_id] = {
        "status": "generating",
        "created_at": datetime.utcnow().isoformat(),
        "analysis_id": request.analysis_id,
        "user_id": current_user.id,
        "pdf_bytes": None,
        "case_id": None,
        "error": None,
    }

    # Generate report in background
    background_tasks.add_task(
        _generate_report_task,
        report_id=report_id,
        analysis_id=request.analysis_id,
        user_id=current_user.id,
        options={
            "include_images": request.include_images,
            "include_dermoscopy": request.include_dermoscopy,
            "include_heatmap": request.include_heatmap,
            "include_biopsy": request.include_biopsy,
        }
    )

    return {
        "report_id": report_id,
        "status": "generating",
        "message": "Report generation started. Use GET /reports/{report_id}/status to check progress."
    }


def _generate_report_task(
    report_id: str,
    analysis_id: int,
    user_id: int,
    options: Dict[str, Any]
):
    """Background task to generate the report."""
    from database import SessionLocal

    db = SessionLocal()
    try:
        # Get analysis data
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id
        ).first()

        if not analysis:
            generated_reports[report_id]["status"] = "failed"
            generated_reports[report_id]["error"] = "Analysis not found"
            return

        # Get user and profile data
        user = db.query(User).filter(User.id == user_id).first()
        profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()

        # Convert to dictionaries
        analysis_dict = {
            "id": analysis.id,
            "user_id": analysis.user_id,
            "image_url": analysis.image_url,
            "image_filename": analysis.image_filename,
            "analysis_type": analysis.analysis_type,
            "predicted_class": analysis.predicted_class,
            "lesion_confidence": analysis.lesion_confidence,
            "binary_confidence": analysis.binary_confidence,
            "lesion_probabilities": analysis.lesion_probabilities,
            "risk_level": analysis.risk_level,
            "risk_recommendation": analysis.risk_recommendation,
            "differential_diagnoses": analysis.differential_diagnoses,
            "treatment_recommendations": analysis.treatment_recommendations,
            "body_location": analysis.body_location,
            "body_sublocation": analysis.body_sublocation,
            "body_side": analysis.body_side,
            "symptom_duration": analysis.symptom_duration,
            "symptom_changes": analysis.symptom_changes,
            "symptom_itching": analysis.symptom_itching,
            "symptom_pain": analysis.symptom_pain,
            "symptom_bleeding": analysis.symptom_bleeding,
            "red_flag_data": analysis.red_flag_data,
            "dermoscopy_data": analysis.dermoscopy_data,
            "explainability_heatmap": analysis.explainability_heatmap,
            "biopsy_performed": analysis.biopsy_performed,
            "biopsy_result": analysis.biopsy_result,
            "biopsy_date": analysis.biopsy_date,
            "pathologist_name": analysis.pathologist_name,
            "prediction_correct": analysis.prediction_correct,
            "accuracy_category": analysis.accuracy_category,
            "model_version": analysis.model_version,
            "created_at": analysis.created_at,
            "dermatologist_name": analysis.dermatologist_name,
        }

        user_dict = {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "age": user.age,
            "gender": user.gender,
        } if user else {}

        profile_dict = {
            "date_of_birth": profile.date_of_birth,
            "gender": profile.gender,
            "skin_type": profile.skin_type,
            "family_history_skin_cancer": profile.family_history_skin_cancer if hasattr(profile, 'family_history_skin_cancer') else False,
            "previous_skin_cancers": profile.previous_skin_cancers if hasattr(profile, 'previous_skin_cancers') else False,
            "immunosuppression": profile.immunosuppression if hasattr(profile, 'immunosuppression') else False,
        } if profile else {}

        # Generate report
        generator = PublicationReportGenerator(output_dir="reports")
        pdf_bytes = generator.generate_report(
            analysis_data=analysis_dict,
            user_data=user_dict,
            profile_data=profile_dict,
            options=options
        )

        # Generate case ID
        case_id = DeidentificationService.generate_case_id(user_id, analysis_id)

        # Update status
        generated_reports[report_id]["status"] = "ready"
        generated_reports[report_id]["pdf_bytes"] = pdf_bytes
        generated_reports[report_id]["case_id"] = case_id
        generated_reports[report_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        generated_reports[report_id]["status"] = "failed"
        generated_reports[report_id]["error"] = str(e)
        import traceback
        print(f"Report generation error: {traceback.format_exc()}")
    finally:
        db.close()


@router.get("/reports/{report_id}/status")
async def get_report_status(
    report_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Check the status of a report generation request.
    """
    if report_id not in generated_reports:
        raise HTTPException(status_code=404, detail="Report not found")

    report = generated_reports[report_id]

    # Verify ownership
    if report["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "report_id": report_id,
        "status": report["status"],
        "case_id": report.get("case_id"),
        "created_at": report["created_at"],
        "completed_at": report.get("completed_at"),
        "error": report.get("error"),
    }


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Download a generated report as PDF.
    """
    if report_id not in generated_reports:
        raise HTTPException(status_code=404, detail="Report not found")

    report = generated_reports[report_id]

    # Verify ownership
    if report["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")

    if report["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Report not ready. Status: {report['status']}"
        )

    if not report.get("pdf_bytes"):
        raise HTTPException(status_code=500, detail="Report data not available")

    # Return PDF as streaming response
    case_id = report.get("case_id", "report")
    filename = f"case_report_{case_id}.pdf"

    return StreamingResponse(
        io.BytesIO(report["pdf_bytes"]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


@router.get("/reports/preview/{analysis_id}")
async def preview_report_data(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
) -> ReportPreviewResponse:
    """
    Preview the data that will be included in a report.

    Useful for showing users what will be in the report before generating.
    """
    # Verify analysis exists and belongs to user
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Get profile
    profile = db.query(UserProfile).filter(
        UserProfile.user_id == current_user.id
    ).first()

    # Generate anonymized preview
    user_dict = {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "age": current_user.age,
        "gender": current_user.gender,
    }

    profile_dict = {
        "date_of_birth": profile.date_of_birth if profile else None,
        "gender": profile.gender if profile else None,
        "skin_type": profile.skin_type if profile else None,
    } if profile else {}

    analysis_dict = {
        "user_id": analysis.user_id,
        "body_location": analysis.body_location,
        "body_sublocation": analysis.body_sublocation,
        "body_side": analysis.body_side,
        "symptom_duration": analysis.symptom_duration,
        "symptom_changes": analysis.symptom_changes,
        "symptom_itching": analysis.symptom_itching,
        "symptom_pain": analysis.symptom_pain,
        "symptom_bleeding": analysis.symptom_bleeding,
        "created_at": analysis.created_at,
    }

    anonymized = anonymize_for_publication(
        current_user.id,
        analysis_id,
        user_dict,
        profile_dict,
        analysis_dict
    )

    return ReportPreviewResponse(
        case_id=anonymized["case_id"],
        analysis_date=anonymized["analysis_date"],
        demographics=anonymized["demographics"],
        clinical_presentation=anonymized["clinical_presentation"],
        diagnosis={
            "predicted_class": analysis.predicted_class,
            "confidence": analysis.lesion_confidence or analysis.binary_confidence or 0,
            "risk_level": analysis.risk_level,
        },
        has_images=bool(analysis.image_url),
        has_dermoscopy=bool(analysis.dermoscopy_data),
        has_biopsy=bool(analysis.biopsy_performed),
        has_abcde=bool(analysis.red_flag_data),
    )


@router.get("/reports/analyses")
async def list_reportable_analyses(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List analyses that can be used for report generation.
    """
    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id,
        AnalysisHistory.predicted_class.isnot(None)
    ).order_by(AnalysisHistory.created_at.desc()).offset(skip).limit(limit).all()

    total = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id,
        AnalysisHistory.predicted_class.isnot(None)
    ).count()

    return {
        "analyses": [
            {
                "id": a.id,
                "image_url": a.image_url,
                "predicted_class": a.predicted_class,
                "confidence": a.lesion_confidence or a.binary_confidence,
                "risk_level": a.risk_level,
                "body_location": a.body_location,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "has_dermoscopy": bool(a.dermoscopy_data),
                "has_biopsy": bool(a.biopsy_performed),
                "has_abcde": bool(a.red_flag_data),
            }
            for a in analyses
        ],
        "total": total,
        "skip": skip,
        "limit": limit,
    }
