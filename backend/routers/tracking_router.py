"""
Lesion Tracking and Sun Exposure Router

Endpoints for:
- Lesion group management
- Lesion comparison over time
- Progression timeline
- Sun exposure logging and analysis
"""

from fastapi import APIRouter, Depends, HTTPException, Form, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from dateutil.relativedelta import relativedelta

from database import get_db, User, AnalysisHistory, LesionGroup, LesionComparison, SunExposure
from auth import get_current_active_user

router = APIRouter(tags=["Lesion Tracking & Sun Exposure"])


# =============================================================================
# LESION GROUPS
# =============================================================================

@router.post("/lesion_groups/")
def create_lesion_group(
    lesion_name: str = Form(...),
    lesion_description: Optional[str] = Form(None),
    body_location: Optional[str] = Form(None),
    body_sublocation: Optional[str] = Form(None),
    body_side: Optional[str] = Form(None),
    first_noticed_date: Optional[str] = Form(None),
    monitoring_frequency: str = Form("monthly"),
    analysis_id: Optional[int] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new lesion group for tracking a specific lesion over time."""
    lesion_group = LesionGroup(
        user_id=current_user.id,
        lesion_name=lesion_name,
        lesion_description=lesion_description,
        body_location=body_location,
        body_sublocation=body_sublocation,
        body_side=body_side,
        monitoring_frequency=monitoring_frequency,
        total_analyses=0,
        is_active=True
    )

    if first_noticed_date:
        try:
            lesion_group.first_noticed_date = datetime.fromisoformat(first_noticed_date.replace('Z', '+00:00'))
        except:
            pass

    frequency_map = {
        "weekly": relativedelta(weeks=1),
        "monthly": relativedelta(months=1),
        "quarterly": relativedelta(months=3),
        "biannual": relativedelta(months=6),
        "annual": relativedelta(years=1)
    }
    lesion_group.next_check_date = datetime.utcnow() + frequency_map.get(monitoring_frequency, relativedelta(months=1))

    db.add(lesion_group)
    db.commit()
    db.refresh(lesion_group)

    if analysis_id:
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == analysis_id,
            AnalysisHistory.user_id == current_user.id
        ).first()

        if analysis:
            analysis.lesion_group_id = lesion_group.id
            lesion_group.total_analyses = 1
            lesion_group.last_analyzed_at = analysis.created_at
            lesion_group.current_risk_level = analysis.risk_level

            if not body_location and analysis.body_location:
                lesion_group.body_location = analysis.body_location
                lesion_group.body_sublocation = analysis.body_sublocation
                lesion_group.body_side = analysis.body_side

            db.commit()
            db.refresh(lesion_group)

    return {
        "id": lesion_group.id,
        "lesion_name": lesion_group.lesion_name,
        "total_analyses": lesion_group.total_analyses,
        "next_check_date": lesion_group.next_check_date,
        "message": "Lesion group created successfully"
    }


@router.get("/lesion_groups/")
def get_lesion_groups(
    include_archived: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all lesion groups for the current user."""
    query = db.query(LesionGroup).filter(LesionGroup.user_id == current_user.id)

    if not include_archived:
        query = query.filter(LesionGroup.archived == False)

    lesion_groups = query.order_by(LesionGroup.last_analyzed_at.desc()).all()

    return [
        {
            "id": group.id,
            "lesion_name": group.lesion_name,
            "lesion_description": group.lesion_description,
            "body_location": group.body_location,
            "body_sublocation": group.body_sublocation,
            "body_side": group.body_side,
            "monitoring_frequency": group.monitoring_frequency,
            "next_check_date": group.next_check_date,
            "current_risk_level": group.current_risk_level,
            "requires_attention": group.requires_attention,
            "attention_reason": group.attention_reason,
            "total_analyses": group.total_analyses,
            "change_detected": group.change_detected,
            "growth_rate": group.growth_rate,
            "is_active": group.is_active,
            "archived": group.archived,
            "first_noticed_date": group.first_noticed_date,
            "last_analyzed_at": group.last_analyzed_at,
            "created_at": group.created_at
        }
        for group in lesion_groups
    ]


@router.get("/lesion_groups/{group_id}")
def get_lesion_group(
    group_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific lesion group."""
    lesion_group = db.query(LesionGroup).filter(
        LesionGroup.id == group_id,
        LesionGroup.user_id == current_user.id
    ).first()

    if not lesion_group:
        raise HTTPException(status_code=404, detail="Lesion group not found")

    analyses = db.query(AnalysisHistory).filter(
        AnalysisHistory.lesion_group_id == group_id
    ).order_by(AnalysisHistory.created_at).all()

    comparisons = db.query(LesionComparison).filter(
        LesionComparison.lesion_group_id == group_id
    ).order_by(LesionComparison.created_at.desc()).all()

    return {
        "id": lesion_group.id,
        "lesion_name": lesion_group.lesion_name,
        "lesion_description": lesion_group.lesion_description,
        "body_location": lesion_group.body_location,
        "body_sublocation": lesion_group.body_sublocation,
        "body_side": lesion_group.body_side,
        "first_noticed_date": lesion_group.first_noticed_date,
        "monitoring_frequency": lesion_group.monitoring_frequency,
        "next_check_date": lesion_group.next_check_date,
        "current_risk_level": lesion_group.current_risk_level,
        "requires_attention": lesion_group.requires_attention,
        "attention_reason": lesion_group.attention_reason,
        "total_analyses": lesion_group.total_analyses,
        "change_detected": lesion_group.change_detected,
        "change_summary": lesion_group.change_summary,
        "growth_rate": lesion_group.growth_rate,
        "is_active": lesion_group.is_active,
        "archived": lesion_group.archived,
        "archive_reason": lesion_group.archive_reason,
        "last_analyzed_at": lesion_group.last_analyzed_at,
        "created_at": lesion_group.created_at,
        "analyses": [
            {
                "id": a.id,
                "image_url": a.image_url,
                "predicted_class": a.predicted_class,
                "lesion_confidence": a.lesion_confidence,
                "risk_level": a.risk_level,
                "created_at": a.created_at
            } for a in analyses
        ],
        "comparisons": [
            {
                "id": c.id,
                "comparison_date": c.created_at,
                "overall_change_assessment": c.overall_change_assessment
            } for c in comparisons
        ]
    }


# =============================================================================
# SUN EXPOSURE TRACKING
# =============================================================================

@router.post("/sun-exposure")
def create_sun_exposure(
    exposure_date: str = Form(...),
    duration_minutes: int = Form(...),
    time_of_day: str = Form(...),
    location: str = Form(...),
    activity: str = Form(...),
    uv_index: float = Form(None),
    uv_index_source: str = Form("manual"),
    weather_conditions: str = Form(None),
    altitude_meters: float = Form(None),
    sun_protection_used: bool = Form(False),
    sunscreen_applied: bool = Form(False),
    spf_level: int = Form(None),
    sunscreen_reapplied: bool = Form(False),
    protective_clothing: bool = Form(False),
    hat_worn: bool = Form(False),
    sunglasses_worn: bool = Form(False),
    shade_sought: bool = Form(False),
    skin_reaction: str = Form("none"),
    reaction_severity: int = Form(0),
    pain_level: int = Form(0),
    peeling_occurred: bool = Form(False),
    intentional_tanning: bool = Form(False),
    indoor_tanning: bool = Form(False),
    notes: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Log a sun exposure event with detailed protection and reaction data."""
    try:
        exposure_datetime = datetime.fromisoformat(exposure_date.replace('Z', '+00:00'))
    except:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO 8601 format.")

    # Calculate protection factor
    protection_factor = 1.0
    if sunscreen_applied and spf_level:
        protection_factor *= (1 - (spf_level / (spf_level + 10)))
    if protective_clothing:
        protection_factor *= 0.5
    if hat_worn:
        protection_factor *= 0.7
    if shade_sought:
        protection_factor *= 0.5

    # Calculate UV dose
    uv_dose = 0
    if uv_index:
        uv_dose = uv_index * (duration_minutes / 60) * (1 - protection_factor)

    # Calculate risk score (0-100)
    risk_score = 0
    if uv_index:
        risk_score = min(uv_index * 5, 50)
        if duration_minutes > 120:
            risk_score += 15
        elif duration_minutes > 60:
            risk_score += 10
        elif duration_minutes > 30:
            risk_score += 5

        if time_of_day == "midday":
            risk_score += 15

        if not sun_protection_used:
            risk_score += 20

        risk_score *= (1 - protection_factor * 0.7)

        if skin_reaction in ["moderate_burn", "severe_burn"]:
            risk_score += 20
        elif skin_reaction == "mild_redness":
            risk_score += 10

        risk_score = min(risk_score, 100)

    sun_exposure = SunExposure(
        user_id=current_user.id,
        exposure_date=exposure_datetime,
        duration_minutes=duration_minutes,
        time_of_day=time_of_day,
        location=location,
        activity=activity,
        uv_index=uv_index,
        uv_index_source=uv_index_source,
        weather_conditions=weather_conditions,
        altitude_meters=altitude_meters,
        sun_protection_used=sun_protection_used,
        sunscreen_applied=sunscreen_applied,
        spf_level=spf_level,
        sunscreen_reapplied=sunscreen_reapplied,
        protective_clothing=protective_clothing,
        hat_worn=hat_worn,
        sunglasses_worn=sunglasses_worn,
        shade_sought=shade_sought,
        exposed_body_areas=[],
        skin_reaction=skin_reaction,
        reaction_severity=reaction_severity,
        pain_level=pain_level,
        peeling_occurred=peeling_occurred,
        intentional_tanning=intentional_tanning,
        indoor_tanning=indoor_tanning,
        notes=notes,
        calculated_uv_dose=uv_dose,
        risk_score=risk_score
    )

    db.add(sun_exposure)
    db.commit()
    db.refresh(sun_exposure)

    return {
        "message": "Sun exposure logged successfully",
        "id": sun_exposure.id,
        "uv_dose": uv_dose,
        "risk_score": risk_score
    }


@router.get("/sun-exposure")
def get_sun_exposures(
    skip: int = 0,
    limit: int = 50,
    start_date: str = None,
    end_date: str = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all sun exposure entries for the current user."""
    query = db.query(SunExposure).filter(SunExposure.user_id == current_user.id)

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            query = query.filter(SunExposure.exposure_date >= start_dt)
        except:
            raise HTTPException(status_code=400, detail="Invalid start_date format")

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            query = query.filter(SunExposure.exposure_date <= end_dt)
        except:
            raise HTTPException(status_code=400, detail="Invalid end_date format")

    total = query.count()
    exposures = query.order_by(SunExposure.exposure_date.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "exposures": [
            {
                "id": exp.id,
                "exposure_date": exp.exposure_date.isoformat(),
                "duration_minutes": exp.duration_minutes,
                "time_of_day": exp.time_of_day,
                "location": exp.location,
                "activity": exp.activity,
                "uv_index": exp.uv_index,
                "sun_protection_used": exp.sun_protection_used,
                "sunscreen_applied": exp.sunscreen_applied,
                "spf_level": exp.spf_level,
                "skin_reaction": exp.skin_reaction,
                "calculated_uv_dose": exp.calculated_uv_dose,
                "risk_score": exp.risk_score
            }
            for exp in exposures
        ]
    }


@router.get("/sun-exposure/{exposure_id}")
def get_sun_exposure(
    exposure_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific sun exposure entry."""
    exposure = db.query(SunExposure).filter(
        SunExposure.id == exposure_id,
        SunExposure.user_id == current_user.id
    ).first()

    if not exposure:
        raise HTTPException(status_code=404, detail="Sun exposure entry not found")

    return {
        "id": exposure.id,
        "exposure_date": exposure.exposure_date.isoformat(),
        "duration_minutes": exposure.duration_minutes,
        "time_of_day": exposure.time_of_day,
        "location": exposure.location,
        "activity": exposure.activity,
        "uv_index": exposure.uv_index,
        "uv_index_source": exposure.uv_index_source,
        "weather_conditions": exposure.weather_conditions,
        "altitude_meters": exposure.altitude_meters,
        "sun_protection_used": exposure.sun_protection_used,
        "sunscreen_applied": exposure.sunscreen_applied,
        "spf_level": exposure.spf_level,
        "sunscreen_reapplied": exposure.sunscreen_reapplied,
        "protective_clothing": exposure.protective_clothing,
        "hat_worn": exposure.hat_worn,
        "sunglasses_worn": exposure.sunglasses_worn,
        "shade_sought": exposure.shade_sought,
        "exposed_body_areas": exposure.exposed_body_areas,
        "skin_reaction": exposure.skin_reaction,
        "reaction_severity": exposure.reaction_severity,
        "pain_level": exposure.pain_level,
        "peeling_occurred": exposure.peeling_occurred,
        "intentional_tanning": exposure.intentional_tanning,
        "indoor_tanning": exposure.indoor_tanning,
        "notes": exposure.notes,
        "calculated_uv_dose": exposure.calculated_uv_dose,
        "risk_score": exposure.risk_score,
        "created_at": exposure.created_at.isoformat() if exposure.created_at else None
    }


@router.delete("/sun-exposure/{exposure_id}")
def delete_sun_exposure(
    exposure_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a sun exposure entry."""
    exposure = db.query(SunExposure).filter(
        SunExposure.id == exposure_id,
        SunExposure.user_id == current_user.id
    ).first()

    if not exposure:
        raise HTTPException(status_code=404, detail="Sun exposure entry not found")

    db.delete(exposure)
    db.commit()

    return {"message": "Sun exposure entry deleted successfully"}


# =============================================================================
# LESION COMPARISON ENDPOINTS
# =============================================================================

@router.post("/lesion_groups/{group_id}/compare")
async def compare_lesions_in_group(
    group_id: int,
    analysis_id_1: int = Form(...),
    analysis_id_2: int = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Compare two analyses within a lesion group to track changes over time.

    Returns detailed comparison of:
    - Classification changes
    - Confidence changes
    - Risk level changes
    - ABCDE score changes
    """
    # Get the lesion group
    group = db.query(LesionGroup).filter(
        LesionGroup.id == group_id,
        LesionGroup.user_id == current_user.id
    ).first()

    if not group:
        raise HTTPException(status_code=404, detail="Lesion group not found")

    # Get both analyses
    analysis_1 = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id_1,
        AnalysisHistory.lesion_group_id == group_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    analysis_2 = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == analysis_id_2,
        AnalysisHistory.lesion_group_id == group_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis_1 or not analysis_2:
        raise HTTPException(status_code=404, detail="One or both analyses not found in this lesion group")

    # Ensure analysis_1 is the earlier one
    if analysis_1.created_at > analysis_2.created_at:
        analysis_1, analysis_2 = analysis_2, analysis_1

    # Get ABCDE data
    abcde_1 = analysis_1.red_flag_data or {}
    abcde_2 = analysis_2.red_flag_data or {}

    # Calculate time difference
    time_diff = analysis_2.created_at - analysis_1.created_at
    days_between = time_diff.days

    # Compare risk levels
    risk_order = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
    risk_1 = risk_order.get(analysis_1.risk_level, 0)
    risk_2 = risk_order.get(analysis_2.risk_level, 0)

    if risk_2 > risk_1:
        risk_change = "increased"
        risk_concern = "high"
    elif risk_2 < risk_1:
        risk_change = "decreased"
        risk_concern = "low"
    else:
        risk_change = "unchanged"
        risk_concern = "moderate"

    # Compare ABCDE scores
    def get_score(data, key):
        return data.get(key, {}).get("overall_score", 0) or 0

    abcde_comparison = {
        "asymmetry": {
            "before": get_score(abcde_1, "asymmetry"),
            "after": get_score(abcde_2, "asymmetry"),
            "change": get_score(abcde_2, "asymmetry") - get_score(abcde_1, "asymmetry")
        },
        "border": {
            "before": get_score(abcde_1, "border"),
            "after": get_score(abcde_2, "border"),
            "change": get_score(abcde_2, "border") - get_score(abcde_1, "border")
        },
        "color": {
            "before": get_score(abcde_1, "color"),
            "after": get_score(abcde_2, "color"),
            "change": get_score(abcde_2, "color") - get_score(abcde_1, "color")
        },
        "diameter": {
            "before": get_score(abcde_1, "diameter"),
            "after": get_score(abcde_2, "diameter"),
            "change": get_score(abcde_2, "diameter") - get_score(abcde_1, "diameter")
        }
    }

    # Calculate overall concern level
    significant_increases = sum(1 for k, v in abcde_comparison.items() if v["change"] > 0.2)

    if significant_increases >= 3 or risk_change == "increased":
        overall_concern = "high"
        recommendation = "Significant changes detected. Dermatologist consultation recommended."
    elif significant_increases >= 1:
        overall_concern = "moderate"
        recommendation = "Some changes detected. Continue monitoring closely."
    else:
        overall_concern = "low"
        recommendation = "Lesion appears stable. Continue routine monitoring."

    return {
        "lesion_group_id": group_id,
        "lesion_name": group.name,
        "comparison": {
            "earlier_analysis": {
                "id": analysis_1.id,
                "date": analysis_1.created_at.isoformat(),
                "predicted_class": analysis_1.predicted_class,
                "confidence": analysis_1.lesion_confidence,
                "risk_level": analysis_1.risk_level
            },
            "later_analysis": {
                "id": analysis_2.id,
                "date": analysis_2.created_at.isoformat(),
                "predicted_class": analysis_2.predicted_class,
                "confidence": analysis_2.lesion_confidence,
                "risk_level": analysis_2.risk_level
            },
            "time_between_days": days_between,
            "classification_changed": analysis_1.predicted_class != analysis_2.predicted_class,
            "confidence_change": (analysis_2.lesion_confidence or 0) - (analysis_1.lesion_confidence or 0),
            "risk_change": risk_change,
            "abcde_comparison": abcde_comparison
        },
        "assessment": {
            "overall_concern": overall_concern,
            "risk_concern": risk_concern,
            "recommendation": recommendation
        }
    }


# =============================================================================
# PROGRESSION TIMELINE ENDPOINT
# =============================================================================

@router.get("/progression/timeline")
async def get_progression_timeline(
    days: int = Query(365, description="Number of days to include"),
    lesion_group_id: int = Query(None, description="Filter by specific lesion group"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get a comprehensive timeline of lesion progression across all or specific lesion groups.

    Returns:
    - Timeline of all analyses
    - Risk level trends
    - Key events (biopsies, high-risk detections, etc.)
    """
    from datetime import timedelta

    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Build query
    query = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id,
        AnalysisHistory.created_at >= cutoff_date
    )

    if lesion_group_id:
        query = query.filter(AnalysisHistory.lesion_group_id == lesion_group_id)

    analyses = query.order_by(AnalysisHistory.created_at).all()

    if not analyses:
        return {
            "period_days": days,
            "total_analyses": 0,
            "message": "No analyses found in this period"
        }

    # Build timeline
    timeline = []
    key_events = []

    for analysis in analyses:
        abcde = analysis.red_flag_data or {}

        entry = {
            "analysis_id": analysis.id,
            "date": analysis.created_at.isoformat(),
            "lesion_group_id": analysis.lesion_group_id,
            "predicted_class": analysis.predicted_class,
            "confidence": analysis.lesion_confidence,
            "risk_level": analysis.risk_level,
            "body_location": analysis.body_location,
            "total_abcde_score": abcde.get("total_score"),
            "biopsy_performed": analysis.biopsy_performed
        }
        timeline.append(entry)

        # Track key events
        if analysis.risk_level in ["high", "very_high"]:
            key_events.append({
                "type": "high_risk_detection",
                "date": analysis.created_at.isoformat(),
                "analysis_id": analysis.id,
                "details": f"High risk {analysis.predicted_class} detected"
            })

        if analysis.biopsy_performed:
            key_events.append({
                "type": "biopsy",
                "date": analysis.biopsy_date.isoformat() if analysis.biopsy_date else analysis.created_at.isoformat(),
                "analysis_id": analysis.id,
                "details": f"Biopsy result: {analysis.biopsy_result or 'pending'}"
            })

    # Calculate risk distribution
    risk_distribution = {
        "low": sum(1 for a in analyses if a.risk_level == "low"),
        "medium": sum(1 for a in analyses if a.risk_level == "medium"),
        "high": sum(1 for a in analyses if a.risk_level == "high"),
        "very_high": sum(1 for a in analyses if a.risk_level == "very_high")
    }

    # Get unique lesion groups
    lesion_groups_in_period = list(set(a.lesion_group_id for a in analyses if a.lesion_group_id))

    # Monthly summary
    monthly_summary = {}
    for analysis in analyses:
        month_key = analysis.created_at.strftime("%Y-%m")
        if month_key not in monthly_summary:
            monthly_summary[month_key] = {"count": 0, "high_risk_count": 0}
        monthly_summary[month_key]["count"] += 1
        if analysis.risk_level in ["high", "very_high"]:
            monthly_summary[month_key]["high_risk_count"] += 1

    return {
        "period_days": days,
        "total_analyses": len(analyses),
        "unique_lesion_groups": len(lesion_groups_in_period),
        "timeline": timeline,
        "key_events": sorted(key_events, key=lambda x: x["date"], reverse=True),
        "risk_distribution": risk_distribution,
        "monthly_summary": [
            {"month": month, "analyses": data["count"], "high_risk": data["high_risk_count"]}
            for month, data in sorted(monthly_summary.items())
        ],
        "first_analysis_date": analyses[0].created_at.isoformat(),
        "last_analysis_date": analyses[-1].created_at.isoformat()
    }
