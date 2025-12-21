"""
Clinical Trials Router

Endpoints for:
- Listing clinical trials with filters
- Getting personalized trial matches
- Expressing interest in trials
- Managing trial enrollments
- Syncing trials from ClinicalTrials.gov (admin)
"""

from fastapi import APIRouter, Depends, HTTPException, Form, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from datetime import datetime
from typing import Optional, List

from database import (
    get_db, User, UserProfile, ClinicalTrial,
    TrialMatch, TrialInterest, AnalysisHistory
)
from auth import get_current_active_user

router = APIRouter(tags=["Clinical Trials"])


@router.get("/clinical-trials/test")
async def test_endpoint(
    current_user: User = Depends(get_current_active_user),
):
    """Simple test endpoint."""
    return {"user_id": current_user.id, "username": current_user.username}


# =============================================================================
# TRIAL LISTING ENDPOINTS
# =============================================================================

@router.get("/clinical-trials")
async def list_clinical_trials(
    status: Optional[str] = Query(None, description="Filter by status (Recruiting, Active, etc.)"),
    condition: Optional[str] = Query(None, description="Filter by condition"),
    phase: Optional[str] = Query(None, description="Filter by phase"),
    state: Optional[str] = Query(None, description="Filter by state"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List all clinical trials with optional filters.
    """
    try:
        query = db.query(ClinicalTrial)

        # Apply filters
        if status:
            query = query.filter(ClinicalTrial.status.ilike(f"%{status}%"))
        else:
            # Default to recruiting trials
            query = query.filter(ClinicalTrial.status.in_([
                "Recruiting", "RECRUITING", "Active, not recruiting"
            ]))

        if condition:
            # Search in conditions JSON array
            query = query.filter(
                ClinicalTrial.conditions.cast(str).ilike(f"%{condition}%")
            )

        if phase:
            query = query.filter(ClinicalTrial.phase.ilike(f"%{phase}%"))

        if state:
            # Search in locations JSON
            query = query.filter(
                ClinicalTrial.locations.cast(str).ilike(f"%{state}%")
            )

        # Get total count
        total = query.count()

        # Get paginated results
        trials = query.order_by(desc(ClinicalTrial.synced_at)).offset(skip).limit(limit).all()

        return {
            "trials": [
                {
                    "id": t.id,
                    "nct_id": t.nct_id,
                    "title": t.title,
                    "brief_summary": t.brief_summary[:500] + "..." if t.brief_summary and len(t.brief_summary) > 500 else t.brief_summary,
                    "phase": t.phase,
                    "status": t.status,
                    "conditions": t.conditions,
                    "sponsor": t.sponsor,
                    "target_enrollment": t.target_enrollment,
                    "locations_count": len(t.locations) if t.locations else 0,
                    "url": t.url,
                }
                for t in trials
            ],
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list trials: {str(e)}")


# NOTE: /clinical-trials/{trial_id} moved to end of file to avoid route conflicts
# Static routes must come before path parameter routes in FastAPI

# =============================================================================
# MATCHING ENDPOINTS
# =============================================================================

@router.get("/clinical-trials/matches")
async def get_trial_matches(
    min_score: int = Query(20, ge=0, le=100, description="Minimum match score"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get personalized clinical trial matches for the current user.
    Matches are based on the user's diagnosis history and conditions.
    """
    try:
        from clinical_trials_matcher import find_matches_for_user, create_or_update_matches

        # First, update matches for this user
        try:
            create_or_update_matches(db, current_user.id)
        except Exception as match_err:
            # Log but don't fail - we can still return existing matches
            print(f"Error updating matches: {match_err}")

        # Get matches from database
        matches = db.query(TrialMatch).filter(
            TrialMatch.user_id == current_user.id,
            TrialMatch.match_score >= min_score,
            TrialMatch.dismissed == False
        ).order_by(desc(TrialMatch.match_score)).offset(skip).limit(limit).all()

        total = db.query(TrialMatch).filter(
            TrialMatch.user_id == current_user.id,
            TrialMatch.match_score >= min_score,
            TrialMatch.dismissed == False
        ).count()

        return {
            "matches": [
                {
                    "match_id": m.id,
                    "match_score": m.match_score,
                    "match_reasons": m.match_reasons,
                    "unmet_criteria": m.unmet_criteria,
                    "matched_conditions": m.matched_conditions,
                    "distance_miles": m.distance_miles,
                    "nearest_location": m.nearest_location,
                    "status": m.status,
                    "matched_at": m.matched_at.isoformat() if m.matched_at else None,
                    # Genetic matching fields (NEW)
                    "genetic_score": m.genetic_score,
                    "matched_biomarkers": m.matched_biomarkers or [],
                    "missing_biomarkers": m.missing_biomarkers or [],
                    "excluded_biomarkers_found": m.excluded_biomarkers_found or [],
                    "genetic_eligible": m.genetic_eligible if m.genetic_eligible is not None else True,
                    "genetic_match_type": m.genetic_match_type or "none",
                    "trial": {
                        "id": m.trial.id,
                        "nct_id": m.trial.nct_id,
                        "title": m.trial.title,
                        "phase": m.trial.phase,
                        "status": m.trial.status,
                        "conditions": m.trial.conditions,
                        "sponsor": m.trial.sponsor,
                        "url": m.trial.url,
                        # Trial biomarker requirements (NEW)
                        "required_biomarkers": m.trial.required_biomarkers or [],
                        "excluded_biomarkers": m.trial.excluded_biomarkers or [],
                        "targeted_therapy_trial": m.trial.targeted_therapy_trial,
                        "requires_genetic_testing": m.trial.requires_genetic_testing,
                    }
                }
                for m in matches
            ],
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get matches: {str(e)}")


@router.post("/clinical-trials/matches/{match_id}/dismiss")
async def dismiss_match(
    match_id: int,
    reason: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Dismiss a trial match (hide it from the user's matches).
    """
    match = db.query(TrialMatch).filter(
        TrialMatch.id == match_id,
        TrialMatch.user_id == current_user.id
    ).first()

    if not match:
        raise HTTPException(status_code=404, detail="Match not found")

    match.dismissed = True
    match.dismissed_at = datetime.utcnow()
    match.dismiss_reason = reason
    db.commit()

    return {"message": "Match dismissed", "match_id": match_id}


# =============================================================================
# INTEREST ENDPOINTS
# =============================================================================

from pydantic import BaseModel

class InterestRequest(BaseModel):
    interest_level: str
    preferred_contact: str = "email"
    notes: Optional[str] = None

@router.post("/clinical-trials/{trial_id}/interest")
async def express_interest(
    trial_id: int,
    request: InterestRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Express interest in a clinical trial.
    """
    interest_level = request.interest_level
    preferred_contact = request.preferred_contact
    notes = request.notes
    # Validate interest level
    if interest_level not in ["high", "medium", "exploring"]:
        raise HTTPException(status_code=400, detail="Invalid interest level")

    # Check trial exists
    trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    # Check if already expressed interest
    existing = db.query(TrialInterest).filter(
        TrialInterest.user_id == current_user.id,
        TrialInterest.trial_id == trial_id
    ).first()

    if existing:
        # Update existing interest
        existing.interest_level = interest_level
        existing.preferred_contact = preferred_contact
        existing.notes = notes
        existing.updated_at = datetime.utcnow()
        db.commit()

        return {
            "message": "Interest updated",
            "interest_id": existing.id,
            "trial_id": trial_id,
        }

    # Get match if exists
    match = db.query(TrialMatch).filter(
        TrialMatch.user_id == current_user.id,
        TrialMatch.trial_id == trial_id
    ).first()

    # Create new interest
    interest = TrialInterest(
        user_id=current_user.id,
        trial_id=trial_id,
        match_id=match.id if match else None,
        interest_level=interest_level,
        preferred_contact=preferred_contact,
        contact_email=current_user.email,
        notes=notes,
    )

    db.add(interest)

    # Update match status if exists
    if match:
        match.status = "interested"
        match.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(interest)

    return {
        "message": "Interest expressed successfully",
        "interest_id": interest.id,
        "trial_id": trial_id,
        "trial_title": trial.title,
        "contact_info": {
            "name": trial.contact_name,
            "email": trial.contact_email,
            "phone": trial.contact_phone,
        }
    }


@router.get("/clinical-trials/interests")
async def get_user_interests(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all trials the user has expressed interest in.
    """
    interests = db.query(TrialInterest).filter(
        TrialInterest.user_id == current_user.id,
        TrialInterest.withdrawn == False
    ).order_by(desc(TrialInterest.expressed_at)).all()

    return {
        "interests": [
            {
                "id": i.id,
                "trial_id": i.trial_id,
                "interest_level": i.interest_level,
                "preferred_contact": i.preferred_contact,
                "notes": i.notes,
                "status": i.enrollment_status or "pending",
                "expressed_at": i.expressed_at.isoformat() if i.expressed_at else None,
                "contacted_trial": i.contacted_trial,
                "contact_response": i.contact_response,
                "enrolled": i.enrolled,
                "trial": {
                    "id": i.trial.id,
                    "nct_id": i.trial.nct_id,
                    "title": i.trial.title,
                    "brief_summary": i.trial.brief_summary,
                    "phase": i.trial.phase,
                    "status": i.trial.status,
                    "conditions": i.trial.conditions or [],
                    "sponsor": i.trial.sponsor,
                    "contact_email": i.trial.contact_email,
                    "url": i.trial.url,
                } if i.trial else None
            }
            for i in interests
            if i.trial  # Only include interests with valid trials
        ],
        "total": len(interests),
    }


@router.delete("/clinical-trials/interests/{interest_id}")
async def withdraw_interest(
    interest_id: int,
    reason: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Withdraw interest in a clinical trial.
    """
    interest = db.query(TrialInterest).filter(
        TrialInterest.id == interest_id,
        TrialInterest.user_id == current_user.id
    ).first()

    if not interest:
        raise HTTPException(status_code=404, detail="Interest not found")

    interest.withdrawn = True
    interest.withdrawn_at = datetime.utcnow()
    interest.withdrawal_reason = reason

    # Update match status if exists
    if interest.match_id:
        match = db.query(TrialMatch).filter(TrialMatch.id == interest.match_id).first()
        if match:
            match.status = "declined"
            match.updated_at = datetime.utcnow()

    db.commit()

    return {"message": "Interest withdrawn", "interest_id": interest_id}


@router.put("/clinical-trials/interests/{interest_id}/contact")
async def update_contact_status(
    interest_id: int,
    contacted: bool = Form(...),
    contact_method: Optional[str] = Form(None),
    contact_response: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update contact status for an interest (track if user contacted the trial).
    """
    interest = db.query(TrialInterest).filter(
        TrialInterest.id == interest_id,
        TrialInterest.user_id == current_user.id
    ).first()

    if not interest:
        raise HTTPException(status_code=404, detail="Interest not found")

    interest.contacted_trial = contacted
    if contacted:
        interest.contacted_at = datetime.utcnow()
        interest.contact_method = contact_method
    interest.contact_response = contact_response
    interest.updated_at = datetime.utcnow()

    # Update match status
    if interest.match_id:
        match = db.query(TrialMatch).filter(TrialMatch.id == interest.match_id).first()
        if match:
            match.status = "contacted"
            match.updated_at = datetime.utcnow()

    db.commit()

    return {"message": "Contact status updated", "interest_id": interest_id}


# =============================================================================
# STATISTICS ENDPOINTS
# =============================================================================

@router.get("/clinical-trials/stats")
async def get_trial_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get clinical trial statistics for the current user.
    """
    # Count matches
    total_matches = db.query(TrialMatch).filter(
        TrialMatch.user_id == current_user.id,
        TrialMatch.dismissed == False
    ).count()

    high_score_matches = db.query(TrialMatch).filter(
        TrialMatch.user_id == current_user.id,
        TrialMatch.match_score >= 60,
        TrialMatch.dismissed == False
    ).count()

    # Count interests
    total_interests = db.query(TrialInterest).filter(
        TrialInterest.user_id == current_user.id,
        TrialInterest.withdrawn == False
    ).count()

    contacted = db.query(TrialInterest).filter(
        TrialInterest.user_id == current_user.id,
        TrialInterest.contacted_trial == True
    ).count()

    enrolled = db.query(TrialInterest).filter(
        TrialInterest.user_id == current_user.id,
        TrialInterest.enrolled == True
    ).count()

    # Count available trials
    available_trials = db.query(ClinicalTrial).filter(
        ClinicalTrial.status.in_(["Recruiting", "RECRUITING"])
    ).count()

    return {
        "available_trials": available_trials,
        "user_stats": {
            "total_matches": total_matches,
            "high_score_matches": high_score_matches,
            "interests_expressed": total_interests,
            "trials_contacted": contacted,
            "trials_enrolled": enrolled,
        }
    }


# =============================================================================
# SYNC ENDPOINTS (Admin)
# =============================================================================

@router.post("/clinical-trials/sync")
async def sync_trials(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Trigger a sync of clinical trials from ClinicalTrials.gov.
    Admin only endpoint.
    """
    # Check if user is admin or professional
    if current_user.role not in ["admin", "dermatologist"]:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        from clinical_trials_sync import sync_dermatology_trials
        import asyncio

        # Run sync
        stats = await sync_dermatology_trials()

        return {
            "message": "Sync completed",
            "stats": stats,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/clinical-trials/sync/status")
async def get_sync_status(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the status of the last trial sync.
    """
    # Get most recently synced trial
    latest = db.query(ClinicalTrial).order_by(desc(ClinicalTrial.synced_at)).first()

    total_trials = db.query(ClinicalTrial).count()
    recruiting = db.query(ClinicalTrial).filter(
        ClinicalTrial.status.in_(["Recruiting", "RECRUITING"])
    ).count()

    return {
        "total_trials": total_trials,
        "recruiting_trials": recruiting,
        "last_sync": latest.synced_at.isoformat() if latest and latest.synced_at else None,
    }


# =============================================================================
# NEW TRIAL ALERTS
# =============================================================================

@router.get("/clinical-trials/alerts/new-matches")
async def check_new_trial_matches(
    since_days: int = Query(7, description="Check for trials added in last N days"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Check for new trials that match the user's conditions since last check.
    Returns new matching trials for notification purposes.
    """
    from datetime import timedelta
    from clinical_trials_matcher import get_user_conditions, match_conditions, CONDITION_MAPPINGS

    # Get user's conditions from analysis history
    user_conditions = get_user_conditions(db, current_user.id)

    if not user_conditions:
        return {
            "new_matches": [],
            "message": "Complete some skin analyses to get personalized trial alerts"
        }

    # Get trials added in the last N days
    cutoff_date = datetime.utcnow() - timedelta(days=since_days)
    new_trials = db.query(ClinicalTrial).filter(
        ClinicalTrial.created_at >= cutoff_date,
        ClinicalTrial.status.in_(["Recruiting", "RECRUITING"])
    ).all()

    # Find matching trials
    new_matches = []
    for trial in new_trials:
        score, matched_conditions, exact_match = match_conditions(
            user_conditions, trial.conditions or []
        )
        if score > 0:
            new_matches.append({
                "trial_id": trial.id,
                "nct_id": trial.nct_id,
                "title": trial.title,
                "phase": trial.phase,
                "conditions": trial.conditions,
                "matched_conditions": matched_conditions,
                "match_score": score,
                "exact_match": exact_match,
                "added_date": trial.created_at.isoformat() if trial.created_at else None,
            })

    # Sort by match score
    new_matches.sort(key=lambda x: x["match_score"], reverse=True)

    return {
        "new_matches": new_matches[:20],  # Top 20
        "total_new_matches": len(new_matches),
        "checked_since": cutoff_date.isoformat(),
        "user_conditions": list(user_conditions),
    }


@router.post("/clinical-trials/alerts/subscribe")
async def subscribe_to_alerts(
    conditions: List[str] = Form(None, description="Additional conditions to watch"),
    frequency: str = Form("weekly", description="Alert frequency: daily, weekly, monthly"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Subscribe to alerts for new matching clinical trials.
    """
    # Get or create user profile
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)

    # Store alert preferences (using existing JSON field or create new one)
    alert_prefs = {
        "enabled": True,
        "frequency": frequency,
        "custom_conditions": conditions or [],
        "subscribed_at": datetime.utcnow().isoformat(),
    }

    # Store in profile (you may need to add this field to UserProfile model)
    profile.clinical_trial_alerts = alert_prefs
    db.commit()

    return {
        "message": "Subscribed to clinical trial alerts",
        "frequency": frequency,
        "conditions_watched": conditions,
    }


# =============================================================================
# SHARE WITH DOCTOR
# =============================================================================

@router.get("/clinical-trials/{trial_id}/share")
async def get_shareable_trial_summary(
    trial_id: int,
    format: str = Query("text", description="Format: text, html, or pdf"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Generate a shareable summary of a clinical trial for sharing with healthcare provider.
    """
    trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    # Get user's match info if available
    match = db.query(TrialMatch).filter(
        TrialMatch.trial_id == trial_id,
        TrialMatch.user_id == current_user.id
    ).first()

    # Build shareable summary
    if format == "html":
        summary = f"""
        <html>
        <head><style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #7c3aed; }}
            .section {{ margin: 20px 0; padding: 15px; background: #f5f3ff; border-radius: 8px; }}
            .label {{ font-weight: bold; color: #4c1d95; }}
        </style></head>
        <body>
        <h1>Clinical Trial Information</h1>
        <div class="section">
            <p class="label">Trial ID:</p>
            <p>{trial.nct_id}</p>
            <p class="label">Title:</p>
            <p>{trial.title}</p>
            <p class="label">Phase:</p>
            <p>{trial.phase or 'Not specified'}</p>
            <p class="label">Status:</p>
            <p>{trial.status}</p>
        </div>
        <div class="section">
            <p class="label">Conditions:</p>
            <p>{', '.join(trial.conditions or [])}</p>
        </div>
        <div class="section">
            <p class="label">Eligibility:</p>
            <p>Age: {trial.min_age or 'No min'} - {trial.max_age or 'No max'} years</p>
            <p>Gender: {trial.gender or 'All'}</p>
        </div>
        <div class="section">
            <p class="label">Contact:</p>
            <p>{trial.contact_name or 'Not provided'}</p>
            <p>{trial.contact_email or ''}</p>
            <p>{trial.contact_phone or ''}</p>
        </div>
        <div class="section">
            <p class="label">More Information:</p>
            <p><a href="{trial.url}">{trial.url}</a></p>
        </div>
        {"<div class='section'><p class='label'>Patient Match Score:</p><p>" + str(match.match_score) + "%</p></div>" if match else ""}
        <p style="color: #6b7280; font-size: 12px; margin-top: 30px;">
            Generated from Skin Disease Analysis App on {datetime.utcnow().strftime('%Y-%m-%d')}
        </p>
        </body></html>
        """
    else:
        # Plain text format
        summary = f"""
CLINICAL TRIAL INFORMATION
==========================

Trial ID: {trial.nct_id}
Title: {trial.title}
Phase: {trial.phase or 'Not specified'}
Status: {trial.status}
Sponsor: {trial.sponsor or 'Not specified'}

CONDITIONS
----------
{chr(10).join(['• ' + c for c in (trial.conditions or [])])}

ELIGIBILITY
-----------
Age: {trial.min_age or 'No minimum'} - {trial.max_age or 'No maximum'} years
Gender: {trial.gender or 'All'}

CONTACT INFORMATION
-------------------
Name: {trial.contact_name or 'Not provided'}
Email: {trial.contact_email or 'Not provided'}
Phone: {trial.contact_phone or 'Not provided'}

MORE INFORMATION
----------------
{trial.url}

{"PATIENT MATCH SCORE: " + str(match.match_score) + "%" if match else ""}

---
Generated from Skin Disease Analysis App
Date: {datetime.utcnow().strftime('%Y-%m-%d')}
"""

    return {
        "trial_id": trial.id,
        "nct_id": trial.nct_id,
        "format": format,
        "summary": summary,
        "share_subject": f"Clinical Trial: {trial.nct_id} - {trial.title[:50]}...",
    }


# =============================================================================
# ELIGIBILITY CHECKER
# =============================================================================

@router.get("/clinical-trials/{trial_id}/eligibility-questions")
async def get_eligibility_questions(
    trial_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get eligibility questions for a specific trial based on its criteria.
    """
    trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    # Get user profile for pre-filling
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    # Build eligibility questions based on trial criteria
    questions = []

    # Age question
    questions.append({
        "id": "age",
        "question": "What is your age?",
        "type": "number",
        "required": True,
        "criteria": f"Trial requires age {trial.min_age or 0} - {trial.max_age or 999}",
        "min": trial.min_age or 0,
        "max": trial.max_age or 999,
        "prefilled": current_user.age if hasattr(current_user, 'age') and current_user.age else None,
    })

    # Gender question
    if trial.gender and trial.gender.lower() != 'all':
        questions.append({
            "id": "gender",
            "question": "What is your biological sex?",
            "type": "select",
            "required": True,
            "options": ["Male", "Female", "Other"],
            "criteria": f"Trial is for {trial.gender} participants",
            "prefilled": current_user.gender if hasattr(current_user, 'gender') else None,
        })

    # Diagnosis confirmation
    if trial.conditions:
        questions.append({
            "id": "diagnosis",
            "question": f"Have you been diagnosed with any of these conditions?",
            "type": "multiselect",
            "required": True,
            "options": trial.conditions[:10],  # Top 10 conditions
            "criteria": "Must have relevant diagnosis",
        })

    # Biopsy confirmation (for cancer trials)
    cancer_keywords = ['melanoma', 'carcinoma', 'cancer', 'tumor']
    if trial.conditions and any(kw in ' '.join(trial.conditions).lower() for kw in cancer_keywords):
        questions.append({
            "id": "biopsy_confirmed",
            "question": "Has your condition been confirmed by biopsy or pathology?",
            "type": "boolean",
            "required": True,
            "criteria": "Cancer trials typically require biopsy confirmation",
        })

    # Treatment history
    questions.append({
        "id": "prior_treatment",
        "question": "Have you received prior treatment for this condition?",
        "type": "select",
        "required": False,
        "options": ["No prior treatment", "Currently on treatment", "Previously treated", "Treatment failed"],
        "criteria": "Some trials require treatment-naive or treatment-experienced patients",
    })

    # Location/travel
    if trial.locations:
        nearest = trial.locations[0] if trial.locations else None
        questions.append({
            "id": "can_travel",
            "question": f"Can you travel to trial sites for regular visits?",
            "type": "boolean",
            "required": True,
            "criteria": f"Nearest site: {nearest.get('city', 'Unknown') if nearest else 'Unknown'}, {nearest.get('state', '') if nearest else ''}",
        })

    # General health
    questions.append({
        "id": "general_health",
        "question": "How would you describe your general health?",
        "type": "select",
        "required": False,
        "options": ["Excellent", "Good", "Fair", "Poor"],
        "criteria": "Most trials require adequate general health",
    })

    return {
        "trial_id": trial.id,
        "nct_id": trial.nct_id,
        "title": trial.title,
        "questions": questions,
        "total_questions": len(questions),
        "raw_eligibility_criteria": trial.eligibility_criteria[:2000] if trial.eligibility_criteria else None,
    }


@router.post("/clinical-trials/{trial_id}/check-eligibility")
async def check_eligibility(
    trial_id: int,
    answers: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Check eligibility based on user's answers to eligibility questions.
    Returns eligibility status and any concerns.
    """
    trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    eligible = True
    concerns = []
    passed = []

    # Check age
    if 'age' in answers:
        user_age = int(answers['age'])
        min_age = trial.min_age or 0
        max_age = trial.max_age or 999

        if min_age <= user_age <= max_age:
            passed.append(f"Age {user_age} is within required range ({min_age}-{max_age})")
        else:
            eligible = False
            concerns.append(f"Age {user_age} is outside required range ({min_age}-{max_age})")

    # Check gender
    if 'gender' in answers and trial.gender and trial.gender.lower() != 'all':
        if answers['gender'].lower() == trial.gender.lower():
            passed.append(f"Gender matches trial requirement ({trial.gender})")
        else:
            eligible = False
            concerns.append(f"Trial is only for {trial.gender} participants")

    # Check diagnosis
    if 'diagnosis' in answers:
        selected = answers['diagnosis'] if isinstance(answers['diagnosis'], list) else [answers['diagnosis']]
        if selected:
            passed.append(f"Confirmed diagnosis: {', '.join(selected)}")
        else:
            concerns.append("No relevant diagnosis selected - may affect eligibility")

    # Check biopsy
    if 'biopsy_confirmed' in answers:
        if answers['biopsy_confirmed']:
            passed.append("Condition confirmed by biopsy")
        else:
            concerns.append("Biopsy confirmation may be required - check with trial coordinator")

    # Check travel
    if 'can_travel' in answers:
        if answers['can_travel']:
            passed.append("Can travel to trial site")
        else:
            eligible = False
            concerns.append("Unable to travel to trial sites")

    # Overall assessment
    if eligible and not concerns:
        status = "likely_eligible"
        message = "Based on your answers, you appear to meet the basic eligibility criteria."
    elif eligible and concerns:
        status = "possibly_eligible"
        message = "You may be eligible, but there are some items to discuss with the trial coordinator."
    else:
        status = "likely_ineligible"
        message = "Based on your answers, you may not meet some eligibility criteria."

    return {
        "trial_id": trial.id,
        "nct_id": trial.nct_id,
        "status": status,
        "message": message,
        "eligible": eligible,
        "passed_criteria": passed,
        "concerns": concerns,
        "recommendation": "Contact the trial coordinator to confirm your eligibility" if status != "likely_ineligible" else "Consider other trials that may be a better fit",
        "contact_email": trial.contact_email,
        "contact_phone": trial.contact_phone,
    }


# =============================================================================
# TRIAL LOCATIONS MAP DATA
# =============================================================================

@router.get("/clinical-trials/{trial_id}/locations")
async def get_trial_locations(
    trial_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all locations for a trial with coordinates for map display.
    """
    trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    # Get user's location if available
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    user_lat = profile.latitude if profile else None
    user_lng = profile.longitude if profile else None

    locations = []
    for loc in (trial.locations or []):
        location_data = {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country", "USA"),
            "zip": loc.get("zip"),
            "status": loc.get("status", "Recruiting"),
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
        }

        # Calculate distance if we have coordinates
        if user_lat and user_lng and loc.get("lat") and loc.get("lng"):
            from clinical_trials_matcher import haversine_distance
            distance = haversine_distance(user_lat, user_lng, loc["lat"], loc["lng"])
            location_data["distance_miles"] = round(distance, 1)

        locations.append(location_data)

    # Sort by distance if available
    locations.sort(key=lambda x: x.get("distance_miles", float("inf")))

    return {
        "trial_id": trial.id,
        "nct_id": trial.nct_id,
        "title": trial.title,
        "total_locations": len(locations),
        "locations": locations,
        "user_location": {
            "lat": user_lat,
            "lng": user_lng,
        } if user_lat and user_lng else None,
    }


# =============================================================================
# TRIAL DETAIL (must be last to avoid route conflicts with static paths)
# =============================================================================

@router.get("/clinical-trials/{trial_id}")
async def get_trial_detail(
    trial_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific trial.
    NOTE: This endpoint is defined last to avoid route conflicts with static paths
    like /clinical-trials/matches, /clinical-trials/interests, etc.
    """
    trial = db.query(ClinicalTrial).filter(ClinicalTrial.id == trial_id).first()

    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    # Check if user has expressed interest
    interest = db.query(TrialInterest).filter(
        TrialInterest.user_id == current_user.id,
        TrialInterest.trial_id == trial_id
    ).first()

    # Check if user has a match
    match = db.query(TrialMatch).filter(
        TrialMatch.user_id == current_user.id,
        TrialMatch.trial_id == trial_id
    ).first()

    return {
        "trial": {
            "id": trial.id,
            "nct_id": trial.nct_id,
            "title": trial.title,
            "brief_summary": trial.brief_summary,
            "detailed_description": trial.detailed_description,
            "phase": trial.phase,
            "status": trial.status,
            "study_type": trial.study_type,
            "conditions": trial.conditions,
            "interventions": trial.interventions,
            "eligibility_criteria": trial.eligibility_criteria,
            "min_age": trial.min_age,
            "max_age": trial.max_age,
            "gender": trial.gender,
            "locations": trial.locations,
            "contact_name": trial.contact_name,
            "contact_email": trial.contact_email,
            "contact_phone": trial.contact_phone,
            "principal_investigator": trial.principal_investigator,
            "target_enrollment": trial.target_enrollment,
            "sponsor": trial.sponsor,
            "collaborators": trial.collaborators,
            "start_date": trial.start_date.isoformat() if trial.start_date else None,
            "completion_date": trial.completion_date.isoformat() if trial.completion_date else None,
            "url": trial.url,
            "synced_at": trial.synced_at.isoformat() if trial.synced_at else None,
            # Biomarker requirements (NEW)
            "required_biomarkers": trial.required_biomarkers or [],
            "excluded_biomarkers": trial.excluded_biomarkers or [],
            "genetic_requirements": trial.genetic_requirements or [],
            "biomarker_keywords": trial.biomarker_keywords or [],
            "requires_genetic_testing": trial.requires_genetic_testing,
            "targeted_therapy_trial": trial.targeted_therapy_trial,
        },
        "user_interest": {
            "expressed": interest is not None,
            "interest_id": interest.id if interest else None,
            "interest_level": interest.interest_level if interest else None,
            "expressed_at": interest.expressed_at.isoformat() if interest else None,
        } if interest else None,
        "match": {
            "match_score": match.match_score,
            "match_reasons": match.match_reasons,
            "distance_miles": match.distance_miles,
            # Genetic match info (NEW)
            "genetic_score": match.genetic_score,
            "matched_biomarkers": match.matched_biomarkers or [],
            "missing_biomarkers": match.missing_biomarkers or [],
            "genetic_match_type": match.genetic_match_type,
        } if match else None,
    }
