"""
Teledermatology Router

Endpoints for:
- Dermatologist directory
- Video consultations
- Referrals
- Consultation notes
- Second opinions with workflow
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import json

from database import (
    get_db, User, DermatologistProfile, VideoConsultation,
    Referral, ConsultationNote, SecondOpinion, UserProfile
)
from auth import get_current_active_user

router = APIRouter(tags=["Teledermatology"])

# Try to load workflow services
try:
    from second_opinion_workflow import (
        SecondOpinionWorkflowService,
        SecondOpinionPaymentService,
        DermatologistAssignmentAlgorithm,
        NotificationService,
        SLATracker
    )
    workflow_service = SecondOpinionWorkflowService()
    payment_service = SecondOpinionPaymentService()
    assignment_algorithm = DermatologistAssignmentAlgorithm()
    notification_service = NotificationService()
    sla_tracker = SLATracker()
    WORKFLOW_AVAILABLE = True
except Exception:
    WORKFLOW_AVAILABLE = False


# =============================================================================
# DERMATOLOGIST DIRECTORY
# =============================================================================

@router.get("/dermatologists")
async def list_dermatologists(
    city: Optional[str] = None,
    state: Optional[str] = None,
    specialization: Optional[str] = None,
    accepts_video: Optional[bool] = None,
    accepts_referrals: Optional[bool] = None,
    min_rating: Optional[float] = None,
    language: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Search and filter dermatologists in the directory."""
    try:
        query = db.query(DermatologistProfile).filter(DermatologistProfile.is_active == True)

        if city:
            query = query.filter(DermatologistProfile.city.ilike(f"%{city}%"))
        if state:
            query = query.filter(DermatologistProfile.state.ilike(f"%{state}%"))
        if accepts_video is not None:
            query = query.filter(DermatologistProfile.accepts_video_consultations == accepts_video)
        if accepts_referrals is not None:
            query = query.filter(DermatologistProfile.accepts_referrals == accepts_referrals)
        if min_rating:
            query = query.filter(DermatologistProfile.average_rating >= min_rating)
        if specialization:
            query = query.filter(DermatologistProfile.specializations.contains([specialization]))
        if language:
            query = query.filter(DermatologistProfile.languages_spoken.contains([language]))

        total = query.count()
        dermatologists = query.order_by(DermatologistProfile.average_rating.desc()).offset(skip).limit(limit).all()

        return {
            "dermatologists": [
                {
                    "id": d.id,
                    "full_name": d.full_name,
                    "credentials": d.credentials,
                    "practice_name": d.practice_name,
                    "city": d.city,
                    "state": d.state,
                    "specializations": d.specializations or [],
                    "languages_spoken": d.languages_spoken or [],
                    "accepts_video_consultations": d.accepts_video_consultations,
                    "accepts_referrals": d.accepts_referrals,
                    "availability_status": d.availability_status,
                    "typical_wait_time_days": d.typical_wait_time_days,
                    "average_rating": d.average_rating,
                    "total_reviews": d.total_reviews,
                    "photo_url": d.photo_url,
                    "bio": d.bio[:200] + "..." if d.bio and len(d.bio) > 200 else d.bio,
                    "is_verified": d.is_verified
                }
                for d in dermatologists
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        print(f"Error listing dermatologists: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list dermatologists: {str(e)}")


@router.get("/dermatologists/{dermatologist_id}")
async def get_dermatologist_profile(
    dermatologist_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed profile of a specific dermatologist."""
    try:
        dermatologist = db.query(DermatologistProfile).filter(
            DermatologistProfile.id == dermatologist_id
        ).first()

        if not dermatologist:
            raise HTTPException(status_code=404, detail="Dermatologist not found")

        return {
            "id": dermatologist.id,
            "full_name": dermatologist.full_name,
            "credentials": dermatologist.credentials,
            "email": dermatologist.email,
            "phone_number": dermatologist.phone_number,
            "practice_name": dermatologist.practice_name,
            "practice_address": dermatologist.practice_address,
            "city": dermatologist.city,
            "state": dermatologist.state,
            "country": dermatologist.country,
            "zip_code": dermatologist.zip_code,
            "specializations": dermatologist.specializations or [],
            "languages_spoken": dermatologist.languages_spoken or [],
            "board_certifications": dermatologist.board_certifications or [],
            "accepts_video_consultations": dermatologist.accepts_video_consultations,
            "accepts_referrals": dermatologist.accepts_referrals,
            "accepts_second_opinions": dermatologist.accepts_second_opinions,
            "availability_status": dermatologist.availability_status,
            "typical_wait_time_days": dermatologist.typical_wait_time_days,
            "consultation_duration_minutes": dermatologist.consultation_duration_minutes,
            "available_days": dermatologist.available_days or [],
            "available_hours": dermatologist.available_hours,
            "timezone": dermatologist.timezone,
            "average_rating": dermatologist.average_rating,
            "total_reviews": dermatologist.total_reviews,
            "total_consultations": dermatologist.total_consultations,
            "video_platform": dermatologist.video_platform,
            "booking_url": dermatologist.booking_url,
            "bio": dermatologist.bio,
            "photo_url": dermatologist.photo_url,
            "years_experience": dermatologist.years_experience,
            "medical_school": dermatologist.medical_school,
            "residency": dermatologist.residency,
            "is_verified": dermatologist.is_verified
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting dermatologist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dermatologist: {str(e)}")


@router.post("/dermatologists")
async def create_dermatologist_profile(
    full_name: str = Form(...),
    credentials: str = Form(None),
    email: str = Form(...),
    phone_number: str = Form(None),
    practice_name: str = Form(None),
    practice_address: str = Form(None),
    city: str = Form(None),
    state: str = Form(None),
    zip_code: str = Form(None),
    specializations: str = Form(None),
    languages_spoken: str = Form(None),
    accepts_video_consultations: bool = Form(True),
    accepts_referrals: bool = Form(True),
    bio: str = Form(None),
    years_experience: int = Form(None),
    db: Session = Depends(get_db)
):
    """Create a new dermatologist profile in the directory."""
    try:
        specs = json.loads(specializations) if specializations else []
        langs = json.loads(languages_spoken) if languages_spoken else ["English"]

        dermatologist = DermatologistProfile(
            full_name=full_name,
            credentials=credentials,
            email=email,
            phone_number=phone_number,
            practice_name=practice_name,
            practice_address=practice_address,
            city=city,
            state=state,
            zip_code=zip_code,
            specializations=specs,
            languages_spoken=langs,
            accepts_video_consultations=accepts_video_consultations,
            accepts_referrals=accepts_referrals,
            bio=bio,
            years_experience=years_experience,
            is_verified=False,
            is_active=True
        )

        db.add(dermatologist)
        db.commit()
        db.refresh(dermatologist)

        return {
            "message": "Dermatologist profile created successfully",
            "dermatologist_id": dermatologist.id,
            "full_name": dermatologist.full_name
        }
    except Exception as e:
        db.rollback()
        print(f"Error creating dermatologist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create dermatologist: {str(e)}")


# =============================================================================
# VIDEO CONSULTATIONS
# =============================================================================

@router.post("/consultations/book")
async def book_video_consultation(
    dermatologist_id: int = Form(...),
    scheduled_datetime: str = Form(...),
    consultation_type: str = Form("initial"),
    consultation_reason: str = Form(...),
    analysis_id: Optional[int] = Form(None),
    lesion_group_id: Optional[int] = Form(None),
    patient_notes: str = Form(None),
    patient_questions: str = Form(None),
    duration_minutes: int = Form(30),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Book a video consultation with a dermatologist."""
    try:
        dermatologist = db.query(DermatologistProfile).filter(
            DermatologistProfile.id == dermatologist_id,
            DermatologistProfile.is_active == True
        ).first()

        if not dermatologist:
            raise HTTPException(status_code=404, detail="Dermatologist not found")

        if not dermatologist.accepts_video_consultations:
            raise HTTPException(status_code=400, detail="This dermatologist does not accept video consultations")

        try:
            scheduled_dt = datetime.fromisoformat(scheduled_datetime.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO format")

        questions = json.loads(patient_questions) if patient_questions else []

        consultation = VideoConsultation(
            user_id=current_user.id,
            dermatologist_id=dermatologist_id,
            analysis_id=analysis_id,
            lesion_group_id=lesion_group_id,
            consultation_type=consultation_type,
            consultation_reason=consultation_reason,
            scheduled_datetime=scheduled_dt,
            duration_minutes=duration_minutes,
            timezone=dermatologist.timezone,
            status="scheduled",
            patient_notes=patient_notes,
            patient_questions=questions,
            video_platform=dermatologist.video_platform or "zoom"
        )

        db.add(consultation)
        db.commit()
        db.refresh(consultation)

        return {
            "message": "Consultation booked successfully",
            "consultation_id": consultation.id,
            "dermatologist_name": dermatologist.full_name,
            "scheduled_datetime": scheduled_dt.isoformat(),
            "duration_minutes": duration_minutes,
            "status": "scheduled",
            "video_platform": consultation.video_platform,
            "next_steps": [
                "You will receive a confirmation email shortly",
                "Video meeting link will be sent 24 hours before the appointment",
                "Prepare any relevant photos or documents to share during the consultation"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error booking consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to book consultation: {str(e)}")


@router.get("/consultations")
async def list_user_consultations(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all consultations for the current user."""
    try:
        query = db.query(VideoConsultation).filter(VideoConsultation.user_id == current_user.id)

        if status:
            query = query.filter(VideoConsultation.status == status)

        total = query.count()
        consultations = query.order_by(VideoConsultation.scheduled_datetime.desc()).offset(skip).limit(limit).all()

        result = []
        for c in consultations:
            dermatologist = db.query(DermatologistProfile).filter(
                DermatologistProfile.id == c.dermatologist_id
            ).first()

            result.append({
                "id": c.id,
                "dermatologist_id": c.dermatologist_id,
                "dermatologist_name": dermatologist.full_name if dermatologist else "Unknown",
                "consultation_type": c.consultation_type,
                "consultation_reason": c.consultation_reason,
                "scheduled_datetime": c.scheduled_datetime.isoformat() if c.scheduled_datetime else None,
                "duration_minutes": c.duration_minutes,
                "status": c.status,
                "video_platform": c.video_platform,
                "video_meeting_url": c.video_meeting_url if c.status in ["confirmed", "in_progress"] else None,
                "analysis_id": c.analysis_id,
                "lesion_group_id": c.lesion_group_id,
                "created_at": c.created_at.isoformat() if c.created_at else None
            })

        return {
            "consultations": result,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        print(f"Error listing consultations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list consultations: {str(e)}")


@router.get("/consultations/{consultation_id}")
async def get_consultation_details(
    consultation_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific consultation."""
    try:
        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.user_id == current_user.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        dermatologist = db.query(DermatologistProfile).filter(
            DermatologistProfile.id == consultation.dermatologist_id
        ).first()

        return {
            "id": consultation.id,
            "user_id": consultation.user_id,
            "dermatologist": {
                "id": dermatologist.id if dermatologist else None,
                "full_name": dermatologist.full_name if dermatologist else "Unknown",
                "credentials": dermatologist.credentials if dermatologist else None,
                "photo_url": dermatologist.photo_url if dermatologist else None
            },
            "consultation_type": consultation.consultation_type,
            "consultation_reason": consultation.consultation_reason,
            "scheduled_datetime": consultation.scheduled_datetime.isoformat() if consultation.scheduled_datetime else None,
            "duration_minutes": consultation.duration_minutes,
            "timezone": consultation.timezone,
            "status": consultation.status,
            "video_platform": consultation.video_platform,
            "video_meeting_url": consultation.video_meeting_url,
            "video_meeting_id": consultation.video_meeting_id,
            "video_meeting_password": consultation.video_meeting_password,
            "patient_notes": consultation.patient_notes,
            "patient_questions": consultation.patient_questions,
            "attachments": consultation.attachments,
            "analysis_id": consultation.analysis_id,
            "lesion_group_id": consultation.lesion_group_id,
            "consultation_fee": consultation.consultation_fee,
            "payment_status": consultation.payment_status,
            "patient_rating": consultation.patient_rating,
            "created_at": consultation.created_at.isoformat() if consultation.created_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get consultation: {str(e)}")


@router.put("/consultations/{consultation_id}/cancel")
async def cancel_consultation(
    consultation_id: int,
    cancellation_reason: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Cancel a scheduled consultation."""
    try:
        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.user_id == current_user.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        if consultation.status == "completed":
            raise HTTPException(status_code=400, detail="Cannot cancel completed consultation")

        consultation.status = "cancelled"
        consultation.cancellation_reason = cancellation_reason
        consultation.cancelled_at = datetime.utcnow()

        db.commit()

        return {
            "message": "Consultation cancelled successfully",
            "consultation_id": consultation_id
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error cancelling consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel consultation: {str(e)}")


@router.put("/consultations/{consultation_id}/reschedule")
async def reschedule_consultation(
    consultation_id: int,
    new_datetime: str = Form(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Reschedule a consultation to a new date/time."""
    try:
        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.user_id == current_user.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        if consultation.status in ["completed", "cancelled"]:
            raise HTTPException(status_code=400, detail="Cannot reschedule this consultation")

        try:
            new_dt = datetime.fromisoformat(new_datetime.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")

        consultation.scheduled_datetime = new_dt
        consultation.status = "rescheduled"
        db.commit()

        return {
            "message": "Consultation rescheduled successfully",
            "consultation_id": consultation_id,
            "new_datetime": new_dt.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error rescheduling consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reschedule: {str(e)}")


@router.post("/consultations/{consultation_id}/rate")
async def rate_consultation(
    consultation_id: int,
    rating: int = Form(...),
    review_text: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Rate a completed consultation."""
    try:
        if rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.user_id == current_user.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        consultation.patient_rating = rating
        consultation.patient_review = review_text
        consultation.rating_submitted_at = datetime.utcnow()

        # Update dermatologist's average rating
        if consultation.dermatologist_id:
            dermatologist = db.query(DermatologistProfile).filter(
                DermatologistProfile.id == consultation.dermatologist_id
            ).first()

            if dermatologist:
                total_reviews = dermatologist.total_reviews or 0
                current_avg = dermatologist.average_rating or 0

                new_total = total_reviews + 1
                new_avg = ((current_avg * total_reviews) + rating) / new_total

                dermatologist.total_reviews = new_total
                dermatologist.average_rating = round(new_avg, 2)

        db.commit()

        return {
            "message": "Rating submitted successfully",
            "consultation_id": consultation_id,
            "rating": rating
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error rating consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit rating: {str(e)}")


# =============================================================================
# REFERRALS
# =============================================================================

@router.post("/referrals")
async def create_referral(
    dermatologist_id: Optional[int] = Form(None),
    referral_reason: str = Form(...),
    primary_concern: str = Form(...),
    clinical_summary: str = Form(None),
    urgency_level: str = Form("routine"),
    analysis_id: Optional[int] = Form(None),
    lesion_group_id: Optional[int] = Form(None),
    referring_provider_name: str = Form(None),
    referring_provider_specialty: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new referral to a dermatologist."""
    try:
        referral = Referral(
            user_id=current_user.id,
            dermatologist_id=dermatologist_id,
            referring_provider_name=referring_provider_name,
            referring_provider_specialty=referring_provider_specialty,
            referral_reason=referral_reason,
            primary_concern=primary_concern,
            clinical_summary=clinical_summary,
            urgency_level=urgency_level,
            analysis_id=analysis_id,
            lesion_group_id=lesion_group_id,
            status="pending"
        )

        db.add(referral)
        db.commit()
        db.refresh(referral)

        return {
            "message": "Referral created successfully",
            "referral_id": referral.id,
            "status": "pending",
            "urgency_level": urgency_level
        }
    except Exception as e:
        db.rollback()
        print(f"Error creating referral: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create referral: {str(e)}")


@router.get("/referrals")
async def list_referrals(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all referrals for the current user."""
    try:
        query = db.query(Referral).filter(Referral.user_id == current_user.id)

        if status:
            query = query.filter(Referral.status == status)

        total = query.count()
        referrals = query.order_by(Referral.created_at.desc()).offset(skip).limit(limit).all()

        result = []
        for r in referrals:
            dermatologist = None
            if r.dermatologist_id:
                dermatologist = db.query(DermatologistProfile).filter(
                    DermatologistProfile.id == r.dermatologist_id
                ).first()

            result.append({
                "id": r.id,
                "dermatologist_id": r.dermatologist_id,
                "dermatologist_name": dermatologist.full_name if dermatologist else None,
                "referral_reason": r.referral_reason,
                "primary_concern": r.primary_concern,
                "urgency_level": r.urgency_level,
                "status": r.status,
                "referring_provider_name": r.referring_provider_name,
                "analysis_id": r.analysis_id,
                "lesion_group_id": r.lesion_group_id,
                "appointment_scheduled_date": r.appointment_scheduled_date.isoformat() if r.appointment_scheduled_date else None,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })

        return {
            "referrals": result,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        print(f"Error listing referrals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list referrals: {str(e)}")


@router.get("/referrals/{referral_id}")
async def get_referral_details(
    referral_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific referral."""
    try:
        referral = db.query(Referral).filter(
            Referral.id == referral_id,
            Referral.user_id == current_user.id
        ).first()

        if not referral:
            raise HTTPException(status_code=404, detail="Referral not found")

        dermatologist = None
        if referral.dermatologist_id:
            dermatologist = db.query(DermatologistProfile).filter(
                DermatologistProfile.id == referral.dermatologist_id
            ).first()

        return {
            "id": referral.id,
            "user_id": referral.user_id,
            "dermatologist": {
                "id": dermatologist.id if dermatologist else None,
                "full_name": dermatologist.full_name if dermatologist else None,
                "practice_name": dermatologist.practice_name if dermatologist else None
            } if dermatologist else None,
            "referring_provider_name": referral.referring_provider_name,
            "referring_provider_specialty": referral.referring_provider_specialty,
            "referring_provider_contact": referral.referring_provider_contact,
            "referral_date": referral.referral_date.isoformat() if referral.referral_date else None,
            "referral_reason": referral.referral_reason,
            "primary_concern": referral.primary_concern,
            "clinical_summary": referral.clinical_summary,
            "urgency_level": referral.urgency_level,
            "analysis_id": referral.analysis_id,
            "lesion_group_id": referral.lesion_group_id,
            "supporting_documents": referral.supporting_documents,
            "status": referral.status,
            "status_notes": referral.status_notes,
            "appointment_scheduled_date": referral.appointment_scheduled_date.isoformat() if referral.appointment_scheduled_date else None,
            "appointment_completed_date": referral.appointment_completed_date.isoformat() if referral.appointment_completed_date else None,
            "insurance_authorization_required": referral.insurance_authorization_required,
            "insurance_authorization_number": referral.insurance_authorization_number,
            "insurance_approved": referral.insurance_approved,
            "dermatologist_accepted": referral.dermatologist_accepted,
            "dermatologist_response": referral.dermatologist_response,
            "dermatologist_diagnosis": referral.dermatologist_diagnosis,
            "treatment_provided": referral.treatment_provided,
            "outcome_report": referral.outcome_report,
            "created_at": referral.created_at.isoformat() if referral.created_at else None,
            "updated_at": referral.updated_at.isoformat() if referral.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting referral: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get referral: {str(e)}")


@router.put("/referrals/{referral_id}/status")
async def update_referral_status(
    referral_id: int,
    status: str = Form(...),
    status_notes: str = Form(None),
    appointment_date: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update the status of a referral."""
    try:
        referral = db.query(Referral).filter(
            Referral.id == referral_id,
            Referral.user_id == current_user.id
        ).first()

        if not referral:
            raise HTTPException(status_code=404, detail="Referral not found")

        valid_statuses = ["pending", "accepted", "appointment_scheduled", "seen", "declined", "cancelled"]
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")

        referral.status = status
        referral.status_notes = status_notes
        referral.status_updated_at = datetime.utcnow()

        if appointment_date:
            try:
                referral.appointment_scheduled_date = datetime.fromisoformat(appointment_date.replace('Z', '+00:00'))
            except ValueError:
                pass

        if status == "seen":
            referral.patient_seen = True
            referral.appointment_completed_date = datetime.utcnow()

        db.commit()

        return {
            "message": "Referral status updated successfully",
            "referral_id": referral_id,
            "status": status
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error updating referral: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update referral: {str(e)}")


# =============================================================================
# CONSULTATION NOTES
# =============================================================================

@router.post("/consultation-notes")
async def create_consultation_note(
    consultation_id: Optional[int] = Form(None),
    referral_id: Optional[int] = Form(None),
    dermatologist_id: int = Form(...),
    note_type: str = Form("consultation"),
    chief_complaint: str = Form(None),
    history_of_present_illness: str = Form(None),
    physical_examination: str = Form(None),
    diagnosis: str = Form(None),
    differential_diagnoses: str = Form(None),
    treatment_plan: str = Form(None),
    prescriptions: str = Form(None),
    follow_up_recommended: bool = Form(False),
    follow_up_timeframe: str = Form(None),
    full_soap_note: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create consultation notes for a patient visit."""
    try:
        diff_diagnoses = json.loads(differential_diagnoses) if differential_diagnoses else None
        presc = json.loads(prescriptions) if prescriptions else None

        note = ConsultationNote(
            user_id=current_user.id,
            dermatologist_id=dermatologist_id,
            consultation_id=consultation_id,
            referral_id=referral_id,
            note_type=note_type,
            chief_complaint=chief_complaint,
            history_of_present_illness=history_of_present_illness,
            physical_examination=physical_examination,
            diagnosis=diagnosis,
            differential_diagnoses=diff_diagnoses,
            treatment_plan=treatment_plan,
            prescriptions=presc,
            follow_up_recommended=follow_up_recommended,
            follow_up_timeframe=follow_up_timeframe,
            full_soap_note=full_soap_note
        )

        db.add(note)
        db.commit()
        db.refresh(note)

        return {
            "message": "Consultation note created successfully",
            "note_id": note.id
        }
    except Exception as e:
        db.rollback()
        print(f"Error creating consultation note: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create note: {str(e)}")


@router.get("/consultation-notes")
async def list_consultation_notes(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all consultation notes for the current user."""
    try:
        query = db.query(ConsultationNote).filter(ConsultationNote.user_id == current_user.id)
        total = query.count()
        notes = query.order_by(ConsultationNote.created_at.desc()).offset(skip).limit(limit).all()

        return {
            "notes": [
                {
                    "id": n.id,
                    "note_type": n.note_type,
                    "diagnosis": n.diagnosis,
                    "treatment_plan": n.treatment_plan,
                    "follow_up_recommended": n.follow_up_recommended,
                    "created_at": n.created_at.isoformat() if n.created_at else None
                }
                for n in notes
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        print(f"Error listing consultation notes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list notes: {str(e)}")


@router.get("/consultation-notes/{note_id}")
async def get_consultation_note(
    note_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a specific consultation note."""
    try:
        note = db.query(ConsultationNote).filter(
            ConsultationNote.id == note_id,
            ConsultationNote.user_id == current_user.id
        ).first()

        if not note:
            raise HTTPException(status_code=404, detail="Consultation note not found")

        return {
            "id": note.id,
            "consultation_id": note.consultation_id,
            "referral_id": note.referral_id,
            "dermatologist_id": note.dermatologist_id,
            "note_type": note.note_type,
            "chief_complaint": note.chief_complaint,
            "history_of_present_illness": note.history_of_present_illness,
            "physical_examination": note.physical_examination,
            "diagnosis": note.diagnosis,
            "differential_diagnoses": note.differential_diagnoses,
            "treatment_plan": note.treatment_plan,
            "prescriptions": note.prescriptions,
            "follow_up_recommended": note.follow_up_recommended,
            "follow_up_timeframe": note.follow_up_timeframe,
            "full_soap_note": note.full_soap_note,
            "created_at": note.created_at.isoformat() if note.created_at else None,
            "updated_at": note.updated_at.isoformat() if note.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting consultation note: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get note: {str(e)}")


# =============================================================================
# SECOND OPINIONS
# =============================================================================

@router.post("/second-opinions")
async def request_second_opinion(
    original_diagnosis: str = Form(...),
    original_provider_name: str = Form(None),
    original_diagnosis_date: str = Form(None),
    original_treatment_plan: str = Form(None),
    reason_for_second_opinion: str = Form(...),
    specific_questions: str = Form(None),
    concerns: str = Form(None),
    analysis_id: Optional[int] = Form(None),
    lesion_group_id: Optional[int] = Form(None),
    dermatologist_id: Optional[int] = Form(None),
    urgency: str = Form("routine"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Request a second opinion on a diagnosis or treatment plan."""
    try:
        questions = json.loads(specific_questions) if specific_questions else []

        diagnosis_date = None
        if original_diagnosis_date:
            try:
                diagnosis_date = datetime.fromisoformat(original_diagnosis_date.replace('Z', '+00:00'))
            except ValueError:
                pass

        second_opinion = SecondOpinion(
            user_id=current_user.id,
            original_diagnosis=original_diagnosis,
            original_provider_name=original_provider_name,
            original_diagnosis_date=diagnosis_date,
            original_treatment_plan=original_treatment_plan,
            reason_for_second_opinion=reason_for_second_opinion,
            specific_questions=questions,
            concerns=concerns,
            analysis_id=analysis_id,
            lesion_group_id=lesion_group_id,
            dermatologist_id=dermatologist_id,
            urgency=urgency,
            status="submitted"
        )

        db.add(second_opinion)
        db.commit()
        db.refresh(second_opinion)

        return {
            "message": "Second opinion request submitted successfully",
            "second_opinion_id": second_opinion.id,
            "status": "submitted"
        }
    except Exception as e:
        db.rollback()
        print(f"Error creating second opinion request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create request: {str(e)}")


@router.get("/second-opinions")
async def list_second_opinions(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """List all second opinion requests for the current user."""
    try:
        query = db.query(SecondOpinion).filter(SecondOpinion.user_id == current_user.id)

        if status:
            query = query.filter(SecondOpinion.status == status)

        total = query.count()
        opinions = query.order_by(SecondOpinion.created_at.desc()).offset(skip).limit(limit).all()

        result = []
        for o in opinions:
            dermatologist = None
            if o.dermatologist_id:
                dermatologist = db.query(DermatologistProfile).filter(
                    DermatologistProfile.id == o.dermatologist_id
                ).first()

            result.append({
                "id": o.id,
                "original_diagnosis": o.original_diagnosis,
                "reason_for_second_opinion": o.reason_for_second_opinion,
                "urgency": o.urgency,
                "status": o.status,
                "dermatologist_id": o.dermatologist_id,
                "dermatologist_name": dermatologist.full_name if dermatologist else None,
                "second_opinion_diagnosis": o.second_opinion_diagnosis,
                "agrees_with_original": o.agrees_with_original_diagnosis,
                "created_at": o.created_at.isoformat() if o.created_at else None
            })

        return {
            "second_opinions": result,
            "total": total,
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        print(f"Error listing second opinions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list second opinions: {str(e)}")


@router.get("/second-opinions/{opinion_id}")
async def get_second_opinion_details(
    opinion_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a second opinion request."""
    try:
        opinion = db.query(SecondOpinion).filter(
            SecondOpinion.id == opinion_id,
            SecondOpinion.user_id == current_user.id
        ).first()

        if not opinion:
            raise HTTPException(status_code=404, detail="Second opinion request not found")

        dermatologist = None
        if opinion.dermatologist_id:
            dermatologist = db.query(DermatologistProfile).filter(
                DermatologistProfile.id == opinion.dermatologist_id
            ).first()

        return {
            "id": opinion.id,
            "user_id": opinion.user_id,
            "original_diagnosis": opinion.original_diagnosis,
            "original_provider_name": opinion.original_provider_name,
            "original_diagnosis_date": opinion.original_diagnosis_date.isoformat() if opinion.original_diagnosis_date else None,
            "original_treatment_plan": opinion.original_treatment_plan,
            "reason_for_second_opinion": opinion.reason_for_second_opinion,
            "specific_questions": opinion.specific_questions,
            "concerns": opinion.concerns,
            "analysis_id": opinion.analysis_id,
            "lesion_group_id": opinion.lesion_group_id,
            "dermatologist": {
                "id": dermatologist.id if dermatologist else None,
                "full_name": dermatologist.full_name if dermatologist else None
            } if dermatologist else None,
            "urgency": opinion.urgency,
            "status": opinion.status,
            "second_opinion_date": opinion.second_opinion_date.isoformat() if opinion.second_opinion_date else None,
            "second_opinion_diagnosis": opinion.second_opinion_diagnosis,
            "second_opinion_notes": opinion.second_opinion_notes,
            "second_opinion_treatment_plan": opinion.second_opinion_treatment_plan,
            "agrees_with_original_diagnosis": opinion.agrees_with_original_diagnosis,
            "diagnosis_confidence_level": opinion.diagnosis_confidence_level,
            "differences_from_original": opinion.differences_from_original,
            "recommended_action": opinion.recommended_action,
            "recommended_next_steps": opinion.recommended_next_steps,
            "additional_tests_needed": opinion.additional_tests_needed,
            "biopsy_recommended": opinion.biopsy_recommended,
            "patient_satisfied": opinion.patient_satisfied,
            "patient_rating": opinion.patient_rating,
            "fee": opinion.fee,
            "payment_status": opinion.payment_status,
            "created_at": opinion.created_at.isoformat() if opinion.created_at else None,
            "updated_at": opinion.updated_at.isoformat() if opinion.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting second opinion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get second opinion: {str(e)}")


@router.get("/second-opinions/pricing")
async def get_second_opinion_pricing(
    urgency: str = "routine",
    specialty: str = None,
    current_user: User = Depends(get_current_active_user)
):
    """Get pricing information for a second opinion request."""
    if not WORKFLOW_AVAILABLE:
        base_prices = {"routine": 75.0, "urgent": 150.0, "emergency": 300.0}
        base = base_prices.get(urgency, 75.0)
        return {
            "base_price": base,
            "specialty_fee": 0.0,
            "platform_fee": base * 0.15,
            "total": base * 1.15,
            "currency": "USD"
        }

    pricing = payment_service.calculate_pricing(urgency, specialty)
    return pricing


@router.post("/second-opinions/{opinion_id}/rate")
async def rate_second_opinion(
    opinion_id: int,
    rating: int = Form(...),
    satisfied: bool = Form(True),
    feedback: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Rate a completed second opinion."""
    try:
        if rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

        opinion = db.query(SecondOpinion).filter(
            SecondOpinion.id == opinion_id,
            SecondOpinion.user_id == current_user.id
        ).first()

        if not opinion:
            raise HTTPException(status_code=404, detail="Second opinion request not found")

        opinion.patient_rating = rating
        opinion.patient_satisfied = satisfied
        opinion.patient_feedback = feedback
        opinion.rating_submitted_at = datetime.utcnow()

        db.commit()

        return {
            "message": "Rating submitted successfully",
            "opinion_id": opinion_id,
            "rating": rating
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error rating second opinion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit rating: {str(e)}")
