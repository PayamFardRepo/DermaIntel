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
    Referral, ConsultationNote, SecondOpinion, UserProfile, AnalysisHistory
)
from auth import get_current_active_user, get_current_ops_user
from database import Notification

router = APIRouter(tags=["Teledermatology"])


# Helper to get current dermatologist user
async def get_current_dermatologist_user(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Verify that the current user is a dermatologist."""
    if current_user.role != "dermatologist":
        raise HTTPException(
            status_code=403,
            detail="Dermatologist access required."
        )
    # Get dermatologist profile
    profile = db.query(DermatologistProfile).filter(
        DermatologistProfile.user_id == current_user.id
    ).first()
    if not profile:
        raise HTTPException(
            status_code=404,
            detail="Dermatologist profile not found."
        )
    return {"user": current_user, "profile": profile}

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

@router.post("/consultations")
async def create_consultation_request(
    consultation_type: str = Form("initial"),
    consultation_reason: str = Form(...),
    scheduled_datetime: str = Form(...),
    duration_minutes: int = Form(30),
    dermatologist_id: Optional[int] = Form(None),
    analysis_id: Optional[int] = Form(None),
    lesion_group_id: Optional[int] = Form(None),
    patient_notes: str = Form(None),
    patient_questions: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Request a consultation. Dermatologist is optional - if not provided,
    one will be assigned automatically or the request will be queued.
    """
    try:
        try:
            scheduled_dt = datetime.fromisoformat(scheduled_datetime.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

        questions = json.loads(patient_questions) if patient_questions else []

        # If dermatologist_id provided, verify they exist and accept consultations
        dermatologist = None
        dermatologist_name = None
        if dermatologist_id:
            dermatologist = db.query(DermatologistProfile).filter(
                DermatologistProfile.id == dermatologist_id,
                DermatologistProfile.is_active == True
            ).first()
            if dermatologist:
                dermatologist_name = dermatologist.full_name
                if not dermatologist.accepts_video_consultations:
                    raise HTTPException(status_code=400, detail="This dermatologist does not accept video consultations")

        # Create consultation - pending assignment if no dermatologist
        consultation = VideoConsultation(
            user_id=current_user.id,
            dermatologist_id=dermatologist_id if dermatologist else None,
            analysis_id=analysis_id,
            lesion_group_id=lesion_group_id,
            consultation_type=consultation_type,
            consultation_reason=consultation_reason,
            scheduled_datetime=scheduled_dt,
            duration_minutes=duration_minutes,
            timezone=dermatologist.timezone if dermatologist else "America/New_York",
            status="pending_assignment" if not dermatologist else "scheduled",
            patient_notes=patient_notes,
            patient_questions=questions,
            video_platform=dermatologist.video_platform if dermatologist else "zoom"
        )

        db.add(consultation)
        db.commit()
        db.refresh(consultation)

        if dermatologist:
            return {
                "message": "Consultation booked successfully",
                "consultation_id": consultation.id,
                "dermatologist_name": dermatologist_name,
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
        else:
            return {
                "message": "Consultation request submitted successfully",
                "consultation_id": consultation.id,
                "dermatologist_name": None,
                "scheduled_datetime": scheduled_dt.isoformat(),
                "duration_minutes": duration_minutes,
                "status": "pending_assignment",
                "next_steps": [
                    "Your request has been received",
                    "A dermatologist will be assigned to your consultation shortly",
                    "You will receive a confirmation once assigned"
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error creating consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create consultation: {str(e)}")


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
                "dermatologist_name": dermatologist.full_name if dermatologist else ("Pending Assignment" if c.status == "pending_assignment" else "Unknown"),
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


# =============================================================================
# OPERATIONS DASHBOARD - Platform Management
# =============================================================================

@router.get("/ops/consultations")
async def ops_list_all_consultations(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_ops_user),
    db: Session = Depends(get_db)
):
    """
    Operations dashboard: List all consultations across all users.
    Requires ops_staff or admin role.
    """
    try:
        query = db.query(VideoConsultation)

        if status:
            query = query.filter(VideoConsultation.status == status)

        total = query.count()
        consultations = query.order_by(VideoConsultation.created_at.desc()).offset(skip).limit(limit).all()

        result = []
        for c in consultations:
            # Get patient info
            patient = db.query(User).filter(User.id == c.user_id).first()
            patient_profile = db.query(UserProfile).filter(UserProfile.user_id == c.user_id).first() if patient else None

            # Get dermatologist info
            dermatologist = db.query(DermatologistProfile).filter(
                DermatologistProfile.id == c.dermatologist_id
            ).first() if c.dermatologist_id else None

            result.append({
                "id": c.id,
                "patient": {
                    "id": patient.id if patient else None,
                    "username": patient.username if patient else "Unknown",
                    "full_name": patient.full_name if patient else "Unknown",
                    "email": patient.email if patient else None
                },
                "dermatologist": {
                    "id": dermatologist.id if dermatologist else None,
                    "full_name": dermatologist.full_name if dermatologist else None,
                    "credentials": dermatologist.credentials if dermatologist else None
                } if dermatologist else None,
                "consultation_type": c.consultation_type,
                "consultation_reason": c.consultation_reason,
                "scheduled_datetime": c.scheduled_datetime.isoformat() if c.scheduled_datetime else None,
                "duration_minutes": c.duration_minutes,
                "status": c.status,
                "created_at": c.created_at.isoformat() if c.created_at else None
            })

        # Count by status for dashboard stats
        pending_count = db.query(VideoConsultation).filter(VideoConsultation.status == "pending_assignment").count()
        scheduled_count = db.query(VideoConsultation).filter(VideoConsultation.status == "scheduled").count()
        completed_count = db.query(VideoConsultation).filter(VideoConsultation.status == "completed").count()

        return {
            "consultations": result,
            "total": total,
            "skip": skip,
            "limit": limit,
            "stats": {
                "pending_assignment": pending_count,
                "scheduled": scheduled_count,
                "completed": completed_count
            }
        }
    except Exception as e:
        print(f"Error listing ops consultations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list consultations: {str(e)}")


@router.get("/ops/dermatologists/available")
async def ops_list_available_dermatologists(
    current_user: User = Depends(get_current_ops_user),
    db: Session = Depends(get_db)
):
    """
    Operations dashboard: List all active dermatologists who can be assigned.
    Includes their current workload for informed assignment.
    Requires ops_staff or admin role.
    """
    try:
        dermatologists = db.query(DermatologistProfile).filter(
            DermatologistProfile.is_active == True,
            DermatologistProfile.accepts_video_consultations == True
        ).all()

        result = []
        for d in dermatologists:
            # Count current scheduled consultations
            scheduled_count = db.query(VideoConsultation).filter(
                VideoConsultation.dermatologist_id == d.id,
                VideoConsultation.status.in_(["scheduled", "confirmed"])
            ).count()

            # Count completed consultations (for experience)
            completed_count = db.query(VideoConsultation).filter(
                VideoConsultation.dermatologist_id == d.id,
                VideoConsultation.status == "completed"
            ).count()

            result.append({
                "id": d.id,
                "full_name": d.full_name,
                "credentials": d.credentials,
                "specializations": d.specializations,
                "average_rating": d.average_rating,
                "total_reviews": d.total_reviews,
                "years_experience": d.years_experience,
                "video_platform": d.video_platform,
                "timezone": d.timezone,
                "workload": {
                    "scheduled_consultations": scheduled_count,
                    "completed_consultations": completed_count
                }
            })

        # Sort by least workload first (for load balancing)
        result.sort(key=lambda x: x["workload"]["scheduled_consultations"])

        return {
            "dermatologists": result,
            "total": len(result)
        }
    except Exception as e:
        print(f"Error listing available dermatologists: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list dermatologists: {str(e)}")


@router.post("/ops/consultations/{consultation_id}/assign")
async def ops_assign_dermatologist(
    consultation_id: int,
    dermatologist_id: int = Form(...),
    notes: str = Form(None),
    current_user: User = Depends(get_current_ops_user),
    db: Session = Depends(get_db)
):
    """
    Operations dashboard: Assign a dermatologist to a pending consultation.
    Requires ops_staff or admin role.
    """
    try:
        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        if consultation.status not in ["pending_assignment", "scheduled"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot assign dermatologist to consultation with status '{consultation.status}'"
            )

        dermatologist = db.query(DermatologistProfile).filter(
            DermatologistProfile.id == dermatologist_id,
            DermatologistProfile.is_active == True
        ).first()

        if not dermatologist:
            raise HTTPException(status_code=404, detail="Dermatologist not found or inactive")

        if not dermatologist.accepts_video_consultations:
            raise HTTPException(status_code=400, detail="This dermatologist does not accept video consultations")

        # Assign the dermatologist
        consultation.dermatologist_id = dermatologist_id
        consultation.status = "scheduled"
        consultation.video_platform = dermatologist.video_platform or "zoom"
        consultation.timezone = dermatologist.timezone

        db.commit()

        # Get patient info for response
        patient = db.query(User).filter(User.id == consultation.user_id).first()

        # Create notification for patient
        patient_notification = Notification(
            user_id=consultation.user_id,
            notification_type="consultation_assigned",
            title="Dermatologist Assigned",
            message=f"Your consultation request has been assigned to {dermatologist.full_name}, {dermatologist.credentials}. You will receive scheduling details soon.",
            priority="high",
            data=json.dumps({
                "consultation_id": consultation_id,
                "dermatologist_id": dermatologist.id,
                "dermatologist_name": dermatologist.full_name,
                "dermatologist_credentials": dermatologist.credentials
            })
        )
        db.add(patient_notification)

        # Create notification for dermatologist (if they have a user account)
        if dermatologist.user_id:
            derm_notification = Notification(
                user_id=dermatologist.user_id,
                notification_type="new_consultation_assignment",
                title="New Patient Assigned",
                message=f"A new consultation has been assigned to you. Patient: {patient.full_name if patient else 'Unknown'}",
                priority="high",
                data=json.dumps({
                    "consultation_id": consultation_id,
                    "patient_id": patient.id if patient else None,
                    "patient_name": patient.full_name if patient else "Unknown",
                    "consultation_reason": consultation.consultation_reason
                })
            )
            db.add(derm_notification)

        db.commit()

        return {
            "message": "Dermatologist assigned successfully",
            "consultation_id": consultation_id,
            "dermatologist": {
                "id": dermatologist.id,
                "full_name": dermatologist.full_name,
                "credentials": dermatologist.credentials
            },
            "patient": {
                "id": patient.id if patient else None,
                "username": patient.username if patient else None
            },
            "status": "scheduled",
            "scheduled_datetime": consultation.scheduled_datetime.isoformat() if consultation.scheduled_datetime else None,
            "notifications_sent": True,
            "next_steps": [
                "Patient has been notified of the assignment",
                "Dermatologist has been notified of new patient",
                "Video meeting link will be generated",
                "Reminder will be sent 24 hours before appointment"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error assigning dermatologist: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assign dermatologist: {str(e)}")


@router.get("/ops/consultations/{consultation_id}")
async def ops_get_consultation_details(
    consultation_id: int,
    current_user: User = Depends(get_current_ops_user),
    db: Session = Depends(get_db)
):
    """
    Operations dashboard: Get full details of a consultation for review.
    Requires ops_staff or admin role.
    """
    try:
        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")

        # Get patient info
        patient = db.query(User).filter(User.id == consultation.user_id).first()
        patient_profile = db.query(UserProfile).filter(UserProfile.user_id == consultation.user_id).first()

        # Get dermatologist info
        dermatologist = db.query(DermatologistProfile).filter(
            DermatologistProfile.id == consultation.dermatologist_id
        ).first() if consultation.dermatologist_id else None

        # Get related analysis if any
        analysis = None
        if consultation.analysis_id:
            analysis_record = db.query(AnalysisHistory).filter(
                AnalysisHistory.id == consultation.analysis_id
            ).first()
            if analysis_record:
                analysis = {
                    "id": analysis_record.id,
                    "condition": analysis_record.condition,
                    "confidence": analysis_record.confidence,
                    "created_at": analysis_record.created_at.isoformat() if analysis_record.created_at else None
                }

        return {
            "id": consultation.id,
            "patient": {
                "id": patient.id if patient else None,
                "username": patient.username if patient else "Unknown",
                "full_name": patient.full_name if patient else "Unknown",
                "email": patient.email if patient else None,
                "phone": getattr(patient_profile, 'phone', None) if patient_profile else None
            },
            "dermatologist": {
                "id": dermatologist.id,
                "full_name": dermatologist.full_name,
                "credentials": dermatologist.credentials,
                "specializations": dermatologist.specializations
            } if dermatologist else None,
            "consultation_type": consultation.consultation_type,
            "consultation_reason": consultation.consultation_reason,
            "scheduled_datetime": consultation.scheduled_datetime.isoformat() if consultation.scheduled_datetime else None,
            "duration_minutes": consultation.duration_minutes,
            "timezone": consultation.timezone,
            "status": consultation.status,
            "video_platform": consultation.video_platform,
            "patient_notes": consultation.patient_notes,
            "patient_questions": consultation.patient_questions,
            "related_analysis": analysis,
            "created_at": consultation.created_at.isoformat() if consultation.created_at else None,
            "updated_at": consultation.updated_at.isoformat() if consultation.updated_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting consultation details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get consultation details: {str(e)}")


# =============================================================================
# DERMATOLOGIST DASHBOARD ENDPOINTS
# =============================================================================

@router.get("/my/consultations")
async def get_dermatologist_consultations(
    status: Optional[str] = None,
    derm_data: dict = Depends(get_current_dermatologist_user),
    db: Session = Depends(get_db)
):
    """
    Dermatologist dashboard: View all consultations assigned to the logged-in dermatologist.
    """
    try:
        dermatologist = derm_data["profile"]
        user = derm_data["user"]

        query = db.query(VideoConsultation).filter(
            VideoConsultation.dermatologist_id == dermatologist.id
        )

        if status:
            query = query.filter(VideoConsultation.status == status)

        consultations = query.order_by(VideoConsultation.scheduled_datetime.desc()).all()

        result = []
        for c in consultations:
            patient = db.query(User).filter(User.id == c.user_id).first()
            patient_profile = db.query(UserProfile).filter(UserProfile.user_id == c.user_id).first()

            result.append({
                "id": c.id,
                "patient": {
                    "id": patient.id if patient else None,
                    "full_name": patient.full_name if patient else "Unknown",
                    "email": patient.email if patient else None,
                    "age": patient_profile.age if patient_profile else None,
                    "skin_type": patient_profile.skin_type if patient_profile else None
                },
                "consultation_type": c.consultation_type,
                "consultation_reason": c.consultation_reason,
                "patient_notes": c.patient_notes,
                "patient_questions": c.patient_questions,
                "scheduled_datetime": c.scheduled_datetime.isoformat() if c.scheduled_datetime else None,
                "duration_minutes": c.duration_minutes,
                "status": c.status,
                "video_platform": c.video_platform,
                "meeting_link": c.meeting_link,
                "created_at": c.created_at.isoformat() if c.created_at else None
            })

        # Count by status
        all_consultations = db.query(VideoConsultation).filter(
            VideoConsultation.dermatologist_id == dermatologist.id
        ).all()

        stats = {
            "total": len(all_consultations),
            "scheduled": len([c for c in all_consultations if c.status == "scheduled"]),
            "confirmed": len([c for c in all_consultations if c.status == "confirmed"]),
            "in_progress": len([c for c in all_consultations if c.status == "in_progress"]),
            "completed": len([c for c in all_consultations if c.status == "completed"]),
            "cancelled": len([c for c in all_consultations if c.status == "cancelled"])
        }

        return {
            "consultations": result,
            "stats": stats,
            "dermatologist": {
                "id": dermatologist.id,
                "full_name": dermatologist.full_name,
                "credentials": dermatologist.credentials
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting dermatologist consultations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get consultations: {str(e)}")


@router.get("/my/consultations/{consultation_id}")
async def get_dermatologist_consultation_details(
    consultation_id: int,
    derm_data: dict = Depends(get_current_dermatologist_user),
    db: Session = Depends(get_db)
):
    """
    Dermatologist: Get full details of an assigned consultation including patient history.
    """
    try:
        dermatologist = derm_data["profile"]

        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.dermatologist_id == dermatologist.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found or not assigned to you")

        patient = db.query(User).filter(User.id == consultation.user_id).first()
        patient_profile = db.query(UserProfile).filter(UserProfile.user_id == consultation.user_id).first()

        # Get patient's recent analysis history for context
        recent_analyses = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == consultation.user_id
        ).order_by(AnalysisHistory.created_at.desc()).limit(5).all()

        analyses_summary = []
        for a in recent_analyses:
            analyses_summary.append({
                "id": a.id,
                "predicted_class": a.predicted_class,
                "confidence": a.confidence,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "body_location": a.body_location
            })

        return {
            "consultation": {
                "id": consultation.id,
                "consultation_type": consultation.consultation_type,
                "consultation_reason": consultation.consultation_reason,
                "patient_notes": consultation.patient_notes,
                "patient_questions": consultation.patient_questions,
                "scheduled_datetime": consultation.scheduled_datetime.isoformat() if consultation.scheduled_datetime else None,
                "duration_minutes": consultation.duration_minutes,
                "status": consultation.status,
                "video_platform": consultation.video_platform,
                "meeting_link": consultation.meeting_link,
                "video_platform_url": consultation.video_platform_url,
                "created_at": consultation.created_at.isoformat() if consultation.created_at else None
            },
            "patient": {
                "id": patient.id if patient else None,
                "full_name": patient.full_name if patient else "Unknown",
                "email": patient.email if patient else None,
                "age": patient_profile.age if patient_profile else None,
                "skin_type": patient_profile.skin_type if patient_profile else None,
                "skin_concerns": patient_profile.skin_concerns if patient_profile else None,
                "allergies": patient_profile.allergies if patient_profile else None,
                "medical_conditions": patient_profile.medical_conditions if patient_profile else None,
                "current_medications": patient_profile.current_medications if patient_profile else None
            },
            "recent_analyses": analyses_summary
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting consultation details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get consultation details: {str(e)}")


@router.post("/my/consultations/{consultation_id}/start")
async def start_consultation(
    consultation_id: int,
    derm_data: dict = Depends(get_current_dermatologist_user),
    db: Session = Depends(get_db)
):
    """
    Dermatologist: Start a scheduled consultation (changes status to in_progress).
    """
    try:
        dermatologist = derm_data["profile"]

        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.dermatologist_id == dermatologist.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found or not assigned to you")

        if consultation.status not in ["scheduled", "confirmed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot start consultation with status '{consultation.status}'"
            )

        consultation.status = "in_progress"
        consultation.actual_start_time = datetime.utcnow()
        db.commit()

        return {
            "message": "Consultation started",
            "consultation_id": consultation_id,
            "status": "in_progress",
            "started_at": consultation.actual_start_time.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error starting consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start consultation: {str(e)}")


@router.post("/my/consultations/{consultation_id}/complete")
async def complete_consultation(
    consultation_id: int,
    notes: str = Form(None),
    diagnosis: str = Form(None),
    recommendations: str = Form(None),
    follow_up_needed: bool = Form(False),
    derm_data: dict = Depends(get_current_dermatologist_user),
    db: Session = Depends(get_db)
):
    """
    Dermatologist: Complete a consultation and add notes.
    """
    try:
        dermatologist = derm_data["profile"]

        consultation = db.query(VideoConsultation).filter(
            VideoConsultation.id == consultation_id,
            VideoConsultation.dermatologist_id == dermatologist.id
        ).first()

        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found or not assigned to you")

        if consultation.status not in ["in_progress", "scheduled", "confirmed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot complete consultation with status '{consultation.status}'"
            )

        consultation.status = "completed"
        consultation.dermatologist_notes = notes
        consultation.dermatologist_diagnosis = diagnosis
        consultation.dermatologist_recommendations = recommendations
        consultation.follow_up_needed = follow_up_needed
        consultation.actual_end_time = datetime.utcnow()
        db.commit()

        # Create notification for patient
        patient_notification = Notification(
            user_id=consultation.user_id,
            notification_type="consultation_completed",
            title="Consultation Completed",
            message=f"Your consultation with {dermatologist.full_name} has been completed. View the notes and recommendations.",
            priority="normal",
            data=json.dumps({
                "consultation_id": consultation_id,
                "dermatologist_name": dermatologist.full_name,
                "follow_up_needed": follow_up_needed
            })
        )
        db.add(patient_notification)
        db.commit()

        return {
            "message": "Consultation completed",
            "consultation_id": consultation_id,
            "status": "completed",
            "completed_at": consultation.actual_end_time.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error completing consultation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to complete consultation: {str(e)}")


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
    """Get detailed information about a specific referral, including AI analysis data."""
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

        # Fetch AI analysis data if analysis_id is present
        ai_analysis = None
        if referral.analysis_id:
            analysis = db.query(AnalysisHistory).filter(
                AnalysisHistory.id == referral.analysis_id
            ).first()
            if analysis:
                ai_analysis = {
                    "id": analysis.id,
                    "image_url": analysis.image_url,
                    "predicted_class": analysis.predicted_class,
                    "lesion_confidence": analysis.lesion_confidence,
                    "risk_level": analysis.risk_level,
                    "body_location": analysis.body_location,
                    "analysis_date": analysis.created_at.isoformat() if analysis.created_at else None,
                    # AI Diagnostic Reasoning
                    "differential_diagnoses": analysis.differential_diagnoses,
                    "clinical_decision_support": analysis.clinical_decision_support,
                    "treatment_recommendations": analysis.treatment_recommendations,
                    # ABCDE Analysis
                    "abcde_analysis": getattr(analysis, 'red_flag_data', None),
                    # Multimodal insights
                    "multimodal_risk_factors": getattr(analysis, 'multimodal_risk_factors', None),
                    "multimodal_recommendations": getattr(analysis, 'multimodal_recommendations', None),
                    # Confidence breakdown
                    "confidence_adjustments": getattr(analysis, 'confidence_adjustments', None),
                    "data_sources_used": getattr(analysis, 'data_sources_used', None),
                }

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
            # Include full AI analysis data for dermatologist review
            "ai_analysis": ai_analysis,
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
    """Get detailed information about a second opinion request, including AI analysis data."""
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

        # Fetch AI analysis data if analysis_id is present
        ai_analysis = None
        if opinion.analysis_id:
            analysis = db.query(AnalysisHistory).filter(
                AnalysisHistory.id == opinion.analysis_id
            ).first()
            if analysis:
                ai_analysis = {
                    "id": analysis.id,
                    "image_url": analysis.image_url,
                    "predicted_class": analysis.predicted_class,
                    "lesion_confidence": analysis.lesion_confidence,
                    "risk_level": analysis.risk_level,
                    "body_location": analysis.body_location,
                    "analysis_date": analysis.created_at.isoformat() if analysis.created_at else None,
                    # AI Diagnostic Reasoning
                    "differential_diagnoses": analysis.differential_diagnoses,
                    "clinical_decision_support": analysis.clinical_decision_support,
                    "treatment_recommendations": analysis.treatment_recommendations,
                    # ABCDE Analysis
                    "abcde_analysis": getattr(analysis, 'red_flag_data', None),
                    # Multimodal insights
                    "multimodal_risk_factors": getattr(analysis, 'multimodal_risk_factors', None),
                    "multimodal_recommendations": getattr(analysis, 'multimodal_recommendations', None),
                    # Confidence breakdown
                    "confidence_adjustments": getattr(analysis, 'confidence_adjustments', None),
                    "data_sources_used": getattr(analysis, 'data_sources_used', None),
                }

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
            # Include full AI analysis data for dermatologist review
            "ai_analysis": ai_analysis,
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
