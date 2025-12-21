"""
Appointments and Providers Router

Endpoints for:
- Appointment booking and management
- Provider availability
- Waitlist management
- Appointment reminders
"""

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, date as dt_date
from typing import Optional
import uuid

from database import get_db, User
from auth import get_current_active_user
from appointment_scheduler import (
    AppointmentScheduler, AppointmentType, AppointmentStatus,
    APPOINTMENT_CONFIGS, get_appointment_types_list
)

router = APIRouter(tags=["Appointments"])

# Initialize the appointment scheduler
appointment_scheduler = AppointmentScheduler()


# =============================================================================
# APPOINTMENT TYPES
# =============================================================================

@router.get("/appointments/types")
async def get_appointment_types():
    """Get all available appointment types with configurations."""
    return {
        "appointment_types": get_appointment_types_list()
    }


# =============================================================================
# AVAILABLE SLOTS
# =============================================================================

@router.get("/appointments/slots")
async def get_available_appointment_slots(
    start_date: str,
    end_date: str,
    appointment_type: str,
    provider_id: Optional[str] = None,
    is_telemedicine: bool = False,
    current_user: User = Depends(get_current_active_user)
):
    """Get available appointment slots within a date range."""
    try:
        apt_type = AppointmentType(appointment_type)
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        slots = appointment_scheduler.get_available_slots(
            start_date=start,
            end_date=end,
            appointment_type=apt_type,
            provider_id=provider_id,
            is_telemedicine=is_telemedicine
        )

        return {
            "total_slots": len(slots),
            "slots": [
                {
                    "start_time": slot.start_time.isoformat(),
                    "end_time": slot.end_time.isoformat(),
                    "date": slot.start_time.date().isoformat(),
                    "time": slot.start_time.time().isoformat(),
                    "provider_id": slot.provider_id,
                    "provider_name": slot.provider_name,
                    "is_telemedicine": slot.is_telemedicine,
                    "location": slot.location
                }
                for slot in slots
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# APPOINTMENT BOOKING
# =============================================================================

@router.post("/appointments/book")
async def book_appointment(
    request: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Book a new appointment."""
    from database import Appointment as DBAppointment

    try:
        apt_type = AppointmentType(request.get("appointment_type"))
        apt_date = datetime.strptime(request.get("appointment_date"), "%Y-%m-%d").date()
        start_time_str = request.get("start_time")
        start_time = datetime.strptime(start_time_str, "%H:%M").time()

        config = APPOINTMENT_CONFIGS[apt_type]
        duration = config["duration_minutes"]

        start_dt = datetime.combine(apt_date, start_time)
        end_dt = start_dt + timedelta(minutes=duration)
        end_time = end_dt.time()

        appointment_id = f"APT-{uuid.uuid4().hex[:12].upper()}"

        provider_id = request.get("provider_id", "PROV-001")
        provider_name = "Dr. Sarah Johnson, MD, FAAD"

        is_telemedicine = request.get("is_telemedicine", False)
        telemedicine_link = None
        location = "Main Clinic - 123 Medical Center Dr"

        if is_telemedicine:
            telemedicine_link = f"https://telehealth.example.com/join/{appointment_id}"
            location = None

        patient_name = current_user.full_name or current_user.username
        patient_email = current_user.email

        new_appointment = DBAppointment(
            user_id=current_user.id,
            appointment_id=appointment_id,
            provider_id=provider_id,
            provider_name=provider_name,
            appointment_date=apt_date,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration,
            appointment_type=apt_type.value,
            status="scheduled",
            is_telemedicine=is_telemedicine,
            location=location,
            telemedicine_link=telemedicine_link,
            patient_name=patient_name,
            patient_email=patient_email,
            patient_phone=request.get("patient_phone"),
            reason_for_visit=request.get("reason_for_visit", ""),
            patient_notes=request.get("patient_notes"),
            related_analysis_ids=request.get("related_analysis_ids", []),
            related_lesion_ids=request.get("related_lesion_ids", []),
            reminder_settings={
                "24_hours_before": True,
                "2_hours_before": True,
                "email": True,
                "push": True
            }
        )

        db.add(new_appointment)
        db.commit()
        db.refresh(new_appointment)

        return {
            "success": True,
            "appointment_id": appointment_id,
            "appointment_date": apt_date.isoformat(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration,
            "appointment_type": apt_type.value,
            "provider_name": provider_name,
            "is_telemedicine": is_telemedicine,
            "telemedicine_link": telemedicine_link,
            "location": location,
            "status": "scheduled",
            "preparation": config["preparation"],
            "message": "Appointment booked successfully!"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to book appointment: {str(e)}")


# =============================================================================
# APPOINTMENT MANAGEMENT
# =============================================================================

@router.get("/appointments")
async def get_user_appointments(
    include_past: bool = False,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all appointments for the current user."""
    from database import Appointment as DBAppointment

    try:
        query = db.query(DBAppointment).filter(DBAppointment.user_id == current_user.id)

        if not include_past:
            query = query.filter(DBAppointment.appointment_date >= dt_date.today())

        if status:
            query = query.filter(DBAppointment.status == status)

        appointments = query.order_by(
            DBAppointment.appointment_date,
            DBAppointment.start_time
        ).all()

        return {
            "total": len(appointments),
            "appointments": [
                {
                    "appointment_id": apt.appointment_id,
                    "appointment_date": apt.appointment_date.isoformat() if apt.appointment_date else None,
                    "start_time": apt.start_time.isoformat() if apt.start_time else None,
                    "end_time": apt.end_time.isoformat() if apt.end_time else None,
                    "duration_minutes": apt.duration_minutes,
                    "appointment_type": apt.appointment_type,
                    "status": apt.status,
                    "provider_name": apt.provider_name,
                    "is_telemedicine": apt.is_telemedicine,
                    "telemedicine_link": apt.telemedicine_link,
                    "location": apt.location,
                    "reason_for_visit": apt.reason_for_visit,
                    "created_at": apt.created_at.isoformat() if apt.created_at else None
                }
                for apt in appointments
            ]
        }
    except Exception as e:
        return {"total": 0, "appointments": []}


@router.get("/appointments/{appointment_id}")
async def get_appointment_detail(
    appointment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific appointment."""
    from database import Appointment as DBAppointment

    appointment = db.query(DBAppointment).filter(
        DBAppointment.appointment_id == appointment_id,
        DBAppointment.user_id == current_user.id
    ).first()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    return {
        "appointment_id": appointment.appointment_id,
        "appointment_date": appointment.appointment_date.isoformat() if appointment.appointment_date else None,
        "start_time": appointment.start_time.isoformat() if appointment.start_time else None,
        "end_time": appointment.end_time.isoformat() if appointment.end_time else None,
        "duration_minutes": appointment.duration_minutes,
        "appointment_type": appointment.appointment_type,
        "status": appointment.status,
        "provider_id": appointment.provider_id,
        "provider_name": appointment.provider_name,
        "is_telemedicine": appointment.is_telemedicine,
        "telemedicine_link": appointment.telemedicine_link,
        "location": appointment.location,
        "reason_for_visit": appointment.reason_for_visit,
        "patient_notes": appointment.patient_notes,
        "related_analysis_ids": appointment.related_analysis_ids,
        "related_lesion_ids": appointment.related_lesion_ids,
        "reminder_settings": appointment.reminder_settings,
        "created_at": appointment.created_at.isoformat() if appointment.created_at else None
    }


@router.post("/appointments/{appointment_id}/cancel")
async def cancel_appointment(
    appointment_id: str,
    cancellation_reason: str = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Cancel an appointment."""
    from database import Appointment as DBAppointment

    appointment = db.query(DBAppointment).filter(
        DBAppointment.appointment_id == appointment_id,
        DBAppointment.user_id == current_user.id
    ).first()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    if appointment.status == "cancelled":
        raise HTTPException(status_code=400, detail="Appointment already cancelled")

    appointment.status = "cancelled"
    appointment.cancellation_reason = cancellation_reason
    appointment.cancelled_at = datetime.utcnow()
    db.commit()

    return {
        "success": True,
        "message": "Appointment cancelled successfully",
        "appointment_id": appointment_id
    }


@router.post("/appointments/{appointment_id}/confirm")
async def confirm_appointment(
    appointment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Confirm an appointment."""
    from database import Appointment as DBAppointment

    appointment = db.query(DBAppointment).filter(
        DBAppointment.appointment_id == appointment_id,
        DBAppointment.user_id == current_user.id
    ).first()

    if not appointment:
        raise HTTPException(status_code=404, detail="Appointment not found")

    appointment.status = "confirmed"
    appointment.confirmed_at = datetime.utcnow()
    db.commit()

    return {
        "success": True,
        "message": "Appointment confirmed",
        "appointment_id": appointment_id
    }


# =============================================================================
# PROVIDERS
# =============================================================================

@router.get("/providers")
async def get_providers(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of available providers."""
    from database import ProviderAvailability

    providers = db.query(
        ProviderAvailability.provider_id,
        ProviderAvailability.provider_name,
        ProviderAvailability.location
    ).distinct(ProviderAvailability.provider_id).filter(
        ProviderAvailability.is_active == True
    ).all()

    if not providers:
        return {
            "providers": [
                {
                    "provider_id": "PROV-001",
                    "provider_name": "Dr. Sarah Johnson, MD, FAAD",
                    "location": "Main Clinic - 123 Medical Center Dr",
                    "specialties": ["General Dermatology", "Skin Cancer", "Cosmetic Dermatology"],
                    "accepts_telemedicine": True
                }
            ]
        }

    return {
        "providers": [
            {
                "provider_id": p.provider_id,
                "provider_name": p.provider_name,
                "location": p.location
            }
            for p in providers
        ]
    }


@router.get("/providers/{provider_id}/availability")
async def get_provider_availability(
    provider_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get a provider's weekly availability schedule."""
    from database import ProviderAvailability, ProviderBlockedTime

    availability = db.query(ProviderAvailability).filter(
        ProviderAvailability.provider_id == provider_id,
        ProviderAvailability.is_active == True
    ).all()

    blocked_times = db.query(ProviderBlockedTime).filter(
        ProviderBlockedTime.provider_id == provider_id,
        ProviderBlockedTime.end_datetime > datetime.utcnow()
    ).all()

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    return {
        "provider_id": provider_id,
        "weekly_schedule": [
            {
                "day": days[a.day_of_week],
                "day_of_week": a.day_of_week,
                "start_time": a.start_time.isoformat() if a.start_time else None,
                "end_time": a.end_time.isoformat() if a.end_time else None,
                "break_start": a.break_start.isoformat() if a.break_start else None,
                "break_end": a.break_end.isoformat() if a.break_end else None,
                "location": a.location,
                "is_telemedicine_day": a.is_telemedicine_day
            }
            for a in sorted(availability, key=lambda x: x.day_of_week)
        ],
        "blocked_times": [
            {
                "id": b.id,
                "start": b.start_datetime.isoformat(),
                "end": b.end_datetime.isoformat(),
                "reason": b.reason,
                "notes": b.notes
            }
            for b in blocked_times
        ]
    }
