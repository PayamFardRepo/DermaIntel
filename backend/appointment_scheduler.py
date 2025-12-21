"""
Appointment Scheduling System

Comprehensive appointment management for dermatology consultations with:
- Appointment booking and management
- Automated reminders (email/SMS placeholders)
- Calendar integration (iCal/Google Calendar export)
- Provider availability management
- Appointment types and durations
- Waitlist management
- Cancellation and rescheduling
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import uuid
import json
from icalendar import Calendar, Event, Alarm
import pytz


class AppointmentType(Enum):
    """Types of dermatology appointments"""
    INITIAL_CONSULTATION = "initial_consultation"
    FOLLOW_UP = "follow_up"
    SKIN_CHECK = "skin_check"
    MOLE_MAPPING = "mole_mapping"
    BIOPSY = "biopsy"
    SURGERY = "surgery"
    COSMETIC = "cosmetic"
    TELEMEDICINE = "telemedicine"
    URGENT = "urgent"
    PROCEDURE = "procedure"


class AppointmentStatus(Enum):
    """Status of an appointment"""
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"
    WAITLISTED = "waitlisted"


class ReminderType(Enum):
    """Types of reminders"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class RecurrencePattern(Enum):
    """Recurrence patterns for follow-up appointments"""
    NONE = "none"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    BIANNUAL = "biannual"
    ANNUAL = "annual"


# Appointment type configurations
APPOINTMENT_CONFIGS = {
    AppointmentType.INITIAL_CONSULTATION: {
        "duration_minutes": 30,
        "description": "Initial dermatology consultation for new patients",
        "preparation": [
            "Bring list of current medications",
            "Bring photos of skin concerns if available",
            "Arrive 15 minutes early to complete paperwork"
        ],
        "allows_telemedicine": True
    },
    AppointmentType.FOLLOW_UP: {
        "duration_minutes": 15,
        "description": "Follow-up visit to review treatment progress",
        "preparation": [
            "Note any changes since last visit",
            "Bring current medications"
        ],
        "allows_telemedicine": True
    },
    AppointmentType.SKIN_CHECK: {
        "duration_minutes": 20,
        "description": "Full body skin examination for skin cancer screening",
        "preparation": [
            "Remove nail polish from fingers and toes",
            "Wear loose, comfortable clothing",
            "Note any new or changing spots"
        ],
        "allows_telemedicine": False
    },
    AppointmentType.MOLE_MAPPING: {
        "duration_minutes": 45,
        "description": "Comprehensive mole documentation and photography",
        "preparation": [
            "Avoid applying lotions or makeup",
            "Wear minimal jewelry",
            "Allow extra time for photography"
        ],
        "allows_telemedicine": False
    },
    AppointmentType.BIOPSY: {
        "duration_minutes": 30,
        "description": "Skin biopsy procedure for diagnostic purposes",
        "preparation": [
            "Stop blood thinners if advised by doctor",
            "Eat normally before appointment",
            "Arrange transportation if needed"
        ],
        "allows_telemedicine": False
    },
    AppointmentType.SURGERY: {
        "duration_minutes": 60,
        "description": "Surgical procedure for skin lesion removal",
        "preparation": [
            "Follow pre-operative instructions",
            "Arrange for someone to drive you home",
            "Wear comfortable, loose clothing"
        ],
        "allows_telemedicine": False
    },
    AppointmentType.COSMETIC: {
        "duration_minutes": 30,
        "description": "Cosmetic dermatology consultation or procedure",
        "preparation": [
            "Avoid sun exposure before appointment",
            "Stop retinoids if advised",
            "Come with clean skin"
        ],
        "allows_telemedicine": True
    },
    AppointmentType.TELEMEDICINE: {
        "duration_minutes": 20,
        "description": "Virtual dermatology consultation",
        "preparation": [
            "Ensure good lighting for video call",
            "Have clear photos of skin concerns ready",
            "Test your camera and microphone",
            "Find a private, quiet location"
        ],
        "allows_telemedicine": True
    },
    AppointmentType.URGENT: {
        "duration_minutes": 30,
        "description": "Urgent dermatology visit for acute concerns",
        "preparation": [
            "Document when symptoms started",
            "List any treatments tried",
            "Note any allergies"
        ],
        "allows_telemedicine": True
    },
    AppointmentType.PROCEDURE: {
        "duration_minutes": 45,
        "description": "In-office dermatological procedure",
        "preparation": [
            "Follow specific procedure instructions",
            "Inform us of any medication changes",
            "Arrange transportation if needed"
        ],
        "allows_telemedicine": False
    }
}


@dataclass
class TimeSlot:
    """Represents an available time slot"""
    start_time: datetime
    end_time: datetime
    provider_id: Optional[str] = None
    provider_name: Optional[str] = None
    is_available: bool = True
    appointment_types: List[AppointmentType] = field(default_factory=list)
    is_telemedicine: bool = False
    location: Optional[str] = None


@dataclass
class Provider:
    """Healthcare provider information"""
    provider_id: str
    name: str
    title: str  # MD, DO, NP, PA
    specialties: List[str]
    available_appointment_types: List[AppointmentType]
    default_location: str
    telemedicine_enabled: bool = True
    timezone: str = "America/New_York"

    # Working hours (day of week -> list of (start_hour, end_hour) tuples)
    working_hours: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)

    # Blocked dates (vacations, conferences, etc.)
    blocked_dates: List[date] = field(default_factory=list)

    # Lunch breaks and other regular breaks
    break_times: List[Tuple[time, time]] = field(default_factory=list)


@dataclass
class Appointment:
    """Represents a scheduled appointment"""
    appointment_id: str
    user_id: int
    provider_id: Optional[str]

    # Timing
    appointment_date: date
    start_time: time
    end_time: time
    duration_minutes: int

    # Type and status (appointment_type must come before fields with defaults)
    appointment_type: AppointmentType

    timezone: str = "America/New_York"
    status: AppointmentStatus = AppointmentStatus.SCHEDULED
    is_telemedicine: bool = False

    # Location
    location: Optional[str] = None
    room: Optional[str] = None
    telemedicine_link: Optional[str] = None

    # Patient information
    patient_name: str = ""
    patient_phone: Optional[str] = None
    patient_email: Optional[str] = None

    # Reason and notes
    reason_for_visit: str = ""
    patient_notes: Optional[str] = None
    provider_notes: Optional[str] = None

    # Related records
    related_analysis_ids: List[int] = field(default_factory=list)
    related_lesion_ids: List[str] = field(default_factory=list)

    # Reminders
    reminder_settings: Dict[str, Any] = field(default_factory=dict)
    reminders_sent: List[Dict[str, Any]] = field(default_factory=list)

    # Recurrence
    recurrence_pattern: RecurrencePattern = RecurrencePattern.NONE
    recurrence_end_date: Optional[date] = None
    parent_appointment_id: Optional[str] = None

    # Insurance
    insurance_verified: bool = False
    copay_amount: Optional[float] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    confirmed_at: Optional[datetime] = None
    checked_in_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    cancellation_reason: Optional[str] = None


@dataclass
class WaitlistEntry:
    """Entry in the appointment waitlist"""
    waitlist_id: str
    user_id: int
    provider_id: Optional[str]
    appointment_type: AppointmentType
    preferred_dates: List[date]
    preferred_times: List[str]  # "morning", "afternoon", "evening"
    flexibility: str  # "flexible", "somewhat_flexible", "specific"
    reason: str
    priority: int = 5  # 1-10, 1 being highest priority
    created_at: datetime = field(default_factory=datetime.utcnow)
    notified_slots: List[str] = field(default_factory=list)
    status: str = "active"  # active, fulfilled, expired, cancelled


class AppointmentScheduler:
    """
    Manages appointment scheduling, reminders, and calendar integration
    """

    def __init__(self, default_timezone: str = "America/New_York"):
        self.timezone = pytz.timezone(default_timezone)
        self.providers: Dict[str, Provider] = {}
        self.appointments: Dict[str, Appointment] = {}
        self.waitlist: Dict[str, WaitlistEntry] = {}

        # Initialize with a default provider
        self._setup_default_provider()

    def _setup_default_provider(self):
        """Setup a default dermatology provider"""
        default_provider = Provider(
            provider_id="PROV-001",
            name="Dr. Sarah Johnson",
            title="MD, FAAD",
            specialties=["General Dermatology", "Skin Cancer", "Cosmetic Dermatology"],
            available_appointment_types=list(AppointmentType),
            default_location="Main Clinic - 123 Medical Center Dr",
            telemedicine_enabled=True,
            working_hours={
                0: [(9, 12), (13, 17)],  # Monday
                1: [(9, 12), (13, 17)],  # Tuesday
                2: [(9, 12), (13, 17)],  # Wednesday
                3: [(9, 12), (13, 17)],  # Thursday
                4: [(9, 12), (13, 16)],  # Friday
            },
            break_times=[(time(12, 0), time(13, 0))]  # Lunch break
        )
        self.providers[default_provider.provider_id] = default_provider

    def get_appointment_types(self) -> List[Dict[str, Any]]:
        """Get all available appointment types with their configurations"""
        return [
            {
                "type": apt_type.value,
                "name": apt_type.value.replace("_", " ").title(),
                "duration_minutes": config["duration_minutes"],
                "description": config["description"],
                "preparation": config["preparation"],
                "allows_telemedicine": config["allows_telemedicine"]
            }
            for apt_type, config in APPOINTMENT_CONFIGS.items()
        ]

    def get_available_slots(
        self,
        start_date: date,
        end_date: date,
        appointment_type: AppointmentType,
        provider_id: Optional[str] = None,
        is_telemedicine: bool = False
    ) -> List[TimeSlot]:
        """
        Get available appointment slots within a date range
        """
        if provider_id and provider_id not in self.providers:
            return []

        providers_to_check = [self.providers[provider_id]] if provider_id else list(self.providers.values())
        config = APPOINTMENT_CONFIGS[appointment_type]
        duration = timedelta(minutes=config["duration_minutes"])

        # Check if telemedicine is allowed for this appointment type
        if is_telemedicine and not config["allows_telemedicine"]:
            return []

        available_slots = []
        current_date = start_date

        while current_date <= end_date:
            day_of_week = current_date.weekday()

            for provider in providers_to_check:
                # Skip if provider doesn't offer this appointment type
                if appointment_type not in provider.available_appointment_types:
                    continue

                # Skip blocked dates
                if current_date in provider.blocked_dates:
                    continue

                # Skip if telemedicine requested but provider doesn't support it
                if is_telemedicine and not provider.telemedicine_enabled:
                    continue

                # Get working hours for this day
                if day_of_week not in provider.working_hours:
                    continue

                for start_hour, end_hour in provider.working_hours[day_of_week]:
                    current_time = datetime.combine(current_date, time(start_hour, 0))
                    end_time = datetime.combine(current_date, time(end_hour, 0))

                    while current_time + duration <= end_time:
                        slot_end = current_time + duration

                        # Check if this slot is during a break
                        is_during_break = False
                        for break_start, break_end in provider.break_times:
                            break_start_dt = datetime.combine(current_date, break_start)
                            break_end_dt = datetime.combine(current_date, break_end)
                            if current_time < break_end_dt and slot_end > break_start_dt:
                                is_during_break = True
                                break

                        if not is_during_break:
                            # Check if slot is already booked
                            is_booked = self._is_slot_booked(
                                provider.provider_id,
                                current_date,
                                current_time.time(),
                                slot_end.time()
                            )

                            if not is_booked:
                                available_slots.append(TimeSlot(
                                    start_time=current_time,
                                    end_time=slot_end,
                                    provider_id=provider.provider_id,
                                    provider_name=f"{provider.name}, {provider.title}",
                                    is_available=True,
                                    appointment_types=[appointment_type],
                                    is_telemedicine=is_telemedicine,
                                    location=None if is_telemedicine else provider.default_location
                                ))

                        current_time += timedelta(minutes=15)  # 15-minute slot intervals

            current_date += timedelta(days=1)

        return available_slots

    def _is_slot_booked(
        self,
        provider_id: str,
        appointment_date: date,
        start_time: time,
        end_time: time
    ) -> bool:
        """Check if a time slot is already booked"""
        for appointment in self.appointments.values():
            if (appointment.provider_id == provider_id and
                appointment.appointment_date == appointment_date and
                appointment.status not in [AppointmentStatus.CANCELLED, AppointmentStatus.RESCHEDULED]):

                # Check for overlap
                apt_start = appointment.start_time
                apt_end = appointment.end_time

                if start_time < apt_end and end_time > apt_start:
                    return True

        return False

    def book_appointment(
        self,
        user_id: int,
        appointment_type: AppointmentType,
        appointment_date: date,
        start_time: time,
        provider_id: Optional[str] = None,
        is_telemedicine: bool = False,
        patient_name: str = "",
        patient_phone: Optional[str] = None,
        patient_email: Optional[str] = None,
        reason_for_visit: str = "",
        patient_notes: Optional[str] = None,
        related_analysis_ids: Optional[List[int]] = None,
        related_lesion_ids: Optional[List[str]] = None,
        reminder_preferences: Optional[Dict[str, bool]] = None
    ) -> Appointment:
        """
        Book a new appointment
        """
        config = APPOINTMENT_CONFIGS[appointment_type]
        duration_minutes = config["duration_minutes"]

        # Calculate end time
        start_datetime = datetime.combine(appointment_date, start_time)
        end_datetime = start_datetime + timedelta(minutes=duration_minutes)
        end_time = end_datetime.time()

        # Use default provider if not specified
        if not provider_id:
            provider_id = list(self.providers.keys())[0] if self.providers else None

        # Verify slot is available
        if provider_id and self._is_slot_booked(provider_id, appointment_date, start_time, end_time):
            raise ValueError("Selected time slot is no longer available")

        # Generate appointment ID
        appointment_id = f"APT-{uuid.uuid4().hex[:12].upper()}"

        # Generate telemedicine link if needed
        telemedicine_link = None
        if is_telemedicine:
            telemedicine_link = f"https://telehealth.example.com/join/{appointment_id}"

        # Setup default reminders
        default_reminders = {
            "24_hours_before": True,
            "2_hours_before": True,
            "email": True,
            "sms": patient_phone is not None,
            "push": True
        }
        if reminder_preferences:
            default_reminders.update(reminder_preferences)

        # Get provider location
        location = None
        if provider_id and provider_id in self.providers:
            location = self.providers[provider_id].default_location

        appointment = Appointment(
            appointment_id=appointment_id,
            user_id=user_id,
            provider_id=provider_id,
            appointment_date=appointment_date,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration_minutes,
            appointment_type=appointment_type,
            status=AppointmentStatus.SCHEDULED,
            is_telemedicine=is_telemedicine,
            location=None if is_telemedicine else location,
            telemedicine_link=telemedicine_link,
            patient_name=patient_name,
            patient_phone=patient_phone,
            patient_email=patient_email,
            reason_for_visit=reason_for_visit,
            patient_notes=patient_notes,
            related_analysis_ids=related_analysis_ids or [],
            related_lesion_ids=related_lesion_ids or [],
            reminder_settings=default_reminders
        )

        self.appointments[appointment_id] = appointment

        return appointment

    def confirm_appointment(self, appointment_id: str) -> Appointment:
        """Confirm an appointment"""
        if appointment_id not in self.appointments:
            raise ValueError("Appointment not found")

        appointment = self.appointments[appointment_id]
        appointment.status = AppointmentStatus.CONFIRMED
        appointment.confirmed_at = datetime.utcnow()
        appointment.updated_at = datetime.utcnow()

        return appointment

    def cancel_appointment(
        self,
        appointment_id: str,
        reason: str,
        notify_waitlist: bool = True
    ) -> Appointment:
        """Cancel an appointment"""
        if appointment_id not in self.appointments:
            raise ValueError("Appointment not found")

        appointment = self.appointments[appointment_id]
        appointment.status = AppointmentStatus.CANCELLED
        appointment.cancelled_at = datetime.utcnow()
        appointment.cancellation_reason = reason
        appointment.updated_at = datetime.utcnow()

        # Notify waitlist if enabled
        if notify_waitlist:
            self._notify_waitlist_of_opening(appointment)

        return appointment

    def reschedule_appointment(
        self,
        appointment_id: str,
        new_date: date,
        new_start_time: time,
        reason: Optional[str] = None
    ) -> Appointment:
        """Reschedule an existing appointment"""
        if appointment_id not in self.appointments:
            raise ValueError("Appointment not found")

        old_appointment = self.appointments[appointment_id]

        # Mark old appointment as rescheduled
        old_appointment.status = AppointmentStatus.RESCHEDULED
        old_appointment.updated_at = datetime.utcnow()

        # Create new appointment with same details
        new_appointment = self.book_appointment(
            user_id=old_appointment.user_id,
            appointment_type=old_appointment.appointment_type,
            appointment_date=new_date,
            start_time=new_start_time,
            provider_id=old_appointment.provider_id,
            is_telemedicine=old_appointment.is_telemedicine,
            patient_name=old_appointment.patient_name,
            patient_phone=old_appointment.patient_phone,
            patient_email=old_appointment.patient_email,
            reason_for_visit=old_appointment.reason_for_visit,
            patient_notes=f"Rescheduled from {old_appointment.appointment_date}. {reason or ''}",
            related_analysis_ids=old_appointment.related_analysis_ids,
            related_lesion_ids=old_appointment.related_lesion_ids
        )

        new_appointment.parent_appointment_id = appointment_id

        # Notify waitlist of the opening from the old slot
        self._notify_waitlist_of_opening(old_appointment)

        return new_appointment

    def check_in(self, appointment_id: str) -> Appointment:
        """Check in for an appointment"""
        if appointment_id not in self.appointments:
            raise ValueError("Appointment not found")

        appointment = self.appointments[appointment_id]
        appointment.status = AppointmentStatus.CHECKED_IN
        appointment.checked_in_at = datetime.utcnow()
        appointment.updated_at = datetime.utcnow()

        return appointment

    def complete_appointment(
        self,
        appointment_id: str,
        provider_notes: Optional[str] = None
    ) -> Appointment:
        """Mark appointment as completed"""
        if appointment_id not in self.appointments:
            raise ValueError("Appointment not found")

        appointment = self.appointments[appointment_id]
        appointment.status = AppointmentStatus.COMPLETED
        appointment.completed_at = datetime.utcnow()
        appointment.updated_at = datetime.utcnow()

        if provider_notes:
            appointment.provider_notes = provider_notes

        return appointment

    def add_to_waitlist(
        self,
        user_id: int,
        appointment_type: AppointmentType,
        preferred_dates: List[date],
        preferred_times: List[str],
        flexibility: str,
        reason: str,
        provider_id: Optional[str] = None,
        priority: int = 5
    ) -> WaitlistEntry:
        """Add a patient to the waitlist"""
        waitlist_id = f"WL-{uuid.uuid4().hex[:12].upper()}"

        entry = WaitlistEntry(
            waitlist_id=waitlist_id,
            user_id=user_id,
            provider_id=provider_id,
            appointment_type=appointment_type,
            preferred_dates=preferred_dates,
            preferred_times=preferred_times,
            flexibility=flexibility,
            reason=reason,
            priority=priority
        )

        self.waitlist[waitlist_id] = entry
        return entry

    def _notify_waitlist_of_opening(self, cancelled_appointment: Appointment):
        """Notify waitlist patients of a newly available slot"""
        # Find matching waitlist entries
        matching_entries = []

        for entry in self.waitlist.values():
            if entry.status != "active":
                continue

            # Check appointment type match
            if entry.appointment_type != cancelled_appointment.appointment_type:
                continue

            # Check provider preference
            if entry.provider_id and entry.provider_id != cancelled_appointment.provider_id:
                continue

            # Check date preference
            if entry.flexibility == "specific":
                if cancelled_appointment.appointment_date not in entry.preferred_dates:
                    continue

            matching_entries.append(entry)

        # Sort by priority (lower number = higher priority)
        matching_entries.sort(key=lambda x: x.priority)

        # Record notification (in real system, would send actual notification)
        for entry in matching_entries[:5]:  # Notify top 5 matches
            entry.notified_slots.append(cancelled_appointment.appointment_id)

    def get_user_appointments(
        self,
        user_id: int,
        include_past: bool = False,
        status_filter: Optional[List[AppointmentStatus]] = None
    ) -> List[Appointment]:
        """Get all appointments for a user"""
        appointments = []
        today = date.today()

        for appointment in self.appointments.values():
            if appointment.user_id != user_id:
                continue

            # Filter by date
            if not include_past and appointment.appointment_date < today:
                continue

            # Filter by status
            if status_filter and appointment.status not in status_filter:
                continue

            appointments.append(appointment)

        # Sort by date and time
        appointments.sort(key=lambda x: (x.appointment_date, x.start_time))

        return appointments

    def generate_ical(self, appointment: Appointment) -> str:
        """Generate iCal format for calendar integration"""
        cal = Calendar()
        cal.add('prodid', '-//Skin Analyzer//Appointment System//EN')
        cal.add('version', '2.0')
        cal.add('calscale', 'GREGORIAN')
        cal.add('method', 'PUBLISH')

        event = Event()

        # Basic event details
        event.add('uid', f"{appointment.appointment_id}@skinanalyzer.com")
        event.add('dtstamp', datetime.utcnow())

        # Combine date and time
        start_dt = datetime.combine(appointment.appointment_date, appointment.start_time)
        end_dt = datetime.combine(appointment.appointment_date, appointment.end_time)

        # Localize to timezone
        tz = pytz.timezone(appointment.timezone)
        start_dt = tz.localize(start_dt)
        end_dt = tz.localize(end_dt)

        event.add('dtstart', start_dt)
        event.add('dtend', end_dt)

        # Title and description
        apt_type_name = appointment.appointment_type.value.replace("_", " ").title()
        event.add('summary', f"Dermatology: {apt_type_name}")

        description_parts = [
            f"Appointment Type: {apt_type_name}",
            f"Reason: {appointment.reason_for_visit}" if appointment.reason_for_visit else "",
        ]

        if appointment.is_telemedicine:
            description_parts.append(f"\nTelemedicine Link: {appointment.telemedicine_link}")
            description_parts.append("\nPlease join the video call at your appointment time.")
        else:
            description_parts.append(f"\nLocation: {appointment.location}")

        # Add preparation instructions
        config = APPOINTMENT_CONFIGS[appointment.appointment_type]
        if config["preparation"]:
            description_parts.append("\n\nPreparation:")
            for prep in config["preparation"]:
                description_parts.append(f"• {prep}")

        event.add('description', "\n".join(filter(None, description_parts)))

        # Location
        if appointment.is_telemedicine:
            event.add('location', appointment.telemedicine_link or "Telemedicine")
        else:
            event.add('location', appointment.location or "Clinic")

        # Add reminders/alarms
        # 24 hours before
        alarm_24h = Alarm()
        alarm_24h.add('action', 'DISPLAY')
        alarm_24h.add('description', f"Reminder: Dermatology appointment tomorrow")
        alarm_24h.add('trigger', timedelta(hours=-24))
        event.add_component(alarm_24h)

        # 2 hours before
        alarm_2h = Alarm()
        alarm_2h.add('action', 'DISPLAY')
        alarm_2h.add('description', f"Reminder: Dermatology appointment in 2 hours")
        alarm_2h.add('trigger', timedelta(hours=-2))
        event.add_component(alarm_2h)

        cal.add_component(event)

        return cal.to_ical().decode('utf-8')

    def generate_google_calendar_url(self, appointment: Appointment) -> str:
        """Generate Google Calendar add event URL"""
        apt_type_name = appointment.appointment_type.value.replace("_", " ").title()

        # Format dates for Google Calendar
        start_dt = datetime.combine(appointment.appointment_date, appointment.start_time)
        end_dt = datetime.combine(appointment.appointment_date, appointment.end_time)

        date_format = "%Y%m%dT%H%M%S"
        dates = f"{start_dt.strftime(date_format)}/{end_dt.strftime(date_format)}"

        # Build description
        description_parts = [f"Appointment Type: {apt_type_name}"]
        if appointment.reason_for_visit:
            description_parts.append(f"Reason: {appointment.reason_for_visit}")
        if appointment.is_telemedicine and appointment.telemedicine_link:
            description_parts.append(f"Join: {appointment.telemedicine_link}")

        # URL encode the parameters
        import urllib.parse

        params = {
            "action": "TEMPLATE",
            "text": f"Dermatology: {apt_type_name}",
            "dates": dates,
            "details": "\n".join(description_parts),
            "location": appointment.telemedicine_link if appointment.is_telemedicine else (appointment.location or ""),
        }

        base_url = "https://calendar.google.com/calendar/render"
        return f"{base_url}?{urllib.parse.urlencode(params)}"

    def get_upcoming_reminders(self, hours_ahead: int = 48) -> List[Dict[str, Any]]:
        """Get appointments that need reminders sent"""
        reminders = []
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)

        for appointment in self.appointments.values():
            if appointment.status not in [AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED]:
                continue

            apt_datetime = datetime.combine(appointment.appointment_date, appointment.start_time)

            if now < apt_datetime <= cutoff:
                # Check reminder settings
                time_until = apt_datetime - now
                hours_until = time_until.total_seconds() / 3600

                reminder_needed = False
                reminder_type = None

                if 23 <= hours_until <= 25 and appointment.reminder_settings.get("24_hours_before"):
                    reminder_needed = True
                    reminder_type = "24_hours"
                elif 1.5 <= hours_until <= 2.5 and appointment.reminder_settings.get("2_hours_before"):
                    reminder_needed = True
                    reminder_type = "2_hours"

                if reminder_needed:
                    # Check if reminder already sent
                    already_sent = any(
                        r.get("type") == reminder_type
                        for r in appointment.reminders_sent
                    )

                    if not already_sent:
                        reminders.append({
                            "appointment_id": appointment.appointment_id,
                            "user_id": appointment.user_id,
                            "reminder_type": reminder_type,
                            "appointment_datetime": apt_datetime,
                            "appointment_type": appointment.appointment_type.value,
                            "is_telemedicine": appointment.is_telemedicine,
                            "patient_email": appointment.patient_email,
                            "patient_phone": appointment.patient_phone
                        })

        return reminders

    def mark_reminder_sent(
        self,
        appointment_id: str,
        reminder_type: str,
        channel: str  # email, sms, push
    ):
        """Record that a reminder was sent"""
        if appointment_id in self.appointments:
            self.appointments[appointment_id].reminders_sent.append({
                "type": reminder_type,
                "channel": channel,
                "sent_at": datetime.utcnow().isoformat()
            })


# Convenience functions for API use
def get_appointment_types_list() -> List[Dict[str, Any]]:
    """Get list of appointment types for API"""
    scheduler = AppointmentScheduler()
    return scheduler.get_appointment_types()


def check_slot_availability(
    start_date: str,
    end_date: str,
    appointment_type: str,
    provider_id: Optional[str] = None,
    is_telemedicine: bool = False
) -> List[Dict[str, Any]]:
    """Check available slots for API"""
    scheduler = AppointmentScheduler()

    apt_type = AppointmentType(appointment_type)
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    slots = scheduler.get_available_slots(
        start_date=start,
        end_date=end,
        appointment_type=apt_type,
        provider_id=provider_id,
        is_telemedicine=is_telemedicine
    )

    return [
        {
            "start_time": slot.start_time.isoformat(),
            "end_time": slot.end_time.isoformat(),
            "provider_id": slot.provider_id,
            "provider_name": slot.provider_name,
            "is_telemedicine": slot.is_telemedicine,
            "location": slot.location
        }
        for slot in slots
    ]
