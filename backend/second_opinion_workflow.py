"""
Complete Second Opinion Workflow System

Comprehensive second opinion management including:
- Patient UI support for requesting second opinions
- Smart dermatologist assignment algorithm
- Dermatologist review interface
- Multi-channel notification system (push/email/SMS)
- Response time SLA tracking
- Payment/credits system for second opinions
"""

import os
import json
import smtplib
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from sqlalchemy.orm import Session


class SecondOpinionStatus(Enum):
    """Status states for second opinion requests"""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    PAYMENT_PENDING = "payment_pending"
    PAYMENT_COMPLETED = "payment_completed"
    ASSIGNING = "assigning"
    ASSIGNED = "assigned"
    UNDER_REVIEW = "under_review"
    ADDITIONAL_INFO_NEEDED = "additional_info_needed"
    PENDING_PATIENT_RESPONSE = "pending_patient_response"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class UrgencyLevel(Enum):
    """Urgency levels for second opinion requests"""
    ROUTINE = "routine"          # 7 days SLA
    SEMI_URGENT = "semi_urgent"  # 3 days SLA
    URGENT = "urgent"            # 24 hours SLA
    EMERGENCY = "emergency"      # 4 hours SLA (rare)


class NotificationType(Enum):
    """Types of notifications"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class PaymentStatus(Enum):
    """Payment status for second opinions"""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class SLAConfig:
    """SLA configuration for different urgency levels"""
    urgency: UrgencyLevel
    response_hours: int
    review_hours: int
    warning_threshold_percent: float  # When to send warning (e.g., 0.75 = 75%)
    escalation_threshold_percent: float  # When to escalate (e.g., 0.90 = 90%)


@dataclass
class DermatologistScore:
    """Score for dermatologist assignment ranking"""
    dermatologist_id: int
    total_score: float
    specialty_match: float
    availability_score: float
    rating_score: float
    workload_score: float
    response_time_score: float
    expertise_match: float
    reasons: List[str]


@dataclass
class NotificationTemplate:
    """Template for notifications"""
    template_id: str
    channel: NotificationType
    subject: str
    body_template: str
    variables: List[str]


@dataclass
class SecondOpinionPricing:
    """Pricing configuration for second opinions"""
    base_price: float
    urgency_multipliers: Dict[str, float]
    specialty_premiums: Dict[str, float]
    consultation_type_prices: Dict[str, float]
    discount_codes: Dict[str, float]


class SLATracker:
    """
    Tracks and enforces SLA for second opinion responses.
    """

    def __init__(self):
        # SLA configurations by urgency
        self.sla_configs = {
            UrgencyLevel.ROUTINE: SLAConfig(
                urgency=UrgencyLevel.ROUTINE,
                response_hours=168,  # 7 days
                review_hours=120,    # 5 days for review
                warning_threshold_percent=0.75,
                escalation_threshold_percent=0.90
            ),
            UrgencyLevel.SEMI_URGENT: SLAConfig(
                urgency=UrgencyLevel.SEMI_URGENT,
                response_hours=72,   # 3 days
                review_hours=48,     # 2 days for review
                warning_threshold_percent=0.60,
                escalation_threshold_percent=0.85
            ),
            UrgencyLevel.URGENT: SLAConfig(
                urgency=UrgencyLevel.URGENT,
                response_hours=24,   # 1 day
                review_hours=18,     # 18 hours for review
                warning_threshold_percent=0.50,
                escalation_threshold_percent=0.75
            ),
            UrgencyLevel.EMERGENCY: SLAConfig(
                urgency=UrgencyLevel.EMERGENCY,
                response_hours=4,    # 4 hours
                review_hours=3,      # 3 hours for review
                warning_threshold_percent=0.40,
                escalation_threshold_percent=0.60
            ),
        }

    def get_sla_config(self, urgency: str) -> SLAConfig:
        """Get SLA configuration for urgency level"""
        try:
            urgency_enum = UrgencyLevel(urgency)
            return self.sla_configs[urgency_enum]
        except (ValueError, KeyError):
            return self.sla_configs[UrgencyLevel.ROUTINE]

    def calculate_deadline(self, submitted_at: datetime, urgency: str) -> datetime:
        """Calculate response deadline based on urgency"""
        config = self.get_sla_config(urgency)
        return submitted_at + timedelta(hours=config.response_hours)

    def get_sla_status(self, submitted_at: datetime, urgency: str,
                       completed_at: Optional[datetime] = None) -> Dict[str, Any]:
        """Get detailed SLA status"""
        config = self.get_sla_config(urgency)
        deadline = self.calculate_deadline(submitted_at, urgency)
        now = datetime.utcnow()

        if completed_at:
            # Already completed
            time_taken = (completed_at - submitted_at).total_seconds() / 3600
            sla_met = completed_at <= deadline
            return {
                "status": "completed",
                "sla_met": sla_met,
                "deadline": deadline.isoformat(),
                "completed_at": completed_at.isoformat(),
                "time_taken_hours": round(time_taken, 2),
                "sla_hours": config.response_hours,
                "hours_remaining": 0,
                "percent_elapsed": 100,
                "warning_level": "none"
            }

        # Still pending
        elapsed = (now - submitted_at).total_seconds() / 3600
        remaining = max(0, config.response_hours - elapsed)
        percent_elapsed = min(100, (elapsed / config.response_hours) * 100)

        # Determine warning level
        if percent_elapsed >= config.escalation_threshold_percent * 100:
            warning_level = "critical"
        elif percent_elapsed >= config.warning_threshold_percent * 100:
            warning_level = "warning"
        else:
            warning_level = "normal"

        return {
            "status": "pending",
            "sla_met": None,  # Not yet determined
            "deadline": deadline.isoformat(),
            "completed_at": None,
            "time_taken_hours": round(elapsed, 2),
            "sla_hours": config.response_hours,
            "hours_remaining": round(remaining, 2),
            "percent_elapsed": round(percent_elapsed, 1),
            "warning_level": warning_level,
            "is_overdue": now > deadline
        }

    def get_breach_risk_score(self, submitted_at: datetime, urgency: str) -> float:
        """Calculate risk score for SLA breach (0-1, higher = more risk)"""
        status = self.get_sla_status(submitted_at, urgency)

        if status["status"] == "completed":
            return 0.0 if status["sla_met"] else 1.0

        if status["is_overdue"]:
            return 1.0

        return status["percent_elapsed"] / 100


class DermatologistAssignmentAlgorithm:
    """
    Smart algorithm for assigning dermatologists to second opinion requests.

    Considers:
    - Specialty match (e.g., Mohs surgery, melanoma specialist)
    - Current workload
    - Response time history
    - Availability status
    - Patient location preferences (by specialty, location, availability)
    - Rating and experience
    - Geographic proximity for in-person consultations
    """

    def __init__(self):
        # Weight factors for scoring
        self.weights = {
            "specialty_match": 0.25,
            "availability": 0.20,
            "rating": 0.15,
            "workload": 0.15,
            "response_time": 0.10,
            "expertise": 0.10,
            "location": 0.05  # Added location weight
        }

        # Specialty mappings for conditions
        self.condition_specialties = {
            "melanoma": ["melanoma_specialist", "surgical_oncology", "dermatopathology"],
            "basal_cell_carcinoma": ["mohs_surgery", "surgical_dermatology"],
            "squamous_cell_carcinoma": ["mohs_surgery", "surgical_dermatology"],
            "psoriasis": ["medical_dermatology", "autoimmune", "phototherapy"],
            "eczema": ["medical_dermatology", "pediatric_dermatology", "allergy"],
            "acne": ["medical_dermatology", "cosmetic_dermatology"],
            "rosacea": ["medical_dermatology", "cosmetic_dermatology"],
            "skin_cancer": ["mohs_surgery", "surgical_oncology", "dermatopathology"],
            "atopic_dermatitis": ["medical_dermatology", "pediatric_dermatology", "allergy"],
            "contact_dermatitis": ["medical_dermatology", "allergy", "occupational_dermatology"],
            "fungal_infection": ["medical_dermatology", "infectious_disease"],
            "bacterial_infection": ["medical_dermatology", "infectious_disease"],
            "viral_infection": ["medical_dermatology", "infectious_disease"],
            "autoimmune": ["medical_dermatology", "autoimmune", "rheumatology"],
            "pigmentation_disorder": ["cosmetic_dermatology", "medical_dermatology"],
            "hair_loss": ["medical_dermatology", "hair_restoration"],
            "nail_disorder": ["medical_dermatology", "nail_specialist"],
        }

    def find_best_dermatologists(self, request: Dict[str, Any],
                                 available_dermatologists: List[Dict[str, Any]],
                                 limit: int = 5) -> List[DermatologistScore]:
        """
        Find and rank best dermatologists for a second opinion request.

        Args:
            request: Second opinion request details
            available_dermatologists: List of available dermatologists
            limit: Maximum number of results

        Returns:
            Ranked list of dermatologist scores
        """
        scores = []

        for derm in available_dermatologists:
            score = self._calculate_dermatologist_score(request, derm)
            scores.append(score)

        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)

        return scores[:limit]

    def _calculate_dermatologist_score(self, request: Dict[str, Any],
                                       derm: Dict[str, Any]) -> DermatologistScore:
        """Calculate comprehensive score for a dermatologist"""
        reasons = []

        # Specialty match
        specialty_score = self._calculate_specialty_match(
            request.get("original_diagnosis", ""),
            request.get("dermatologist_specialty_requested"),
            derm.get("specializations", [])
        )
        if specialty_score > 0.8:
            reasons.append(f"Excellent specialty match: {derm.get('specializations', [])[:2]}")

        # Availability score
        availability_score = self._calculate_availability_score(
            derm.get("availability_status", "available"),
            derm.get("typical_wait_time_days", 7),
            request.get("urgency", "routine")
        )
        if availability_score > 0.8:
            reasons.append("Currently available with short wait time")

        # Rating score
        rating_score = self._calculate_rating_score(
            derm.get("average_rating", 0),
            derm.get("total_reviews", 0)
        )
        if rating_score > 0.8:
            reasons.append(f"Highly rated: {derm.get('average_rating', 0)}/5 stars")

        # Workload score
        workload_score = self._calculate_workload_score(
            derm.get("current_queue_size", 0),
            derm.get("max_queue_size", 20)
        )
        if workload_score > 0.8:
            reasons.append("Low current workload")

        # Response time score
        response_time_score = self._calculate_response_time_score(
            derm.get("avg_response_hours", 48),
            request.get("urgency", "routine")
        )
        if response_time_score > 0.8:
            reasons.append("Fast response history")

        # Expertise match
        expertise_score = self._calculate_expertise_match(
            request,
            derm.get("years_experience", 0),
            derm.get("board_certifications", []),
            derm.get("total_consultations", 0)
        )
        if expertise_score > 0.8:
            reasons.append("Extensive relevant experience")

        # Location match (for in-person or proximity preference)
        location_score = self._calculate_location_score(
            request.get("patient_location"),
            derm,
            request.get("consultation_method", "chart_review"),
            request.get("preferred_distance_miles", 50)
        )
        if location_score > 0.8:
            distance = self._calculate_distance(
                request.get("patient_location", {}),
                {"latitude": derm.get("latitude"), "longitude": derm.get("longitude")}
            )
            if distance is not None and distance < 25:
                reasons.append(f"Nearby location: {derm.get('city', 'Local')}")
            elif derm.get("state") == request.get("patient_location", {}).get("state"):
                reasons.append(f"Same state: {derm.get('state')}")

        # Calculate weighted total
        total_score = (
            specialty_score * self.weights["specialty_match"] +
            availability_score * self.weights["availability"] +
            rating_score * self.weights["rating"] +
            workload_score * self.weights["workload"] +
            response_time_score * self.weights["response_time"] +
            expertise_score * self.weights["expertise"] +
            location_score * self.weights["location"]
        )

        return DermatologistScore(
            dermatologist_id=derm.get("id", 0),
            total_score=round(total_score, 3),
            specialty_match=round(specialty_score, 3),
            availability_score=round(availability_score, 3),
            rating_score=round(rating_score, 3),
            workload_score=round(workload_score, 3),
            response_time_score=round(response_time_score, 3),
            expertise_match=round(expertise_score, 3),
            reasons=reasons
        )

    def _calculate_specialty_match(self, diagnosis: str,
                                   requested_specialty: Optional[str],
                                   derm_specialties: List[str]) -> float:
        """Calculate specialty match score"""
        if not derm_specialties:
            return 0.3  # Base score for general dermatologists

        # Check if requested specialty matches
        if requested_specialty:
            if requested_specialty.lower() in [s.lower() for s in derm_specialties]:
                return 1.0
            return 0.5

        # Match based on diagnosis
        diagnosis_lower = diagnosis.lower()
        needed_specialties = []

        for condition, specialties in self.condition_specialties.items():
            if condition in diagnosis_lower:
                needed_specialties.extend(specialties)

        if not needed_specialties:
            return 0.6  # No specific match needed

        derm_specialties_lower = [s.lower() for s in derm_specialties]
        matches = sum(1 for s in needed_specialties if s.lower() in derm_specialties_lower)

        if matches > 0:
            return min(1.0, 0.6 + (matches * 0.2))

        return 0.4

    def _calculate_availability_score(self, status: str, wait_days: int,
                                      urgency: str) -> float:
        """Calculate availability score based on status and wait time"""
        status_scores = {
            "available": 1.0,
            "limited": 0.7,
            "busy": 0.4,
            "unavailable": 0.0
        }

        base_score = status_scores.get(status.lower(), 0.5)

        # Adjust for wait time based on urgency
        urgency_max_wait = {
            "emergency": 1,
            "urgent": 3,
            "semi_urgent": 7,
            "routine": 14
        }

        max_wait = urgency_max_wait.get(urgency, 14)
        wait_penalty = min(1.0, wait_days / max_wait)
        wait_score = 1.0 - (wait_penalty * 0.5)

        return base_score * wait_score

    def _calculate_rating_score(self, rating: float, review_count: int) -> float:
        """Calculate rating score with review count weight"""
        if review_count == 0:
            return 0.5  # No reviews yet

        # Normalize rating to 0-1
        rating_normalized = rating / 5.0

        # Weight by review count (more reviews = more reliable)
        confidence = min(1.0, review_count / 50)
        weighted_rating = (rating_normalized * confidence) + (0.5 * (1 - confidence))

        return weighted_rating

    def _calculate_workload_score(self, current: int, max_size: int) -> float:
        """Calculate workload score"""
        if max_size == 0:
            return 0.5

        utilization = current / max_size
        return max(0, 1.0 - utilization)

    def _calculate_response_time_score(self, avg_hours: float, urgency: str) -> float:
        """Calculate response time score"""
        urgency_targets = {
            "emergency": 4,
            "urgent": 24,
            "semi_urgent": 72,
            "routine": 168
        }

        target = urgency_targets.get(urgency, 168)

        if avg_hours <= target * 0.5:
            return 1.0
        elif avg_hours <= target:
            return 0.8
        elif avg_hours <= target * 1.5:
            return 0.5
        else:
            return 0.2

    def _calculate_expertise_match(self, request: Dict[str, Any],
                                   years_exp: int, certifications: List[str],
                                   total_consultations: int) -> float:
        """Calculate expertise match score"""
        score = 0.0

        # Years of experience
        if years_exp >= 15:
            score += 0.3
        elif years_exp >= 10:
            score += 0.25
        elif years_exp >= 5:
            score += 0.15
        else:
            score += 0.1

        # Board certifications
        if certifications:
            score += min(0.3, len(certifications) * 0.1)

        # Total consultations experience
        if total_consultations >= 1000:
            score += 0.4
        elif total_consultations >= 500:
            score += 0.3
        elif total_consultations >= 100:
            score += 0.2
        else:
            score += 0.1

        return min(1.0, score)

    def _calculate_location_score(self, patient_location: Optional[Dict[str, Any]],
                                  derm: Dict[str, Any],
                                  consultation_method: str,
                                  preferred_distance: int = 50) -> float:
        """
        Calculate location match score based on proximity and consultation type.

        For video/chart review: Location matters less, but same timezone/state is a bonus
        For in-person: Location is critical
        """
        # If no patient location provided, return neutral score
        if not patient_location:
            return 0.5

        # Video consultations don't need proximity, but timezone match helps
        if consultation_method in ["video", "chart_review"]:
            # Timezone/country match is a small bonus
            if derm.get("country") == patient_location.get("country"):
                base_score = 0.7
            else:
                base_score = 0.5

            # Same state is a bonus (familiar with local healthcare)
            if derm.get("state") == patient_location.get("state"):
                base_score += 0.2

            return min(1.0, base_score)

        # In-person consultations need proximity
        distance = self._calculate_distance(patient_location, {
            "latitude": derm.get("latitude"),
            "longitude": derm.get("longitude")
        })

        if distance is None:
            # Fall back to state/city matching
            if derm.get("state") == patient_location.get("state"):
                if derm.get("city") == patient_location.get("city"):
                    return 0.9
                return 0.6
            return 0.3

        # Score based on distance
        if distance <= 10:
            return 1.0
        elif distance <= 25:
            return 0.9
        elif distance <= preferred_distance:
            return 0.7
        elif distance <= preferred_distance * 2:
            return 0.4
        else:
            return 0.2

    def _calculate_distance(self, loc1: Dict[str, Any], loc2: Dict[str, Any]) -> Optional[float]:
        """
        Calculate distance in miles between two locations using Haversine formula.

        Returns:
            Distance in miles, or None if coordinates not available
        """
        import math

        lat1 = loc1.get("latitude")
        lon1 = loc1.get("longitude")
        lat2 = loc2.get("latitude")
        lon2 = loc2.get("longitude")

        if None in (lat1, lon1, lat2, lon2):
            return None

        try:
            # Earth's radius in miles
            R = 3959

            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)

            a = (math.sin(delta_lat / 2) ** 2 +
                 math.cos(lat1_rad) * math.cos(lat2_rad) *
                 math.sin(delta_lon / 2) ** 2)

            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            return R * c
        except (TypeError, ValueError):
            return None

    def find_nearby_dermatologists(self, patient_location: Dict[str, Any],
                                   available_dermatologists: List[Dict[str, Any]],
                                   max_distance_miles: float = 50,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find dermatologists within a certain distance from patient.

        Args:
            patient_location: Dict with latitude, longitude, city, state
            available_dermatologists: List of available dermatologists
            max_distance_miles: Maximum distance to search
            limit: Maximum number of results

        Returns:
            List of dermatologists with distance info, sorted by distance
        """
        results = []

        for derm in available_dermatologists:
            distance = self._calculate_distance(patient_location, {
                "latitude": derm.get("latitude"),
                "longitude": derm.get("longitude")
            })

            if distance is not None and distance <= max_distance_miles:
                derm_copy = derm.copy()
                derm_copy["distance_miles"] = round(distance, 1)
                results.append(derm_copy)
            elif distance is None:
                # Fall back to state matching
                if derm.get("state") == patient_location.get("state"):
                    derm_copy = derm.copy()
                    derm_copy["distance_miles"] = None  # Unknown but same state
                    results.append(derm_copy)

        # Sort by distance (None values at end)
        results.sort(key=lambda x: (x.get("distance_miles") is None, x.get("distance_miles") or 9999))

        return results[:limit]

    def auto_assign(self, request: Dict[str, Any],
                    available_dermatologists: List[Dict[str, Any]]) -> Optional[int]:
        """
        Automatically assign the best dermatologist.
        Returns dermatologist_id or None if no suitable match.
        """
        if not available_dermatologists:
            return None

        scores = self.find_best_dermatologists(request, available_dermatologists, limit=1)

        if scores and scores[0].total_score >= 0.5:
            return scores[0].dermatologist_id

        return None


class NotificationService:
    """
    Multi-channel notification service for second opinion workflow.

    Supports:
    - Email notifications
    - SMS notifications (via Twilio)
    - Push notifications (via Firebase/Expo)
    - In-app notifications
    """

    def __init__(self):
        # Load configuration from environment
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@skinclassifier.com")

        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER", "")

        self.firebase_key = os.getenv("FIREBASE_SERVER_KEY", "")

        # Notification templates
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, NotificationTemplate]:
        """Load notification templates"""
        return {
            # Patient notifications
            "second_opinion_submitted": NotificationTemplate(
                template_id="second_opinion_submitted",
                channel=NotificationType.EMAIL,
                subject="Second Opinion Request Submitted",
                body_template="""
Dear {patient_name},

Your second opinion request has been successfully submitted.

Request ID: {request_id}
Original Diagnosis: {original_diagnosis}
Urgency: {urgency}
Expected Response: Within {sla_hours} hours

We are currently matching you with a qualified specialist. You will receive another notification once a dermatologist has been assigned.

Track your request status at: {tracking_url}

Thank you for choosing our service.

Best regards,
SkinClassifier Team
                """,
                variables=["patient_name", "request_id", "original_diagnosis", "urgency", "sla_hours", "tracking_url"]
            ),

            "dermatologist_assigned": NotificationTemplate(
                template_id="dermatologist_assigned",
                channel=NotificationType.EMAIL,
                subject="Specialist Assigned to Your Second Opinion",
                body_template="""
Dear {patient_name},

Great news! A specialist has been assigned to review your second opinion request.

Assigned Specialist: Dr. {dermatologist_name}
Credentials: {credentials}
Specializations: {specializations}
Rating: {rating}/5 stars

Expected Review Completion: {expected_completion}

Dr. {dermatologist_name} will review your case and provide a detailed second opinion. You may be contacted if additional information is needed.

Track your request at: {tracking_url}

Best regards,
SkinClassifier Team
                """,
                variables=["patient_name", "dermatologist_name", "credentials", "specializations", "rating", "expected_completion", "tracking_url"]
            ),

            "additional_info_needed": NotificationTemplate(
                template_id="additional_info_needed",
                channel=NotificationType.EMAIL,
                subject="Additional Information Needed for Your Second Opinion",
                body_template="""
Dear {patient_name},

The reviewing specialist needs additional information to complete your second opinion.

Request ID: {request_id}
Specialist: Dr. {dermatologist_name}

Information Requested:
{requested_info}

Please provide this information as soon as possible to avoid delays in your review.

Submit additional information at: {submit_url}

Best regards,
SkinClassifier Team
                """,
                variables=["patient_name", "request_id", "dermatologist_name", "requested_info", "submit_url"]
            ),

            "second_opinion_completed": NotificationTemplate(
                template_id="second_opinion_completed",
                channel=NotificationType.EMAIL,
                subject="Your Second Opinion is Ready",
                body_template="""
Dear {patient_name},

Your second opinion review has been completed.

Request ID: {request_id}
Reviewed by: Dr. {dermatologist_name}
Review Date: {review_date}

Summary:
{summary}

To view the full second opinion report, including:
- Detailed diagnosis review
- Treatment recommendations
- Additional testing recommendations (if any)
- Follow-up guidance

View full report at: {report_url}

If you have questions about this second opinion, you can schedule a follow-up consultation with Dr. {dermatologist_name}.

Best regards,
SkinClassifier Team
                """,
                variables=["patient_name", "request_id", "dermatologist_name", "review_date", "summary", "report_url"]
            ),

            # Dermatologist notifications
            "new_case_assigned": NotificationTemplate(
                template_id="new_case_assigned",
                channel=NotificationType.EMAIL,
                subject="New Second Opinion Case Assigned",
                body_template="""
Dear Dr. {dermatologist_name},

A new second opinion case has been assigned to you.

Case ID: {request_id}
Patient: {patient_identifier}
Original Diagnosis: {original_diagnosis}
Urgency: {urgency}
SLA Deadline: {deadline}

Reason for Second Opinion:
{reason}

Patient Questions:
{questions}

Review the case at: {review_url}

Please complete your review within the SLA deadline. The patient is counting on your expertise.

Best regards,
SkinClassifier Platform
                """,
                variables=["dermatologist_name", "request_id", "patient_identifier", "original_diagnosis", "urgency", "deadline", "reason", "questions", "review_url"]
            ),

            "sla_warning": NotificationTemplate(
                template_id="sla_warning",
                channel=NotificationType.EMAIL,
                subject="URGENT: SLA Warning for Second Opinion Case",
                body_template="""
Dear Dr. {dermatologist_name},

This is an urgent reminder about a pending second opinion case.

Case ID: {request_id}
SLA Deadline: {deadline}
Time Remaining: {time_remaining} hours

Current Status: {current_status}

Please complete your review as soon as possible to meet the SLA commitment.

Complete review at: {review_url}

If you are unable to complete this review, please escalate immediately.

SkinClassifier Platform
                """,
                variables=["dermatologist_name", "request_id", "deadline", "time_remaining", "current_status", "review_url"]
            ),

            # SMS templates
            "sms_submission_confirm": NotificationTemplate(
                template_id="sms_submission_confirm",
                channel=NotificationType.SMS,
                subject="",
                body_template="SkinClassifier: Your second opinion request #{request_id} has been submitted. Track at: {short_url}",
                variables=["request_id", "short_url"]
            ),

            "sms_opinion_ready": NotificationTemplate(
                template_id="sms_opinion_ready",
                channel=NotificationType.SMS,
                subject="",
                body_template="SkinClassifier: Your second opinion is ready! View the report: {short_url}",
                variables=["short_url"]
            ),
        }

    async def send_notification(self, template_id: str, recipient: Dict[str, Any],
                                variables: Dict[str, str],
                                channels: List[NotificationType] = None) -> Dict[str, bool]:
        """
        Send notification using specified template.

        Args:
            template_id: Template identifier
            recipient: Recipient info (email, phone, device_token)
            variables: Template variables
            channels: List of channels to use (default: all available)

        Returns:
            Dict of channel -> success status
        """
        if template_id not in self.templates:
            return {"error": f"Template {template_id} not found"}

        template = self.templates[template_id]
        results = {}

        # Default to all available channels
        if channels is None:
            channels = [NotificationType.EMAIL, NotificationType.IN_APP]

        for channel in channels:
            if channel == NotificationType.EMAIL and recipient.get("email"):
                results["email"] = await self._send_email(template, recipient["email"], variables)

            elif channel == NotificationType.SMS and recipient.get("phone"):
                results["sms"] = await self._send_sms(template, recipient["phone"], variables)

            elif channel == NotificationType.PUSH and recipient.get("device_token"):
                results["push"] = await self._send_push(template, recipient["device_token"], variables)

            elif channel == NotificationType.IN_APP:
                results["in_app"] = await self._create_in_app_notification(template, recipient, variables)

        return results

    async def _send_email(self, template: NotificationTemplate, email: str,
                          variables: Dict[str, str]) -> bool:
        """Send email notification"""
        if not self.smtp_user or not self.smtp_password:
            print("Email not configured - skipping")
            return False

        try:
            # Format template
            body = template.body_template
            for var, value in variables.items():
                body = body.replace(f"{{{var}}}", str(value))

            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = email
            msg["Subject"] = template.subject

            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            return True
        except Exception as e:
            print(f"Email send error: {e}")
            return False

    async def _send_sms(self, template: NotificationTemplate, phone: str,
                        variables: Dict[str, str]) -> bool:
        """Send SMS notification via Twilio"""
        if not self.twilio_sid or not self.twilio_token:
            print("Twilio not configured - skipping SMS")
            return False

        try:
            # Format message
            body = template.body_template
            for var, value in variables.items():
                body = body.replace(f"{{{var}}}", str(value))

            # Use Twilio REST API
            from urllib.request import Request, urlopen
            from urllib.parse import urlencode
            import base64

            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.twilio_sid}/Messages.json"
            data = urlencode({
                "To": phone,
                "From": self.twilio_phone,
                "Body": body
            }).encode()

            credentials = base64.b64encode(f"{self.twilio_sid}:{self.twilio_token}".encode()).decode()

            req = Request(url, data=data)
            req.add_header("Authorization", f"Basic {credentials}")

            response = urlopen(req)
            return response.status == 201

        except Exception as e:
            print(f"SMS send error: {e}")
            return False

    async def _send_push(self, template: NotificationTemplate, device_token: str,
                         variables: Dict[str, str]) -> bool:
        """Send push notification via Firebase"""
        if not self.firebase_key:
            print("Firebase not configured - skipping push")
            return False

        try:
            from urllib.request import Request, urlopen

            # Format message
            body = template.body_template
            for var, value in variables.items():
                body = body.replace(f"{{{var}}}", str(value))

            url = "https://fcm.googleapis.com/fcm/send"
            data = json.dumps({
                "to": device_token,
                "notification": {
                    "title": template.subject,
                    "body": body[:200]  # Truncate for push
                },
                "data": variables
            }).encode()

            req = Request(url, data=data)
            req.add_header("Authorization", f"key={self.firebase_key}")
            req.add_header("Content-Type", "application/json")

            response = urlopen(req)
            return response.status == 200

        except Exception as e:
            print(f"Push notification error: {e}")
            return False

    async def _create_in_app_notification(self, template: NotificationTemplate,
                                          recipient: Dict[str, Any],
                                          variables: Dict[str, str]) -> bool:
        """Create in-app notification (stored in database)"""
        # This would typically store in a notifications table
        # For now, return True to indicate success
        return True

    async def send_sla_warning(self, request: Dict[str, Any],
                               dermatologist: Dict[str, Any],
                               time_remaining_hours: float):
        """Send SLA warning notification to dermatologist"""
        variables = {
            "dermatologist_name": dermatologist.get("full_name", "Doctor"),
            "request_id": str(request.get("id", "")),
            "deadline": request.get("deadline", ""),
            "time_remaining": str(round(time_remaining_hours, 1)),
            "current_status": request.get("status", ""),
            "review_url": f"https://app.skinclassifier.com/dermatologist/review/{request.get('id', '')}"
        }

        await self.send_notification(
            "sla_warning",
            {"email": dermatologist.get("email"), "phone": dermatologist.get("phone_number")},
            variables,
            [NotificationType.EMAIL, NotificationType.SMS, NotificationType.PUSH]
        )


class SecondOpinionPaymentService:
    """
    Payment and credits system for second opinions.
    """

    def __init__(self):
        self.stripe_key = os.getenv("STRIPE_SECRET_KEY", "")

        # Pricing configuration
        self.pricing = SecondOpinionPricing(
            base_price=99.00,
            urgency_multipliers={
                "routine": 1.0,
                "semi_urgent": 1.5,
                "urgent": 2.0,
                "emergency": 3.0
            },
            specialty_premiums={
                "mohs_surgery": 50.00,
                "dermatopathology": 75.00,
                "melanoma_specialist": 50.00,
                "pediatric_dermatology": 25.00,
                "surgical_oncology": 75.00
            },
            consultation_type_prices={
                "chart_review": 99.00,
                "video": 149.00,
                "in_person": 199.00,
                "comprehensive": 249.00
            },
            discount_codes={
                "FIRST10": 0.10,    # 10% off first opinion
                "FOLLOWUP": 0.25,   # 25% off follow-up
                "HEALTHCARE": 0.20  # 20% off for healthcare workers
            }
        )

        # Credits packages
        self.credit_packages = [
            {"credits": 1, "price": 99.00, "savings": 0},
            {"credits": 3, "price": 249.00, "savings": 48.00},
            {"credits": 5, "price": 399.00, "savings": 96.00},
            {"credits": 10, "price": 749.00, "savings": 241.00},
        ]

    def calculate_price(self, urgency: str, specialty: Optional[str] = None,
                       consultation_type: str = "chart_review",
                       discount_code: Optional[str] = None) -> Dict[str, Any]:
        """Calculate price for a second opinion"""
        # Base price by consultation type
        base = self.pricing.consultation_type_prices.get(consultation_type, 99.00)

        # Urgency multiplier
        multiplier = self.pricing.urgency_multipliers.get(urgency, 1.0)
        subtotal = base * multiplier

        # Specialty premium
        specialty_premium = 0.0
        if specialty:
            specialty_premium = self.pricing.specialty_premiums.get(specialty.lower(), 0.0)
        subtotal += specialty_premium

        # Apply discount
        discount_amount = 0.0
        if discount_code and discount_code.upper() in self.pricing.discount_codes:
            discount_percent = self.pricing.discount_codes[discount_code.upper()]
            discount_amount = subtotal * discount_percent

        total = subtotal - discount_amount

        return {
            "base_price": base,
            "urgency_multiplier": multiplier,
            "specialty_premium": specialty_premium,
            "subtotal": round(subtotal, 2),
            "discount_code": discount_code,
            "discount_amount": round(discount_amount, 2),
            "total": round(total, 2),
            "currency": "USD"
        }

    async def create_payment_intent(self, user_id: int, amount: float,
                                    request_id: int) -> Dict[str, Any]:
        """Create Stripe payment intent"""
        if not self.stripe_key:
            # Demo mode
            import secrets
            return {
                "payment_intent_id": f"pi_demo_{secrets.token_hex(12)}",
                "client_secret": f"secret_demo_{secrets.token_hex(16)}",
                "amount": amount,
                "currency": "usd",
                "demo_mode": True
            }

        try:
            import stripe
            stripe.api_key = self.stripe_key

            intent = stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Cents
                currency="usd",
                metadata={
                    "user_id": user_id,
                    "request_id": request_id,
                    "type": "second_opinion"
                }
            )

            return {
                "payment_intent_id": intent.id,
                "client_secret": intent.client_secret,
                "amount": amount,
                "currency": "usd"
            }
        except ImportError:
            return {"error": "Stripe not installed"}
        except Exception as e:
            return {"error": str(e)}

    def get_credit_balance(self, user_credits: int) -> Dict[str, Any]:
        """Get user's credit balance and packages"""
        return {
            "current_credits": user_credits,
            "can_request": user_credits > 0,
            "packages": self.credit_packages
        }

    async def use_credit(self, user_id: int, db: Session) -> bool:
        """Deduct one credit from user's balance"""
        from database import User

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        # Check if user has credits field
        credits = getattr(user, 'second_opinion_credits', 0)
        if credits <= 0:
            return False

        user.second_opinion_credits = credits - 1
        db.commit()
        return True

    async def refund_credit(self, user_id: int, db: Session) -> bool:
        """Refund credit (e.g., if request cancelled)"""
        from database import User

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        credits = getattr(user, 'second_opinion_credits', 0)
        user.second_opinion_credits = credits + 1
        db.commit()
        return True


class SecondOpinionWorkflowService:
    """
    Main service coordinating the complete second opinion workflow.
    """

    def __init__(self):
        self.sla_tracker = SLATracker()
        self.assignment_algorithm = DermatologistAssignmentAlgorithm()
        self.notification_service = NotificationService()
        self.payment_service = SecondOpinionPaymentService()

    async def submit_request(self, request_data: Dict[str, Any],
                            user: Dict[str, Any],
                            db: Session) -> Dict[str, Any]:
        """
        Submit a new second opinion request.

        Handles:
        1. Validation
        2. Payment/credit check
        3. Request creation
        4. Auto-assignment (optional)
        5. Notifications
        """
        from database import SecondOpinion, DermatologistProfile

        # Calculate price
        pricing = self.payment_service.calculate_price(
            urgency=request_data.get("urgency", "routine"),
            specialty=request_data.get("dermatologist_specialty_requested"),
            consultation_type=request_data.get("consultation_method", "chart_review"),
            discount_code=request_data.get("discount_code")
        )

        # Check payment method
        use_credits = request_data.get("use_credits", False)
        payment_required = not use_credits

        # Create request
        opinion = SecondOpinion(
            user_id=user["id"],
            original_diagnosis=request_data["original_diagnosis"],
            original_provider_name=request_data.get("original_provider_name"),
            original_diagnosis_date=request_data.get("original_diagnosis_date"),
            original_treatment_plan=request_data.get("original_treatment_plan"),
            reason_for_second_opinion=request_data["reason_for_second_opinion"],
            specific_questions=request_data.get("specific_questions"),
            concerns=request_data.get("concerns"),
            analysis_id=request_data.get("analysis_id"),
            lesion_group_id=request_data.get("lesion_group_id"),
            dermatologist_specialty_requested=request_data.get("dermatologist_specialty_requested"),
            urgency=request_data.get("urgency", "routine"),
            patient_anxiety_level=request_data.get("patient_anxiety_level"),
            consultation_method=request_data.get("consultation_method", "chart_review"),
            status=SecondOpinionStatus.PAYMENT_PENDING.value if payment_required else SecondOpinionStatus.SUBMITTED.value,
            fee=pricing["total"],
            payment_status=PaymentStatus.PENDING.value if payment_required else PaymentStatus.NOT_REQUIRED.value
        )

        db.add(opinion)
        db.commit()
        db.refresh(opinion)

        # Calculate SLA
        sla_status = self.sla_tracker.get_sla_status(
            opinion.created_at,
            opinion.urgency
        )

        # Auto-assign if payment not required or using credits
        if not payment_required:
            await self._try_auto_assign(opinion, db)

        # Send confirmation notification
        await self.notification_service.send_notification(
            "second_opinion_submitted",
            {"email": user.get("email"), "phone": user.get("phone")},
            {
                "patient_name": user.get("full_name", "Patient"),
                "request_id": str(opinion.id),
                "original_diagnosis": opinion.original_diagnosis,
                "urgency": opinion.urgency.upper(),
                "sla_hours": str(self.sla_tracker.get_sla_config(opinion.urgency).response_hours),
                "tracking_url": f"https://app.skinclassifier.com/second-opinion/{opinion.id}"
            }
        )

        return {
            "request_id": opinion.id,
            "status": opinion.status,
            "pricing": pricing,
            "sla": sla_status,
            "payment_required": payment_required,
            "message": "Second opinion request submitted successfully"
        }

    async def _try_auto_assign(self, opinion, db: Session) -> Optional[int]:
        """Try to auto-assign a dermatologist"""
        from database import DermatologistProfile

        # Get available dermatologists
        available = db.query(DermatologistProfile).filter(
            DermatologistProfile.is_active == True,
            DermatologistProfile.accepts_second_opinions == True,
            DermatologistProfile.availability_status != "unavailable"
        ).all()

        if not available:
            return None

        derm_list = [
            {
                "id": d.id,
                "specializations": d.specializations or [],
                "availability_status": d.availability_status,
                "typical_wait_time_days": d.typical_wait_time_days or 7,
                "average_rating": d.average_rating or 0,
                "total_reviews": d.total_reviews or 0,
                "years_experience": d.years_experience or 0,
                "board_certifications": d.board_certifications or [],
                "total_consultations": d.total_consultations or 0
            }
            for d in available
        ]

        request_data = {
            "original_diagnosis": opinion.original_diagnosis,
            "dermatologist_specialty_requested": opinion.dermatologist_specialty_requested,
            "urgency": opinion.urgency
        }

        assigned_id = self.assignment_algorithm.auto_assign(request_data, derm_list)

        if assigned_id:
            opinion.dermatologist_id = assigned_id
            opinion.status = SecondOpinionStatus.ASSIGNED.value
            db.commit()

            # Notify dermatologist
            dermatologist = db.query(DermatologistProfile).filter(
                DermatologistProfile.id == assigned_id
            ).first()

            if dermatologist:
                await self._notify_dermatologist_assignment(opinion, dermatologist)

        return assigned_id

    async def _notify_dermatologist_assignment(self, opinion, dermatologist):
        """Notify dermatologist of new assignment"""
        deadline = self.sla_tracker.calculate_deadline(opinion.created_at, opinion.urgency)

        questions = opinion.specific_questions or []
        questions_str = "\n".join([f"- {q}" for q in questions]) if questions else "No specific questions"

        await self.notification_service.send_notification(
            "new_case_assigned",
            {"email": dermatologist.email, "phone": dermatologist.phone_number},
            {
                "dermatologist_name": dermatologist.full_name,
                "request_id": str(opinion.id),
                "patient_identifier": f"Patient #{opinion.user_id}",
                "original_diagnosis": opinion.original_diagnosis,
                "urgency": opinion.urgency.upper(),
                "deadline": deadline.strftime("%Y-%m-%d %H:%M UTC"),
                "reason": opinion.reason_for_second_opinion,
                "questions": questions_str,
                "review_url": f"https://app.skinclassifier.com/dermatologist/review/{opinion.id}"
            }
        )

    async def complete_review(self, opinion_id: int, review_data: Dict[str, Any],
                             dermatologist_id: int, db: Session) -> Dict[str, Any]:
        """
        Complete the second opinion review (dermatologist submits review).
        """
        from database import SecondOpinion, User

        opinion = db.query(SecondOpinion).filter(
            SecondOpinion.id == opinion_id,
            SecondOpinion.dermatologist_id == dermatologist_id
        ).first()

        if not opinion:
            raise ValueError("Second opinion request not found or not assigned to you")

        # Update opinion with review data
        opinion.second_opinion_date = datetime.utcnow()
        opinion.second_opinion_diagnosis = review_data.get("diagnosis")
        opinion.second_opinion_notes = review_data.get("notes")
        opinion.second_opinion_treatment_plan = review_data.get("treatment_plan")
        opinion.agrees_with_original_diagnosis = review_data.get("agrees_with_original")
        opinion.diagnosis_confidence_level = review_data.get("confidence_level")
        opinion.differences_from_original = review_data.get("differences")
        opinion.recommended_action = review_data.get("recommended_action")
        opinion.recommended_next_steps = review_data.get("next_steps")
        opinion.additional_tests_needed = review_data.get("additional_tests")
        opinion.biopsy_recommended = review_data.get("biopsy_recommended", False)
        opinion.follow_up_needed = review_data.get("follow_up_needed", False)
        opinion.status = SecondOpinionStatus.COMPLETED.value

        db.commit()

        # Get SLA status
        sla_status = self.sla_tracker.get_sla_status(
            opinion.created_at,
            opinion.urgency,
            opinion.second_opinion_date
        )

        # Notify patient
        user = db.query(User).filter(User.id == opinion.user_id).first()
        if user:
            await self.notification_service.send_notification(
                "second_opinion_completed",
                {"email": user.email, "phone": getattr(user, 'phone', None)},
                {
                    "patient_name": user.full_name or "Patient",
                    "request_id": str(opinion.id),
                    "dermatologist_name": review_data.get("dermatologist_name", "Specialist"),
                    "review_date": opinion.second_opinion_date.strftime("%Y-%m-%d"),
                    "summary": self._generate_summary(review_data),
                    "report_url": f"https://app.skinclassifier.com/second-opinion/{opinion.id}/report"
                },
                [NotificationType.EMAIL, NotificationType.SMS, NotificationType.PUSH]
            )

        return {
            "message": "Second opinion review completed",
            "opinion_id": opinion_id,
            "sla_status": sla_status
        }

    def _generate_summary(self, review_data: Dict[str, Any]) -> str:
        """Generate brief summary for notification"""
        agrees = review_data.get("agrees_with_original")
        action = review_data.get("recommended_action", "")

        if agrees:
            summary = "The specialist agrees with your original diagnosis. "
        else:
            summary = "The specialist has a different assessment. "

        action_summaries = {
            "proceed_as_planned": "Treatment plan is appropriate.",
            "modify_treatment": "Some treatment modifications are recommended.",
            "additional_testing": "Additional testing is recommended.",
            "specialist_referral": "Referral to a specialist is recommended.",
            "urgent_intervention": "Urgent follow-up is recommended."
        }

        summary += action_summaries.get(action, "Please review the full report for details.")
        return summary

    async def check_sla_breaches(self, db: Session) -> List[Dict[str, Any]]:
        """
        Check for SLA warnings and breaches.
        Should be called periodically (e.g., every hour).
        """
        from database import SecondOpinion, DermatologistProfile

        pending_opinions = db.query(SecondOpinion).filter(
            SecondOpinion.status.in_([
                SecondOpinionStatus.ASSIGNED.value,
                SecondOpinionStatus.UNDER_REVIEW.value
            ])
        ).all()

        warnings = []
        for opinion in pending_opinions:
            sla_status = self.sla_tracker.get_sla_status(
                opinion.created_at,
                opinion.urgency
            )

            if sla_status["warning_level"] in ["warning", "critical"]:
                dermatologist = db.query(DermatologistProfile).filter(
                    DermatologistProfile.id == opinion.dermatologist_id
                ).first()

                if dermatologist:
                    await self.notification_service.send_sla_warning(
                        {
                            "id": opinion.id,
                            "deadline": sla_status["deadline"],
                            "status": opinion.status
                        },
                        {
                            "full_name": dermatologist.full_name,
                            "email": dermatologist.email,
                            "phone_number": dermatologist.phone_number
                        },
                        sla_status["hours_remaining"]
                    )

                warnings.append({
                    "opinion_id": opinion.id,
                    "warning_level": sla_status["warning_level"],
                    "hours_remaining": sla_status["hours_remaining"],
                    "dermatologist_id": opinion.dermatologist_id
                })

        return warnings

    def get_dashboard_stats(self, user_id: int, db: Session,
                           is_dermatologist: bool = False) -> Dict[str, Any]:
        """Get dashboard statistics for second opinions"""
        from database import SecondOpinion

        if is_dermatologist:
            query = db.query(SecondOpinion).filter(
                SecondOpinion.dermatologist_id == user_id
            )
        else:
            query = db.query(SecondOpinion).filter(
                SecondOpinion.user_id == user_id
            )

        total = query.count()
        pending = query.filter(SecondOpinion.status.in_([
            SecondOpinionStatus.SUBMITTED.value,
            SecondOpinionStatus.ASSIGNED.value,
            SecondOpinionStatus.UNDER_REVIEW.value
        ])).count()
        completed = query.filter(
            SecondOpinion.status == SecondOpinionStatus.COMPLETED.value
        ).count()

        # Calculate average response time for completed
        completed_opinions = query.filter(
            SecondOpinion.status == SecondOpinionStatus.COMPLETED.value,
            SecondOpinion.second_opinion_date.isnot(None)
        ).all()

        avg_response_hours = 0
        sla_compliance_rate = 0

        if completed_opinions:
            response_times = []
            met_sla = 0

            for o in completed_opinions:
                hours = (o.second_opinion_date - o.created_at).total_seconds() / 3600
                response_times.append(hours)

                sla_config = self.sla_tracker.get_sla_config(o.urgency)
                if hours <= sla_config.response_hours:
                    met_sla += 1

            avg_response_hours = sum(response_times) / len(response_times)
            sla_compliance_rate = (met_sla / len(completed_opinions)) * 100

        return {
            "total_requests": total,
            "pending": pending,
            "completed": completed,
            "cancelled": query.filter(
                SecondOpinion.status == SecondOpinionStatus.CANCELLED.value
            ).count(),
            "average_response_hours": round(avg_response_hours, 1),
            "sla_compliance_rate": round(sla_compliance_rate, 1)
        }


# Global service instance
_workflow_service = None

def get_second_opinion_workflow_service() -> SecondOpinionWorkflowService:
    """Get or create global workflow service instance"""
    global _workflow_service
    if _workflow_service is None:
        _workflow_service = SecondOpinionWorkflowService()
    return _workflow_service
