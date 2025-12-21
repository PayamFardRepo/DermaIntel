"""
Advanced Teledermatology Module

Features:
- WebRTC signaling for live video consultations
- Real-time annotation/drawing support
- Multi-specialist consensus workflow
- Automated triage routing based on AI analysis
- Enhanced store-and-forward capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum
import json
import uuid
import asyncio

from database import (
    get_db, User, DermatologistProfile, Notification,
    ConsensusCase, ConsensusOpinion, ConsensusAssignment
)
from auth import get_current_active_user

router = APIRouter(prefix="/teledermatology/advanced", tags=["Advanced Teledermatology"])


# =============================================================================
# ENUMS AND MODELS
# =============================================================================

class TriagePriority(str, Enum):
    EMERGENCY = "emergency"  # Immediate - possible melanoma, severe burns
    URGENT = "urgent"  # 24-48 hours - suspicious lesions, infections
    STANDARD = "standard"  # 1-2 weeks - routine evaluations
    LOW = "low"  # Non-urgent - cosmetic concerns


class ConsensusStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    CONSENSUS_REACHED = "consensus_reached"
    DISAGREEMENT = "disagreement"
    ESCALATED = "escalated"


class VideoSessionStatus(str, Enum):
    WAITING = "waiting"
    CONNECTING = "connecting"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class AnnotationType(str, Enum):
    DRAWING = "drawing"
    ARROW = "arrow"
    CIRCLE = "circle"
    TEXT = "text"
    MEASUREMENT = "measurement"


# Pydantic Models
class TriageRequest(BaseModel):
    analysis_id: Optional[int] = None
    chief_complaint: str
    symptom_duration: Optional[str] = None
    pain_level: Optional[int] = None
    is_spreading: Optional[bool] = False
    has_fever: Optional[bool] = False
    immunocompromised: Optional[bool] = False
    previous_skin_cancer: Optional[bool] = False
    ai_risk_score: Optional[float] = None
    ai_predicted_conditions: Optional[List[str]] = []


class ConsensusRequest(BaseModel):
    case_id: str
    specialist_ids: List[int]
    case_summary: str
    images: List[str]
    ai_analysis: Optional[Dict[str, Any]] = None
    urgency: str = "standard"
    deadline_hours: int = 72


class SpecialistOpinion(BaseModel):
    case_id: str
    specialist_id: int
    diagnosis: str
    confidence: float
    differential_diagnoses: List[str] = []
    recommended_actions: List[str] = []
    notes: str = ""
    agrees_with_ai: Optional[bool] = None


class StoreForwardCase(BaseModel):
    patient_summary: str
    chief_complaint: str
    history_present_illness: str
    relevant_history: Optional[str] = None
    current_medications: Optional[List[str]] = []
    allergies: Optional[List[str]] = []
    images: List[str]
    ai_analysis_id: Optional[int] = None
    preferred_specialist_id: Optional[int] = None
    urgency: str = "standard"


class Annotation(BaseModel):
    type: AnnotationType
    data: Dict[str, Any]  # coordinates, color, size, text, etc.
    timestamp: float
    author_id: int
    author_name: str


# =============================================================================
# IN-MEMORY STORES (In production, use Redis)
# =============================================================================

# WebRTC signaling
video_sessions: Dict[str, Dict[str, Any]] = {}
session_connections: Dict[str, List[WebSocket]] = {}

# Consensus workflows
consensus_cases: Dict[str, Dict[str, Any]] = {}

# Triage queue
triage_queue: Dict[str, List[Dict[str, Any]]] = {
    "emergency": [],
    "urgent": [],
    "standard": [],
    "low": []
}

# Store-and-forward cases
store_forward_cases: Dict[str, Dict[str, Any]] = {}

# Real-time annotations
session_annotations: Dict[str, List[Dict[str, Any]]] = {}


# =============================================================================
# AUTOMATED TRIAGE ROUTING
# =============================================================================

@router.post("/triage/assess")
async def assess_triage_priority(
    request: TriageRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Automatically assess triage priority based on symptoms and AI analysis.
    Uses a scoring system to determine urgency.
    """
    priority_score = 0
    risk_factors = []
    routing_recommendations = []

    # AI risk score contribution (0-40 points)
    if request.ai_risk_score:
        if request.ai_risk_score >= 0.8:
            priority_score += 40
            risk_factors.append("High AI malignancy risk score")
        elif request.ai_risk_score >= 0.6:
            priority_score += 25
            risk_factors.append("Elevated AI risk score")
        elif request.ai_risk_score >= 0.4:
            priority_score += 15
            risk_factors.append("Moderate AI risk score")

    # AI predicted conditions (0-30 points)
    high_risk_conditions = ["melanoma", "mel", "bcc", "scc", "akiec", "squamous_cell_carcinoma", "basal_cell_carcinoma"]
    urgent_conditions = ["cellulitis", "herpes_zoster", "impetigo", "abscess", "necrotizing_fasciitis"]

    if request.ai_predicted_conditions:
        for condition in request.ai_predicted_conditions:
            condition_lower = condition.lower()
            if any(hrc in condition_lower for hrc in high_risk_conditions):
                priority_score += 30
                risk_factors.append(f"Possible malignancy: {condition}")
                break
            elif any(uc in condition_lower for uc in urgent_conditions):
                priority_score += 20
                risk_factors.append(f"Urgent condition: {condition}")

    # Symptom-based scoring
    if request.has_fever:
        priority_score += 15
        risk_factors.append("Fever present")

    if request.is_spreading:
        priority_score += 15
        risk_factors.append("Condition is spreading")

    if request.pain_level and request.pain_level >= 7:
        priority_score += 10
        risk_factors.append(f"Severe pain (level {request.pain_level})")

    # Risk factor scoring
    if request.immunocompromised:
        priority_score += 20
        risk_factors.append("Patient is immunocompromised")

    if request.previous_skin_cancer:
        priority_score += 15
        risk_factors.append("History of skin cancer")

    # Determine priority level
    if priority_score >= 70:
        priority = TriagePriority.EMERGENCY
        wait_time = "Immediate - within 24 hours"
        routing_recommendations = [
            "Route to on-call dermatologist",
            "Consider emergency department referral",
            "Flag for same-day video consultation"
        ]
    elif priority_score >= 45:
        priority = TriagePriority.URGENT
        wait_time = "24-48 hours"
        routing_recommendations = [
            "Schedule priority video consultation",
            "Assign to available specialist",
            "Monitor for symptom progression"
        ]
    elif priority_score >= 20:
        priority = TriagePriority.STANDARD
        wait_time = "1-2 weeks"
        routing_recommendations = [
            "Add to standard consultation queue",
            "Store-and-forward review acceptable",
            "Patient education materials recommended"
        ]
    else:
        priority = TriagePriority.LOW
        wait_time = "2-4 weeks or as convenient"
        routing_recommendations = [
            "Schedule routine appointment",
            "Asynchronous consultation appropriate",
            "Self-monitoring instructions provided"
        ]

    # Create triage case
    triage_id = f"TRI-{uuid.uuid4().hex[:8].upper()}"
    triage_case = {
        "id": triage_id,
        "user_id": current_user.id,
        "priority": priority.value,
        "priority_score": priority_score,
        "risk_factors": risk_factors,
        "routing_recommendations": routing_recommendations,
        "estimated_wait_time": wait_time,
        "request_data": request.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "status": "pending_assignment"
    }

    # Add to appropriate queue
    triage_queue[priority.value].append(triage_case)

    return {
        "triage_id": triage_id,
        "priority": priority.value,
        "priority_score": priority_score,
        "estimated_wait_time": wait_time,
        "risk_factors": risk_factors,
        "routing_recommendations": routing_recommendations,
        "queue_position": len(triage_queue[priority.value]),
        "message": f"Your case has been triaged as {priority.value.upper()}. {routing_recommendations[0]}"
    }


@router.get("/triage/queue")
async def get_triage_queue(
    priority: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get triage queue (for specialists/admins)."""
    # In production, check if user is a specialist
    if priority:
        return {
            "priority": priority,
            "cases": triage_queue.get(priority, []),
            "count": len(triage_queue.get(priority, []))
        }

    return {
        "queues": {
            "emergency": {"cases": triage_queue["emergency"], "count": len(triage_queue["emergency"])},
            "urgent": {"cases": triage_queue["urgent"], "count": len(triage_queue["urgent"])},
            "standard": {"cases": triage_queue["standard"], "count": len(triage_queue["standard"])},
            "low": {"cases": triage_queue["low"], "count": len(triage_queue["low"])}
        },
        "total_pending": sum(len(q) for q in triage_queue.values())
    }


# =============================================================================
# MULTI-SPECIALIST CONSENSUS WORKFLOW (Database-backed)
# =============================================================================

@router.post("/consensus/create")
async def create_consensus_case_db(
    request: ConsensusRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a multi-specialist consensus case.
    Multiple specialists review and provide opinions, then consensus is determined.
    """
    case_id = request.case_id or f"CON-{uuid.uuid4().hex[:8].upper()}"
    deadline = datetime.utcnow() + timedelta(hours=request.deadline_hours)

    # Get available dermatologists if none specified
    specialist_ids = request.specialist_ids
    if not specialist_ids or specialist_ids == [1, 2, 3]:  # Default placeholder IDs
        # Get actual dermatologists from database
        dermatologists = db.query(DermatologistProfile).filter(
            DermatologistProfile.accepts_second_opinions == True,
            DermatologistProfile.availability_status.in_(["accepting", "limited"])
        ).limit(3).all()

        if dermatologists:
            specialist_ids = [d.id for d in dermatologists]
        else:
            # Create demo dermatologists if none exist
            specialist_ids = _ensure_demo_dermatologists(db)

    # Create consensus case in database
    db_case = ConsensusCase(
        case_id=case_id,
        requesting_user_id=current_user.id,
        case_summary=request.case_summary,
        images=request.images,
        specialist_ids=specialist_ids,
        required_opinions=len(specialist_ids),
        urgency=request.urgency,
        deadline=deadline,
        status="pending"
    )
    db.add(db_case)
    db.flush()  # Get the ID

    # Create assignments and notifications for each specialist
    notifications_sent = 0
    for specialist_id in specialist_ids:
        # Create assignment
        assignment = ConsensusAssignment(
            case_id=db_case.id,
            specialist_id=specialist_id,
            status="assigned",
            notification_sent=True,
            notification_sent_at=datetime.utcnow()
        )
        db.add(assignment)

        # Get specialist details for notification
        specialist = db.query(DermatologistProfile).filter(
            DermatologistProfile.id == specialist_id
        ).first()

        # Create notification for specialist (if they have a linked user account)
        if specialist and specialist.user_id:
            notification = Notification(
                user_id=specialist.user_id,
                notification_type="consensus_request",
                title="New Consensus Review Request",
                message=f"You have been assigned to review case {case_id}. Urgency: {request.urgency.upper()}. Please review and submit your opinion by {deadline.strftime('%Y-%m-%d %H:%M')}.",
                data={
                    "case_id": case_id,
                    "db_case_id": db_case.id,
                    "urgency": request.urgency,
                    "deadline": deadline.isoformat()
                },
                priority="high" if request.urgency == "urgent" else "normal"
            )
            db.add(notification)
            notifications_sent += 1

    db.commit()
    db.refresh(db_case)

    return {
        "case_id": case_id,
        "db_id": db_case.id,
        "status": "created",
        "specialist_count": len(specialist_ids),
        "specialists_notified": notifications_sent,
        "deadline": deadline.isoformat(),
        "message": f"Consensus case created. {len(specialist_ids)} specialists have been assigned."
    }


def _ensure_demo_dermatologists(db: Session) -> List[int]:
    """Create demo dermatologists if none exist."""
    existing = db.query(DermatologistProfile).count()
    if existing >= 3:
        derms = db.query(DermatologistProfile).limit(3).all()
        return [d.id for d in derms]

    # Create demo dermatologists
    demo_derms = [
        DermatologistProfile(
            full_name="Dr. Sarah Chen",
            credentials="MD, FAAD",
            email="dr.chen@dermclinic.example",
            practice_name="Advanced Dermatology Associates",
            city="Boston",
            state="MA",
            specializations=["Dermoscopy", "Skin Cancer", "Melanoma"],
            accepts_second_opinions=True,
            availability_status="accepting"
        ),
        DermatologistProfile(
            full_name="Dr. Michael Roberts",
            credentials="MD, PhD",
            email="dr.roberts@skincare.example",
            practice_name="University Dermatology Center",
            city="New York",
            state="NY",
            specializations=["Mohs Surgery", "Skin Cancer", "Dermatopathology"],
            accepts_second_opinions=True,
            availability_status="accepting"
        ),
        DermatologistProfile(
            full_name="Dr. Emily Watson",
            credentials="DO, FAOCD",
            email="dr.watson@dermatology.example",
            practice_name="Comprehensive Skin Care",
            city="Chicago",
            state="IL",
            specializations=["General Dermatology", "Pediatric", "Cosmetic"],
            accepts_second_opinions=True,
            availability_status="accepting"
        )
    ]

    for derm in demo_derms:
        db.add(derm)
    db.commit()

    return [d.id for d in demo_derms]


@router.post("/consensus/opinion")
async def submit_specialist_opinion_db(
    opinion: SpecialistOpinion,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit a specialist's opinion for a consensus case."""
    # Find the case
    db_case = db.query(ConsensusCase).filter(
        ConsensusCase.case_id == opinion.case_id
    ).first()

    if not db_case:
        raise HTTPException(status_code=404, detail="Consensus case not found")

    # Verify specialist is assigned to this case
    if opinion.specialist_id not in db_case.specialist_ids:
        raise HTTPException(status_code=403, detail="Not authorized for this case")

    # Check if opinion already exists
    existing_opinion = db.query(ConsensusOpinion).filter(
        ConsensusOpinion.case_id == db_case.id,
        ConsensusOpinion.specialist_id == opinion.specialist_id
    ).first()

    if existing_opinion and existing_opinion.submitted:
        raise HTTPException(status_code=400, detail="Opinion already submitted")

    # Create or update opinion
    if existing_opinion:
        existing_opinion.diagnosis = opinion.diagnosis
        existing_opinion.confidence = opinion.confidence
        existing_opinion.differential_diagnoses = opinion.differential_diagnoses
        existing_opinion.recommended_actions = opinion.recommended_actions
        existing_opinion.clinical_notes = opinion.notes
        existing_opinion.agrees_with_ai = opinion.agrees_with_ai
        existing_opinion.submitted = True
        existing_opinion.submitted_at = datetime.utcnow()
        db_opinion = existing_opinion
    else:
        db_opinion = ConsensusOpinion(
            case_id=db_case.id,
            specialist_id=opinion.specialist_id,
            diagnosis=opinion.diagnosis,
            confidence=opinion.confidence,
            differential_diagnoses=opinion.differential_diagnoses,
            recommended_actions=opinion.recommended_actions,
            clinical_notes=opinion.notes,
            agrees_with_ai=opinion.agrees_with_ai,
            submitted=True,
            submitted_at=datetime.utcnow()
        )
        db.add(db_opinion)

    # Update assignment status
    assignment = db.query(ConsensusAssignment).filter(
        ConsensusAssignment.case_id == db_case.id,
        ConsensusAssignment.specialist_id == opinion.specialist_id
    ).first()
    if assignment:
        assignment.status = "completed"
        assignment.completed_at = datetime.utcnow()

    # Update case status
    db_case.status = "in_review"

    db.commit()

    # Check if all opinions are in
    opinions_count = db.query(ConsensusOpinion).filter(
        ConsensusOpinion.case_id == db_case.id,
        ConsensusOpinion.submitted == True
    ).count()

    consensus_result = None
    if opinions_count >= db_case.required_opinions:
        # Calculate consensus
        consensus_result = _calculate_consensus_from_db(db, db_case.id)
        db_case.status = consensus_result["status"]
        db_case.consensus_diagnosis = consensus_result["primary_diagnosis"]
        db_case.consensus_confidence = consensus_result["average_confidence"]
        db_case.agreement_ratio = consensus_result["agreement_ratio"]
        db_case.recommended_actions = consensus_result["recommended_actions"]
        db_case.completed_at = datetime.utcnow()

        # Notify requesting user
        notification = Notification(
            user_id=db_case.requesting_user_id,
            notification_type="consensus_complete",
            title="Consensus Review Complete",
            message=f"Your consensus review for case {db_case.case_id} is complete. Diagnosis: {consensus_result['primary_diagnosis']} ({int(consensus_result['agreement_ratio']*100)}% agreement)",
            data={
                "case_id": db_case.case_id,
                "diagnosis": consensus_result["primary_diagnosis"],
                "agreement_ratio": consensus_result["agreement_ratio"]
            },
            priority="high"
        )
        db.add(notification)
        db.commit()

    return {
        "case_id": opinion.case_id,
        "opinion_recorded": True,
        "opinions_received": opinions_count,
        "opinions_expected": db_case.required_opinions,
        "current_status": db_case.status,
        "consensus_result": consensus_result
    }


def _calculate_consensus_from_db(db: Session, case_id: int) -> Dict[str, Any]:
    """Calculate consensus from database opinions."""
    opinions = db.query(ConsensusOpinion).filter(
        ConsensusOpinion.case_id == case_id,
        ConsensusOpinion.submitted == True
    ).all()

    if not opinions:
        return {"status": "pending", "message": "No opinions yet"}

    diagnoses = [op.diagnosis.lower() for op in opinions]
    confidences = [op.confidence for op in opinions]

    # Find most common diagnosis
    diagnosis_counts = {}
    for d in diagnoses:
        diagnosis_counts[d] = diagnosis_counts.get(d, 0) + 1

    most_common = max(diagnosis_counts.items(), key=lambda x: x[1])
    agreement_ratio = most_common[1] / len(diagnoses)
    avg_confidence = sum(confidences) / len(confidences)

    # Collect all recommended actions
    all_actions = []
    for op in opinions:
        if op.recommended_actions:
            all_actions.extend(op.recommended_actions)

    # Get unique actions with counts
    action_counts = {}
    for action in all_actions:
        action_counts[action] = action_counts.get(action, 0) + 1

    # Actions recommended by majority
    majority_actions = [a for a, c in action_counts.items() if c >= len(opinions) / 2]

    if agreement_ratio >= 0.75:
        status = "consensus_reached"
        message = f"Strong consensus reached: {most_common[0].title()}"
    elif agreement_ratio >= 0.5:
        status = "consensus_reached"
        message = f"Majority consensus: {most_common[0].title()}"
    else:
        status = "disagreement"
        message = "No clear consensus - case may need escalation"

    return {
        "status": status,
        "primary_diagnosis": most_common[0].title(),
        "agreement_ratio": agreement_ratio,
        "average_confidence": avg_confidence,
        "diagnosis_distribution": diagnosis_counts,
        "recommended_actions": majority_actions,
        "message": message,
        "calculated_at": datetime.utcnow().isoformat()
    }


@router.get("/consensus/{case_id}")
async def get_consensus_case_db(
    case_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get details of a consensus case."""
    db_case = db.query(ConsensusCase).filter(
        ConsensusCase.case_id == case_id
    ).first()

    if not db_case:
        raise HTTPException(status_code=404, detail="Consensus case not found")

    # Get opinions
    opinions = db.query(ConsensusOpinion).filter(
        ConsensusOpinion.case_id == db_case.id
    ).all()

    # Get specialist details
    specialists = []
    for spec_id in db_case.specialist_ids:
        spec = db.query(DermatologistProfile).filter(
            DermatologistProfile.id == spec_id
        ).first()
        if spec:
            specialists.append({
                "id": spec.id,
                "name": spec.full_name,
                "credentials": spec.credentials,
                "specializations": spec.specializations
            })

    return {
        "case_id": db_case.case_id,
        "status": db_case.status,
        "case_summary": db_case.case_summary,
        "urgency": db_case.urgency,
        "deadline": db_case.deadline.isoformat() if db_case.deadline else None,
        "created_at": db_case.created_at.isoformat(),
        "specialists": specialists,
        "opinions_received": len([o for o in opinions if o.submitted]),
        "opinions_expected": db_case.required_opinions,
        "consensus_diagnosis": db_case.consensus_diagnosis,
        "consensus_confidence": db_case.consensus_confidence,
        "agreement_ratio": db_case.agreement_ratio,
        "recommended_actions": db_case.recommended_actions,
        "opinions": [
            {
                "specialist_id": o.specialist_id,
                "diagnosis": o.diagnosis,
                "confidence": o.confidence,
                "differential_diagnoses": o.differential_diagnoses,
                "recommended_actions": o.recommended_actions,
                "notes": o.clinical_notes,
                "submitted_at": o.submitted_at.isoformat() if o.submitted_at else None
            }
            for o in opinions if o.submitted
        ] if current_user.id == db_case.requesting_user_id or current_user.account_type == "professional" else []
    }


@router.get("/consensus/user/cases")
async def get_user_consensus_cases(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all consensus cases for the current user."""
    cases = db.query(ConsensusCase).filter(
        ConsensusCase.requesting_user_id == current_user.id
    ).order_by(ConsensusCase.created_at.desc()).all()

    return {
        "cases": [
            {
                "case_id": c.case_id,
                "status": c.status,
                "urgency": c.urgency,
                "created_at": c.created_at.isoformat(),
                "deadline": c.deadline.isoformat() if c.deadline else None,
                "consensus_diagnosis": c.consensus_diagnosis,
                "agreement_ratio": c.agreement_ratio,
                "opinions_received": len([o for o in c.opinions if o.submitted]),
                "opinions_expected": c.required_opinions
            }
            for c in cases
        ],
        "total": len(cases)
    }


@router.get("/consensus/specialist/assigned")
async def get_specialist_assigned_cases(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all consensus cases assigned to the current specialist."""
    # Find dermatologist profile linked to this user
    specialist = db.query(DermatologistProfile).filter(
        DermatologistProfile.user_id == current_user.id
    ).first()

    if not specialist:
        # Check if user is a professional - they might be able to see cases
        if current_user.account_type != "professional":
            raise HTTPException(status_code=403, detail="Not a registered specialist")
        # Create a profile for them
        specialist = DermatologistProfile(
            full_name=current_user.full_name or current_user.username,
            email=current_user.email,
            user_id=current_user.id,
            accepts_second_opinions=True,
            availability_status="accepting"
        )
        db.add(specialist)
        db.commit()
        db.refresh(specialist)

    # Get assignments
    assignments = db.query(ConsensusAssignment).filter(
        ConsensusAssignment.specialist_id == specialist.id,
        ConsensusAssignment.declined == False
    ).all()

    cases = []
    for assignment in assignments:
        case = db.query(ConsensusCase).filter(
            ConsensusCase.id == assignment.case_id
        ).first()
        if case:
            # Check if already submitted opinion
            opinion = db.query(ConsensusOpinion).filter(
                ConsensusOpinion.case_id == case.id,
                ConsensusOpinion.specialist_id == specialist.id,
                ConsensusOpinion.submitted == True
            ).first()

            cases.append({
                "case_id": case.case_id,
                "db_id": case.id,
                "status": case.status,
                "case_summary": case.case_summary,
                "urgency": case.urgency,
                "deadline": case.deadline.isoformat() if case.deadline else None,
                "created_at": case.created_at.isoformat(),
                "assignment_status": assignment.status,
                "opinion_submitted": opinion is not None,
                "images": case.images or []
            })

    return {
        "specialist_id": specialist.id,
        "specialist_name": specialist.full_name,
        "assigned_cases": cases,
        "pending_count": len([c for c in cases if not c["opinion_submitted"]]),
        "completed_count": len([c for c in cases if c["opinion_submitted"]])
    }


@router.post("/consensus/{case_id}/escalate")
async def escalate_consensus_case_db(
    case_id: str,
    reason: str = "No consensus reached",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Escalate a consensus case to a senior specialist."""
    db_case = db.query(ConsensusCase).filter(
        ConsensusCase.case_id == case_id
    ).first()

    if not db_case:
        raise HTTPException(status_code=404, detail="Consensus case not found")

    db_case.status = "escalated"
    db_case.escalated = True
    db_case.escalation_reason = reason

    # Create notification for requesting user
    notification = Notification(
        user_id=db_case.requesting_user_id,
        notification_type="consensus_escalated",
        title="Consensus Case Escalated",
        message=f"Your case {case_id} has been escalated for senior specialist review. Reason: {reason}",
        data={"case_id": case_id, "reason": reason},
        priority="high"
    )
    db.add(notification)
    db.commit()

    return {
        "case_id": case_id,
        "status": "escalated",
        "message": "Case has been escalated for senior specialist review"
    }


# =============================================================================
# ENHANCED STORE-AND-FORWARD
# =============================================================================

@router.post("/store-forward/create")
async def create_store_forward_case(
    case: StoreForwardCase,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create an enhanced store-and-forward case with structured data.
    """
    case_id = f"SAF-{uuid.uuid4().hex[:8].upper()}"

    sf_case = {
        "case_id": case_id,
        "patient_id": current_user.id,
        "patient_summary": case.patient_summary,
        "chief_complaint": case.chief_complaint,
        "history_present_illness": case.history_present_illness,
        "relevant_history": case.relevant_history,
        "current_medications": case.current_medications,
        "allergies": case.allergies,
        "images": case.images,
        "image_count": len(case.images),
        "ai_analysis_id": case.ai_analysis_id,
        "preferred_specialist_id": case.preferred_specialist_id,
        "urgency": case.urgency,
        "status": "submitted",
        "specialist_response": None,
        "created_at": datetime.utcnow().isoformat(),
        "timeline": [
            {
                "event": "case_created",
                "timestamp": datetime.utcnow().isoformat(),
                "actor": "patient"
            }
        ]
    }

    store_forward_cases[case_id] = sf_case

    # Auto-route based on urgency
    if case.urgency == "urgent":
        sf_case["auto_routed"] = True
        sf_case["routing_notes"] = "Auto-routed to next available specialist"

    return {
        "case_id": case_id,
        "status": "submitted",
        "estimated_response_time": "24-48 hours" if case.urgency == "urgent" else "3-5 business days",
        "message": "Your case has been submitted for specialist review"
    }


@router.get("/store-forward/{case_id}")
async def get_store_forward_case(
    case_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get store-and-forward case details."""
    case = store_forward_cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return case


@router.post("/store-forward/{case_id}/respond")
async def respond_to_store_forward(
    case_id: str,
    diagnosis: str,
    recommendations: List[str],
    follow_up_needed: bool = False,
    follow_up_timeframe: Optional[str] = None,
    prescriptions: Optional[List[str]] = None,
    referral_needed: bool = False,
    notes: str = "",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Specialist response to store-and-forward case."""
    case = store_forward_cases.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    case["specialist_response"] = {
        "specialist_id": current_user.id,
        "diagnosis": diagnosis,
        "recommendations": recommendations,
        "follow_up_needed": follow_up_needed,
        "follow_up_timeframe": follow_up_timeframe,
        "prescriptions": prescriptions or [],
        "referral_needed": referral_needed,
        "notes": notes,
        "responded_at": datetime.utcnow().isoformat()
    }

    case["status"] = "responded"
    case["timeline"].append({
        "event": "specialist_responded",
        "timestamp": datetime.utcnow().isoformat(),
        "actor": "specialist"
    })

    return {
        "case_id": case_id,
        "status": "responded",
        "message": "Response recorded and patient will be notified"
    }


# =============================================================================
# WEBRTC VIDEO SESSIONS WITH ANNOTATIONS
# =============================================================================

@router.post("/video/create-session")
async def create_video_session(
    specialist_id: int,
    scheduled_time: Optional[str] = None,
    session_type: str = "consultation",
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new video consultation session."""
    session_id = f"VID-{uuid.uuid4().hex[:8].upper()}"

    session = {
        "session_id": session_id,
        "patient_id": current_user.id,
        "specialist_id": specialist_id,
        "session_type": session_type,
        "scheduled_time": scheduled_time,
        "status": VideoSessionStatus.WAITING.value,
        "created_at": datetime.utcnow().isoformat(),
        "ice_candidates": {"patient": [], "specialist": []},
        "sdp_offers": {},
        "annotations": [],
        "recording_enabled": False,
        "participants": []
    }

    video_sessions[session_id] = session
    session_annotations[session_id] = []

    return {
        "session_id": session_id,
        "status": "created",
        "join_url": f"/teledermatology/video/{session_id}",
        "message": "Video session created. Share the session ID with your specialist."
    }


@router.get("/video/{session_id}")
async def get_video_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get video session details."""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.post("/video/{session_id}/signal")
async def signal_video_session(
    session_id: str,
    signal_type: str,  # offer, answer, ice-candidate
    signal_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Handle WebRTC signaling."""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    role = "patient" if current_user.id == session["patient_id"] else "specialist"

    if signal_type == "offer":
        session["sdp_offers"][role] = signal_data
    elif signal_type == "answer":
        session["sdp_offers"][f"{role}_answer"] = signal_data
    elif signal_type == "ice-candidate":
        session["ice_candidates"][role].append(signal_data)

    # Update status
    if session["sdp_offers"].get("patient") and session["sdp_offers"].get("specialist"):
        session["status"] = VideoSessionStatus.CONNECTING.value

    return {
        "session_id": session_id,
        "signal_received": signal_type,
        "status": session["status"]
    }


@router.post("/video/{session_id}/annotation")
async def add_video_annotation(
    session_id: str,
    annotation: Annotation,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Add annotation during video session."""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    annotation_data = {
        **annotation.dict(),
        "id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat()
    }

    session["annotations"].append(annotation_data)
    session_annotations[session_id].append(annotation_data)

    return {
        "annotation_id": annotation_data["id"],
        "session_id": session_id,
        "status": "added"
    }


@router.get("/video/{session_id}/annotations")
async def get_video_annotations(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all annotations for a video session."""
    annotations = session_annotations.get(session_id, [])
    return {
        "session_id": session_id,
        "annotations": annotations,
        "count": len(annotations)
    }


@router.post("/video/{session_id}/end")
async def end_video_session(
    session_id: str,
    summary: Optional[str] = None,
    follow_up_actions: Optional[List[str]] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """End a video session and save summary."""
    session = video_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session["status"] = VideoSessionStatus.ENDED.value
    session["ended_at"] = datetime.utcnow().isoformat()
    session["summary"] = summary
    session["follow_up_actions"] = follow_up_actions or []

    # Calculate duration
    start_time = datetime.fromisoformat(session["created_at"])
    end_time = datetime.utcnow()
    session["duration_minutes"] = (end_time - start_time).seconds // 60

    return {
        "session_id": session_id,
        "status": "ended",
        "duration_minutes": session["duration_minutes"],
        "annotations_count": len(session["annotations"]),
        "message": "Video session ended successfully"
    }


# =============================================================================
# WEBSOCKET FOR REAL-TIME SIGNALING
# =============================================================================

@router.websocket("/video/{session_id}/ws")
async def video_websocket(
    websocket: WebSocket,
    session_id: str
):
    """WebSocket endpoint for real-time video signaling and annotations."""
    await websocket.accept()

    session = video_sessions.get(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    # Add to connections
    if session_id not in session_connections:
        session_connections[session_id] = []
    session_connections[session_id].append(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")

            if message_type == "signal":
                # Broadcast signal to other participants
                for conn in session_connections[session_id]:
                    if conn != websocket:
                        await conn.send_json(data)

            elif message_type == "annotation":
                # Store and broadcast annotation
                annotation = {
                    **data.get("annotation", {}),
                    "id": str(uuid.uuid4()),
                    "timestamp": datetime.utcnow().isoformat()
                }
                session["annotations"].append(annotation)

                for conn in session_connections[session_id]:
                    await conn.send_json({
                        "type": "annotation",
                        "annotation": annotation
                    })

            elif message_type == "clear_annotations":
                # Clear all annotations
                session["annotations"] = []
                for conn in session_connections[session_id]:
                    await conn.send_json({"type": "annotations_cleared"})

            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        session_connections[session_id].remove(websocket)
        if not session_connections[session_id]:
            del session_connections[session_id]


# =============================================================================
# SPECIALIST AVAILABILITY & ROUTING
# =============================================================================

@router.get("/specialists/available")
async def get_available_specialists(
    urgency: str = "standard",
    specialization: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get available specialists for routing based on urgency and specialization."""
    # In production, this would query the database
    # For now, return mock data structure
    return {
        "urgency": urgency,
        "available_specialists": [],
        "next_available_slot": datetime.utcnow() + timedelta(hours=2),
        "estimated_wait_time": "2-4 hours" if urgency == "urgent" else "24-48 hours",
        "message": "Specialists are being notified"
    }


@router.post("/route/automatic")
async def automatic_routing(
    triage_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Automatically route a triaged case to the best available specialist."""
    # Find the triage case
    triage_case = None
    for priority_queue in triage_queue.values():
        for case in priority_queue:
            if case["id"] == triage_id:
                triage_case = case
                break

    if not triage_case:
        raise HTTPException(status_code=404, detail="Triage case not found")

    # Routing logic (simplified)
    routing_result = {
        "triage_id": triage_id,
        "routing_method": "automatic",
        "assigned_specialist_id": None,  # Would be filled by actual routing
        "routing_factors": [
            "Priority level",
            "Specialist availability",
            "Specialization match",
            "Response time history"
        ],
        "status": "pending_assignment",
        "routed_at": datetime.utcnow().isoformat()
    }

    triage_case["routing_result"] = routing_result

    return routing_result
