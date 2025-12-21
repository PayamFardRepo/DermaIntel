"""
Clinic Management API Module

This module provides API endpoints for the hybrid user role system:
- Clinic CRUD operations
- Staff management
- Patient-clinic linking
- Invitation system
- Shared analysis management
- Clinical notes
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from typing import List, Optional
from datetime import datetime, timedelta
import secrets
import string

from database import (
    get_db, User, Clinic, ClinicStaff, ClinicPatient,
    SharedAnalysis, ClinicNote, ClinicInvitation, AnalysisHistory
)
from models import (
    ClinicCreate, ClinicUpdate, ClinicResponse, ClinicListResponse,
    ClinicStaffCreate, ClinicStaffInvite, ClinicStaffUpdate, ClinicStaffResponse,
    ClinicPatientCreate, ClinicPatientUpdate, ClinicPatientResponse, PatientClinicListResponse,
    ClinicInvitationCreate, ClinicInvitationResponse, AcceptInvitationRequest,
    SharedAnalysisCreate, SharedAnalysisResponse, BulkShareRequest,
    ClinicNoteCreate, ClinicNoteUpdate, ClinicNoteResponse,
    ClinicDashboardStats, PatientSummaryForClinic, ProviderDashboard, UserRoleSummary,
    UserUpdate, ProfessionalVerificationRequest, ProfessionalVerificationResponse
)
from auth import get_current_user

router = APIRouter(prefix="/api/clinics", tags=["Clinic Management"])


# ============================================================================
# Helper Functions
# ============================================================================

def generate_clinic_code(length: int = 8) -> str:
    """Generate a unique clinic code"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


def generate_invitation_code(length: int = 12) -> str:
    """Generate a unique invitation code"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))


def check_clinic_admin(db: Session, clinic_id: int, user_id: int) -> bool:
    """Check if user is admin of the clinic"""
    staff = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic_id,
        ClinicStaff.user_id == user_id,
        ClinicStaff.role == "admin",
        ClinicStaff.is_active == True
    ).first()
    return staff is not None


def check_clinic_staff(db: Session, clinic_id: int, user_id: int) -> bool:
    """Check if user is any staff member of the clinic"""
    staff = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic_id,
        ClinicStaff.user_id == user_id,
        ClinicStaff.is_active == True
    ).first()
    return staff is not None


def check_clinic_provider(db: Session, clinic_id: int, user_id: int) -> bool:
    """Check if user is a provider (admin or provider role) at the clinic"""
    staff = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic_id,
        ClinicStaff.user_id == user_id,
        ClinicStaff.role.in_(["admin", "provider"]),
        ClinicStaff.is_active == True
    ).first()
    return staff is not None


# ============================================================================
# Clinic CRUD Endpoints
# ============================================================================

@router.post("/", response_model=ClinicResponse, status_code=status.HTTP_201_CREATED)
async def create_clinic(
    clinic_data: ClinicCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new clinic. The creating user becomes the clinic admin.
    Requires verified professional status.
    """
    # Check if user is a verified professional
    if not current_user.is_verified_professional:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only verified healthcare professionals can create clinics"
        )

    # Generate unique clinic code
    clinic_code = generate_clinic_code()
    while db.query(Clinic).filter(Clinic.clinic_code == clinic_code).first():
        clinic_code = generate_clinic_code()

    # Create clinic
    new_clinic = Clinic(
        name=clinic_data.name,
        address=clinic_data.address,
        phone=clinic_data.phone,
        email=clinic_data.email,
        description=clinic_data.description,
        specialty=clinic_data.specialty,
        website=clinic_data.website,
        clinic_code=clinic_code,
        created_by_user_id=current_user.id,
        is_verified=False,  # Requires admin verification
        is_active=True
    )
    db.add(new_clinic)
    db.flush()  # Get the clinic ID

    # Add creator as admin
    admin_staff = ClinicStaff(
        clinic_id=new_clinic.id,
        user_id=current_user.id,
        role="admin",
        title="Practice Administrator",
        is_active=True
    )
    db.add(admin_staff)
    db.commit()
    db.refresh(new_clinic)

    return ClinicResponse(
        id=new_clinic.id,
        name=new_clinic.name,
        address=new_clinic.address,
        phone=new_clinic.phone,
        email=new_clinic.email,
        description=new_clinic.description,
        specialty=new_clinic.specialty,
        website=new_clinic.website,
        clinic_code=new_clinic.clinic_code,
        is_verified=new_clinic.is_verified,
        is_active=new_clinic.is_active,
        created_by_user_id=new_clinic.created_by_user_id,
        created_at=new_clinic.created_at,
        updated_at=new_clinic.updated_at,
        staff_count=1,
        patient_count=0
    )


@router.get("/", response_model=ClinicListResponse)
async def list_my_clinics(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List clinics where current user is a staff member"""
    query = db.query(Clinic).join(ClinicStaff).filter(
        ClinicStaff.user_id == current_user.id,
        ClinicStaff.is_active == True,
        Clinic.is_active == True
    )

    total = query.count()
    clinics = query.offset((page - 1) * page_size).limit(page_size).all()

    clinic_responses = []
    for clinic in clinics:
        staff_count = db.query(ClinicStaff).filter(
            ClinicStaff.clinic_id == clinic.id,
            ClinicStaff.is_active == True
        ).count()
        patient_count = db.query(ClinicPatient).filter(
            ClinicPatient.clinic_id == clinic.id,
            ClinicPatient.is_active == True
        ).count()

        clinic_responses.append(ClinicResponse(
            id=clinic.id,
            name=clinic.name,
            address=clinic.address,
            phone=clinic.phone,
            email=clinic.email,
            description=clinic.description,
            specialty=clinic.specialty,
            website=clinic.website,
            clinic_code=clinic.clinic_code,
            is_verified=clinic.is_verified,
            is_active=clinic.is_active,
            created_by_user_id=clinic.created_by_user_id,
            created_at=clinic.created_at,
            updated_at=clinic.updated_at,
            staff_count=staff_count,
            patient_count=patient_count
        ))

    return ClinicListResponse(
        clinics=clinic_responses,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{clinic_id}", response_model=ClinicResponse)
async def get_clinic(
    clinic_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get clinic details (must be staff or linked patient)"""
    clinic = db.query(Clinic).filter(Clinic.id == clinic_id).first()
    if not clinic:
        raise HTTPException(status_code=404, detail="Clinic not found")

    # Check access
    is_staff = check_clinic_staff(db, clinic_id, current_user.id)
    is_patient = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic_id,
        ClinicPatient.patient_user_id == current_user.id,
        ClinicPatient.is_active == True
    ).first() is not None

    if not is_staff and not is_patient:
        raise HTTPException(status_code=403, detail="Access denied")

    staff_count = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic.id,
        ClinicStaff.is_active == True
    ).count()
    patient_count = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic.id,
        ClinicPatient.is_active == True
    ).count()

    return ClinicResponse(
        id=clinic.id,
        name=clinic.name,
        address=clinic.address,
        phone=clinic.phone,
        email=clinic.email,
        description=clinic.description,
        specialty=clinic.specialty,
        website=clinic.website,
        clinic_code=clinic.clinic_code,
        is_verified=clinic.is_verified,
        is_active=clinic.is_active,
        created_by_user_id=clinic.created_by_user_id,
        created_at=clinic.created_at,
        updated_at=clinic.updated_at,
        staff_count=staff_count,
        patient_count=patient_count
    )


@router.put("/{clinic_id}", response_model=ClinicResponse)
async def update_clinic(
    clinic_id: int,
    clinic_data: ClinicUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update clinic details (admin only)"""
    if not check_clinic_admin(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Admin access required")

    clinic = db.query(Clinic).filter(Clinic.id == clinic_id).first()
    if not clinic:
        raise HTTPException(status_code=404, detail="Clinic not found")

    # Update fields
    update_data = clinic_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(clinic, field, value)
    clinic.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(clinic)

    staff_count = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic.id,
        ClinicStaff.is_active == True
    ).count()
    patient_count = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic.id,
        ClinicPatient.is_active == True
    ).count()

    return ClinicResponse(
        id=clinic.id,
        name=clinic.name,
        address=clinic.address,
        phone=clinic.phone,
        email=clinic.email,
        description=clinic.description,
        specialty=clinic.specialty,
        website=clinic.website,
        clinic_code=clinic.clinic_code,
        is_verified=clinic.is_verified,
        is_active=clinic.is_active,
        created_by_user_id=clinic.created_by_user_id,
        created_at=clinic.created_at,
        updated_at=clinic.updated_at,
        staff_count=staff_count,
        patient_count=patient_count
    )


# ============================================================================
# Staff Management Endpoints
# ============================================================================

@router.get("/{clinic_id}/staff", response_model=List[ClinicStaffResponse])
async def list_clinic_staff(
    clinic_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all staff members of a clinic"""
    if not check_clinic_staff(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    staff_list = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic_id
    ).all()

    responses = []
    for staff in staff_list:
        user = db.query(User).filter(User.id == staff.user_id).first()
        responses.append(ClinicStaffResponse(
            id=staff.id,
            clinic_id=staff.clinic_id,
            user_id=staff.user_id,
            role=staff.role,
            title=staff.title,
            department=staff.department,
            is_active=staff.is_active,
            joined_at=staff.joined_at,
            user_name=user.full_name or user.username if user else None,
            user_email=user.email if user else None,
            is_verified_professional=user.is_verified_professional if user else False
        ))

    return responses


@router.post("/{clinic_id}/staff", response_model=ClinicStaffResponse, status_code=status.HTTP_201_CREATED)
async def add_clinic_staff(
    clinic_id: int,
    staff_data: ClinicStaffCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Add a staff member to the clinic (admin only)"""
    if not check_clinic_admin(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Admin access required")

    # Check if user exists
    user = db.query(User).filter(User.id == staff_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if already staff
    existing = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic_id,
        ClinicStaff.user_id == staff_data.user_id
    ).first()
    if existing:
        if existing.is_active:
            raise HTTPException(status_code=400, detail="User is already a staff member")
        # Reactivate
        existing.is_active = True
        existing.role = staff_data.role
        existing.title = staff_data.title
        existing.department = staff_data.department
        db.commit()
        db.refresh(existing)
        return ClinicStaffResponse(
            id=existing.id,
            clinic_id=existing.clinic_id,
            user_id=existing.user_id,
            role=existing.role,
            title=existing.title,
            department=existing.department,
            is_active=existing.is_active,
            joined_at=existing.joined_at,
            user_name=user.full_name or user.username,
            user_email=user.email,
            is_verified_professional=user.is_verified_professional
        )

    # Create new staff record
    new_staff = ClinicStaff(
        clinic_id=clinic_id,
        user_id=staff_data.user_id,
        role=staff_data.role,
        title=staff_data.title,
        department=staff_data.department,
        is_active=True
    )
    db.add(new_staff)
    db.commit()
    db.refresh(new_staff)

    return ClinicStaffResponse(
        id=new_staff.id,
        clinic_id=new_staff.clinic_id,
        user_id=new_staff.user_id,
        role=new_staff.role,
        title=new_staff.title,
        department=new_staff.department,
        is_active=new_staff.is_active,
        joined_at=new_staff.joined_at,
        user_name=user.full_name or user.username,
        user_email=user.email,
        is_verified_professional=user.is_verified_professional
    )


@router.put("/{clinic_id}/staff/{staff_id}", response_model=ClinicStaffResponse)
async def update_clinic_staff(
    clinic_id: int,
    staff_id: int,
    staff_data: ClinicStaffUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a staff member's role/info (admin only)"""
    if not check_clinic_admin(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Admin access required")

    staff = db.query(ClinicStaff).filter(
        ClinicStaff.id == staff_id,
        ClinicStaff.clinic_id == clinic_id
    ).first()
    if not staff:
        raise HTTPException(status_code=404, detail="Staff member not found")

    # Prevent removing the last admin
    if staff_data.role and staff_data.role != "admin" and staff.role == "admin":
        admin_count = db.query(ClinicStaff).filter(
            ClinicStaff.clinic_id == clinic_id,
            ClinicStaff.role == "admin",
            ClinicStaff.is_active == True
        ).count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot remove the last admin")

    update_data = staff_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(staff, field, value)

    db.commit()
    db.refresh(staff)

    user = db.query(User).filter(User.id == staff.user_id).first()

    return ClinicStaffResponse(
        id=staff.id,
        clinic_id=staff.clinic_id,
        user_id=staff.user_id,
        role=staff.role,
        title=staff.title,
        department=staff.department,
        is_active=staff.is_active,
        joined_at=staff.joined_at,
        user_name=user.full_name or user.username if user else None,
        user_email=user.email if user else None,
        is_verified_professional=user.is_verified_professional if user else False
    )


@router.delete("/{clinic_id}/staff/{staff_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_clinic_staff(
    clinic_id: int,
    staff_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove a staff member from the clinic (admin only, or self-removal)"""
    staff = db.query(ClinicStaff).filter(
        ClinicStaff.id == staff_id,
        ClinicStaff.clinic_id == clinic_id
    ).first()
    if not staff:
        raise HTTPException(status_code=404, detail="Staff member not found")

    # Allow self-removal or admin removal
    is_admin = check_clinic_admin(db, clinic_id, current_user.id)
    is_self = staff.user_id == current_user.id

    if not is_admin and not is_self:
        raise HTTPException(status_code=403, detail="Access denied")

    # Prevent removing the last admin
    if staff.role == "admin":
        admin_count = db.query(ClinicStaff).filter(
            ClinicStaff.clinic_id == clinic_id,
            ClinicStaff.role == "admin",
            ClinicStaff.is_active == True
        ).count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot remove the last admin")

    staff.is_active = False
    db.commit()


# ============================================================================
# Patient Management Endpoints
# ============================================================================

@router.get("/{clinic_id}/patients", response_model=List[ClinicPatientResponse])
async def list_clinic_patients(
    clinic_id: int,
    search: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List patients linked to the clinic (staff only)"""
    if not check_clinic_staff(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    query = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic_id,
        ClinicPatient.is_active == True
    )

    patients = query.offset((page - 1) * page_size).limit(page_size).all()

    responses = []
    for patient in patients:
        user = db.query(User).filter(User.id == patient.patient_user_id).first()
        shared_count = db.query(SharedAnalysis).filter(
            SharedAnalysis.clinic_id == clinic_id,
            SharedAnalysis.patient_user_id == patient.patient_user_id
        ).count()

        # Find last visit (last shared analysis or note)
        last_shared = db.query(SharedAnalysis).filter(
            SharedAnalysis.clinic_id == clinic_id,
            SharedAnalysis.patient_user_id == patient.patient_user_id
        ).order_by(SharedAnalysis.shared_at.desc()).first()

        responses.append(ClinicPatientResponse(
            id=patient.id,
            clinic_id=patient.clinic_id,
            patient_user_id=patient.patient_user_id,
            consent_level=patient.consent_level,
            patient_notes=patient.patient_notes,
            is_active=patient.is_active,
            linked_at=patient.linked_at,
            updated_at=patient.updated_at,
            patient_name=user.full_name or user.username if user else None,
            patient_email=user.email if user else None,
            total_shared_analyses=shared_count,
            last_visit=last_shared.shared_at if last_shared else None
        ))

    return responses


@router.post("/{clinic_id}/patients/invite", response_model=ClinicInvitationResponse, status_code=status.HTTP_201_CREATED)
async def invite_patient(
    clinic_id: int,
    invitation_data: ClinicInvitationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Invite a patient to link with the clinic"""
    if not check_clinic_staff(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    clinic = db.query(Clinic).filter(Clinic.id == clinic_id).first()
    if not clinic:
        raise HTTPException(status_code=404, detail="Clinic not found")

    # Generate unique invitation code
    invitation_code = generate_invitation_code()
    while db.query(ClinicInvitation).filter(ClinicInvitation.invitation_code == invitation_code).first():
        invitation_code = generate_invitation_code()

    # Create invitation
    invitation = ClinicInvitation(
        clinic_id=clinic_id,
        patient_email=invitation_data.patient_email,
        invitation_code=invitation_code,
        consent_level=invitation_data.consent_level,
        message=invitation_data.message,
        invited_by_user_id=current_user.id,
        status="pending",
        expires_at=datetime.utcnow() + timedelta(days=invitation_data.expires_in_days)
    )
    db.add(invitation)
    db.commit()
    db.refresh(invitation)

    return ClinicInvitationResponse(
        id=invitation.id,
        clinic_id=invitation.clinic_id,
        clinic_name=clinic.name,
        invitation_code=invitation.invitation_code,
        patient_email=invitation.patient_email,
        consent_level=invitation.consent_level,
        message=invitation.message,
        status=invitation.status,
        created_at=invitation.created_at,
        expires_at=invitation.expires_at,
        responded_at=invitation.responded_at
    )


# ============================================================================
# Patient-Side Endpoints
# ============================================================================

@router.get("/invitations/pending", response_model=List[ClinicInvitationResponse])
async def get_my_pending_invitations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get pending invitations for the current user"""
    invitations = db.query(ClinicInvitation).filter(
        ClinicInvitation.patient_email == current_user.email,
        ClinicInvitation.status == "pending",
        ClinicInvitation.expires_at > datetime.utcnow()
    ).all()

    responses = []
    for inv in invitations:
        clinic = db.query(Clinic).filter(Clinic.id == inv.clinic_id).first()
        responses.append(ClinicInvitationResponse(
            id=inv.id,
            clinic_id=inv.clinic_id,
            clinic_name=clinic.name if clinic else "Unknown",
            invitation_code=inv.invitation_code,
            patient_email=inv.patient_email,
            consent_level=inv.consent_level,
            message=inv.message,
            status=inv.status,
            created_at=inv.created_at,
            expires_at=inv.expires_at,
            responded_at=inv.responded_at
        ))

    return responses


@router.post("/invitations/respond", response_model=ClinicPatientResponse)
async def respond_to_invitation(
    response_data: AcceptInvitationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Accept or decline a clinic invitation"""
    invitation = db.query(ClinicInvitation).filter(
        ClinicInvitation.invitation_code == response_data.invitation_code
    ).first()

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found")

    if invitation.patient_email != current_user.email:
        raise HTTPException(status_code=403, detail="This invitation is not for you")

    if invitation.status != "pending":
        raise HTTPException(status_code=400, detail=f"Invitation already {invitation.status}")

    if invitation.expires_at < datetime.utcnow():
        invitation.status = "expired"
        db.commit()
        raise HTTPException(status_code=400, detail="Invitation has expired")

    invitation.responded_at = datetime.utcnow()

    if not response_data.accept:
        invitation.status = "declined"
        db.commit()
        raise HTTPException(status_code=200, detail="Invitation declined")

    invitation.status = "accepted"

    # Check if already linked
    existing = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == invitation.clinic_id,
        ClinicPatient.patient_user_id == current_user.id
    ).first()

    if existing:
        existing.is_active = True
        existing.consent_level = response_data.consent_level or invitation.consent_level
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        patient_record = existing
    else:
        # Create patient-clinic link
        patient_record = ClinicPatient(
            clinic_id=invitation.clinic_id,
            patient_user_id=current_user.id,
            consent_level=response_data.consent_level or invitation.consent_level,
            is_active=True
        )
        db.add(patient_record)
        db.commit()
        db.refresh(patient_record)

    clinic = db.query(Clinic).filter(Clinic.id == invitation.clinic_id).first()

    return ClinicPatientResponse(
        id=patient_record.id,
        clinic_id=patient_record.clinic_id,
        patient_user_id=patient_record.patient_user_id,
        consent_level=patient_record.consent_level,
        patient_notes=patient_record.patient_notes,
        is_active=patient_record.is_active,
        linked_at=patient_record.linked_at,
        updated_at=patient_record.updated_at,
        patient_name=current_user.full_name or current_user.username,
        patient_email=current_user.email
    )


@router.get("/my-clinics", response_model=PatientClinicListResponse)
async def get_my_clinics_as_patient(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get clinics where current user is a patient"""
    patient_links = db.query(ClinicPatient).filter(
        ClinicPatient.patient_user_id == current_user.id,
        ClinicPatient.is_active == True
    ).all()

    clinics = []
    for link in patient_links:
        clinic = db.query(Clinic).filter(Clinic.id == link.clinic_id).first()
        if clinic:
            clinics.append(ClinicResponse(
                id=clinic.id,
                name=clinic.name,
                address=clinic.address,
                phone=clinic.phone,
                email=clinic.email,
                description=clinic.description,
                specialty=clinic.specialty,
                website=clinic.website,
                clinic_code=clinic.clinic_code,
                is_verified=clinic.is_verified,
                is_active=clinic.is_active,
                created_by_user_id=clinic.created_by_user_id,
                created_at=clinic.created_at,
                updated_at=clinic.updated_at
            ))

    return PatientClinicListResponse(clinics=clinics, total=len(clinics))


@router.put("/my-clinics/{clinic_id}/consent", response_model=ClinicPatientResponse)
async def update_my_consent(
    clinic_id: int,
    update_data: ClinicPatientUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update consent level for a clinic (patient self-management)"""
    patient_link = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic_id,
        ClinicPatient.patient_user_id == current_user.id
    ).first()

    if not patient_link:
        raise HTTPException(status_code=404, detail="Not linked to this clinic")

    if update_data.consent_level:
        patient_link.consent_level = update_data.consent_level
    if update_data.is_active is not None:
        patient_link.is_active = update_data.is_active
    patient_link.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(patient_link)

    return ClinicPatientResponse(
        id=patient_link.id,
        clinic_id=patient_link.clinic_id,
        patient_user_id=patient_link.patient_user_id,
        consent_level=patient_link.consent_level,
        patient_notes=patient_link.patient_notes,
        is_active=patient_link.is_active,
        linked_at=patient_link.linked_at,
        updated_at=patient_link.updated_at,
        patient_name=current_user.full_name or current_user.username,
        patient_email=current_user.email
    )


# ============================================================================
# Shared Analysis Endpoints
# ============================================================================

@router.post("/share-analysis", response_model=SharedAnalysisResponse, status_code=status.HTTP_201_CREATED)
async def share_analysis_with_clinic(
    share_data: SharedAnalysisCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Share an analysis with a clinic (patient action)"""
    # Verify the analysis belongs to the user
    analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.id == share_data.analysis_id,
        AnalysisHistory.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found or not yours")

    # Verify patient is linked to the clinic
    patient_link = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == share_data.clinic_id,
        ClinicPatient.patient_user_id == current_user.id,
        ClinicPatient.is_active == True
    ).first()

    if not patient_link:
        raise HTTPException(status_code=403, detail="Not linked to this clinic")

    # Check if already shared
    existing = db.query(SharedAnalysis).filter(
        SharedAnalysis.analysis_id == share_data.analysis_id,
        SharedAnalysis.clinic_id == share_data.clinic_id
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Analysis already shared with this clinic")

    # Create share record
    shared = SharedAnalysis(
        analysis_id=share_data.analysis_id,
        clinic_id=share_data.clinic_id,
        patient_user_id=current_user.id,
        shared_by_user_id=current_user.id,
        share_notes=share_data.share_notes
    )
    db.add(shared)
    db.commit()
    db.refresh(shared)

    return SharedAnalysisResponse(
        id=shared.id,
        analysis_id=shared.analysis_id,
        clinic_id=shared.clinic_id,
        patient_user_id=shared.patient_user_id,
        shared_by_user_id=shared.shared_by_user_id,
        share_notes=shared.share_notes,
        shared_at=shared.shared_at,
        viewed_at=shared.viewed_at,
        viewed_by_user_id=shared.viewed_by_user_id,
        analysis_type=analysis.analysis_type,
        predicted_class=analysis.predicted_class,
        risk_level=analysis.risk_level,
        created_at=analysis.created_at
    )


@router.get("/{clinic_id}/shared-analyses", response_model=List[SharedAnalysisResponse])
async def get_clinic_shared_analyses(
    clinic_id: int,
    patient_id: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get analyses shared with the clinic (staff only)"""
    if not check_clinic_staff(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    query = db.query(SharedAnalysis).filter(SharedAnalysis.clinic_id == clinic_id)

    if patient_id:
        query = query.filter(SharedAnalysis.patient_user_id == patient_id)

    shared_list = query.order_by(SharedAnalysis.shared_at.desc()).offset(
        (page - 1) * page_size
    ).limit(page_size).all()

    responses = []
    for shared in shared_list:
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == shared.analysis_id
        ).first()

        responses.append(SharedAnalysisResponse(
            id=shared.id,
            analysis_id=shared.analysis_id,
            clinic_id=shared.clinic_id,
            patient_user_id=shared.patient_user_id,
            shared_by_user_id=shared.shared_by_user_id,
            share_notes=shared.share_notes,
            shared_at=shared.shared_at,
            viewed_at=shared.viewed_at,
            viewed_by_user_id=shared.viewed_by_user_id,
            analysis_type=analysis.analysis_type if analysis else None,
            predicted_class=analysis.predicted_class if analysis else None,
            risk_level=analysis.risk_level if analysis else None,
            created_at=analysis.created_at if analysis else None
        ))

    return responses


@router.post("/{clinic_id}/shared-analyses/{shared_id}/view")
async def mark_analysis_viewed(
    clinic_id: int,
    shared_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Mark a shared analysis as viewed by clinic staff"""
    if not check_clinic_staff(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    shared = db.query(SharedAnalysis).filter(
        SharedAnalysis.id == shared_id,
        SharedAnalysis.clinic_id == clinic_id
    ).first()

    if not shared:
        raise HTTPException(status_code=404, detail="Shared analysis not found")

    if not shared.viewed_at:
        shared.viewed_at = datetime.utcnow()
        shared.viewed_by_user_id = current_user.id
        db.commit()

    return {"message": "Marked as viewed"}


# ============================================================================
# Clinical Notes Endpoints
# ============================================================================

@router.post("/{clinic_id}/notes", response_model=ClinicNoteResponse, status_code=status.HTTP_201_CREATED)
async def create_clinical_note(
    clinic_id: int,
    note_data: ClinicNoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a clinical note for a patient (provider only)"""
    if not check_clinic_provider(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Provider access required")

    # Verify patient is linked to clinic
    patient_link = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic_id,
        ClinicPatient.patient_user_id == note_data.patient_user_id,
        ClinicPatient.is_active == True
    ).first()

    if not patient_link:
        raise HTTPException(status_code=404, detail="Patient not linked to this clinic")

    # Get staff record for title
    staff = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic_id,
        ClinicStaff.user_id == current_user.id
    ).first()

    note = ClinicNote(
        clinic_id=clinic_id,
        patient_user_id=note_data.patient_user_id,
        provider_user_id=current_user.id,
        related_analysis_id=note_data.related_analysis_id,
        note_type=note_data.note_type,
        note_content=note_data.note_content,
        is_private=note_data.is_private
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    return ClinicNoteResponse(
        id=note.id,
        clinic_id=note.clinic_id,
        patient_user_id=note.patient_user_id,
        provider_user_id=note.provider_user_id,
        related_analysis_id=note.related_analysis_id,
        note_type=note.note_type,
        note_content=note.note_content,
        is_private=note.is_private,
        created_at=note.created_at,
        updated_at=note.updated_at,
        provider_name=current_user.full_name or current_user.username,
        provider_title=staff.title if staff else None
    )


@router.get("/{clinic_id}/patients/{patient_id}/notes", response_model=List[ClinicNoteResponse])
async def get_patient_notes(
    clinic_id: int,
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get clinical notes for a patient"""
    is_staff = check_clinic_staff(db, clinic_id, current_user.id)
    is_patient = current_user.id == patient_id

    if not is_staff and not is_patient:
        raise HTTPException(status_code=403, detail="Access denied")

    query = db.query(ClinicNote).filter(
        ClinicNote.clinic_id == clinic_id,
        ClinicNote.patient_user_id == patient_id
    )

    # Patients can't see private notes
    if is_patient and not is_staff:
        query = query.filter(ClinicNote.is_private == False)

    notes = query.order_by(ClinicNote.created_at.desc()).all()

    responses = []
    for note in notes:
        provider = db.query(User).filter(User.id == note.provider_user_id).first()
        staff = db.query(ClinicStaff).filter(
            ClinicStaff.clinic_id == clinic_id,
            ClinicStaff.user_id == note.provider_user_id
        ).first()

        responses.append(ClinicNoteResponse(
            id=note.id,
            clinic_id=note.clinic_id,
            patient_user_id=note.patient_user_id,
            provider_user_id=note.provider_user_id,
            related_analysis_id=note.related_analysis_id,
            note_type=note.note_type,
            note_content=note.note_content,
            is_private=note.is_private,
            created_at=note.created_at,
            updated_at=note.updated_at,
            provider_name=provider.full_name or provider.username if provider else None,
            provider_title=staff.title if staff else None
        ))

    return responses


# ============================================================================
# Dashboard Endpoints
# ============================================================================

@router.get("/{clinic_id}/dashboard", response_model=ClinicDashboardStats)
async def get_clinic_dashboard(
    clinic_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get clinic dashboard statistics"""
    if not check_clinic_staff(db, clinic_id, current_user.id):
        raise HTTPException(status_code=403, detail="Access denied")

    # Total and active patients
    total_patients = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic_id
    ).count()
    active_patients = db.query(ClinicPatient).filter(
        ClinicPatient.clinic_id == clinic_id,
        ClinicPatient.is_active == True
    ).count()

    # Shared analyses
    total_shared = db.query(SharedAnalysis).filter(
        SharedAnalysis.clinic_id == clinic_id
    ).count()

    # This month
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    analyses_this_month = db.query(SharedAnalysis).filter(
        SharedAnalysis.clinic_id == clinic_id,
        SharedAnalysis.shared_at >= month_start
    ).count()

    # Pending reviews (unviewed shared analyses)
    pending_reviews = db.query(SharedAnalysis).filter(
        SharedAnalysis.clinic_id == clinic_id,
        SharedAnalysis.viewed_at == None
    ).count()

    # Staff count
    staff_count = db.query(ClinicStaff).filter(
        ClinicStaff.clinic_id == clinic_id,
        ClinicStaff.is_active == True
    ).count()

    # Recent activity (last 10 shared analyses)
    recent = db.query(SharedAnalysis).filter(
        SharedAnalysis.clinic_id == clinic_id
    ).order_by(SharedAnalysis.shared_at.desc()).limit(10).all()

    recent_activity = []
    for shared in recent:
        patient = db.query(User).filter(User.id == shared.patient_user_id).first()
        analysis = db.query(AnalysisHistory).filter(
            AnalysisHistory.id == shared.analysis_id
        ).first()
        recent_activity.append({
            "type": "analysis_shared",
            "patient_name": patient.full_name or patient.username if patient else "Unknown",
            "analysis_type": analysis.analysis_type if analysis else None,
            "predicted_class": analysis.predicted_class if analysis else None,
            "timestamp": shared.shared_at.isoformat()
        })

    return ClinicDashboardStats(
        total_patients=total_patients,
        active_patients=active_patients,
        total_analyses_shared=total_shared,
        analyses_this_month=analyses_this_month,
        pending_reviews=pending_reviews,
        staff_count=staff_count,
        recent_activity=recent_activity
    )


# ============================================================================
# User Role Summary
# ============================================================================

@router.get("/user/role-summary", response_model=UserRoleSummary)
async def get_user_role_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get summary of user's roles across clinics"""
    # Clinics as staff
    staff_records = db.query(ClinicStaff).filter(
        ClinicStaff.user_id == current_user.id,
        ClinicStaff.is_active == True
    ).all()

    clinics_as_staff = []
    for staff in staff_records:
        clinic = db.query(Clinic).filter(Clinic.id == staff.clinic_id).first()
        if clinic:
            clinics_as_staff.append(ClinicResponse(
                id=clinic.id,
                name=clinic.name,
                address=clinic.address,
                phone=clinic.phone,
                email=clinic.email,
                description=clinic.description,
                specialty=clinic.specialty,
                website=clinic.website,
                clinic_code=clinic.clinic_code,
                is_verified=clinic.is_verified,
                is_active=clinic.is_active,
                created_by_user_id=clinic.created_by_user_id,
                created_at=clinic.created_at,
                updated_at=clinic.updated_at
            ))

    # Clinics as patient
    patient_records = db.query(ClinicPatient).filter(
        ClinicPatient.patient_user_id == current_user.id,
        ClinicPatient.is_active == True
    ).all()

    clinics_as_patient = []
    for patient in patient_records:
        clinic = db.query(Clinic).filter(Clinic.id == patient.clinic_id).first()
        if clinic:
            clinics_as_patient.append(ClinicResponse(
                id=clinic.id,
                name=clinic.name,
                address=clinic.address,
                phone=clinic.phone,
                email=clinic.email,
                description=clinic.description,
                specialty=clinic.specialty,
                website=clinic.website,
                clinic_code=clinic.clinic_code,
                is_verified=clinic.is_verified,
                is_active=clinic.is_active,
                created_by_user_id=clinic.created_by_user_id,
                created_at=clinic.created_at,
                updated_at=clinic.updated_at
            ))

    # Pending invitations
    invitations = db.query(ClinicInvitation).filter(
        ClinicInvitation.patient_email == current_user.email,
        ClinicInvitation.status == "pending",
        ClinicInvitation.expires_at > datetime.utcnow()
    ).all()

    pending_invitations = []
    for inv in invitations:
        clinic = db.query(Clinic).filter(Clinic.id == inv.clinic_id).first()
        pending_invitations.append(ClinicInvitationResponse(
            id=inv.id,
            clinic_id=inv.clinic_id,
            clinic_name=clinic.name if clinic else "Unknown",
            invitation_code=inv.invitation_code,
            patient_email=inv.patient_email,
            consent_level=inv.consent_level,
            message=inv.message,
            status=inv.status,
            created_at=inv.created_at,
            expires_at=inv.expires_at,
            responded_at=inv.responded_at
        ))

    return UserRoleSummary(
        user_id=current_user.id,
        account_type=current_user.account_type or "personal",
        display_mode=current_user.display_mode or "simple",
        is_verified_professional=current_user.is_verified_professional or False,
        clinics_as_staff=clinics_as_staff,
        clinics_as_patient=clinics_as_patient,
        pending_invitations=pending_invitations
    )
