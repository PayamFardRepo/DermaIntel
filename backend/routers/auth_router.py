"""
Authentication and User Management Router

Endpoints for:
- User registration and login
- User profile management
- Professional verification
- User settings
"""

from fastapi import APIRouter, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
from typing import Optional

from database import get_db, User, UserProfile, AnalysisHistory
from auth import (
    authenticate_user, create_access_token, get_current_active_user,
    get_password_hash, ACCESS_TOKEN_EXPIRE_MINUTES
)
from models import UserCreate, UserResponse, UserLogin, Token, UserProfileCreate

router = APIRouter(tags=["Authentication & Users"])


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

@router.post("/register", response_model=UserResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account."""
    # Check if user already exists
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )

    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    # Validate account_type and display_mode
    valid_account_types = ["personal", "professional"]
    valid_display_modes = ["simple", "professional"]

    account_type = user.account_type if user.account_type in valid_account_types else "personal"
    display_mode = user.display_mode if user.display_mode in valid_display_modes else "simple"

    # Create new user with role settings
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        account_type=account_type,
        display_mode=display_mode
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@router.post("/login", response_model=Token)
def login_user(form_data: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return access token."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# =============================================================================
# USER PROFILE ENDPOINTS
# =============================================================================

@router.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current authenticated user."""
    return current_user


@router.put("/me/settings")
def update_user_settings(
    display_mode: Optional[str] = None,
    account_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update user display mode and account type settings."""
    updated = False

    if display_mode:
        if display_mode in ["simple", "professional"]:
            current_user.display_mode = display_mode
            updated = True
        else:
            raise HTTPException(status_code=400, detail="Invalid display_mode. Must be 'simple' or 'professional'")

    if account_type:
        if account_type in ["personal", "professional"]:
            current_user.account_type = account_type
            updated = True
        else:
            raise HTTPException(status_code=400, detail="Invalid account_type. Must be 'personal' or 'professional'")

    if updated:
        db.commit()
        db.refresh(current_user)

    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "display_mode": current_user.display_mode or "simple",
        "account_type": current_user.account_type or "personal",
        "is_verified_professional": current_user.is_verified_professional or False,
        "message": "Settings updated successfully" if updated else "No changes made"
    }


@router.get("/me/extended")
def get_extended_user_info(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get extended user info including profile and analysis stats."""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()
    total_analyses = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == current_user.id).count()
    last_analysis = db.query(AnalysisHistory).filter(
        AnalysisHistory.user_id == current_user.id
    ).order_by(AnalysisHistory.created_at.desc()).first()

    last_analysis_date = last_analysis.created_at if last_analysis else None

    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "created_at": current_user.created_at,
        "profile": profile,
        "total_analyses": total_analyses,
        "last_analysis_date": last_analysis_date
    }


# =============================================================================
# PROFESSIONAL VERIFICATION
# =============================================================================

@router.post("/me/professional-verification")
def request_professional_verification(
    license_number: str = Form(...),
    license_state: str = Form(...),
    npi_number: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit professional verification request.
    In production, this triggers a verification workflow.
    """
    current_user.professional_license_number = license_number
    current_user.professional_license_state = license_state
    if npi_number:
        current_user.npi_number = npi_number
    current_user.account_type = "professional"

    db.commit()
    db.refresh(current_user)

    return {
        "message": "Professional verification request submitted",
        "status": "pending",
        "license_number": license_number,
        "license_state": license_state,
        "npi_number": npi_number,
        "note": "Your verification request will be reviewed. You will be notified once verified."
    }


@router.get("/me/professional-status")
def get_professional_status(current_user: User = Depends(get_current_active_user)):
    """Get current professional verification status."""
    return {
        "is_verified_professional": current_user.is_verified_professional or False,
        "professional_license_number": current_user.professional_license_number,
        "professional_license_state": current_user.professional_license_state,
        "npi_number": current_user.npi_number,
        "verification_date": current_user.verification_date.isoformat() if current_user.verification_date else None,
        "account_type": current_user.account_type or "personal",
        "display_mode": current_user.display_mode or "simple"
    }


# =============================================================================
# USER PROFILE DATA
# =============================================================================

@router.get("/profile")
def get_user_profile(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user profile data."""
    profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return profile


@router.post("/profile")
def create_or_update_profile(
    profile_data: UserProfileCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create or update user profile."""
    existing_profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

    if existing_profile:
        for field, value in profile_data.model_dump(exclude_unset=True).items():
            setattr(existing_profile, field, value)
        existing_profile.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing_profile)
        return existing_profile
    else:
        new_profile = UserProfile(
            user_id=current_user.id,
            **profile_data.model_dump(exclude_unset=True)
        )
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        return new_profile


@router.put("/users/me/location")
async def update_user_location(
    city: str = Form(None),
    state: str = Form(None),
    country: str = Form("USA"),
    zip_code: str = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    preferred_distance_miles: int = Form(50),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update user's location for dermatologist matching."""
    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == current_user.id).first()

        if not profile:
            profile = UserProfile(user_id=current_user.id)
            db.add(profile)

        if city:
            profile.city = city
        if state:
            profile.state = state
        if country:
            profile.country = country
        if zip_code:
            profile.zip_code = zip_code
        if latitude is not None:
            profile.latitude = latitude
        if longitude is not None:
            profile.longitude = longitude
        if preferred_distance_miles:
            profile.preferred_distance_miles = preferred_distance_miles

        db.commit()
        db.refresh(profile)

        return {
            "message": "Location updated successfully",
            "location": {
                "city": profile.city,
                "state": profile.state,
                "country": profile.country,
                "zip_code": profile.zip_code,
                "has_coordinates": profile.latitude is not None,
                "preferred_distance_miles": profile.preferred_distance_miles
            }
        }

    except Exception as e:
        db.rollback()
        print(f"Error updating location: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update location: {str(e)}")
