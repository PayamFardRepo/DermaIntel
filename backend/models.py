from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List, Any
from datetime import datetime

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str
    account_type: Optional[str] = "personal"  # "personal" or "professional"
    display_mode: Optional[str] = "simple"    # "simple" or "professional"

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    account_type: Optional[str] = "personal"
    display_mode: Optional[str] = "simple"
    is_verified_professional: Optional[bool] = False

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None


# Analysis History Models
class AnalysisHistoryBase(BaseModel):
    image_filename: Optional[str] = None
    analysis_type: str
    is_lesion: Optional[bool] = None
    binary_confidence: Optional[float] = None
    binary_probabilities: Optional[Dict[str, float]] = None
    predicted_class: Optional[str] = None
    lesion_confidence: Optional[float] = None
    lesion_probabilities: Optional[Dict[str, float]] = None
    risk_level: Optional[str] = None
    risk_recommendation: Optional[str] = None
    image_quality_score: Optional[float] = None
    image_quality_passed: Optional[bool] = None
    quality_issues: Optional[List[Dict[str, Any]]] = None
    processing_time_seconds: Optional[float] = None
    model_version: Optional[str] = None


class AnalysisHistoryCreate(AnalysisHistoryBase):
    pass


class AnalysisHistoryResponse(AnalysisHistoryBase):
    id: int
    user_id: int
    image_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# User Profile Models
class UserProfileBase(BaseModel):
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    emergency_contact: Optional[str] = None
    medical_history: Optional[str] = None
    skin_type: Optional[str] = None
    family_history: Optional[str] = None
    ethnicity: Optional[str] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    privacy_settings: Optional[Dict[str, Any]] = None


class UserProfileCreate(UserProfileBase):
    pass


class UserProfileUpdate(UserProfileBase):
    pass


class UserProfileResponse(UserProfileBase):
    id: int
    user_id: int
    total_analyses: int
    last_analysis_date: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Extended User Response with Profile
class UserWithProfileResponse(UserResponse):
    profile: Optional[UserProfileResponse] = None
    total_analyses: int = 0
    last_analysis_date: Optional[datetime] = None


# Analysis Statistics
class AnalysisStatsResponse(BaseModel):
    total_analyses: int
    lesion_detections: int
    non_lesion_detections: int
    most_common_diagnosis: Optional[str] = None
    average_confidence: float
    recent_analyses: List[AnalysisHistoryResponse]
    monthly_analysis_counts: Dict[str, int]


# ============================================================================
# Clinical Context Models - For Bayesian Prior Adjustment
# ============================================================================

class FitzpatrickSkinType(str):
    """Fitzpatrick skin type classification (I-VI)"""
    TYPE_I = "I"      # Very fair, always burns, never tans
    TYPE_II = "II"    # Fair, usually burns, tans minimally
    TYPE_III = "III"  # Medium, sometimes burns, tans uniformly
    TYPE_IV = "IV"    # Olive, rarely burns, tans easily
    TYPE_V = "V"      # Brown, very rarely burns, tans very easily
    TYPE_VI = "VI"    # Dark brown/black, never burns


class LesionDuration(str):
    """How long the lesion has been present"""
    NEW = "new"                    # Less than 1 month
    RECENT = "recent"              # 1-6 months
    MONTHS = "months"              # 6-12 months
    ONE_YEAR = "one_year"          # 1-2 years
    YEARS = "years"                # 2-5 years
    LONG_TERM = "long_term"        # More than 5 years
    UNKNOWN = "unknown"            # Patient doesn't know


class ABCDEChanges(BaseModel):
    """ABCDE criteria changes - Evolution tracking"""
    asymmetry_changed: Optional[bool] = None       # A - Shape becoming asymmetric
    border_changed: Optional[bool] = None          # B - Border becoming irregular
    color_changed: Optional[bool] = None           # C - Color changes (multiple colors, darkening)
    diameter_changed: Optional[bool] = None        # D - Growing larger (>6mm is concerning)
    evolving: Optional[bool] = None                # E - Any noticeable evolution


class LesionSymptoms(BaseModel):
    """Symptoms associated with the lesion"""
    itching: Optional[bool] = None
    bleeding: Optional[bool] = None
    pain: Optional[bool] = None
    crusting: Optional[bool] = None
    oozing: Optional[bool] = None
    ulceration: Optional[bool] = None


class ClinicalContext(BaseModel):
    """
    Clinical context information collected before analysis.
    Used to calculate Bayesian priors for risk adjustment.
    """
    # Patient Demographics
    patient_age: Optional[int] = None                          # Age in years
    fitzpatrick_skin_type: Optional[str] = None                # I, II, III, IV, V, or VI

    # Lesion History
    lesion_duration: Optional[str] = None                      # How long present
    is_new_lesion: Optional[bool] = None                       # Appeared recently?
    has_changed_recently: Optional[bool] = None                # Changed in past 3 months?

    # ABCDE Evolution Criteria
    abcde_changes: Optional[ABCDEChanges] = None

    # Location
    body_location: Optional[str] = None                        # General location
    is_sun_exposed_area: Optional[bool] = None                 # Chronically sun exposed?
    is_high_risk_location: Optional[bool] = None               # Palm, sole, nail, mucosa?

    # Symptoms
    symptoms: Optional[LesionSymptoms] = None

    # Medical History
    personal_history_melanoma: Optional[bool] = None           # Previous melanoma?
    personal_history_skin_cancer: Optional[bool] = None        # Previous BCC/SCC?
    personal_history_atypical_moles: Optional[bool] = None     # Dysplastic nevi?
    family_history_melanoma: Optional[bool] = None             # First-degree relative?
    family_history_skin_cancer: Optional[bool] = None          # Family BCC/SCC?

    # Risk Factors
    history_severe_sunburns: Optional[bool] = None             # Blistering sunburns?
    uses_tanning_beds: Optional[bool] = None                   # Indoor tanning?
    immunosuppressed: Optional[bool] = None                    # Transplant, HIV, etc?
    many_moles: Optional[bool] = None                          # >50 moles?

    # Additional Notes
    patient_concerns: Optional[str] = None                     # What worries the patient?
    additional_notes: Optional[str] = None                     # Other relevant info


class ClinicalContextResponse(ClinicalContext):
    """Response model with calculated risk factors"""
    calculated_risk_multiplier: Optional[float] = None
    risk_factors_identified: Optional[List[str]] = None
    risk_level: Optional[str] = None  # low, moderate, high, very_high


# ============================================================================
# Hybrid User Role System Models
# ============================================================================

class UserUpdate(BaseModel):
    """Update user account settings"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    display_mode: Optional[str] = None  # "simple" or "professional"
    account_type: Optional[str] = None  # "personal" or "professional"


class ProfessionalVerificationRequest(BaseModel):
    """Request to verify professional credentials"""
    license_number: str
    license_state: str
    npi_number: Optional[str] = None
    specialty: Optional[str] = None
    verification_documents: Optional[List[str]] = None  # URLs to uploaded docs


class ProfessionalVerificationResponse(BaseModel):
    """Response for professional verification status"""
    is_verified_professional: bool
    professional_license_number: Optional[str] = None
    professional_license_state: Optional[str] = None
    npi_number: Optional[str] = None
    verification_date: Optional[datetime] = None
    verification_status: str  # "pending", "verified", "rejected"


# ============================================================================
# Clinic Models
# ============================================================================

class ClinicBase(BaseModel):
    """Base clinic model"""
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    description: Optional[str] = None
    specialty: Optional[str] = None
    website: Optional[str] = None


class ClinicCreate(ClinicBase):
    """Create a new clinic"""
    pass


class ClinicUpdate(BaseModel):
    """Update clinic information"""
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    description: Optional[str] = None
    specialty: Optional[str] = None
    website: Optional[str] = None
    is_active: Optional[bool] = None


class ClinicResponse(ClinicBase):
    """Clinic response with all details"""
    id: int
    clinic_code: str
    is_verified: bool
    is_active: bool
    created_by_user_id: int
    created_at: datetime
    updated_at: datetime
    staff_count: Optional[int] = None
    patient_count: Optional[int] = None

    class Config:
        from_attributes = True


class ClinicListResponse(BaseModel):
    """Paginated list of clinics"""
    clinics: List[ClinicResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Clinic Staff Models
# ============================================================================

class ClinicStaffBase(BaseModel):
    """Base clinic staff model"""
    role: str  # "admin", "provider", "staff"
    title: Optional[str] = None
    department: Optional[str] = None


class ClinicStaffCreate(ClinicStaffBase):
    """Add staff member to clinic"""
    user_id: int


class ClinicStaffInvite(BaseModel):
    """Invite a user to join clinic as staff"""
    email: EmailStr
    role: str  # "admin", "provider", "staff"
    title: Optional[str] = None
    message: Optional[str] = None


class ClinicStaffUpdate(BaseModel):
    """Update staff member's role/info"""
    role: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None


class ClinicStaffResponse(ClinicStaffBase):
    """Staff member response"""
    id: int
    clinic_id: int
    user_id: int
    is_active: bool
    joined_at: datetime
    # User details
    user_name: Optional[str] = None
    user_email: Optional[str] = None
    is_verified_professional: Optional[bool] = None

    class Config:
        from_attributes = True


# ============================================================================
# Clinic Patient Models
# ============================================================================

class ConsentLevel(str):
    """Consent levels for patient data sharing"""
    FULL = "full"           # Full access to all analyses and history
    LIMITED = "limited"     # Access to shared analyses only
    VISIT_ONLY = "visit_only"  # Access only during active visit
    REVOKED = "revoked"     # Access revoked


class ClinicPatientBase(BaseModel):
    """Base clinic patient model"""
    consent_level: str = "limited"  # full, limited, visit_only
    patient_notes: Optional[str] = None


class ClinicPatientCreate(ClinicPatientBase):
    """Link patient to clinic"""
    patient_user_id: int


class ClinicPatientUpdate(BaseModel):
    """Update patient-clinic relationship"""
    consent_level: Optional[str] = None
    patient_notes: Optional[str] = None
    is_active: Optional[bool] = None


class ClinicPatientResponse(ClinicPatientBase):
    """Patient-clinic relationship response"""
    id: int
    clinic_id: int
    patient_user_id: int
    is_active: bool
    linked_at: datetime
    updated_at: datetime
    # Patient details (only visible based on consent)
    patient_name: Optional[str] = None
    patient_email: Optional[str] = None
    total_shared_analyses: Optional[int] = None
    last_visit: Optional[datetime] = None

    class Config:
        from_attributes = True


class PatientClinicListResponse(BaseModel):
    """List of clinics a patient is linked to"""
    clinics: List[ClinicResponse]
    total: int


# ============================================================================
# Clinic Invitation Models
# ============================================================================

class ClinicInvitationCreate(BaseModel):
    """Create invitation for patient to join clinic"""
    patient_email: EmailStr
    consent_level: str = "limited"
    message: Optional[str] = None
    expires_in_days: int = 7


class ClinicInvitationResponse(BaseModel):
    """Invitation response"""
    id: int
    clinic_id: int
    clinic_name: str
    invitation_code: str
    patient_email: str
    consent_level: str
    message: Optional[str] = None
    status: str  # "pending", "accepted", "declined", "expired"
    created_at: datetime
    expires_at: datetime
    responded_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AcceptInvitationRequest(BaseModel):
    """Accept or decline invitation"""
    invitation_code: str
    accept: bool
    consent_level: Optional[str] = None  # Can override suggested consent level


# ============================================================================
# Shared Analysis Models
# ============================================================================

class SharedAnalysisCreate(BaseModel):
    """Share an analysis with a clinic"""
    analysis_id: int
    clinic_id: int
    share_notes: Optional[str] = None


class SharedAnalysisResponse(BaseModel):
    """Shared analysis response"""
    id: int
    analysis_id: int
    clinic_id: int
    patient_user_id: int
    shared_by_user_id: int
    share_notes: Optional[str] = None
    shared_at: datetime
    viewed_at: Optional[datetime] = None
    viewed_by_user_id: Optional[int] = None
    # Analysis summary
    analysis_type: Optional[str] = None
    predicted_class: Optional[str] = None
    risk_level: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class BulkShareRequest(BaseModel):
    """Share multiple analyses with a clinic"""
    analysis_ids: List[int]
    clinic_id: int
    share_notes: Optional[str] = None


# ============================================================================
# Clinic Note Models
# ============================================================================

class ClinicNoteBase(BaseModel):
    """Base clinic note model"""
    note_type: str  # "general", "diagnosis", "treatment", "follow_up"
    note_content: str
    is_private: bool = False  # Private notes not visible to patient


class ClinicNoteCreate(ClinicNoteBase):
    """Create a clinical note"""
    patient_user_id: int
    related_analysis_id: Optional[int] = None


class ClinicNoteUpdate(BaseModel):
    """Update a clinical note"""
    note_content: Optional[str] = None
    note_type: Optional[str] = None
    is_private: Optional[bool] = None


class ClinicNoteResponse(ClinicNoteBase):
    """Clinical note response"""
    id: int
    clinic_id: int
    patient_user_id: int
    provider_user_id: int
    related_analysis_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    # Provider info
    provider_name: Optional[str] = None
    provider_title: Optional[str] = None

    class Config:
        from_attributes = True


# ============================================================================
# Dashboard and Summary Models
# ============================================================================

class ClinicDashboardStats(BaseModel):
    """Clinic dashboard statistics"""
    total_patients: int
    active_patients: int
    total_analyses_shared: int
    analyses_this_month: int
    pending_reviews: int
    staff_count: int
    recent_activity: List[Dict[str, Any]]


class PatientSummaryForClinic(BaseModel):
    """Patient summary as seen by clinic"""
    patient_id: int
    patient_name: str
    consent_level: str
    linked_since: datetime
    last_visit: Optional[datetime] = None
    total_shared_analyses: int
    recent_analyses: List[SharedAnalysisResponse]
    notes_count: int
    risk_flags: List[str]  # High-risk findings


class ProviderDashboard(BaseModel):
    """Dashboard for a provider"""
    clinics: List[ClinicResponse]
    total_patients_across_clinics: int
    pending_reviews: int
    recent_shared_analyses: List[SharedAnalysisResponse]
    upcoming_follow_ups: List[Dict[str, Any]]


class UserRoleSummary(BaseModel):
    """Summary of user's roles and access"""
    user_id: int
    account_type: str  # "personal" or "professional"
    display_mode: str  # "simple" or "professional"
    is_verified_professional: bool
    clinics_as_staff: List[ClinicResponse]
    clinics_as_patient: List[ClinicResponse]
    pending_invitations: List[ClinicInvitationResponse]